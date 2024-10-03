# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) Stanford University

import importlib.util
import pickle
import socket
import threading
import time
from datetime import datetime
import torch
import numpy as np
from fairmotion.ops import conversions
from pygame.time import Clock

from real_time_runner import RTRunner
from simple_transformer_with_state import TF_RNN_Past_State
from render_funcs import init_viz, update_height_field_pb, COLOR_OURS
# make deterministic
from learning_utils import set_seed
import constants as cst
import xsensdeviceapi as xda #for windows only
import keyboard
from collections import deque
from threading import Lock
import sys

set_seed(1234567)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
running = False
is_recording = True     # always record imu every 15 sec
record_buffer = None
num_imus = 6
num_float_one_frame = num_imus * 7      # sent from Xsens
FREQ = int(1. / cst.DT)

color = COLOR_OURS

# model_name = "output/model-new-v0-2.pt"
model_name = "output\model-with-dip9and10.pt"
USE_5_SBP = True
WITH_ACC_SUM = True
MULTI_SBP_CORRECTION = False
VIZ_H_MAP = True
MAX_ACC = 10.0

init_grid_np = np.random.uniform(-100.0, 100.0, (cst.GRID_NUM, cst.GRID_NUM))
init_grid_list = list(init_grid_np.flatten())

input_channels_imu = 6 * (9 + 3)
if USE_5_SBP:
    output_channels = 18 * 6 + 3 + 20
else:
    output_channels = 18 * 6 + 3 + 8

# make an aligned T pose, such that front is x, left is y, and up is z (i.e. without heading)
# the IMU sensor at head will be placed the same way, so we can get the T pose's heading (wrt ENU) easily
# the following are the known bone orientations at such a T pose
Rs_aligned_T_pose = np.array([
    1.0, 0, 0, 0, 0, -1, 0, 1, 0,
    1.0, 0, 0, 0, 0, -1, 0, 1, 0,
    1.0, 0, 0, 0, 0, -1, 0, 1, 0,
    1.0, 0, 0, 0, 0, -1, 0, 1, 0,
    1.0, 0, 0, 0, 0, -1, 0, 1, 0,
    1.0, 0, 0, 0, 0, -1, 0, 1, 0,
])
Rs_aligned_T_pose = Rs_aligned_T_pose.reshape((6, 3, 3))
Rs_aligned_T_pose = \
    np.einsum('ij,njk->nik', conversions.A2R(np.array([0, 0, np.pi/2])), Rs_aligned_T_pose)
print(Rs_aligned_T_pose)

# the state at the T pose, dq not necessary actually and will not be used either
s_init_T_pose = np.zeros(cst.n_dofs * 2)
s_init_T_pose[2] = 0.85
s_init_T_pose[3:6] = np.array([1.20919958, 1.20919958, 1.20919958])

class XsPortInfoStr:
    def __str__(self, p):
        return f"Port: {p.portNumber():>2} ({p.portName()}) @ {p.baudrate():>7} Bd, ID: {p.deviceId().toString()}"

class XsDeviceStr:
    def __str__(self, d):
        return f"ID: {d.deviceId().toString()} ({d.productCode()})"

def find_closest_update_rate(supported_update_rates, desired_update_rate):
    if not supported_update_rates:
        return 0

    if len(supported_update_rates) == 1:
        return supported_update_rates[0]

    closest_update_rate = min(supported_update_rates, key=lambda x: abs(x - desired_update_rate))
    return closest_update_rate

class WirelessMasterCallback(xda.XsCallback):
    def __init__(self):
        super().__init__()
        self.m_connectedMTWs = set()
        self.m_mutex = Lock()

    def getWirelessMTWs(self):
        with self.m_mutex:
            return self.m_connectedMTWs.copy()

    def onConnectivityChanged(self, dev, newState):
        with self.m_mutex:
            if newState == xda.XCS_Disconnected:
                print(f"\nEVENT: MTW Disconnected -> {dev.deviceId()}")
                self.m_connectedMTWs.discard(dev)
            elif newState == xda.XCS_Rejected:
                print(f"\nEVENT: MTW Rejected -> {dev.deviceId()}")
                self.m_connectedMTWs.discard(dev)
            elif newState == xda.XCS_PluggedIn:
                print(f"\nEVENT: MTW PluggedIn -> {dev.deviceId()}")
                self.m_connectedMTWs.discard(dev)
            elif newState == xda.XCS_Wireless:
                print(f"\nEVENT: MTW Connected -> {dev.deviceId()}")
                self.m_connectedMTWs.add(dev)
            elif newState == xda.XCS_File:
                print(f"\nEVENT: MTW File -> {dev.deviceId()}")
                self.m_connectedMTWs.discard(dev)
            elif newState == xda.XCS_Unknown:
                print(f"\nEVENT: MTW Unknown -> {dev.deviceId()}")
                self.m_connectedMTWs.discard(dev)
            else:
                print(f"\nEVENT: MTW Error -> {dev.deviceId()}")
                self.m_connectedMTWs.discard(dev)

class MtwCallback(xda.XsCallback):
    def __init__(self, mtwIndex, device):
        super().__init__()
        self.m_packetBuffer = deque(maxlen=300)
        self.m_mutex = Lock()
        self.m_mtwIndex = mtwIndex
        self.m_device = device

    def dataAvailable(self):
        with self.m_mutex:
            return bool(self.m_packetBuffer)

    def getOldestPacket(self):
        with self.m_mutex:
            packet = self.m_packetBuffer[0]
            return packet

    def deleteOldestPacket(self):
        with self.m_mutex:
            self.m_packetBuffer.popleft()

    def getMtwIndex(self):
        return self.m_mtwIndex

    def device(self):
        assert self.m_device is not None
        return self.m_device

    def onLiveDataAvailable(self, _, packet):
        with self.m_mutex:
            # NOTE: Processing of packets should not be done in this thread.
            self.m_packetBuffer.append(packet)
            if len(self.m_packetBuffer) > 300:
                self.deleteOldestPacket()

# Based from TransPose github repo
class IMUSet:
    def __init__(self, desired_update_rate = 80, desired_radio_channel=19):
        self.wireless_master_callback = WirelessMasterCallback()
        self.mtw_callbacks = []
        self.clock = Clock()

        self._is_reading = False
        self._read_thread = None

        self.current_reading = None
        self.counter = 0
        print("Constructing XsControl...")
        self.control = xda.XsControl.construct()
        if self.control is None:
            print("Failed to construct XsControl instance.")
            sys.exit(1)
        try:
            print("Scanning ports...")

            detected_devices = xda.XsScanner_scanPorts()

            print("Finding wireless master...")
            wireless_master_port = next((port for port in detected_devices if port.deviceId().isWirelessMaster()), None)
            if wireless_master_port is None:
                raise RuntimeError("No wireless masters found")

            print(f"Wireless master found @ {wireless_master_port}")

            print("Opening port...")
            if not self.control.openPort(wireless_master_port.portName(), wireless_master_port.baudrate()):
                raise RuntimeError(f"Failed to open port {wireless_master_port}")

            print("Getting XsDevice instance for wireless master...")
            self.wireless_master_device = self.control.device(wireless_master_port.deviceId())
            if self.wireless_master_device is None:
                raise RuntimeError(f"Failed to construct XsDevice instance: {wireless_master_port}")

            print(f"XsDevice instance created @ {self.wireless_master_device}")

            print("Setting config mode...")
            if not self.wireless_master_device.gotoConfig():
                raise RuntimeError(f"Failed to goto config mode: {self.wireless_master_device}")

            print("Attaching callback handler...")
            self.wireless_master_device.addCallbackHandler(self.wireless_master_callback)

            print("Getting the list of the supported update rates...")
            supportUpdateRates = xda.XsDevice.supportedUpdateRates(self.wireless_master_device, xda.XDI_None)

            print("Supported update rates: ", end="")
            for rate in supportUpdateRates:
                print(rate, end=" ")
            print()

            new_update_rate = find_closest_update_rate(supportUpdateRates, desired_update_rate)

            print(f"Setting update rate to {new_update_rate} Hz...")

            if not self.wireless_master_device.setUpdateRate(new_update_rate):
                raise RuntimeError(f"Failed to set update rate: {self.wireless_master_device}")

            print("Disabling radio channel if previously enabled...")

            if self.wireless_master_device.isRadioEnabled():
                if not self.wireless_master_device.disableRadio():
                    raise RuntimeError(f"Failed to disable radio channel: {self.wireless_master_device}")

            print(f"Setting radio channel to {desired_radio_channel} and enabling radio...")
            if not self.wireless_master_device.enableRadio(desired_radio_channel):
                raise RuntimeError(f"Failed to set radio channel: {self.wireless_master_device}")

            print("Waiting for MTW to wirelessly connect...\n")
        except Exception as ex:
            print(ex)
            print("****ABORT****")
            self.shutdown_mtw()
        except:
            print("An unknown fatal error has occurred. Aborting.")
            print("****ABORT****")
            self.shutdown_mtw()     

    def connect_MTw(self):
        wait_for_connections = True
        connected_mtw_count = len(self.wireless_master_callback.getWirelessMTWs())
        while connected_mtw_count < 6:
            time.sleep(0.1)
            next_count = len(self.wireless_master_callback.getWirelessMTWs())
            if next_count != connected_mtw_count:
                print(f"Number of connected MTWs: {next_count}. Press 's' to start measurement.")
                connected_mtw_count = next_count
            # wait_for_connections = not keyboard.is_pressed('s')

    def start_measurement(self):
        try:
            print("Starting measurement...")
            if not self.wireless_master_device.gotoMeasurement():
                raise RuntimeError(f"Failed to goto measurement mode: {self.wireless_master_device}")

            print("Getting XsDevice instances for all MTWs...")
            all_device_ids = self.control.deviceIds()
            # All device IDs are sorted in order
            mtw_device_ids = sorted([device_id for device_id in all_device_ids if device_id.isMtw()])
            print("Device IDs", [str(i) for i in mtw_device_ids])
            mtw_devices = []
            for device_id in mtw_device_ids:
                mtw_device = self.control.device(device_id)
                if mtw_device is not None:
                    mtw_devices.append(mtw_device)
                else:
                    raise RuntimeError("Failed to create an MTW XsDevice instance")

            print("Attaching callback handlers to MTWs...")
            mtw_callbacks = [MtwCallback(i, mtw_devices[i]) for i in range(len(mtw_devices))]
            for i in range(len(mtw_devices)):
                mtw_devices[i].addCallbackHandler(mtw_callbacks[i])

            # print("Creating a log file...")
            # logFileName = "logfile.mtb"
            # if self.wireless_master_device.createLogFile(logFileName) != xda.XRV_OK:
            #     raise RuntimeError("Failed to create a log file. Aborting.")
            # else:
            #     print("Created a log file: %s" % logFileName)

            print("Starting recording...")
            ready_to_record = False

            while not ready_to_record:
                ready_to_record = all([mtw_callbacks[i].dataAvailable() for i in range(len(mtw_callbacks))])
                if not ready_to_record:
                    print("Waiting for data available...")
                    time.sleep(0.5)
                #     optional, enable heading reset before recording data, make sure all sensors have aligned physically the same heading!!
                # else:
                #     print("Do heading reset before recording data, make sure all sensors have aligned physically the same heading!!")
                #     all([mtw_devices[i].resetOrientation(xda.XRM_Heading) for i in range(len(mtw_callbacks))])

            if not self.wireless_master_device.startRecording():
                raise RuntimeError("Failed to start recording. Aborting.")

            print("\nMain loop. Press any key to quit\n")
            print("Waiting for data available...")

            # euler_data = [xda.XsEuler()] * len(mtw_callbacks)
            quat_data = np.zeros((len(mtw_callbacks), 4))
            acc_data = np.zeros((len(mtw_callbacks), 3))
            gyro_data = np.zeros((len(mtw_callbacks), 3))
            mag_data = np.zeros((len(mtw_callbacks), 3))
            print_counter = 0
            while self._is_reading:
                new_data_available = False
                for i in range(len(mtw_callbacks)):
                    if mtw_callbacks[i].dataAvailable():
                        new_data_available = True
                        packet = mtw_callbacks[i].getOldestPacket()
                        # euler_data[i] = packet.orientationEuler()
                        q = xda.XsQuaternion(packet.orientationQuaternion())
                        quat_data[i] = np.array([q.x(),q.y(), q.z(), q.w()]) # np.array(x,y,z,w)
                        acc_data[i] = packet.calibratedAcceleration() # np.array
                        gyro_data[i] = packet.calibratedGyroscopeData()
                        mag_data[i] = packet.calibratedMagneticField()
                        mtw_callbacks[i].deleteOldestPacket()
                if new_data_available:
                    R_s_gn = conversions.Q2R(quat_data)
                    a_s = acc_data
                    # need to do acc offset elsewhere.
                    # a_s_g = np.einsum('ijk,ik->ij', R_s_g, a_s)
                    # # probably doesn't matter, will be taken care by acc offset calibration as well.
                    # a_s_g += np.array([0., 0., -9.8])

                    # if self.counter % 25 == 0:
                    #     print('\n' + str(q_s[0, :]) + str(a_s_g[0, :]))
                    self.counter += 1
                    # everything in global (ENU) frame
                    self.current_reading = np.concatenate((R_s_gn.reshape(-1), a_s.reshape(-1)))
                    # print only 1/x of the data in the screen.
                    # if print_counter % 300 == 0:
                    #     for i in range(len(mtw_callbacks)):
                    #         print(f"[{i}]: ID: {mtw_callbacks[i].device().deviceId()}, "
                    #             f"Acc: {acc_data[i]}",
                    #             f"Gyro: {gyro_data[i]}",
                    #             f"Quat: {quat_data[i]}"
                    #         )
                    print_counter += 1
                self.clock.tick(60)
            print("Setting config mode...")
            if not self.wireless_master_device.gotoConfig():
                raise RuntimeError(f"Failed to goto config mode: {self.wireless_master_device}")

            print("Disabling radio...")
            if not self.wireless_master_device.disableRadio():
                raise RuntimeError(f"Failed to disable radio: {self.wireless_master_device}")
    
        except Exception as ex:
            print(ex)
            print("****ABORT****")
            self.shutdown_mtw()
        except:
            print("An unknown fatal error has occurred. Aborting.")
            print("****ABORT****")
            self.shutdown_mtw()
    def start_reading_thread(self):
        """
        Start reading imu measurements into the buffer.
        """
        if self._read_thread is None:
            self._is_reading = True
            self._read_thread = threading.Thread(target=self.start_measurement)
            self._read_thread.setDaemon(True)
            self._read_thread.start()
        else:
            print('Failed to start reading thread: reading is already start.')
    def shutdown_mtw(self):
        print("Closing XsControl...")
        self.control.close()
        print("Deleting mtw callbacks...")
        print("Successful exit.")
    def stop_reading(self):
        """
        Stop reading imu measurements.
        """
        print("Stopping reading...")
        if self._read_thread is not None:
            self._is_reading = False
            self._read_thread.join()
            self._read_thread = None
            self.shutdown_mtw()


def get_input():
    global running
    while running:
        c = input()
        if c == 'q':
            running = False


def get_mean_readings_3_sec():
    counter = 0
    mean_buffer = []
    while counter <= FREQ * 3:
        clock.tick(FREQ)
        mean_buffer.append(imu_set.current_reading.copy())
        counter += 1

    return np.array(mean_buffer).mean(axis=0)


def get_transformed_current_reading():
    R_and_acc_t = imu_set.current_reading.copy()

    R_Gn_St = R_and_acc_t[: 6*9].reshape((6, 3, 3))
    acc_St = R_and_acc_t[6*9:].reshape((6, 3))

    R_Gp_St = np.einsum('nij,njk->nik', R_Gn_Gp.transpose((0, 2, 1)), R_Gn_St)
    R_Gp_Bt = np.einsum('nij,njk->nik', R_Gp_St, R_B0_S0.transpose((0, 2, 1)))

    acc_Gp = np.einsum('ijk,ik->ij', R_Gp_St, acc_St)
    acc_Gp = acc_Gp - acc_offset_Gp

    acc_Gp = np.clip(acc_Gp, -MAX_ACC, MAX_ACC)

    return np.concatenate((R_Gp_Bt.reshape(-1), acc_Gp.reshape(-1)))


def viz_point(x, ind):
    pb_c.resetBasePositionAndOrientation(
        p_vids[ind],
        x,
        [0., 0, 0, 1]
    )


if __name__ == '__main__':
    imu_set = IMUSet()
    imu_set.connect_MTw()
    ''' Load Character Info Moudle '''
    spec = importlib.util.spec_from_file_location(
        "char_info", "amass_char_info.py")
    char_info = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(char_info)

    pb_c, c1, _, p_vids, h_id, h_b_id = init_viz(char_info,
                                                 init_grid_list,
                                                 viz_h_map=VIZ_H_MAP,
                                                 hmap_scale=cst.GRID_SIZE,
                                                 gui=True,
                                                 compare_gt=False)

    model = TF_RNN_Past_State(
        input_channels_imu, output_channels,
        rnn_hid_size=512,
        tf_hid_size=1024, tf_in_dim=256,
        n_heads=16, tf_layers=4,
        dropout=0.0, in_dropout=0.0,
        past_state_dropout=0.8,
        with_acc_sum=WITH_ACC_SUM,
    )
    model.load_state_dict(torch.load(model_name))
    model = model.cuda()

    clock = Clock()
    imu_set.start_reading_thread()
    time.sleep(10)
    input('Put all imus aligned with your body reference frame and then press any key.')
    print('Keep for 3 seconds ...', end='')

    # calibration: heading reset
    R_and_acc_mean = get_mean_readings_3_sec()

    # R_head = R_and_acc_mean[5*9: 6*9].reshape(3, 3)     # last sensor being head
    R_Gn_Gp = R_and_acc_mean[:6*9].reshape((6, 3, 3))
    # calibration: acceleration offset
    acc_offset_Gp = R_and_acc_mean[6*9:].reshape(6, 3)      # sensor frame (S) and room frame (Gp) align during this

    # R_head = np.array([[0.5,  0.866,  0.0],
    # [-0.866,  0.5,    0.0],
    # [ 0.0,  -0.0,  1.0]])

    # this should be pretty much just z rotation (i.e. only heading)
    # might be different for different sensors...
    print(R_Gn_Gp)

    input('\nWear all imus correctly and press any key.')
    for i in range(12, 0, -1):
        print('\rStand straight in T-pose and be ready. The calibration will begin after %d seconds.' % i, end='')
        time.sleep(1)
    print('\rStand straight in T-pose. Keep the pose for 3 seconds ...', end='')

    # calibration: bone-to-sensor transform
    R_and_acc_mean = get_mean_readings_3_sec()

    R_Gn_S0 = R_and_acc_mean[: 6 * 9].reshape((6, 3, 3))
    R_Gp_B0 = Rs_aligned_T_pose
    R_Gp_S0 = np.einsum('nij,njk->nik', R_Gn_Gp.transpose((0, 2, 1)), R_Gn_S0)
    R_B0_S0 = np.einsum('nij,njk->nik', R_Gp_B0.transpose((0, 2, 1)), R_Gp_S0)

    # # rotate init T pose according to heading reset results
    # nominal_root_R = conversions.A2R(s_init_T_pose[3:6])
    # root_R_init = R_head.dot(nominal_root_R)
    # s_init_T_pose[3:6] = conversions.R2A(root_R_init)

    # use real time runner with online data
    rt_runner = RTRunner(
        c1, model, 40, s_init_T_pose,
        map_bound=cst.MAP_BOUND,
        grid_size=cst.GRID_SIZE,
        play_back_gt=False,
        five_sbp=USE_5_SBP,
        with_acc_sum=WITH_ACC_SUM,
        multi_sbp_terrain_and_correction=MULTI_SBP_CORRECTION,
    )
    last_root_pos = s_init_T_pose[:3]     # assume always start from (0,0,0.9)

    print('\tFinish.\nStart estimating poses. Press q to quit')

    running = True

    get_input_thread = threading.Thread(target=get_input)
    get_input_thread.setDaemon(True)
    get_input_thread.start()

    RB_and_acc_t = get_transformed_current_reading()
    # rt_runner.record_raw_imu(RB_and_acc_t)
    if is_recording:
        record_buffer = RB_and_acc_t.reshape(1, -1)
    t = 1

    while running:
        RB_and_acc_t = get_transformed_current_reading()

        # t does not matter, not used
        res = rt_runner.step(RB_and_acc_t, last_root_pos, s_gt=None, c_gt=None, t=t)

        last_root_pos = res['qdq'][:3]

        viz_locs = res['viz_locs']
        for sbp_i in range(viz_locs.shape[0]):
            viz_point(viz_locs[sbp_i, :], sbp_i)

        if t % 15 == 0 and h_id is not None:
            # TODO: double for loop...
            for ii in range(init_grid_np.shape[0]):
                for jj in range(init_grid_np.shape[1]):
                    init_grid_list[jj * init_grid_np.shape[0] + ii] = \
                        rt_runner.region_height_list[rt_runner.height_region_map[ii, jj]]
            h_id, h_b_id = update_height_field_pb(
                pb_c,
                h_data=init_grid_list,
                scale=cst.GRID_SIZE,
                terrainShape=h_id,
                terrain=h_b_id
            )

        clock.tick(FREQ)

        # print('\r', R_G_Bt.reshape(6,9), acc_G_t, end='')

        t += 1
        # recording
        if is_recording:
            record_buffer = np.concatenate([record_buffer, RB_and_acc_t.reshape(1, -1)], axis=0)

            if t % (FREQ * 15) == 0:
                with open('./imu_recordings/r' + datetime.now().strftime('%m:%d:%T').replace(':', '-') + '.pkl',
                          "wb") as handle:
                    pickle.dump(
                        {"imu": record_buffer, "qdq_init": s_init_T_pose},
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL
                    )

    get_input_thread.join()
    imu_set.stop_reading()
    print('Finish.')
