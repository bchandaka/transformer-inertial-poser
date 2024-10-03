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
import sys, os, json

sys.path.append(os.path.dirname(__file__))
from real_time_runner import RTRunner
from simple_transformer_with_state import TF_RNN_Past_State
from amass_char_info import nimble2smpl_ind, SMPL_JOINT_NAMES, smpl18rsmpl_ind
from render_funcs import init_viz, update_height_field_pb, COLOR_OURS
# make deterministic
from learning_utils import set_seed
import constants as cst
import xsensdeviceapi as xda #for windows only
import keyboard
from collections import deque
from threading import Lock
import smplx
import rerun as rr
from rerun import Material
from scipy.spatial.transform import Rotation as R
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from ih_utils import vis, math, constants
import json
set_seed(1234567)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
running = False
is_recording = True     # always record imu every 15 sec
record_buffer = None
num_imus = 6
num_float_one_frame = num_imus * 7      # sent from Xsens
FREQ = int(1. / cst.DT)

color = COLOR_OURS

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
# s_init_T_pose[3:6] = np.array([3.141592653589793, 0, 0])

def get_input():
    global running
    while running:
        c = input()
        if c == 'q':
            running = False

def get_transformed_current_reading(current_reading, R_Gn_Gp, acc_offset_Gp, R_B0_S0):
    R_and_acc_t = current_reading.copy()

    R_Gn_St = R_and_acc_t[: 6*9].reshape((6, 3, 3))
    acc_St = R_and_acc_t[6*9:].reshape((6, 3))

    R_Gp_St = np.einsum('nij,njk->nik', R_Gn_Gp.transpose((0, 2, 1)), R_Gn_St)
    R_Gp_Bt = np.einsum('nij,njk->nik', R_Gp_St, R_B0_S0.transpose((0, 2, 1)))

    acc_Gp = np.einsum('ijk,ik->ij', R_Gp_St, acc_St)
    acc_Gp = acc_Gp - acc_offset_Gp

    acc_Gp = np.clip(acc_Gp, -MAX_ACC, MAX_ACC)

    return np.concatenate((R_Gp_Bt.reshape(-1), acc_Gp.reshape(-1)))


def viz_point(x, ind, pb_c, p_vids):
    pb_c.resetBasePositionAndOrientation(
        p_vids[ind],
        x,
        [0., 0, 0, 1]
    )
def tip_q2smpl(qdq):
    rotate_frame = R.from_matrix(constants.RH_Z_UP_TO_RH_Y_DOWN)
    global_orient = np.expand_dims((rotate_frame * R.from_rotvec(qdq[3:6])).as_rotvec(), axis=0).astype(np.float32)
    transl = np.expand_dims(rotate_frame.as_matrix() @ qdq[:3], axis=0).astype(np.float32)
    
    q = qdq[3:cst.n_dofs] # including root orientation, 54 length
    smpl_joints = np.zeros((1, 24*3)).astype(np.float32) # not including root
    for c in range(0,18): # fill in all 18 joints including root
        smpl_ind = nimble2smpl_ind[c]*3
        nimble_ind = c*3
        smpl_joints[0, smpl_ind:smpl_ind+3] = q[nimble_ind:nimble_ind+3]
    smpl_joints = np.expand_dims(smpl_joints[0, 3:],axis=0) # remove root joint, size (1,23*3)
    return smpl_joints, global_orient, transl

def run_tip(awinda_dir, smpl_out_file, visualize=False):
    tip_dir = os.path.dirname(__file__)
    os.chdir(tip_dir)

    imu_human_dir = os.path.dirname(tip_dir)
    model_n = smplx.create(os.path.join(imu_human_dir, "body_models", "SMPL_NEUTRAL.pkl"), model_type="smpl", gender="neutral")
    if visualize:
        rr.init("offline_tip",spawn=True)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, timeless=True) # world origin is camera coordinate frame
    # rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, timeless=True) # world origin is camera coordinate frame
    recorded_awinda_dir = awinda_dir
    current_readings = np.loadtxt(os.path.join(recorded_awinda_dir, 'current_reading.txt'))
    reading_t_ns = np.loadtxt(os.path.join(recorded_awinda_dir, 'reading_t_ns.txt'))
    calib_times = json.load(open(os.path.join(recorded_awinda_dir, 'calib', 'calib_time.json')))
    t_pose_start = calib_times['t_pose_start_time']
    start_ind = np.where(reading_t_ns >= t_pose_start)[0][0]
    print('Start index:', start_ind)
    reading_t_ns = reading_t_ns[start_ind:]
    current_readings = current_readings[start_ind:]
    align_R_and_acc_mean = np.loadtxt(os.path.join(recorded_awinda_dir, 'calib', 'align_R_and_acc_mean.txt'))
    t_pose_R_and_acc_mean = np.loadtxt(os.path.join(recorded_awinda_dir, 'calib', 't_pose_R_and_acc_mean.txt'))
    ''' Load Character Info Moudle '''
    spec = importlib.util.spec_from_file_location(
        os.path.join(tip_dir, "char_info"), "amass_char_info.py")
    char_info = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(char_info)
    pb_c, c1, _, p_vids, h_id, h_b_id = init_viz(char_info,
                                                 init_grid_list,
                                                 viz_h_map=VIZ_H_MAP,
                                                 hmap_scale=cst.GRID_SIZE,
                                                 gui=True,
                                                 compare_gt=False)

    model_name = os.path.join(tip_dir, "output\model-with-dip9and10.pt")

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
    # imu_set.start_reading_thread()
    # time.sleep(10)
    # input('Put all imus aligned with your body reference frame and then press any key.')
    # print('Keep for 3 seconds ...', end='')

    # calibration: heading reset
    R_and_acc_mean = align_R_and_acc_mean #get_mean_readings_3_sec()

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

    # input('\nWear all imus correctly and press any key.')
    # for i in range(12, 0, -1):
    #     print('\rStand straight in T-pose and be ready. The calibration will begin after %d seconds.' % i, end='')
    #     time.sleep(1)
    # print('\rStand straight in T-pose. Keep the pose for 3 seconds ...', end='')

    # calibration: bone-to-sensor transform
    R_and_acc_mean = t_pose_R_and_acc_mean #get_mean_readings_3_sec()

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

    # get_input_thread = threading.Thread(target=get_input)
    # get_input_thread.setDaemon(True)
    # get_input_thread.start()
    # start_ind = 3000
    RB_and_acc_t = get_transformed_current_reading(current_readings[0], R_Gn_Gp, acc_offset_Gp, R_B0_S0)
    # rt_runner.record_raw_imu(RB_and_acc_t)
    if is_recording:
        record_buffer = RB_and_acc_t.reshape(1, -1)
    t = 1
    tip_smpl = {} # time: (body_pose, global_orient, transl)

    for i in range(1, len(current_readings)):
        RB_and_acc_t = get_transformed_current_reading(current_readings[i], R_Gn_Gp, acc_offset_Gp, R_B0_S0)

        # t does not matter, not used
        res = rt_runner.step(RB_and_acc_t, last_root_pos, s_gt=None, c_gt=None, t=t)

        last_root_pos = res['qdq'][:3]

        if visualize:
            viz_locs = res['viz_locs']
            for sbp_i in range(viz_locs.shape[0]):
                viz_point(viz_locs[sbp_i, :], sbp_i, pb_c, p_vids)

        
        body_pose, global_orient, transl = tip_q2smpl(res['qdq'])
        output_n = model_n(global_orient=torch.from_numpy(global_orient), 
                           body_pose=torch.from_numpy(body_pose), 
                           transl=torch.from_numpy(transl))
        
        vertices = output_n.vertices[0].detach().cpu().numpy().squeeze()
        joints = output_n.joints[0].detach().cpu().numpy().squeeze()
        faces = model_n.faces
        if i == 1:
            # min_z = np.min(vertices[:, 2])
            min_z = 0
            if visualize:
                vis.log_floor_mesh(height_offset=min_z, subsquares=20)
            tip_smpl[int(reading_t_ns[0])] = (body_pose, global_orient, transl)
        tip_smpl[int(reading_t_ns[i])] = (body_pose, global_orient, transl)

        global_orient_q = (R.from_rotvec(global_orient[0])).as_quat()
        if visualize:
            vis.log_transform("world/person/gorient", int(reading_t_ns[i]), transl[0], global_orient_q)
            vis.vis_refit_smpl_sequence(int(reading_t_ns[i]), "world/person/mesh", joints=joints, vertices=vertices, faces=faces)
        
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

        # clock.tick(FREQ)

        # print('\r', R_G_Bt.reshape(6,9), acc_G_t, end='')

        t += 1
        # recording
    # save tip_smpl
    os.makedirs(os.path.dirname(smpl_out_file), exist_ok=True)
    with open(smpl_out_file, 'wb') as f:
        pickle.dump(tip_smpl, f)

    # get_input_thread.join()
    os.chdir(imu_human_dir)
    print('Finish.')
if __name__ == '__main__':
    awinda_dir = r'C:\projects\imu_human_rewrite\recordings\bhargav_living_room\simple_loop_closure\awinda'
    smpl_out_file = awinda_dir.replace(r'\awinda', r'\smpl\tip.pkl')
    run_tip(awinda_dir, smpl_out_file, visualize=True)