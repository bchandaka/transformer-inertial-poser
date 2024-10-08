# Copyright (c) Facebook, Inc. and its affiliates.
# copied from ScaDiver repo, a lot of these not used

import collections
import numpy as np

''' 
The up direction of the character w.r.t. its root joint.
The up direction in the world frame can be computed by dot(R_root, v_up), 
where R_root is the orientation of the root.
'''
v_up = np.array([0.0, 1.0, 0.0])
''' 
The facing direction of the character w.r.t. its root joint.
The facing direction in the world frame can be computed by dot(R_root, v_face), 
where R_root is the orientation of the root.
'''
v_face = np.array([0.0, 0.0, 1.0])
''' 
The up direction of the world frame, when the character holds its defalult posture (e.g. t-pose).
This information is useful/necessary when comparing a relationship between the character and its environment.
'''
v_up_env = np.array([0.0, 0.0, 1.0])

''' 
Definition of Link/Joint (In our character definition, one joint can only have one link)
'''
root = -1
lhip = 0
lknee = 1
lankle = 2
rhip = 3
rknee = 4
rankle = 5
lowerback = 6
upperback = 7
chest = 8
lowerneck = 9
upperneck = 10
lclavicle = 11
lshoulder = 12
lelbow = 13
lwrist = 14
rclavicle = 15
rshoulder = 16
relbow = 17
rwrist = 18

bullet2smpl = {
    root: 'pelvis',
    lhip: 'left_hip',
    lknee: 'left_knee',
    lankle: 'left_ankle',
    rhip: 'right_hip',
    rknee: 'right_knee',
    rankle: 'right_ankle',
    lowerback: 'spine1',
    upperback: 'spine2',
    chest: 'spine3',
    lowerneck: 'neck',
    upperneck: 'head',
    lclavicle: 'left_collar',
    lshoulder: 'left_shoulder',
    lelbow: 'left_elbow',
    lwrist: 'left_wrist',
    rclavicle: 'right_collar',
    rshoulder: 'right_shoulder',
    relbow: 'right_elbow',
    rwrist: 'right_wrist',
}
    # 0: 'pelvis',
    # 1: 'left_hip',
    # 2: 'left_knee',
    # 3: 'left_ankle',
    # 4: 'right_hip',
    # 5: 'right_knee',
    # 6: 'right_ankle',
    # 7: 'spine1',
    # 8: 'spine2',
    # 9: 'spine3',
    # 10: 'neck',
    # 11: 'head',
    # 12: 'left_collar',
    # 13: 'left_shoulder',
    # 14: 'left_elbow',
    # 15: 'left_wrist',
    # 16: 'right_collar',
    # 17: 'right_shoulder',
    # 18: 'right_elbow',
    # 19: 'right_wrist',
# }
SMPL_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
]
SMPL18_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
]
    
bullet2smpl_ind = {k:SMPL_JOINT_NAMES.index(v) for k,v in bullet2smpl.items()}
smpl18rsmpl_ind = {SMPL18_JOINT_NAMES.index(v):SMPL_JOINT_NAMES.index(v) for v in SMPL18_JOINT_NAMES}
''' 
Definition of the root (base) joint
'''
ROOT = root

''' 
Definition of end effectors
'''
end_effector_indices = [
    lwrist, rwrist, lankle, rankle,
]

'''
Mapping from Bullet indices to Nimble indices
'''
nimble_map = collections.OrderedDict()
nimble_map[root] = 0
nimble_map[lhip] = 1
nimble_map[lknee] = 2
nimble_map[lankle] = 3
nimble_map[rhip] = 17
nimble_map[rknee] = 18
nimble_map[rankle] = 19
nimble_map[lowerback] = 4
nimble_map[upperback] = 5
nimble_map[chest] = 6
nimble_map[lowerneck] = 11
nimble_map[upperneck] = 12
nimble_map[lclavicle] = 7
nimble_map[lshoulder] = 8
nimble_map[lelbow] = 9
nimble_map[lwrist] = 10
nimble_map[rclavicle] = 13
nimble_map[rshoulder] = 14
nimble_map[relbow] = 15
nimble_map[rwrist] = 16

'''
Mapping from Bullet indices to Nimble state indices (no weld joints)
'''
nimble_state_map = collections.OrderedDict()
nimble_state_map[root] = 0
nimble_state_map[lhip] = 1
nimble_state_map[lknee] = 2
nimble_state_map[lankle] = 3
nimble_state_map[rhip] = 15
nimble_state_map[rknee] = 16
nimble_state_map[rankle] = 17
nimble_state_map[lowerback] = 4
nimble_state_map[upperback] = 5
nimble_state_map[chest] = 6
nimble_state_map[lowerneck] = 10
nimble_state_map[upperneck] = 11
nimble_state_map[lclavicle] = 7
nimble_state_map[lshoulder] = 8
nimble_state_map[lelbow] = 9
nimble_state_map[lwrist] = None
nimble_state_map[rclavicle] = 12
nimble_state_map[rshoulder] = 13
nimble_state_map[relbow] = 14
nimble_state_map[rwrist] = None
nimble2smpl_ind = {nimble_state_map[k]:smpl_ind for k,smpl_ind in bullet2smpl_ind.items()}

''' 
Mapping from joint indicies to names
'''
joint_name = collections.OrderedDict()

joint_name[root] = "root"
joint_name[lhip] = "lhip"
joint_name[lknee] = "lknee"
joint_name[lankle] = "lankle"
joint_name[rhip] = "rhip"
joint_name[rknee] = "rknee"
joint_name[rankle] = "rankle"
joint_name[lowerback] = "lowerback"
joint_name[upperback] = "upperback"
joint_name[chest] = "chest"
joint_name[lowerneck] = "lowerneck"
joint_name[upperneck] = "upperneck"
joint_name[lclavicle] = "lclavicle"
joint_name[lshoulder] = "lshoulder"
joint_name[lelbow] = "lelbow"
joint_name[lwrist] = "lwrist"
joint_name[rclavicle] = "rclavicle"
joint_name[rshoulder] = "rshoulder"
joint_name[relbow] = "relbow"
joint_name[rwrist] = "rwrist"

''' 
Mapping from joint names to indicies
'''
joint_idx = collections.OrderedDict()

joint_idx["root"] = root
joint_idx["lhip"] = lhip
joint_idx["lknee"] = lknee
joint_idx["lankle"] = lankle
joint_idx["rhip"] = rhip
joint_idx["rknee"] = rknee
joint_idx["rankle"] = rankle
joint_idx["lowerback"] = lowerback
joint_idx["upperback"] = upperback
joint_idx["chest"] = chest
joint_idx["lowerneck"] = lowerneck
joint_idx["upperneck"] = upperneck
joint_idx["lclavicle"] = lclavicle
joint_idx["lshoulder"] = lshoulder
joint_idx["lelbow"] = lelbow
joint_idx["lwrist"] = lwrist
joint_idx["rclavicle"] = rclavicle
joint_idx["rshoulder"] = rshoulder
joint_idx["relbow"] = relbow
joint_idx["rwrist"] = rwrist

''' 
Mapping from character's joint indicies to bvh's joint names.
Some entry could have no mapping (by assigning None).
'''
bvh_map = collections.OrderedDict()

bvh_map[root] = "root"
bvh_map[lhip] = "lhip"
bvh_map[lknee] = "lknee"
bvh_map[lankle] = "lankle"
bvh_map[rhip] = "rhip"
bvh_map[rknee] = "rknee"
bvh_map[rankle] = "rankle"
bvh_map[lowerback] = "lowerback"
bvh_map[upperback] = "upperback"
bvh_map[chest] = "chest"
bvh_map[lowerneck] = "lowerneck"
bvh_map[upperneck] = "upperneck"
bvh_map[lclavicle] = "lclavicle"
bvh_map[lshoulder] = "lshoulder"
bvh_map[lelbow] = "lelbow"
bvh_map[lwrist] = "lwrist"
bvh_map[rclavicle] = "rclavicle"
bvh_map[rshoulder] = "rshoulder"
bvh_map[relbow] = "relbow"
bvh_map[rwrist] = "rwrist"

''' 
Mapping from bvh's joint names to character's joint indicies.
Some entry could have no mapping (by assigning None).
'''
bvh_map_inv = collections.OrderedDict()

bvh_map_inv["root"] = root
bvh_map_inv["lhip"] = lhip
bvh_map_inv["lknee"] = lknee
bvh_map_inv["lankle"] = lankle
bvh_map_inv["ltoe"] = None
bvh_map_inv["rhip"] = rhip
bvh_map_inv["rknee"] = rknee
bvh_map_inv["rankle"] = rankle
bvh_map_inv["rtoe"] = None
bvh_map_inv["lowerback"] = lowerback
bvh_map_inv["upperback"] = upperback
bvh_map_inv["chest"] = chest
bvh_map_inv["lowerneck"] = lowerneck
bvh_map_inv["upperneck"] = upperneck
bvh_map_inv["lclavicle"] = lclavicle
bvh_map_inv["lshoulder"] = lshoulder
bvh_map_inv["lelbow"] = lelbow
bvh_map_inv["lwrist"] = lwrist
bvh_map_inv["rclavicle"] = rclavicle
bvh_map_inv["rshoulder"] = rshoulder
bvh_map_inv["relbow"] = relbow
bvh_map_inv["rwrist"] = rwrist

bvh_map_inv["lhand"] = None
bvh_map_inv["rhand"] = None

''' 
Definition of PD gains (tuned for Stable PD Controller)
'''
kp = {
    root: 0,
    lhip: 500,
    lknee: 400,
    lankle: 300,
    rhip: 500,
    rknee: 400,
    rankle: 300,
    lowerback: 500,
    upperback: 500,
    chest: 500,
    lowerneck: 200,
    upperneck: 200,
    lclavicle: 400,
    lshoulder: 400,
    lelbow: 300,
    lwrist: 0,
    rclavicle: 400,
    rshoulder: 400,
    relbow: 300,
    rwrist: 0,
}

kd = {}
for k, v in kp.items():
    kd[k] = 0.1 * v
kd[root] = 0

''' 
Definition of PD gains (tuned for Contrained PD Controller).
"cpd_ratio * kp" and "cpd_ratio * kd" will be used respectively.
'''
cpd_ratio = 0.0002

max_force = {
    root: 0,
    lhip: 300,
    lknee: 200,
    lankle: 100,
    rhip: 300,
    rknee: 200,
    rankle: 100,
    lowerback: 300,
    upperback: 300,
    chest: 300,
    lowerneck: 100,
    upperneck: 100,
    lclavicle: 200,
    lshoulder: 200,
    lelbow: 150,
    lwrist: 0,
    rclavicle: 200,
    rshoulder: 200,
    relbow: 150,
    rwrist: 0,
}

''' 
Maximum forces that character can generate when PD controller is used.
'''
contact_allow_map = {
    root: False,
    lhip: False,
    lknee: False,
    lankle: False,
    rhip: False,
    rknee: False,
    rankle: False,
    lowerback: False,
    upperback: False,
    chest: False,
    lowerneck: False,
    upperneck: False,
    lclavicle: False,
    lshoulder: False,
    lelbow: False,
    lwrist: False,
    rclavicle: False,
    rshoulder: False,
    relbow: False,
    rwrist: False,
}

joint_weight = {
    root: 1.0,
    lhip: 0.5,
    lknee: 0.3,
    lankle: 0.2,
    rhip: 0.5,
    rknee: 0.3,
    rankle: 0.2,
    lowerback: 0.4,
    upperback: 0.4,
    chest: 0.3,
    lowerneck: 0.3,
    upperneck: 0.3,
    lclavicle: 0.3,
    lshoulder: 0.3,
    lelbow: 0.2,
    lwrist: 0.0,
    rclavicle: 0.3,
    rshoulder: 0.3,
    relbow: 0.2,
    rwrist: 0.0,
}

''' mu, sigma, lower, upper '''
noise_pose = {
    root: (0.0, 0.1, -0.5, 0.5),
    lhip: (0.0, 0.1, -0.5, 0.5),
    lknee: (0.0, 0.1, -0.5, 0.5),
    lankle: (0.0, 0.1, -0.5, 0.5),
    rhip: (0.0, 0.1, -0.5, 0.5),
    rknee: (0.0, 0.1, -0.5, 0.5),
    rankle: (0.0, 0.1, -0.5, 0.5),
    lowerback: (0.0, 0.1, -0.5, 0.5),
    upperback: (0.0, 0.1, -0.5, 0.5),
    chest: (0.0, 0.1, -0.5, 0.5),
    lowerneck: (0.0, 0.1, -0.5, 0.5),
    upperneck: (0.0, 0.1, -0.5, 0.5),
    lclavicle: (0.0, 0.1, -0.5, 0.5),
    lshoulder: (0.0, 0.1, -0.5, 0.5),
    lelbow: (0.0, 0.1, -0.5, 0.5),
    lwrist: (0.0, 0.1, -0.5, 0.5),
    rclavicle: (0.0, 0.1, -0.5, 0.5),
    rshoulder: (0.0, 0.1, -0.5, 0.5),
    relbow: (0.0, 0.1, -0.5, 0.5),
    rwrist: (0.0, 0.1, -0.5, 0.5),
}

''' mu, sigma, lower, upper '''
noise_vel = {
    root: (0.0, 0.1, -0.5, 0.5),
    lhip: (0.0, 0.1, -0.5, 0.5),
    lknee: (0.0, 0.1, -0.5, 0.5),
    lankle: (0.0, 0.1, -0.5, 0.5),
    rhip: (0.0, 0.1, -0.5, 0.5),
    rknee: (0.0, 0.1, -0.5, 0.5),
    rankle: (0.0, 0.1, -0.5, 0.5),
    lowerback: (0.0, 0.1, -0.5, 0.5),
    upperback: (0.0, 0.1, -0.5, 0.5),
    chest: (0.0, 0.1, -0.5, 0.5),
    lowerneck: (0.0, 0.1, -0.5, 0.5),
    upperneck: (0.0, 0.1, -0.5, 0.5),
    lclavicle: (0.0, 0.1, -0.5, 0.5),
    lshoulder: (0.0, 0.1, -0.5, 0.5),
    lelbow: (0.0, 0.1, -0.5, 0.5),
    lwrist: (0.0, 0.1, -0.5, 0.5),
    rclavicle: (0.0, 0.1, -0.5, 0.5),
    rshoulder: (0.0, 0.1, -0.5, 0.5),
    relbow: (0.0, 0.1, -0.5, 0.5),
    rwrist: (0.0, 0.1, -0.5, 0.5),
}

collison_ignore_pairs = [

]

friction_lateral = 0.8
friction_spinning = 0.3
restitution = 0.0

sum_joint_weight = 0.0
for key, val in joint_weight.items():
    sum_joint_weight += val
for key, val in joint_weight.items():
    joint_weight[key] /= sum_joint_weight
