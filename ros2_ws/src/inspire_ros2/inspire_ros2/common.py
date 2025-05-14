import numpy as np


# follow the order in URDF
JOINT_NAMES = [
    "J_THUMB_PROXIMAL_BASE_R",
    "J_THUMB_PROXIMAL_R",
    "J_INDEX_PROXIMAL_R",
    "J_MIDDLE_PROXIMAL_R",
    "J_RING_PROXIMAL_R",
    "J_PINKY_PROXIMAL_R"
]

FINGER_TIP_NAMES = [
    "J_THUMB_TIP_R",
    "J_INDEX_TIP_R",
    "J_MIDDLE_TIP_R",
    "J_RING_TIP_R",
    "J_PINKY_TIP_R"
]

# follow the order in registers
ACTUATOR_NAMES = [
    "J_PINKY_PROXIMAL_R",
    "J_RING_PROXIMAL_R",
    "J_MIDDLE_PROXIMAL_R",
    "J_INDEX_PROXIMAL_R",
    "J_THUMB_PROXIMAL_R",
    "J_THUMB_PROXIMAL_BASE_R"
]
NUM_ACTUATORS = len(ACTUATOR_NAMES)

PASSIVE_JOINTS = {
    "J_THUMB_INTERMEDIATE_R": {
        "mimic": "J_THUMB_PROXIMAL_R",
        "multiplier": 1.6,
        "offset": 0.0
    },
    "J_THUMB_DISTAL_R": {
        "mimic": "J_THUMB_PROXIMAL_R",
        "multiplier": 2.4,
        "offset": 0.0
    },
    "J_INDEX_INTERMEDIATE_R": {
        "mimic": "J_INDEX_PROXIMAL_R",
        "multiplier": 1,
        "offset": 0.0
    },
    "J_MIDDLE_INTERMEDIATE_R": {
        "mimic": "J_MIDDLE_PROXIMAL_R",
        "multiplier": 1,
        "offset": 0.0
    },
    "J_RING_INTERMEDIATE_R": {
        "mimic": "J_RING_PROXIMAL_R",
        "multiplier": 1,
        "offset": 0.0
    },
    "J_PINKY_INTERMEDIATE_R": {
        "mimic": "J_PINKY_PROXIMAL_R",
        "multiplier": 1,
        "offset": 0.0
    },
}

ACTUATOR_MIN = 0
ACTUATOR_MAX = 1000

JOINT_MIN = [
    -0.1,
    -0.1,
    0,
    0,
    0,
    0
]

JOINT_MAX = [
    1.3,
    0.6,
    1.7,
    1.7,
    1.7,
    1.7
]

JOINT_MIN = np.array(JOINT_MIN)
JOINT_MAX = np.array(JOINT_MAX)
