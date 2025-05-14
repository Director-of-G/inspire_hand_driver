import numpy as np
from sensor_msgs.msg import JointState
from inspire_ros2.common import (
    JOINT_NAMES, JOINT_MIN, JOINT_MAX,
    ACTUATOR_NAMES, NUM_ACTUATORS, ACTUATOR_MIN, ACTUATOR_MAX,
    PASSIVE_JOINTS
)


def map_actuator_to_joint(actuator_values):
    """
    Maps actuator values to joint angles.
    
    Args:
        actuator_values (list): List of actuator values.
    
    Returns:
        list: Mapped joint angles.
    """
    joint_angles = []
    for i in range(len(JOINT_NAMES)):
        actuator_value = actuator_values[i]
        normalized = (actuator_value - ACTUATOR_MIN) / (ACTUATOR_MAX - ACTUATOR_MIN)
        joint_angle = JOINT_MAX[i] - normalized * (JOINT_MAX[i] - JOINT_MIN[i])
        joint_angles.append(joint_angle)
    return joint_angles


def get_passive_joints(act_joints):
    """
    Computes the passive joints based on the active joints.
    
    Args:
        act_joints (dict): Dictionary of active joints.
    
    Returns:
        dict: Dictionary of passive joints.
    """
    passive_joints = {}
    for joint_name, params in PASSIVE_JOINTS.items():
        if params["mimic"] in act_joints:
            act_value = act_joints[params["mimic"]]
            passive_value = act_value * params["multiplier"] + params["offset"]
            passive_joints[joint_name] = passive_value
    return passive_joints


def get_joint_state_msg(actuator_values):
    """
    Creates a JointState message from actuator values.
    
    Args:
        actuator_values (list): List of actuator values.
    
    Returns:
        JointState: JointState message.
    """
    actuator_values = np.array(actuator_values)
    joint_angles = map_actuator_to_joint(actuator_values)
    
    joint_state_msg = JointState()

    act_joints = {j: q for j, q in zip(JOINT_NAMES, joint_angles)}
    passive_joints = get_passive_joints(act_joints)

    joint_state_msg.name = JOINT_NAMES + list(passive_joints.keys())
    joint_state_msg.position = joint_angles + list(passive_joints.values())
    
    return joint_state_msg


def map_joint_to_actuator(joint_names, joint_values):
    """
    Maps joint angles to actuator values.
    
    Args:
        joint_values (list): List of joint angles.
    
    Returns:
        list: Mapped actuator values.
    """
    if len(joint_names) != 6:
        joint_names = JOINT_NAMES
    actuator_values = []
    for i in range(NUM_ACTUATORS):
        actuator_name = ACTUATOR_NAMES[i]
        joint_id = joint_names.index(actuator_name)
        joint_value = joint_values[joint_id]
        joint_min, joint_max = JOINT_MIN[joint_id], JOINT_MAX[joint_id]
        normalized = (joint_max - joint_value) / (joint_max - joint_min)
        actuator_value = ACTUATOR_MIN + normalized * (ACTUATOR_MAX - ACTUATOR_MIN)
        actuator_value = int(min(max(actuator_value, ACTUATOR_MIN), ACTUATOR_MAX))
        actuator_values.append(actuator_value)
        
    return actuator_values


def get_q_from_act_joints(pin_model, act_joints):
    """
        Get the full joint pos vec (nq,) from active joint angles dict
    """
    nq = pin_model.nq
    q = np.zeros(nq)

    for i, name in enumerate(pin_model.names):
        id = pin_model.joints[i].idx_q
        if name in act_joints:
            q[id] = act_joints[name]
        elif name in PASSIVE_JOINTS:
            mimic, multiplier, offset = PASSIVE_JOINTS[name].values()
            q[id] = act_joints[mimic] * multiplier + offset
        else:
            pass

    return q
