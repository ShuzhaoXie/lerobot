#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from functools import cached_property
from typing import Any
import socket
from collections import deque
import inspect

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_irb120 import IRB120Config

logger = logging.getLogger(__name__)


def check_coordinates(coordinates):
    if (len(coordinates) == 2 and
            len(coordinates[0]) == 3 and
            len(coordinates[1]) == 4):
        return coordinates
    elif len(coordinates) == 7:
        return [coordinates[0:3], coordinates[3:7]]
    # log.warning('Received malformed coordinate: %s', str(coordinates))  # Python 3: warn改为warning，修复拼写错误Recieved→Received
    raise NameError('Malformed coordinate!')


class IRB120(Robot):
    """
    ABB IRB 120
    """

    config_class = IRB120Config
    name = "abb_irb120"

    def __init__(self, config: IRB120Config):
        super().__init__(config)
        self.config = config
        self.delay = .08
        self.cameras = make_cameras_from_configs(config.cameras)
    
    def set_units(self, linear, angular):
        units_l = {'millimeters': 1.0,
                   'meters': 1000.0,
                   'inches': 25.4}
        units_a = {'degrees': 1.0,
                   'radians': 57.2957795}
        self.scale_linear = units_l[linear]
        self.scale_angle = units_a[angular]

    # @property
    # def _motors_ft(self) -> dict[str, type]:
    #     return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    # @cached_property
    # def observation_features(self) -> dict[str, type | tuple]:
    #     return {**self._motors_ft, **self._cameras_ft}

    # @cached_property
    # def action_features(self) -> dict[str, type]:
    #     return self._motors_ft

    # @property
    # def is_connected(self) -> bool:
    #     return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect_motion(self, remote):
        print('Attempting to connect to robot motion server at %s', str(remote))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(2.5)
        self.sock.connect(remote)
        self.sock.settimeout(None)
        print('Connected to robot motion server at %s', str(remote))


    def connect_logger(self, remote, maxlen=None):
        self.pose = deque(maxlen=maxlen)
        self.joints = deque(maxlen=maxlen)

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(remote)
        s.setblocking(1)
        try:
            while True:
                data = s.recv(4096).decode().split()  # Python 3: 需解码字节为字符串
                data = list(map(float, data))  # Python 3: map返回迭代器，需转为列表
                if int(data[1]) == 0:
                    self.pose.append([data[2:5], data[5:]])
                # elif int(data[1]) == 1: self.joints.append([a[2:5], a[5:]])
        finally:
            s.shutdown(socket.SHUT_RDWR)

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.connect_motion((self.config.ip, self.config.port))

        # self.bus.connect()
        # if not self.is_calibrated and calibrate:
        #     logger.info(
        #         "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
        #     )
        #     self.calibrate()

        for cam in self.cameras.values():
            cam.connect()
        
        self.set_units()
        self.set_tool()
        self.set_workobject()
        self.set_speed()
        self.set_zone()
        self.is_connected = True
        # self.configure()
        logger.info(f"{self} connected.")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()

        obs_list, gripper = self.get_joints()
        # obs_dict = self.bus.sync_read("Present_Position")
        # obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        obs_dict = {f"joint_{i}": val for i, val in enumerate(obs_list)}
        obs_dict["gripper"] = int(gripper)
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def get_ee_pose_and_joints(self):
        pass
    
    def format_pose(self, pose):
        pose = check_coordinates(pose)
        msg = ''
        for cartesian in pose[0]:
            msg += f"{cartesian * self.scale_linear:+08.1f} "  # Python 3: f-string格式化
        for quaternion in pose[1]:
            msg += f"{quaternion:+08.5f} "  # Python 3: f-string格式化
        msg += "#"
        return msg
    
    def get_cartesian(self):
        """
        Returns the current pose of the robot, in millimeters
        """
        msg = "03 #"
        data = self.send(msg).split()
        r = [float(s) for s in data]
        return [r[2:5], r[5:9]]
    
    def set_cartesian(self, pose):
        """
        Executes a move immediately from the current pose,
        to 'pose', with units of millimeters.
        """
        msg = "01 " + self.format_pose(pose)
        return self.send(msg)
    
    def get_joints(self):
        """
        Returns the current angles of the robots joints, in degrees & 0, 1 indicating off or on
        """
        msg = "04 #"
        data = self.send(msg).split()
        print('abb_new.py line 101: get_joints recived', data, ', scale_angle, ', self.scale_angle, ', data[1]', data[1])
        # print('print(data)' + '-'*10)
        # print(data)
        # import pdb; pdb.set_trace()
        # check data[2] is 0 or 1;
        return [float(s) / self.scale_angle for s in data[2:8]], abs(1-int(data[1]))
    
    def set_joints(self, joints):
        """
        Executes a move immediately, from current joint angles,
        to 'joints', in degrees.
        """
        print("abb_new.py line 77: send the joints info", 'joints (model output)', joints)
        if len(joints) != 6:
            joints = joints[:6]
        # TODO: something wrong? why # at the end?
        msg = "02 "
        for joint in joints:
            msg += f"{joint * self.scale_angle:+08.2f} "  # Python 3: 使用f-string格式化
        msg += "#"
        return self.send(msg)
    
    def set_speed(self, speed=[100, 50, 50, 50]):
        """
        speed: [robot TCP linear speed (mm/s), TCP orientation speed (deg/s),
                external axis linear, external axis orientation]
        """

        if len(speed) != 4:
            return False
        msg = "08 "
        msg += f"{speed[0]:+08.1f} "  # Python 3: f-string格式化
        msg += f"{speed[1]:+08.2f} "
        msg += f"{speed[2]:+08.1f} "
        msg += f"{speed[3]:+08.2f} #"
        self.send(msg)
        
    def set_zone(self,
                 zone_key='z1',
                 point_motion=False,
                 manual_zone=[]):
        zone_dict = {'z0': [.3, .3, .03],
                     'z1': [1, 1, .1],
                     'z5': [5, 8, .8],
                     'z10': [10, 15, 1.5],
                     'z15': [15, 23, 2.3],
                     'z20': [20, 30, 3],
                     'z30': [30, 45, 4.5],
                     'z50': [50, 75, 7.5],
                     'z100': [100, 150, 15],
                     'z200': [200, 300, 30]}
        """
        Sets the motion zone of the robot. This can also be thought of as
        the flyby zone, AKA if the robot is going from point A -> B -> C,
        how close do we have to pass by B to get to C

        zone_key: uses values from RAPID handbook (stored here in zone_dict)
        with keys 'z*', you should probably use these

        point_motion: go to point exactly, and stop briefly before moving on

        manual_zone = [pzone_tcp, pzone_ori, zone_ori]
        pzone_tcp: mm, radius from goal where robot tool centerpoint
                   is not rigidly constrained
        pzone_ori: mm, radius from goal where robot tool orientation
                   is not rigidly constrained
        zone_ori: degrees, zone size for the tool reorientation
        """

        if point_motion:
            zone = [0, 0, 0]
        elif len(manual_zone) == 3:
            zone = manual_zone
        elif zone_key in zone_dict:  # Python 3: dict.keys()返回视图对象，可直接用于in判断
            zone = zone_dict[zone_key]
        else:
            return False

        msg = "09 "
        msg += str(int(point_motion)) + " "
        msg += f"{zone[0]:+08.4f} "
        msg += f"{zone[1]:+08.4f} "
        msg += f"{zone[2]:+08.4f} #"
        self.send(msg)
    
    def set_dio(self, num, value, id=0):
        """
        num = 1, doGripperOpen; num=2, doGripperClose;
        value=0, Reset; value=1,Set;
        """
        msg = f'97 {int(num)} {int(bool(value))} #'  # Python 3: f-string格式化
        return self.send(msg)
    
    def send(self, message, wait_for_response=True):
        """
        Send a formatted message to the robot socket.
        if wait_for_response, we wait for the response and return it
        """
        caller = inspect.stack()[1][3]
        print('%-14s sending: %s', caller, message)
        self.sock.send(message.encode())  # Python 3: 发送字节需编码字符串
        time.sleep(self.delay)
        if not wait_for_response:
            return
        data = self.sock.recv(4096).decode()  # Python 3: 接收字节需解码为字符串
        print('%-14s received: %s', caller, data)  # Python 3: 修复拼写错误recieved→received
        print('%-14s received2: %s', caller, data)
        return data


    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            the action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        joints = []
        for i in range(1, 7):
            joints.append(float(action[f'joint{i}']))
        self.set_joints(joints)
        if action['gripper'] > 0.5:
            self.set_dio(1, 1)
            gripper_state = 1
        else:
            self.set_dio(2, 1)
            gripper_state = 0

        return action # just return the fucking action
        # goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        # if self.config.max_relative_target is not None:
        #     present_pos = self.bus.sync_read("Present_Position")
        #     goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
        #     goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # # Send goal position to the arm
        # self.bus.sync_write("Goal_Position", goal_pos)
        # return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def set_tool(self, tool=[[0, 0, 0], [1, 0, 0, 0]]):
        """
        Sets the tool centerpoint (TCP) of the robot.
        When you command a cartesian move,
        it aligns the TCP frame with the requested frame.

        Offsets are from tool0, which is defined at the intersection of the
        tool flange center axis and the flange face.
        """
        msg = "06 " + self.format_pose(tool)
        self.send(msg)
        self.tool = tool


    def set_workobject(self, work_obj=[[0, 0, 0], [1, 0, 0, 0]]):
        """
        The workobject is a local coordinate frame you can define on the robot,
        then subsequent cartesian moves will be in this coordinate frame.
        """
        msg = "07 " + self.format_pose(work_obj)
        self.send(msg)
        
    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # self.bus.disconnect(self.config.disable_torque_on_disconnect)
        self.send("99 #", False)
        self.sock.shutdown(socket.SHUT_RDWR)
        self.sock.close()

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
