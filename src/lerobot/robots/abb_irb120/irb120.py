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


class IRB120(Robot):
    """
    ABB IRB 120
    """

    config_class = IRB120Config
    name = "abb_irb120"

    def __init__(self, config: IRB120Config):
        super().__init__(config)
        self.config = config
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

        # self.configure()
        logger.info(f"{self} connected.")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()

        
        obs_dict = self.bus.sync_read("Present_Position")
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
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

    def get_joints(self):
        """
        Returns the current angles of the robots joints, in degrees & 0, 1 indicating off or on
        """
        msg = "04 #"
        data = self.send(msg).split()
        print('abb_new.py line 101: get_joints recived', data, ', scale_angle, ', self.scale_angle, ', data[2]', data[2])
        # print('print(data)' + '-'*10)
        # print(data)
        # import pdb; pdb.set_trace()
        return [float(s) / self.scale_angle for s in data[2:8]] + [abs(1-float(data[1]))]
    
    def set_joints(self, joints):
        """
        Executes a move immediately, from current joint angles,
        to 'joints', in degrees.
        """
        print("abb_new.py line 77: send the joints info", 'joints (model output)', joints)
        if len(joints) != 6:
            joints = joints[:6]
        msg = "02 "
        for joint in joints:
            msg += f"{joint * self.scale_angle:+08.2f} "  # Python 3: 使用f-string格式化
        msg += "#"
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

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Send goal position to the arm
        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

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
