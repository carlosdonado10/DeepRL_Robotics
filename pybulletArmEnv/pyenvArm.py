from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np
from box import Box
import yaml
import math

from tf_agents.environments import py_environment
# from tf_agents.environments import tf_environment
# from tf_agents.environments import tf_py_environment
# from tf_agents.environments import utils
from tf_agents.specs import array_spec
# from tf_agents.environments import wrappers
# # from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()

with open('./pybulletArmEnv/env_config.yaml', 'r') as ymlfile:
    cfg = Box(yaml.safe_load(ymlfile))


class PandaArm(object):
    def __init__(self, bullet_client, offset):
        self.bullet_client = bullet_client
        self.bullet_client.setPhysicsEngineParameter(solverResidualThreshold=0)
        self.offset = np.array(offset)
        self.flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES

        self.bullet_client.loadURDF("tray/traybox.urdf", [0 + offset[0], 0 + offset[1], -0.6 + offset[2]],
                                    [-0.5, -0.5, -0.5, 0.5], flags=self.flags)

        self.panda = self.bullet_client.loadURDF("franka_panda/panda.urdf", np.array([0, 0, 0]) + self.offset,
                                                 cfg.panda.init.orn, useFixedBase=True, flags=self.flags)

        self.state_t = 0
        self.action = 0
        self.control_dt = 1. / 240.
        self.finger_target = 0
        self.gripper_height = 0.2
        self._episode_ended = False
        self._init = True

        #Constraint to keep fingers centered
        c = self.bullet_client.createConstraint(self.panda,
                                                9,
                                                self.panda,
                                                10,
                                                jointType=self.bullet_client.JOINT_GEAR,
                                                jointAxis=[1, 0, 0],
                                                parentFramePosition=[0, 0, 0],
                                                childFramePosition=[0, 0, 0])

        self.bullet_client.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

        index = 0
        for j in range(self.bullet_client.getNumJoints(self.panda)):
            self.bullet_client.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
            info = self.bullet_client.getJointInfo(self.panda, j)
            # print("info=",info)
            jointName = info[1]
            jointType = info[2]
            if (jointType == self.bullet_client.JOINT_PRISMATIC):
                self.bullet_client.resetJointState(self.panda, j, cfg.panda.init.jointPositions[index])
                index = index + 1
            if (jointType == self.bullet_client.JOINT_REVOLUTE):
                self.bullet_client.resetJointState(self.panda, j, cfg.panda.init.jointPositions[index])
                index = index + 1
        self.t = 0.

        self._initialize()

    def _initialize(self):
        if not self._init:
            self.bullet_client.removeBody(self.target)
        self._init = False
        self.state_t = 0
        self._episode_ended = False
        self.target = self.bullet_client.loadURDF("cube_small.urdf", np.array([0.1, 0.3, -0.7]) + self.offset,
                                                  flags=self.flags)
        self.pos, self.orn = (tuple(cfg.panda.init.initial_pos),
                              self.bullet_client.getQuaternionFromEuler([math.pi / 2, 0, 0]))
        self._move()


    def _move(self):
        c = cfg.panda.init
        jointPoses = self.bullet_client.calculateInverseKinematics(self.panda, c.pandaEndEffectorIndex, self.pos,
                                                                   self.orn, c.ll,
                                                                   c.ul,
                                                                   c.jr, c.rp, maxNumIterations=20)
        for i in range(c.pandaNumDofs):
            self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                     jointPoses[i], force=5 * 240.)
        # Fingers
        for i in [9, 10]:
            self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                     self.finger_target, force=10)

    def get_image_matrix(self):
        c = cfg.panda.init
        com_p, com_o, _, _, _, _ = self.bullet_client.getLinkState(self.panda, 7)
        rot_matrix = self.bullet_client.getMatrixFromQuaternion(com_o)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        init_camera_vector = (0, 0, 1)  # z-axis
        init_up_vector = (0, 1, 0)  # y-axis
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)
        view_matrix = self.bullet_client.computeViewMatrix(com_p, com_p + 0.1 * camera_vector, up_vector)
        projectionMatrix = self.bullet_client.computeProjectionMatrixFOV(c.fov, c.pixelWidth / c.pixelHeight, c.nearPlane, c.farPlane)
        img = self.bullet_client.getCameraImage(320, 200, view_matrix, projectionMatrix)

        return np.array(img[2], dtype=np.float32)[:, :, 0:3] / 255 #TODO: Test depth input

    def _update_state(self):
        self.state_t += self.control_dt
        if self.state_t > 1:
            self._episode_ended = True

    def compute_reward(self):
        return - np.linalg.norm(np.array(self.pos)-np.array(self.bullet_client.getBasePositionAndOrientation(self.target)[0]))

    def _step(self, action=None):
        if self._episode_ended:
            return self.reset()

        self.bullet_client.stepSimulation()
        self._update_state()

        if action == 0:
            self.pos = (self.pos[0] + 0.01, self.pos[1], self.pos[2])
        elif action == 1:
            self.pos = (self.pos[0] - 0.01, self.pos[1], self.pos[2])
        elif action == 2:
            self.pos = (self.pos[0], self.pos[1] + 0.01, self.pos[2])
        elif action == 3:
            self.pos = (self.pos[0], self.pos[1] - 0.01, self.pos[2])
        elif action == 4:
            self.pos = (self.pos[0], self.pos[1], self.pos[2] + 0.01)
        elif action == 5:
            self.pos = (self.pos[0], self.pos[1], self.pos[2] - 0.01)
        elif action == 6:
            if self.finger_target > 0:
                self.finger_target -= 0.01
        elif action == 7:
            if self.finger_target <= 0.04:
                self.finger_target += 0.01
        if action:
            self._move()

        if self._episode_ended:
            return ts.termination(
                self.get_image_matrix(), self.compute_reward()
            )

        return ts.transition(
            self.get_image_matrix(), self.compute_reward(), discount=0.9
        )


class PandaEnv(PandaArm, py_environment.PyEnvironment):

    def __init__(self, bullet_client, offset):
        PandaArm.__init__(self, bullet_client, offset)
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(1,),
            dtype=np.int32,
            minimum=cfg.pyenv.init.action_specs.minimum,
            maximum=cfg.pyenv.init.action_specs.maximum,
            name=cfg.pyenv.init.action_specs.name
        )

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=tuple([cfg.panda.init.pixelHeight, cfg.panda.init.pixelWidth, 3]),
            dtype=np.float32,
            minimum=cfg.pyenv.init.observation_specs.minimum,
            maximum=1,
            name=cfg.pyenv.init.observation_specs.name
        )

        self._state = 0
        self.episode_ended = False


    def _reset(self):
        self._initialize()
        return ts.restart(self.get_image_matrix())

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

if __name__ == '__main__':

    env = PandaEnv()