import time
import numpy as np
import math
import gym
from gym import Env

import matplotlib.pyplot as plt

useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 11  # 8
pandaNumDofs = 7

ll = [-7] * pandaNumDofs
# upper limits for null space (todo: set them to proper range)
ul = [7] * pandaNumDofs
# joint ranges for null space (todo: set them to proper range)
jr = [7] * pandaNumDofs
# restposes for null space
jointPositions = [0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
rp = jointPositions


class PandaSim(object):
    def __init__(self, bullet_client, offset):
        self.bullet_client = bullet_client
        self.bullet_client.setPhysicsEngineParameter(solverResidualThreshold=0)
        self.offset = np.array(offset)

        # print("offset=",offset)
        flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        self.legos = []

        self.bullet_client.loadURDF("tray/traybox.urdf", [0 + offset[0], 0 + offset[1], -0.6 + offset[2]],
                                    [-0.5, -0.5, -0.5, 0.5], flags=flags)
        self.legos.append(
            self.bullet_client.loadURDF("lego/lego.urdf", np.array([0.1, 0.3, -0.5]) + self.offset, flags=flags))
        self.bullet_client.changeVisualShape(self.legos[0], -1, rgbaColor=[1, 0, 0, 1])
        self.legos.append(
            self.bullet_client.loadURDF("lego/lego.urdf", np.array([-0.1, 0.3, -0.5]) + self.offset, flags=flags))
        self.legos.append(
            self.bullet_client.loadURDF("lego/lego.urdf", np.array([0.1, 0.3, -0.7]) + self.offset, flags=flags))
        self.sphereId = self.bullet_client.loadURDF("sphere_small.urdf", np.array([0, 0.3, -0.6]) + self.offset,
                                                    flags=flags)
        self.bullet_client.loadURDF("cube_small.urdf", np.array([0.1, 0.3, -0.7]) + self.offset, flags=flags)
        self.bullet_client.loadURDF("sphere_small.urdf", np.array([0, 0.3, -0.7]) + self.offset, flags=flags)
        orn = [-0.707107, 0.0, 0.0, 0.707107]  # p.getQuaternionFromEuler([-math.pi/2,math.pi/2,0])
        eul = self.bullet_client.getEulerFromQuaternion([-0.5, -0.5, -0.5, 0.5])
        self.panda = self.bullet_client.loadURDF("franka_panda/panda.urdf", np.array([0, 0, 0]) + self.offset, orn,
                                                 useFixedBase=True, flags=flags)
        index = 0
        self.state = 0
        self.control_dt = 1. / 240.
        self.finger_target = 0
        self.gripper_height = 0.2
        # create a constraint to keep the fingers centered
        c = self.bullet_client.createConstraint(self.panda,
                                                9,
                                                self.panda,
                                                10,
                                                jointType=self.bullet_client.JOINT_GEAR,
                                                jointAxis=[1, 0, 0],
                                                parentFramePosition=[0, 0, 0],
                                                childFramePosition=[0, 0, 0])
        self.bullet_client.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

        for j in range(self.bullet_client.getNumJoints(self.panda)):
            self.bullet_client.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
            info = self.bullet_client.getJointInfo(self.panda, j)
            # print("info=",info)
            jointName = info[1]
            jointType = info[2]
            if (jointType == self.bullet_client.JOINT_PRISMATIC):
                self.bullet_client.resetJointState(self.panda, j, jointPositions[index])
                index = index + 1
            if (jointType == self.bullet_client.JOINT_REVOLUTE):
                self.bullet_client.resetJointState(self.panda, j, jointPositions[index])
                index = index + 1
        self.t = 0.

    def reset(self):
        pass

    def update_state(self):
        keys = self.bullet_client.getKeyboardEvents()
        if len(keys) > 0:
            for k, v in keys.items():
                if v & self.bullet_client.KEY_WAS_TRIGGERED:
                    if (k == ord('1')):
                        self.state = 1
                    elif (k == ord('2')):
                        self.state = 2
                    elif (k == ord('3')):
                        self.state = 3
                    elif (k == ord('4')):
                        self.state = 4
                    elif (k == ord('5')):
                        self.state = 5
                    elif (k == ord('6')):
                        self.state = 6
                if v & self.bullet_client.KEY_WAS_RELEASED:
                    self.state = 0

    def step(self):
        if self.state == 6 and self.finger_target > 0:
            self.finger_target -= 0.01
        if self.state == 5 and self.finger_target < 0.05:
            self.finger_target = 0.04
        self.bullet_client.submitProfileTiming("step")
        self.update_state()
        # print("self.state=",self.state)
        # print("self.finger_target=",self.finger_target)
        alpha = 0.9  # 0.99
        if self.state == 1 or self.state == 2 or self.state == 3 or self.state == 4 or self.state == 7:
            # gripper_height = 0.034
            self.gripper_height = alpha * self.gripper_height + (1. - alpha) * 0.03
            if self.state == 2 or self.state == 3 or self.state == 7:
                self.gripper_height = alpha * self.gripper_height + (1. - alpha) * 0.2

            t = self.t
            self.t += self.control_dt
            pos = [self.offset[0] + 0.2 * math.sin(1.5 * t), self.offset[1] + self.gripper_height,
                   self.offset[2] + -0.6 + 0.1 * math.cos(1.5 * t)]
            if self.state == 3 or self.state == 4:
                pos, o = self.bullet_client.getBasePositionAndOrientation(self.legos[0])
                pos = [pos[0], self.gripper_height, pos[2]]
                self.prev_pos = pos
            if self.state == 7:
                pos = self.prev_pos
                diffX = pos[0] - self.offset[0]
                diffZ = pos[2] - (self.offset[2] - 0.6)
                self.prev_pos = [self.prev_pos[0] - diffX * 0.1, self.prev_pos[1], self.prev_pos[2] - diffZ * 0.1]

            orn = self.bullet_client.getQuaternionFromEuler([math.pi / 2., 0., 0.])
            self.bullet_client.submitProfileTiming("IK")
            jointPoses = self.bullet_client.calculateInverseKinematics(self.panda, pandaEndEffectorIndex, pos, orn, ll,
                                                                       ul,
                                                                       jr, rp, maxNumIterations=20)
            self.bullet_client.submitProfileTiming()
            for i in range(pandaNumDofs):
                self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                         jointPoses[i], force=5 * 240.)
            # target for fingers
        for i in [9, 10]:
            self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                     self.finger_target, force=10)
        self.bullet_client.submitProfileTiming()


class PandaSimAuto(PandaSim):
    def __init__(self, bullet_client, offset):
        PandaSim.__init__(self, bullet_client, offset)
        self.state_t = 0
        self.cur_state = 0
        self.states = [0, 3, 5, 4, 6, 3, 7]
        self.state_durations = [1, 1, 1, 2, 1, 1, 10]

    def update_state(self):
        self.state_t += self.control_dt
        if self.state_t > self.state_durations[self.cur_state]:
            self.cur_state += 1
            if self.cur_state >= len(self.states):
                self.cur_state = 0
            self.state_t = 0
            self.state = self.states[self.cur_state]
            # print("self.state=",self.state)


class PandaManual(Env, PandaSim):
    def __init__(self, bullet_client, offset, input_shape=(320, 200, 1)):
        Env().__init__()
        self.action_space = gym.spaces.Discrete(8)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=input_shape, dtype=np.uint8)


        self.bullet_client = bullet_client
        self.bullet_client.setPhysicsEngineParameter(solverResidualThreshold=0)
        self.offset = np.array(offset)

        flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        self.flags = flags
        orn = [-0.707107, 0.0, 0.0, 0.707107]  # p.getQuaternionFromEuler([-math.pi/2,math.pi/2,0])
        eul = self.bullet_client.getEulerFromQuaternion([-0.5, -0.5, -0.5, 0.5])
        self.bullet_client.loadURDF("tray/traybox.urdf", [0 + offset[0], 0 + offset[1], -0.6 + offset[2]],
                                    [-0.5, -0.5, -0.5, 0.5], flags=flags)
        self.panda = self.bullet_client.loadURDF("franka_panda/panda.urdf", np.array([0, 0, 0]) + self.offset, orn,
                                                 useFixedBase=True, flags=flags)

        index = 0
        self.action = 0
        self.control_dt = 1. / 240.
        self.finger_target = 0
        self.gripper_height = 0.2
        # create a constraint to keep the fingers centered
        c = self.bullet_client.createConstraint(self.panda,
                                                9,
                                                self.panda,
                                                10,
                                                jointType=self.bullet_client.JOINT_GEAR,
                                                jointAxis=[1, 0, 0],
                                                parentFramePosition=[0, 0, 0],
                                                childFramePosition=[0, 0, 0])
        self.bullet_client.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

        for j in range(self.bullet_client.getNumJoints(self.panda)):
            self.bullet_client.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
            info = self.bullet_client.getJointInfo(self.panda, j)
            # print("info=",info)
            jointName = info[1]
            jointType = info[2]
            if (jointType == self.bullet_client.JOINT_PRISMATIC):
                self.bullet_client.resetJointState(self.panda, j, jointPositions[index])
                index = index + 1
            if (jointType == self.bullet_client.JOINT_REVOLUTE):
                self.bullet_client.resetJointState(self.panda, j, jointPositions[index])
                index = index + 1
        self.t = 0.
        self.reset(init=True)

    def reset(self, init=False):
        if not init:
            self.bullet_client.removeBody(self.target)
        self.target = self.bullet_client.loadURDF("cube_small.urdf", np.array([0.1, 0.3, -0.7]) + self.offset, flags=self.flags)
        self.pos, self.orn = ((0.05176176834328308, 0.026489668721649595, -0.5817247161786195), self.bullet_client.getQuaternionFromEuler([math.pi /2 , 0, 0]))
        self.move_arm()

    def move_arm(self):
        self.bullet_client.submitProfileTiming("IK")
        jointPoses = self.bullet_client.calculateInverseKinematics(self.panda, pandaEndEffectorIndex, self.pos, self.orn, ll,
                                                                   ul,
                                                                   jr, rp, maxNumIterations=20)
        self.bullet_client.submitProfileTiming()

        for i in range(pandaNumDofs):
            self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                     jointPoses[i], force=5 * 240.)

        for i in [9, 10]:
            self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                     self.finger_target, force=10)

        self.bullet_client.submitProfileTiming()

    def get_image_matrix(self):

        camTargetPos = [0, 0, 0]
        cameraUp = [0, 0, 0.5]
        cameraPos = [1, 1, 1]

        pitch = -10.0

        roll = 60
        upAxisIndex = 2
        camDistance = 1
        pixelWidth = 320
        pixelHeight = 200
        nearPlane = 0.01
        farPlane = 100

        fov = 30

        viewMatrix = self.bullet_client.computeViewMatrix(
            cameraEyePosition=[1, 1, 0.5],
            cameraTargetPosition=[0.5, .5, 0],
            cameraUpVector=[0, 1, 0]
        )
        aspect = pixelWidth / pixelHeight
        projectionMatrix = self.bullet_client.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)
        img = self.bullet_client.getCameraImage(pixelWidth,
                                          pixelHeight,
                                          viewMatrix,
                                          projectionMatrix,
                                          shadow=1,
                                          lightDirection=[1, 1, 1],
                                          renderer=self.bullet_client.ER_BULLET_HARDWARE_OPENGL)

        show(img[2])



        com_p, com_o, _, _, _, _ = self.bullet_client.getLinkState(self.panda, 7)
        rot_matrix = self.bullet_client.getMatrixFromQuaternion(com_o)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        init_camera_vector = (0, 0, 1)  # z-axis
        init_up_vector = (0, 1, 0)  # y-axis
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)
        view_matrix = self.bullet_client.computeViewMatrix(com_p, com_p + 0.1 * camera_vector, up_vector)
        img = self.bullet_client.getCameraImage(1000, 1000, view_matrix, projectionMatrix)
        show(img[2])


        return img

    def compute_reward(self):
        return - np.linalg.norm(np.array(self.pos)-np.array(self.bullet_client.getBasePositionAndOrientation(self.target)[0]))

    def render_image(self, img):
        plt.subplot(1, 3, 1)
        plt.imshow(img[2])
        plt.title('rgb pixels')

        plt.subplot(1,3,2)
        plt.imshow(img[3])
        plt.title('depth pixels')

        plt.subplot(1,3,3)
        plt.imshow(img[4])
        plt.title('segmentation mask buffer')
        plt.show()
        plt.show()

    def visualize(self):
        pass

    def step(self, action=None):

        self.bullet_client.stepSimulation()

        self.update_state()
        img = self.get_image_matrix()
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
            self.move_arm()

        return img[2], self.compute_reward(), False
