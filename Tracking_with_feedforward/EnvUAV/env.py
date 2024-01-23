import os
import time

from .uav import UAV
from .surrounding import Surrounding
from .controller import Controller
import numpy as np
import pybullet as p


class YawControlEnv:
    def __init__(self,
                 model='cf2x',
                 render=False,
                 random=True,
                 time_step=0.01):
        '''
        :param model: The model/type of the uav.
        :param render: Whether to render the simulation process
        :param random: Whether to use random initialization setting
        :param time_step: time_steps
        '''
        self.render = render
        self.model = model
        self.random = random
        self.time_step = time_step
        self.path = os.path.dirname(os.path.realpath(__file__))

        self.client = None
        self.time = None
        self.surr = None
        self.current_pos = self.last_pos = None
        self.current_ang = self.last_ang = None
        self.current_R = self.last_R = None
        self.current_vel = self.last_vel = None
        self.current_ang_vel = self.last_ang_vel = None
        self.target = None
        self.uav = None
        self.controller = Controller(path=self.path, time_step=self.time_step)

    def close(self):
        p.disconnect(self.client)

    def reset(self, base_pos, base_ang):
        # 若已经存在上一组，则关闭之，开启下一组训练
        if p.isConnected():
            p.disconnect(self.client)
        self.client = p.connect(p. GUI if self.render else p.DIRECT)
        self.time = 0.
        # 构建场景
        self.surr = Surrounding(client=self.client,
                                time_step=self.time_step)
        base_pos = base_pos
        base_ori = base_ang
        self.current_pos = self.last_pos = np.array(base_pos)
        self.current_ang = self.last_ang = np.array(base_ori)
        self.current_R = self.last_R = np.array([[1., 0., 0.],
                                                 [0., 1., 0.],
                                                 [0., 0., 1.]])
        self.current_vel = self.last_vel = np.array([0., 0., 0.])
        self.current_ang_vel = self.last_ang_vel = np.array([0., 0., 0.])
        self.uav = UAV(path=self.path,
                       client=self.client,
                       time_step=self.time_step,
                       base_pos=base_pos,
                       base_ori=p.getQuaternionFromEuler(base_ori))
        self.controller.reset()

    def step(self, target):
        F = self.controller.step(time_index=self.time/self.time_step,
                                 target=target,
                                 pos=self.current_pos,
                                 vel=self.current_vel,
                                 last_R=self.last_R,
                                 current_R=self.current_R,
                                 ang_v=np.matmul(self.current_R.T, self.current_ang_vel))

        self.uav.apply_action(F, self.time)
        p.stepSimulation()
        self.time += self.time_step

        self.last_pos = self.current_pos
        self.last_ang = self.current_ang
        self.last_R = self.current_R
        self.last_vel = self.current_vel
        self.last_ang_vel = self.current_ang_vel

        current_pos, current_ang = p.getBasePositionAndOrientation(self.uav.id)
        current_R = np.reshape(p.getMatrixFromQuaternion(current_ang), [3, 3])
        current_ang = p.getEulerFromQuaternion(current_ang)
        current_vel, current_ang_vel = p.getBaseVelocity(self.uav.id)
        # 在环境当中，我们均以np.array的形式来存储。
        self.current_pos = np.array(current_pos)
        self.current_ang = np.array(current_ang)
        self.current_R = np.array(current_R)
        self.current_vel = np.array(current_vel)
        self.current_ang_vel = np.array(current_ang_vel)
