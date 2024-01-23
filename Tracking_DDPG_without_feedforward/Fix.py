import time
from EnvUAV.env import YawControlEnv
import os
import json
import numpy as np
import matplotlib.pyplot as plt


def main():
    path = os.path.dirname(os.path.realpath(__file__))

    env = YawControlEnv()

    pos = []
    ang = []

    x = []
    x_target = []
    roll = []
    roll_target = []

    y = []
    y_target = []
    pitch = []
    pitch_target = []

    z = []
    z_target = []
    yaw = []
    yaw_target = []

    action = []

    '''
    Fixed1: [10, -10, 5] [0, 0, 0] -> [0, 0, 0, 0]
    Fixed2: [0, 0, 0] [0, 0, 0] -> [10, -10, 5, 90]
    Fixed2: [0, 0, 0] [60, 60, 60] -> [10, -10, 5, -120]
    '''
    # name = 'Fixed1'
    # env.reset(base_pos=np.array([5, -5, 2]), base_ang=np.array([0, 0, 0]))
    # targets = np.array([[0, 0, 0, 0],
    #                     [0, 0, 0, 0]])

    # name = 'Fixed2'
    # env.reset(base_pos=np.array([0, 0, 0]), base_ang=np.array([0, 0, 0]))
    # targets = np.array([[5, -5, 2, np.pi/2],
    #                     [0, 0, 0, 0]])
    #
    # # name = 'Fixed3'
    # env.reset(base_pos=np.array([0, 0, 0]), base_ang=np.array([1, 1, 1])*np.pi/3)
    # targets = np.array([[5, -5, 2, -np.pi/3*2],
    #                     [0, 0, 0, 0]])

    name = 'Test'
    env.reset(base_pos=np.array([1e-15, 1e-15,  1e-15]), base_ang=np.array([0, 0, 0]))
    targets = np.array([[0, 0,  0,  0],
                        [0, 0, 0, 0]])
    for episode in range(1):
        target = targets[episode, :]
        for ep_step in range(500):
            env.step(target)

            pos.append(env.current_pos.tolist())
            ang.append(env.current_ang.tolist())

            x.append(env.current_pos[0])
            x_target.append(0)
            roll.append(env.current_ang[0])
            roll_target.append(target[0])

            y.append(env.current_pos[1])
            y_target.append(0)
            pitch.append(env.current_ang[1])
            pitch_target.append(target[1])

            z.append(env.current_pos[2])
            z_target.append(0)
            yaw.append(env.current_ang[2])
            yaw_target.append(target[2])

    index = np.array(range(len(x))) * 0.01
    zeros = np.zeros_like(index)
    roll = np.array(roll) / np.pi * 180
    pitch = np.array(pitch) / np.pi * 180
    yaw = np.array(yaw) / np.pi * 180
    roll_target = np.array(roll_target) / np.pi * 180
    pitch_target = np.array(pitch_target) / np.pi * 180
    yaw_target = np.array(yaw_target) / np.pi * 180
    plt.subplot(3, 2, 1)
    plt.plot(index, x, label='x')
    plt.plot(index, x_target, label='x_target')
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(index, pitch, label='pitch')
    # plt.plot(index, pitch_target)
    plt.plot(index, zeros)
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(index, y, label='y')
    plt.plot(index, y_target, label='y_target')
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(index, roll, label='roll')
    # plt.plot(index, roll_target)
    plt.plot(index, zeros)
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(index, x, label='x')
    plt.plot(index, y, label='y')
    plt.plot(index, z, label='z')
    plt.plot(index, z_target, label='z_target')
    # plt.plot(index, action)
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.plot(index, yaw, label='yaw')
    # plt.plot(index, yaw_target)
    plt.plot(index, zeros)
    plt.legend()

    plt.show()

    pos = np.array(pos)
    ang = np.array(ang)
    np.save(path + '/Case4_' + name + '_pos.npy', pos)
    np.save(path + '/Case4_' + name + '_ang.npy', ang)

if __name__ == '__main__':
    main()

