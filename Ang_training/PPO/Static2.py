import time
from EnvUAV.env import YawControlEnv
from Agent import PPO
import os
import json
import numpy as np
import matplotlib.pyplot as plt


def main():
    path = os.path.dirname(os.path.realpath(__file__))
    path += '/PPO_0.9/'
    with open(path + 'disr0.9.json', 'r') as f:
        reward_store2 = json.load(f)

    reward_store2 = np.array(reward_store2)

    index = np.array(range(reward_store2.shape[0]))
    for i in range(10):
        print(i, np.argmax(reward_store2[:, i]), 1 - np.max(reward_store2[:, i]))
        plt.plot(index, reward_store2[:, i])
        plt.show()
    # reward_store = np.clip(reward_store, 0, 2)
    # plt.plot(index, np.clip(reward_store[:, 1], 0, 10))
    # plt.plot(index, np.mean(reward_store1, axis=1), label='0.9')
    plt.plot(index, np.max(reward_store2, axis=1), label='0.98')
    plt.plot(index, np.mean(reward_store2, axis=1), label='0.98')
    plt.plot(index, np.min(reward_store2, axis=1), label='0.98')
    # plt.plot(index, np.mean(reward_store3, axis=1), label='0.98')
    # plt.plot(index, np.mean(reward_store4, axis=1), label='0.99')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
 