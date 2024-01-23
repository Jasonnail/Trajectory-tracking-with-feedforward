import time
from EnvUAV.env import YawControlEnv
from Agent import PPO
import os
import json
import argparse


def main(gamma, index):
    # start = time.time()
    # start_time = time.time()
    path = os.path.dirname(os.path.realpath(__file__))
    path += '/PPO_'+str(gamma)+'/'+str(index)
    if not os.path.exists(path):
        os.makedirs(path)
    agent = PPO(path, s_dim=2, gamma=gamma)
    env = YawControlEnv()

    for episode in range(200):
        s = env.reset()
        init_error = s[0]
        for ep_step in range(500):
            a = agent.get_action(s)
            s_, r, done, info = env.step(a[0])
            if done:
                break
            agent.store_transition(s, a, s_, r)
            s = s_
        last_error = s[0]
        print('episode: ', episode,
              ' init_error: ', round(init_error, 3),
              ' last_error: ', round(last_error, 5))
        agent.store_net(str(episode))
        if done:
            episode -= 1
    env.close()


if __name__ == '__main__':
    for i in [10]:
        # start = time.time()
        main(0.9, i)
        # print(time.time() - start)

