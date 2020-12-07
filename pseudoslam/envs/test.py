import numpy as np
import cv2, time
from os import path
from matplotlib import pyplot as plt

import gym
from gym import spaces
from gym.utils import seeding
# from pseudoslam.envs.simulator.pseudoSlam import pseudoSlam

class MonitorEnv(gym.Wrapper):
    def __init__(self, env=None,param={'vel':1,'foot':0,'tau':0}):
        """Record episodes stats prior to EpisodicLifeEnv, etc."""
        gym.Wrapper.__init__(self, env)
        self._current_reward = None
        self._num_steps = None
        self._total_steps = None
        self._episode_rewards = []
        self._episode_lengths = []
        self._num_episodes = 0
        self._num_returned = 0
        self.param = param
        self.current_obs = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs).squeeze()
        # obs = obs.reshape(64,64)
        self.current_obs = obs.copy()
        if self._total_steps is None:
            self._total_steps = sum(self._episode_lengths)

        if self._current_reward is not None:
            self._episode_rewards.append(self._current_reward)
            self._episode_lengths.append(self._num_steps)
            self._num_episodes += 1

        self._current_reward = 0
        self._num_steps = 0

        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        obs = np.squeeze(obs)
        self.current_obs = obs.copy()
        self._current_reward += rew
        self._num_steps += 1
        self._total_steps += 1
        return (obs, rew, done, info)

    def get_episode_rewards(self):
        return self._episode_rewards

    def get_current_rewards(self):
        return self._current_reward
    def get_episode_lengths(self):
        return self._episode_lengths

    def get_total_steps(self):
        return self._total_steps

    def next_episode_results(self):
        for i in range(self._num_returned, len(self._episode_rewards)):
            yield (self._episode_rewards[i], self._episode_lengths[i])
        self._num_returned = len(self._episode_rewards)

    def find_contour(self):
        img = self.env.sim.get_state().copy()
        ret,binary = cv2.threshold(img,-1,1,cv2.THRESH_BINARY)
        binary_=binary.astype(np.uint8)
        # binary = np.dtype(binary,np.uint8)
        # print('max:',np.max(binary),'min:',np.min(binary),binary_.dtype)
        # raise NotImplementedError
        #  gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        contours,hierarchy = cv2.findContours(binary_,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        new_contour = []
        nc = []
        for point in contours[0]:
            # print(point[0],img[point[0][1],point[0][0]])
            # if img[point[0][1],point[0][0]]!=0:
            #     # nc.append(point)
            #     img[point[0][1],point[0][0]]=0
            img[point[0][1],point[0][0]]=50
        # new_contour.append(nc)
        # imag = cv2.drawContours(img,new_contour,-1,100,1)
        # img_ = imag-img
        # print(np.max(img_),np.min(img_))
        # raise NotImplementedError
        return img


if __name__ == '__main__':
    # env = RobotExplorationT0()
    env = gym.make("pseudoslam:RobotExploration-v0")
    env = MonitorEnv(env)
    obs = env.reset()

    while 1:
        pose = env.sim.get_pose()
        plt.figure(1)
        plt.subplot(1,3,1)
        # plt.clf()
        # plt.imshow(env.sim.get_state().copy(), cmap='gray')
        plt.imshow(env.current_obs.copy(), cmap='gray')
        # plt.draw()
        # plt.pause(0.001)
        # plt.figure(1)
        plt.subplot(1,3,2)
        # plt.clf()
        # plt.imshow(env.sim.get_state().copy()-env.find_contour().copy(), cmap='gray')
        plt.imshow(env.sim.get_state().copy(), cmap='gray')
        plt.subplot(1,3,3)
        plt.imshow(env.find_contour(), cmap='gray')
        # plt.imshow(obs.copy(), cmap='gray')
        plt.draw()
        plt.pause(0.001)
        # env.render()
        # print(env.find_contour())
        # epi_cnt += 1
        # act = np.random.randint(3)
        act = 0
        obs, reward, done, info = env.step(act)
        cmd = ['forward', 'left', 'right']
       
        # if epi_cnt > 100 or done:
        #     epi_cnt = 0
        #     env.reset()