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
        self.obstacle = []
        self.pose = None

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
        self.update()
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

    def calc_energy(self,pose,contour):
        (pose_y, pose_x) = (int(pose[0]), int(pose[1]))
        obs_energy = 0

        print(self.obstacle)


    def update(self):
        # 0-y 1-x
        img = self.env.sim.get_state().copy()
        self.obstacle = np.where(img>=90) 
        self.pose = self.env.sim.get_pose()
        # for i in range(len(obstacle[0])):
        #     print('obs:',img[obstacle[0][i],obstacle[1][i]],'xy:',obstacle[0][i],obstacle[1][i])
        # print('obs:',self.obstacle)
        # return obstacle

    def cal_frontier_reward(self,contour,)

    def find_contour(self):
        img = self.env.sim.get_state().copy()
        # print('shape',img.shape)
        h,w = img.shape
        _,binary = cv2.threshold(img,-1,1,cv2.THRESH_BINARY)
        binary_=binary.astype(np.uint8)
        # binary = np.dtype(binary,np.uint8)
        # print('max:',np.max(binary),'min:',np.min(binary),binary_.dtype)
        # raise NotImplementedError
        #  gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        contours,hierarchy = cv2.findContours(binary_,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        new_contour = []
        nc = []
        img_ref = img.copy()
        for point in contours[0]:
            img[point[0][1],point[0][0]]=100
        img_diff = img-img_ref
        _,binary = cv2.threshold(img_diff,90,100,cv2.THRESH_BINARY)
        binary_c = binary.astype(np.uint8)
        contours_,hierarchy = cv2.findContours(binary_c,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        contour_img = np.zeros((h,w),dtype=np.uint8)
        cons = []
        for con in contours_:
            if len(con)>5:
                cons.append(con)
        # print(len(cons))
        # cv2.drawContours(contour_img,[contours_[1]],-1,255,1)
        cv2.drawContours(contour_img,cons,-1,255,1)
        return contour_img


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