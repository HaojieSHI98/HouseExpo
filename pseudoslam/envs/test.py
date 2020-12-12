import numpy as np
import cv2, time
from os import path
from matplotlib import pyplot as plt

import gym
from gym import spaces
from gym.utils import seeding
# from pseudoslam.envs.simulator.pseudoSlam import pseudoSlam

class MonitorEnv(gym.Wrapper):
    def __init__(self, env=None,param={'goal':1}):
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
        self.future_pose = {}
        self.action2command = {0:'forward',1:'left',2:'right'}
        self.calc_step_length = 0.1
        self.contour = None
        self.dists = {}
        self.goalxy = np.zeros(6)

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
        self.update(action)
        info = {}

        info['goal'] = self.param['goal']*self.calc_simple_rewards(action)
        # print(self.param['goal'])
        # info['goalxy'] = self.goalxy
        # print(info)
        # print('rew',rew)
        # self.calc_simple_reward(0)
        # self.calc_simple_reward(1)
        # self.calc_simple_reward(2)
        obs = np.squeeze(obs)
        self.current_obs = obs.copy()
        self._current_reward += rew
        self._num_steps += 1
        self._total_steps += 1
        # raise NotImplemented
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

    # def calc_dist(a,b):
    #     return (a[0]-b[0])*(a[0]-b[0])+(a[0]-b[0])*(a[0]-b[0])
    def calc_energy(self,pose,contour):
        (pose_y, pose_x) = (int(pose[0]), int(pose[1]))
        obs_energy = self.calc_obs_energy(pose)

        print(self.obstacle)


    def next_pose(self,pose,action):
        motion = self.env.sim.motionChoice[action]
        dv= motion[0]*self.calc_step_length # forward motion
        dtheta= motion[1]*self.calc_step_length
        theta= pose[2] + dtheta
        theta= np.arctan2(np.sin(theta),np.cos(theta))
        y= pose[0] - np.sin(theta)*dv
        x= pose[1] + np.cos(theta)*dv
        target_pose = np.array([y,x,theta])
        return target_pose

    def calc_simple_dist(self,pose):
        if pose[0]==self.pose[0] and pose[1]==self.pose[1] and pose[2]==self.pose[2]:
            self.goalxy = np.zeros(6)
            dist = []
            xy_list = []
            # i = 0
            dist_min = 1e6
            for con in self.contour:
                contour = np.asarray(con).reshape(-1,2)
                a = contour[:,0]-pose[1]
                b = contour[:,1]-pose[0]
                c =np.mean(np.power(a,2)+np.power(b,2))
                # print(c )
                dist.append(c)
                xy_list.append((np.mean(a),np.mean(b)))
                # if dist_min>c or i==0:
                #     dist_min = c
            while(len(dist)<3):
                dist.append(1e6)
                xy_list.append((0,0))
            dist_arr = np.asarray(dist)
            sort_arg = np.argsort(dist_arr)
            for i in range(3):
                index = sort_arg[i]
                # print(xy_list[index])
                self.goalxy[2*i] = xy_list[index][0]/100
                self.goalxy[2*i+1] = xy_list[index][1]/100
            dist_min = np.min(dist_arr)
            return dist_min
        else:
            # print('pose',pose)
            i = 0
            dist_min = 1e6
            for con in self.contour:
                contour = np.asarray(con).reshape(-1,2)
                a = contour[:,0]-pose[1]
                b = contour[:,1]-pose[0]
                c =np.mean(np.power(a,2)+np.power(b,2))
                # print(c)
                if dist_min>c or i==0:
                    dist_min = c
                i += 1
            # print(dist_min)
            return dist_min

    def calc_simple_rewards(self,action):
        rewards = np.zeros(3)
        for i in range(3):
            rewards[i] = np.exp(self.calc_simple_reward(i))
        print('rewards',rewards)
        reward = rewards[action]/(np.sum(rewards)+1e-5)

        return reward

        
    def calc_simple_reward(self,action):
        command = self.action2command[action]
        # print(self.dists['now'] ,self.dists['forward'] )
        if action == 0:
            reward = self.dists['forward']-self.dists['now']
            # return self.dists['forward']-self.dists['now']
        else:
            reward = self.dists[command]-self.dists['forward']
        reward = max(reward,-100)
        return reward
            # return self.dists[command]-self.dists['forward']
        # return reward


    def update(self,action):
        # 0-y 1-x
        img = self.env.sim.get_state().copy()
        self.obstacle = np.where(img>=90) 
        self.pose = self.env.sim.get_pose()
        self.future_pose['forward'] = self.next_pose(self.pose,'forward')
        left = self.next_pose(self.pose,'left')
        self.future_pose['left'] = self.next_pose(left,'forward')
        right = self.next_pose(self.pose,'right')
        self.future_pose['right'] = self.next_pose(right,'right')
        self.dists['now'] = self.calc_simple_dist(self.pose)
        self.dists['forward'] = self.calc_simple_dist(self.future_pose['forward'])
        self.dists['left'] = self.calc_simple_dist(self.future_pose['left'])
        self.dists['right'] = self.calc_simple_dist(self.future_pose['right'])
        # print(self.pose,self.future_pose)
        # print(self.dists['now'] ,self.dists['forward'] )
        # command = self.action2command[action]
        # self.future_pose[command] = self.next_pose(self.pose,command)
        # if action != 0:
        #     self.future_pose['forward'] = self.next_pose(self.pose,'forward')
        # else:
        #     self.future_pose[command] = self.next_pose(self.future_pose[command],'forward')
        self.find_contour()
        # a = self.obstacle[0]-self.pose[0]
        # b = self.obstacle[1]-self.pose[1]
        # c = np.power(a,2)+np.power(b,2)
        # c[np.where(c>)
        # print(c)
        # print(np.power(a,2)+np.pow(b,2))
            # print('obs:',img[obstacle[0][i],obstacle[1][i]],'xy:',obstacle[0][i],obstacle[1][i])
        # print('obs:',self.obstacle)
        # return obstacle

    # def cal_frontier_reward(self,contour,)

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
        pose =  self.env.sim.get_pose()
        self.contour = cons
        # print(pose*180/3.14)
        # print(len(cons))
        # cv2.drawContours(contour_img,[contours_[1]],-1,255,1)
        # for con in cons:
        #     for p in con:
        #         contour_img[p[0][1],p[0][0]]=255
        cv2.drawContours(contour_img,cons,-1,255,1)
        return contour_img
        # return contour


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
        act = 1
        obs, reward, done, info = env.step(act)
        cmd = ['forward', 'left', 'right']
       
        # if epi_cnt > 100 or done:
        #     epi_cnt = 0
        #     env.reset()