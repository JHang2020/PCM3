import torch
import torch.nn as nn
import numpy as np
import math
import random
trunk_ori_index = [4, 3, 21, 2, 1]
left_hand_ori_index = [9, 10, 11, 12, 24, 25]
right_hand_ori_index = [5, 6, 7, 8, 22, 23]
left_leg_ori_index = [17, 18, 19, 20]
right_leg_ori_index = [13, 14, 15, 16]

trunk = [i - 1 for i in trunk_ori_index]
left_hand = [i - 1 for i in left_hand_ori_index]
right_hand = [i - 1 for i in right_hand_ori_index]
left_leg = [i - 1 for i in left_leg_ori_index]
right_leg = [i - 1 for i in right_leg_ori_index]
body_parts = [trunk, left_hand, right_hand, left_leg, right_leg]

#temoral-frame mask
class TemporalMask:
    def __init__(self, mask_ratio, person_num, joint_num, channel_num):
        self.mask_ratio = mask_ratio
        self.person_num = person_num
        self.joint_num = joint_num
        self.channel_num = channel_num
        
    def __call__(self, data, frame):
        '''
        given a data: N,C,T,V,M
                frame: the number of the valid frames in data 
        return Mask: N,C,T,V,M 
        randomly mask the frame*ratio frames
        '''
        #data N,C,T,V,M
        N,C,T,V,M = data.shape
        data = data.permute((0,2,4,3,1))##N,T,M,V,C
        #data = data.reshape(N,T,C*V*M)
        size, max_frame, feature_dim = N,T,V*C*M
        #data = data.view(size, max_frame, self.person_num, self.joint_num, self.channel_num)#N,T,M,V,C

        mask_idx = torch.tensor((frame * (1 - self.mask_ratio))).reshape((1, 1, 1, 1, 1)).repeat(size, max_frame, self.person_num, self.joint_num, self.channel_num)
        frame_idx = torch.arange(max_frame).reshape((1, max_frame, 1, 1, 1)).repeat(size, 1, self.person_num, self.joint_num, self.channel_num)
        mask = (frame_idx < mask_idx).float()
        randper = np.random.permutation(range(frame))
        rand = torch.arange(max_frame)
        rand[0:frame] = torch.from_numpy(randper)
        mask = mask[:,rand,...]
        #N,T,M,V,C -> N C T V M
        #mask = mask.permute((0,4,1,3,2))
        #trans_data = data * mask
        #trans_data = trans_data.view(size, max_frame, feature_dim)
        return mask #N,T,M,V,C
    
#random joint mask different for frames
class Jointmask:
    def __init__(self, mask_ratio, person_num, joint_num, channel_num):
        self.mask_ratio = mask_ratio
        self.person_num = person_num
        self.joint_num = joint_num
        self.channel_num = channel_num
    def __call__(self, data, frame=None):
        N,C,T,V,M = data.shape
        data = data.permute((0,2,4,3,1))##N,T,M,V,C
        mask = torch.ones((T,V))
        mask_joint_num = int(V*self.mask_ratio)
        for i in range(T):
            rand = random.sample(range(0,V),mask_joint_num)
            mask[i][rand] = 0.0
        mask = mask.reshape(1,T,1,V,1).repeat(N,1,M,1,C)
        mask[:,:,1,:,:] = 1.0
        return mask

#random topology joints mask same clip frames
class Jointmask3:
    def __init__(self, mask_ratio, person_num, joint_num, channel_num):
        #self.mask_ratio = mask_ratio
        self.mask_part_num = int(5*mask_ratio)
        if self.mask_part_num == 0:
            print('WARNING: Too small mask ratio in the implenmentation!')
        self.person_num = person_num
        self.joint_num = joint_num
        self.channel_num = channel_num
        self.chunk_num = 8
        self.clip_length = 64 // self.chunk_num
    def __call__(self, data, frame=None):
        N,C,T,V,M = data.shape
        data = data.permute((0,2,4,3,1))##N,T,M,V,C
        mask = torch.ones((N,T,V))
        for i in range(self.chunk_num):
            parts_idx = random.sample(body_parts, self.mask_part_num)
            spa_idx = []
            for part_idx in parts_idx:
                spa_idx += part_idx
            spa_idx.sort()
            mask[:,i*self.clip_length:(i+1)*self.clip_length,spa_idx] = 0.0

        mask = mask.reshape(N,T,1,V,1).repeat(1,1,M,1,C)
        mask[:,:,1,:,:] = 1.0#set the second people visable
        return mask#NTMVC

#random joint mask same for frames
class Jointmask2:
    def __init__(self, mask_ratio, person_num, joint_num, channel_num):
        self.mask_ratio = mask_ratio
        self.person_num = person_num
        self.joint_num = joint_num
        self.channel_num = channel_num
    def __call__(self, data, frame=None):
        N,C,T,V,M = data.shape
        data = data.permute((0,2,4,3,1))##N,T,M,V,C
        mask = torch.ones((N,T,V))
        mask_joint_num = int(V*self.mask_ratio)
        
        rand = random.sample(range(0,V),mask_joint_num)
        mask[:,:,rand] = 0.0
        mask = mask.reshape(N,T,1,V,1).repeat(1,1,M,1,C)
        return mask

#random joint mask same for clip frames
class Jointmask4:
    def __init__(self, mask_ratio, person_num, joint_num, channel_num):
        self.mask_ratio = mask_ratio
        self.person_num = person_num
        self.joint_num = joint_num
        self.channel_num = channel_num
    def __call__(self, data, frame=None):
        N,C,T,V,M = data.shape
        data = data.permute((0,2,4,3,1))##N,T,M,V,C
        mask = torch.ones((N,T,V))
        mask_joint_num = int(V*self.mask_ratio)
        for i in range(8):
            rand = random.sample(range(0,V),mask_joint_num)
            mask[:,i*8:(i+1)*8,rand] = 0.0
        
        mask = mask.reshape(N,T,1,V,1).repeat(1,1,M,1,C)
        return mask

#random topology joints mask different frames
class Jointmask5:
    def __init__(self, mask_ratio, person_num, joint_num, channel_num):
        #self.mask_ratio = mask_ratio
        self.mask_part_num = int(5*mask_ratio)
        if self.mask_part_num == 0:
            print('WARNING: Too small mask ratio in the implenmentation!')
        self.person_num = person_num
        self.joint_num = joint_num
        self.channel_num = channel_num
        self.chunk_num = 64
        self.clip_length = 64 // self.chunk_num
    def __call__(self, data, frame=None):
        N,C,T,V,M = data.shape
        data = data.permute((0,2,4,3,1))##N,T,M,V,C
        mask = torch.ones((N,T,V))
        for i in range(self.chunk_num):
            parts_idx = random.sample(body_parts, self.mask_part_num)
            spa_idx = []
            for part_idx in parts_idx:
                spa_idx += part_idx
            spa_idx.sort()
            mask[:,i*self.clip_length:(i+1)*self.clip_length,spa_idx] = 0.0

        mask = mask.reshape(N,T,1,V,1).repeat(1,1,M,1,C)
        mask[:,:,1,:,:] = 1.0#set the second people visable
        return mask#NTMVC

#random topology joints mask random clip frames
class Jointmask6:
    def __init__(self, mask_ratio, person_num, joint_num, channel_num):
        #self.mask_ratio = mask_ratio
        self.mask_part_num = int(5*mask_ratio)
        if self.mask_part_num == 0:
            print('WARNING: Too small mask ratio in the implenmentation!')
        self.person_num = person_num
        self.joint_num = joint_num
        self.channel_num = channel_num
        self.chunk_num = 8
        self.clip_length = 64 // self.chunk_num
    def __call__(self, data, frame=None):
        self.clip_length = random.randint(1,10)
        self.chunk_num = 64//self.clip_length
        
        N,C,T,V,M = data.shape
        data = data.permute((0,2,4,3,1))##N,T,M,V,C
        mask = torch.ones((N,T,V))
        for i in range(self.chunk_num):
            parts_idx = random.sample(body_parts, self.mask_part_num)
            spa_idx = []
            for part_idx in parts_idx:
                spa_idx += part_idx
            spa_idx.sort()
            mask[:,i*self.clip_length:(i+1)*self.clip_length,spa_idx] = 0.0

        mask = mask.reshape(N,T,1,V,1).repeat(1,1,M,1,C)
        mask[:,:,1,:,:] = 1.0#set the second people visable
        return mask#NTMVC

#temporal random clip based 
#spatial combine random and topology
class Jointmask7:
    def __init__(self, mask_ratio, person_num, joint_num, channel_num):
        #self.mask_ratio = mask_ratio
        self.mask_part_num = int(5*mask_ratio)
        if self.mask_part_num == 0:
            print('WARNING: Too small mask ratio in the implenmentation!')
        self.person_num = person_num
        self.joint_num = joint_num
        self.channel_num = channel_num
        self.chunk_num = 8
        self.clip_length = 64 // self.chunk_num
    def __call__(self, data, frame=None):
        self.clip_length = random.randint(1,10)
        self.chunk_num = 64//self.clip_length
        
        N,C,T,V,M = data.shape
        data = data.permute((0,2,4,3,1))##N,T,M,V,C
        mask = torch.ones((N,T,V))
        for i in range(self.chunk_num):
            parts_idx = random.sample(body_parts, self.mask_part_num)
            spa_idx = []
            noise1 = random.sample(range(0,V),2)
            noise2 = random.sample(range(0,V),3)
            for part_idx in parts_idx:
                spa_idx += part_idx
            spa_idx.sort()
            mask[:,i*self.clip_length:(i+1)*self.clip_length,spa_idx] = 0.0
            mask[:,i*self.clip_length:(i+1)*self.clip_length,noise1] = 1.0
            mask[:,i*self.clip_length:(i+1)*self.clip_length,noise2] = 0.0

        mask = mask.reshape(N,T,1,V,1).repeat(1,1,M,1,C)
        mask[:,:,1,:,:] = 1.0#set the second people visable
        return mask#NTMVC

#random joint mask totally random
class Jointmask8:
    def __init__(self, mask_ratio, person_num, joint_num, channel_num):
        self.mask_ratio = mask_ratio
        self.person_num = person_num
        self.joint_num = joint_num
        self.channel_num = channel_num
    def __call__(self, data, ratio=None):
        if ratio!=None:
            self.mask_ratio = ratio
        N,C,T,V,M = data.shape
        data = data.permute((0,2,4,3,1))##N,T,M,V,C
        mask = torch.ones((T*V))
        mask_joint_num = int(T*V*self.mask_ratio)
        
        rand = random.sample(range(0,T*V),mask_joint_num)
        mask[rand] = 0.0
        mask = mask.reshape(1,T,1,V,1).repeat(N,1,M,1,C)
        mask[:,:,1,:,:] = 1.0
        return mask

if __name__ =='__main__':
    J = Jointmask3(0.4,1,25,1)
    T = TemporalMask(0.5,1,10,1)
    mask = J(torch.rand((1,1,64,25,2)))
    print(mask[0,45,0,:,0])
    