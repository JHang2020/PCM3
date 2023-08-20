import torch.nn.functional as F
import torch
import random
import numpy as np
from itertools import permutations


def joint_courruption(input_data,no_aug=False):
    if no_aug:
        return torch.zeros((25,)).float()                                                            

    out = input_data.copy()

    flip_prob  = random.random()
    mask = torch.zeros((25,)).float()
    if flip_prob < 0.5:

        #joint_indicies = np.random.choice(25, random.randint(5, 10), replace=False)
        #original : 15
        joint_indicies = np.random.choice(25, 15,replace=False)
        out[:,:,joint_indicies,:] = 0 
        mask[joint_indicies] = 1.0
        return mask, out
    
    else:
         #joint_indicies = np.random.choice(25, random.randint(5, 10), replace=False)
         joint_indicies = np.random.choice(25, 15,replace=False)
         
         temp = out[:,:,joint_indicies,:]
         Corruption = np.array([
                           [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)],
                           [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)],
                           [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)] ])
         temp = np.dot(temp.transpose([1, 2, 3, 0]), Corruption)
         temp = temp.transpose(3, 0, 1, 2)
         out[:,:,joint_indicies,:] = temp
         mask[joint_indicies] = 1.0
         return mask, out


def pose_augmentation(input_data,no_aug=False):
        if no_aug:
            Shear   = np.array([
                        [1,	0, 	0],
                        [0,   1, 	0],
                        [0,   0,  1]
                        ])
            return torch.from_numpy(Shear.flatten()).float()


        Shear       = np.array([
                      [1,	random.uniform(-1, 1), 	random.uniform(-1, 1)],
                      [random.uniform(-1, 1), 1, 	random.uniform(-1, 1)],
                      [random.uniform(-1, 1), 	random.uniform(-1, 1),      1]
                      ])

        temp_data = input_data.copy()
        result =  np.dot(temp_data.transpose([1, 2, 3, 0]),Shear.transpose())
        output = result.transpose(3, 0, 1, 2)

        return torch.from_numpy(Shear.flatten()).float(), output

def temporal_cropresize(input_d,num_of_frames,l_ratio,output_size):
    #num_of_frames = 200 #for pkummd1
    input_data = input_d.copy()
    C, T, V, M =input_data.shape

    # Temporal crop
    min_crop_length = 64

    scale = np.random.rand(1)*(l_ratio[1]-l_ratio[0])+l_ratio[0]
    temporal_crop_length = np.minimum(np.maximum(int(np.floor(num_of_frames*scale)),min_crop_length),num_of_frames)

    start = np.random.randint(0,num_of_frames-temporal_crop_length+1)
    temporal_context = input_data[:,start:start+temporal_crop_length, :, :]

    # interpolate
    temporal_context = torch.tensor(temporal_context,dtype=torch.float)
    temporal_context=temporal_context.permute(0, 2, 3, 1).contiguous().view(C * V * M,temporal_crop_length)
    temporal_context=temporal_context[None, :, :, None]
    temporal_context= F.interpolate(temporal_context, size=(output_size, 1), mode='bilinear',align_corners=False)
    temporal_context = temporal_context.squeeze(dim=3).squeeze(dim=0) 
    temporal_context=temporal_context.contiguous().view(C, V, M, output_size).permute(0, 3, 1, 2).contiguous().numpy()

    return temporal_context

group_num = 4
permu = list(permutations(list(range(group_num))))
shuffle_num = len(permu)
def temporal_shuffle(input_data, shuffle_list=None):
    '''
    分成8组,组间进行shuffle
    input_data: C T V M
    '''
    if type(input_data) is np.ndarray:
        temporal_context = torch.from_numpy(input_data).float()
    else:
        temporal_context = input_data.clone().detach()
    C,T,V,M = temporal_context.shape
    if T%group_num!=0:
        old_temporal_context = temporal_context.clone()
        new_T = ((T//group_num)+1)*group_num
        temporal_context = torch.zeros(C,new_T,V,M)
        temporal_context[:,:T] = old_temporal_context
        temporal_context[:,T:,...] = old_temporal_context[:,T-1:T,...]
    temporal_context = list(torch.chunk(temporal_context, chunks=group_num, dim=1))
    if shuffle_list==None:
        shuffle_id = random.randint(0,shuffle_num-1)
        shuffle_list = permu[shuffle_id]
        shuffle_list = torch.tensor(shuffle_list)
    else:
        shuffle_id = shuffle_list
        shuffle_list = torch.tensor(permu[shuffle_id]).long().clone().detach()
    temporal_context = torch.stack(temporal_context,dim=0)[shuffle_list]
    temporal_context = torch.chunk(temporal_context, chunks=group_num, dim=0)
    temporal_context_shuffle = torch.cat((temporal_context),dim=2).squeeze(0)# C T V M

    if T%group_num!=0:
        temporal_context_shuffle = temporal_context_shuffle[:,:T,...]
    assert temporal_context_shuffle.shape==input_data.shape
    #return temporal_context_shuffle.clone().detach(), shuffle_list.clone().detach()
    return temporal_context_shuffle.clone().detach(), shuffle_id


def crop_subsequence(input_data,num_of_frames,l_ratio,output_size):


    C, T, V, M =input_data.shape

    if l_ratio[0] == 0.5:
    # if training , sample a random crop

         min_crop_length = 64
         scale = np.random.rand(1)*(l_ratio[1]-l_ratio[0])+l_ratio[0]
         temporal_crop_length = np.minimum(np.maximum(int(np.floor(num_of_frames*scale)),min_crop_length),num_of_frames)

         start = np.random.randint(0,num_of_frames-temporal_crop_length+1)
         temporal_crop = input_data[:,start:start+temporal_crop_length, :, :]

         temporal_crop= torch.tensor(temporal_crop,dtype=torch.float)
         temporal_crop=temporal_crop.permute(0, 2, 3, 1).contiguous().view(C * V * M,temporal_crop_length)
         temporal_crop=temporal_crop[None, :, :, None]
         temporal_crop= F.interpolate(temporal_crop, size=(output_size, 1), mode='bilinear',align_corners=False)
         temporal_crop=temporal_crop.squeeze(dim=3).squeeze(dim=0) 
         temporal_crop=temporal_crop.contiguous().view(C, V, M, output_size).permute(0, 3, 1, 2).contiguous().numpy()

         return temporal_crop

    else:
    # if testing , sample a center crop

        start = int((1-l_ratio[0]) * num_of_frames/2)
        data =input_data[:,start:num_of_frames-start, :, :]
        temporal_crop_length = data.shape[1]

        temporal_crop= torch.tensor(data,dtype=torch.float)
        temporal_crop=temporal_crop.permute(0, 2, 3, 1).contiguous().view(C * V * M,temporal_crop_length)
        temporal_crop=temporal_crop[None, :, :, None]
        temporal_crop= F.interpolate(temporal_crop, size=(output_size, 1), mode='bilinear',align_corners=False)
        temporal_crop=temporal_crop.squeeze(dim=3).squeeze(dim=0) 
        temporal_crop=temporal_crop.contiguous().view(C, V, M, output_size).permute(0, 3, 1, 2).contiguous().numpy()

        return temporal_crop
