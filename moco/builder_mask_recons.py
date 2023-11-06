import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .GRU import BIGRU
import random

def loss_kld(inputs, targets):
    inputs = F.log_softmax(inputs, dim=1)
    targets = F.softmax(targets, dim=1)
    return F.kl_div(inputs, targets, reduction='batchmean')

# initilize weight
def weights_init_gru(model):
    with torch.no_grad():
        for child in list(model.children()):
            print(child)
            for param in list(child.parameters()):
                  if param.dim() == 2:
                        nn.init.xavier_uniform_(param)
    print('GRU weights initialization finished!')

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

class MoCo(nn.Module):
    def __init__(self, skeleton_representation, args_bi_gru, dim=128, K=65536, m=0.999, T=0.07,
                 teacher_T=0.05, student_T=0.1, cmd_weight=1.0, topk=1024, mlp=False, pretrain=True):
        super(MoCo, self).__init__()
        self.pretrain = pretrain
        if pretrain:
            self.register_parameter('cl_prompt', nn.Parameter(torch.zeros((1,256,))) )
            self.register_parameter('mp_prompt', nn.Parameter(torch.zeros((1,256,))) )
            self.register_parameter('base_prompt', nn.Parameter(torch.zeros((1,1,150,))) )
            self.register_parameter('mask_prompt', nn.Parameter(torch.zeros((1,1,150,))) )
            self.register_parameter('recons_prompt', nn.Parameter(torch.zeros((1,1,150,))) )
            self.register_parameter('mix_prompt', nn.Parameter(torch.zeros((1,1,150,))) )
            torch.nn.init.xavier_uniform_(self.cl_prompt)
            torch.nn.init.xavier_uniform_(self.mp_prompt)
            torch.nn.init.xavier_uniform_(self.base_prompt)
            torch.nn.init.xavier_uniform_(self.mask_prompt)
            torch.nn.init.xavier_uniform_(self.recons_prompt)
            torch.nn.init.xavier_uniform_(self.mix_prompt)

        self.Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                     (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                     (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]

        self.swap_mode = 'swap'
        self.spatial_mode = 'semantic'
        if not self.pretrain:
            self.encoder_q = BIGRU(**args_bi_gru)
            weights_init_gru(self.encoder_q)
        else:
            self.K = K
            self.m = m
            self.T = T
            self.teacher_T = teacher_T
            self.student_T = student_T
            self.cmd_weight = cmd_weight
            self.topk = topk
            mlp=mlp
            print(" MoCo parameters",K,m,T,mlp)
            print(" PCM3 parameters: teacher-T %.2f, student-T %.2f, cmd-weight: %.2f, topk: %d"%(teacher_T,student_T,cmd_weight,topk))
            print(skeleton_representation)

            self.local_rank = 0
            self.encoder_q = BIGRU(**args_bi_gru)
            self.encoder_k = BIGRU(**args_bi_gru)

            weights_init_gru(self.encoder_q)
            weights_init_gru(self.encoder_k)

            #projection heads
            if mlp:  # hack: brute-force replacement
                dim_mlp = self.encoder_q.fc.weight.shape[1]
                self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                    nn.ReLU(),
                                                    self.encoder_q.fc)
                self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                    nn.ReLU(),
                                                    self.encoder_k.fc)
                #decoder for reconstruction
                self.linear = nn.Sequential(nn.Linear(dim_mlp, 512), nn.BatchNorm1d(512),
                                            nn.ReLU(inplace=True), nn.Linear(512, 512))
                self.recons_decoder = DEC(input_size=150,frame=64,hidden_size=512)


            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)    # initialize
                param_k.requires_grad = False       # not update by gradient
            

            # create the queue
            self.register_buffer("queue", torch.randn(dim, self.K))
            self.queue = F.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


    @torch.no_grad()
    def ske_swap(self, x, spa_l, spa_u, tem_l, tem_u, p=None):
        '''
        swap a batch skeleton
        T   64 --> 32 --> 16    # 8n
        S   25 --> 25 --> 25 (5 parts)
        '''
        N, C, T, V, M = x.size()
        tem_downsample_ratio = 4

        # generate swap swap idx
        idx = torch.arange(N)
        n = torch.randint(1, N - 1, (1,))
        randidx = (idx + n) % N

        # ------ Spatial ------ #
        if self.spatial_mode == 'semantic':
            Cs = random.randint(spa_l, spa_u)
            # sample the parts index
            parts_idx = random.sample(body_parts, Cs)
            # generate spa_idx
            spa_idx = []
            for part_idx in parts_idx:
                spa_idx += part_idx
            spa_idx.sort()
        else:
            raise ValueError('Not supported operation {}'.format(self.spatial_mode))
        # spa_idx = torch.tensor(spa_idx, dtype=torch.long).cuda()

        # ------ Temporal ------ #
        Ct = random.randint(tem_l, tem_u)
        tem_idx = random.randint(0, T // tem_downsample_ratio - Ct)
        rt = Ct * tem_downsample_ratio

        xst = x.clone()
        if p==None:
            p = random.random()

        if p > 0.5:
            # begin swap
            if self.swap_mode == 'swap':
                xst[:, :, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :] = \
                    xst[randidx][:, :, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :]
            elif self.swap_mode == 'zeros':
                xst[:, :, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :] = 0
            elif self.swap_mode == 'Gaussian':
                xst[:, :, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :] = \
                    torch.randn(N, C, rt, len(spa_idx), M).cuda()
            else:
                raise ValueError('Not supported operation {}'.format(self.swap_mode))
            # generate mask
            mask = torch.zeros(T // tem_downsample_ratio, V).cuda()
            mask[tem_idx:tem_idx + Ct, spa_idx] = 1
        elif p <= 0.5 and p > 0.25:
            N, C, T, V, M = xst.size()
            xst_temp = xst[:, :, : 2 * rt]
            xst_temp = xst_temp.permute(0, 4, 3, 1, 2).contiguous()
            xst_temp = xst_temp.view(N * M, V * C, -1)
            xst_temp = torch.nn.functional.interpolate(xst_temp, size=rt)
            xst_temp = xst_temp.view(N, M, V, C, rt)
            xst_temp = xst_temp.permute(0, 3, 4, 2, 1).contiguous()
            xst[:, :, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :] = \
                    xst_temp[randidx][:, :, :, spa_idx, :]
            mask = torch.zeros(T // tem_downsample_ratio, V).cuda()
            mask[tem_idx:tem_idx + Ct, spa_idx] = 1
        else:
            lamb = random.random()
            xst = xst * (1 - lamb) + xst[randidx] * lamb
            mask = torch.zeros(T // tem_downsample_ratio, V).cuda() + lamb

        return randidx, xst, mask
    
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys,idx=None):
        if self.local_rank!=-1:
            keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        if idx!=None:
            self.index[ptr:ptr + batch_size] = idx
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr
    
    @torch.no_grad()
    def update_ptr(self, batch_size):
        assert self.K % batch_size == 0  # for simplicity
        #simply pass, ptr has been updated in the deque and enque 
        pass
        #self.queue_ptr[0] = (self.queue_ptr[0] + batch_size) % self.K

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        if self.local_rank==-1:
            return x, None
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        if self.local_rank==-1:
            return x
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]
    
    def forward(self, im_q, im_k=None, im_q_nor=None, im_k_nor=None, im_q_mask=None, mask=None, view='joint', knn_eval=False):

        # Permute and Reshape
        N, C, T, V, M = im_q.size()
        im_q = im_q.permute(0,2,3,1,4).reshape(N,T,-1)#N, T, C*V*M
        if not self.pretrain:
            if view == 'joint':
                return self.encoder_q(im_q, knn_eval)
            else:
                raise ValueError        
    
    def forward_baseline(self, im_q, im_k=None, view='joint', knn_eval=False,local_rank=0):
        #每一路都用infonce loss
        #aug使用: base aug

        # Permute and Reshape
        self.local_rank = local_rank
        N, C, T, V, M = im_q.size()
        im_q = im_q.permute(0,2,3,1,4).reshape(N,T,-1)#N, T, C*V*M
        

        if not self.pretrain:
            if view == 'joint':
                return self.encoder_q(im_q, knn_eval)
            else:
                raise ValueError      
        
        # Permute and Reshape
        im_k = im_k.permute(0,2,3,1,4).reshape(N,T,-1)

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = F.normalize(q, dim=1)

        # compute key features for  s1 and  s2  skeleton representations 
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels
    
    def forward_pcm3_woprompt(self, im_q, im_k=None, im_mask=None, view='joint', knn_eval=False,local_rank=0,mask=None):

        if mask!=None:
            im_mask = im_mask.permute(0,2,4,3,1)#NTMVC 
            im_mask = im_mask*mask
            mask_region = (mask==0)[:,:,:,:,0]
            im_mask[:,:,:,:,0][mask_region] = 0.0#self.mask_param[0]
            im_mask[:,:,:,:,1][mask_region] = 0.0#self.mask_param[1]
            im_mask[:,:,:,:,2][mask_region] = 0.0#self.mask_param[2]
            im_mask = im_mask.permute(0,4,1,3,2)#NCTVM
            mask_copy = mask.clone()

        # Permute and Reshape
        self.local_rank = local_rank
        N, C, T, V, M = im_q.size()
        randidx, im_pc, mask = self.ske_swap(im_q, 2, 3, 7, 11)
        lamb = mask.mean()

        im_q = im_q.permute(0,2,3,1,4).reshape(N,T,-1)#N, T, V*C*M
        im_mask = im_mask.permute(0,2,3,1,4).reshape(N,T,-1)#N, T, V*C*M
        im_pc = im_pc.permute(0,2,3,1,4).reshape(N,T,-1)

        if not self.pretrain:
            if view == 'joint':
                return self.encoder_q(im_q, knn_eval)
            else:
                raise ValueError      
        
        # Permute and Reshape
        im_k = im_k.permute(0,2,3,1,4).reshape(N,T,-1)
        
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = F.normalize(q, dim=1)

        seq_feat = self.encoder_q(im_mask,all_seq=True)  # NTC
        seq_feat_copy = seq_feat.clone() 
        N,T,D = seq_feat.shape
        seq_feat = self.linear(seq_feat.reshape(N*T,D)).reshape(N,T,-1)
        recons_data = self.recons_decoder(seq_feat) # N T VCM
        recons_data_copy = recons_data.clone()
        recons_data = recons_data.reshape(N,T,V,C,M).permute(0,3,1,2,4)#->N C T V M

        mask_q = self.encoder_q.fc(seq_feat_copy[:,-1,:])
        mask_q = F.normalize(mask_q, dim=1)

        #im_mask N T VCM
        #recons_data_copy N T VCM
        #mask_copy N T M V C 
        assert im_mask.shape==recons_data_copy.shape
        mask_copy = mask_copy.permute(0,1,3,4,2).reshape(N,T,V*C*M)
        #TODO reshape
        recons_im = recons_data_copy*(1-mask_copy) + im_mask*mask_copy
        recons_q = self.encoder_q(recons_im.detach())
        recons_q = F.normalize(recons_q, dim=1)

        pc = self.encoder_q(im_pc)
        pc = F.normalize(pc, dim=1)

        # compute key features for  s1 and  s2  skeleton representations 
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

            kc = k * (1 - lamb) + k[randidx] * lamb
            kc = F.normalize(kc, dim=1)

        # MOCO
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        #TODO calculate the logits of mask
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        l_pos_mask = torch.einsum('nc,nc->n', [mask_q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_mask = torch.einsum('nc,ck->nk', [mask_q, self.queue.clone().detach()])

        l_pos_recons = torch.einsum('nc,nc->n', [recons_q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_recons = torch.einsum('nc,ck->nk', [recons_q, self.queue.clone().detach()])

        l_pos_mix = torch.einsum('nc,nc->n', [pc, kc]).unsqueeze(-1)
        l_neg_mix = torch.einsum('nc,ck->nk', [pc, self.queue.clone().detach()])

        lk_neg = torch.einsum('nc,ck->nk', [k, self.queue.clone().detach()])
        lk_neg_mix = torch.einsum('nc,ck->nk', [kc, self.queue.clone().detach()])
        
        # Top-k
        
        lk_neg_topk = lk_neg
        topk_idx = torch.arange(0,lk_neg.shape[1]).cuda().unsqueeze(0).repeat(N,1)
        lk_neg_mix_topk = lk_neg_mix
        #from CMD
        loss_cmd = loss_kld(torch.gather(l_neg, -1, topk_idx) / self.student_T, lk_neg_topk / self.teacher_T) + loss_kld(torch.gather(l_neg_mix, -1, topk_idx) / self.student_T, lk_neg_mix_topk / self.teacher_T) + \
             loss_kld(torch.gather(l_neg_mask, -1, topk_idx) / self.student_T, lk_neg_topk / self.teacher_T) + loss_kld(torch.gather(l_neg_recons, -1, topk_idx) / self.student_T, lk_neg_topk / self.teacher_T)
        
        #  logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits_mask = torch.cat([l_pos_mask, l_neg_mask], dim=1)
        logits_recons = torch.cat([l_pos_recons, l_neg_recons], dim=1)
        logits_mix = torch.cat([l_pos_mix,l_neg_mix],dim=1)


        # apply temperature
        logits /= self.T
        logits_mask /= self.T
        logits_recons /= self.T
        logits_mix /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, logits_mix, logits_recons,logits_mask, labels, recons_data, loss_cmd * self.cmd_weight

    def forward_pcm3_wprompt(self, im_q, im_k=None, im_mask=None, view='joint', knn_eval=False,local_rank=0,mask=None):

        if mask!=None:
            im_mask = im_mask.permute(0,2,4,3,1)#NTMVC 
            im_mask = im_mask*mask
            mask_region = (mask==0)[:,:,:,:,0]
            im_mask[:,:,:,:,0][mask_region] = 0.0#self.mask_param[0]
            im_mask[:,:,:,:,1][mask_region] = 0.0#self.mask_param[1]
            im_mask[:,:,:,:,2][mask_region] = 0.0#self.mask_param[2]
            im_mask = im_mask.permute(0,4,1,3,2)#NCTVM
            mask_copy = mask.clone()

        prompt_size = self.cl_prompt.shape[1]//2
        st_idx = random.randint(0,1024-prompt_size-1)

        # Permute and Reshape
        self.local_rank = local_rank
        N, C, T, V, M = im_q.size()
        randidx, im_pc, mask = self.ske_swap(im_q, 2, 3, 7, 11)
        lamb = mask.mean()

        im_q = im_q.permute(0,2,3,1,4).reshape(N,T,-1)#N, T, V*C*M
        im_mask = im_mask.permute(0,2,3,1,4).reshape(N,T,-1)#N, T, V*C*M
        im_pc = im_pc.permute(0,2,3,1,4).reshape(N,T,-1)

        if not self.pretrain:
            if view == 'joint':
                return self.encoder_q(im_q, knn_eval)
            else:
                raise ValueError      
        
        # Permute and Reshape
        im_k = im_k.permute(0,2,3,1,4).reshape(N,T,-1)
        
        # compute query features
        q = self.encoder_q(im_q+self.base_prompt,small_prompt=self.cl_prompt,st_idx=st_idx)  # queries: NxC
        q = F.normalize(q, dim=1)

        seq_feat = self.encoder_q(im_mask+self.mask_prompt,all_seq=True)  # NTC
        seq_feat_copy = seq_feat.clone() 
        N,T,D = seq_feat.shape
        seq_feat = seq_feat.reshape(N*T,D)
        st_idx = random.randint(0,1024-prompt_size-1)
        seq_feat[:,st_idx+1024:st_idx+1024+prompt_size] += self.mp_prompt[:, prompt_size:]
        seq_feat[:,st_idx:st_idx+prompt_size] += self.mp_prompt[:, 0:prompt_size]

        seq_feat_cl = seq_feat.clone().detach()#stop gradient of the encoder
        seq_feat_cl = self.linear(seq_feat_cl.reshape(N*T,D)).reshape(N,T,-1)
        recons_data_cl = self.recons_decoder(seq_feat_cl) # N T VCM


        seq_feat = self.linear(seq_feat).reshape(N,T,-1)
        recons_data = self.recons_decoder(seq_feat) # N T VCM
        recons_data_copy = recons_data.clone()
        recons_data = recons_data.reshape(N,T,V,C,M).permute(0,3,1,2,4)#->N C T V M


        seq_feat_copy = seq_feat_copy[:,-1,:]
        st_idx = random.randint(0,1024-prompt_size-1)
        seq_feat_copy[:,st_idx+1024:st_idx+1024+prompt_size] += self.cl_prompt[:, prompt_size:]
        seq_feat_copy[:,st_idx:st_idx+prompt_size] += self.cl_prompt[:, 0:prompt_size]

        mask_q = self.encoder_q.fc(seq_feat_copy)
        mask_q = F.normalize(mask_q, dim=1)

        #im_mask N T VCM
        #recons_data_cl N T VCM
        #mask_copy N T M V C 
        assert im_mask.shape==recons_data_cl.shape
        mask_copy = mask_copy.permute(0,1,3,4,2).reshape(N,T,V*C*M)
        #TODO reshape
        recons_im = recons_data_cl*(1-mask_copy) + im_mask*mask_copy
        st_idx = random.randint(0,1024-prompt_size-1)
        recons_q = self.encoder_q(recons_im+self.recons_prompt,small_prompt=self.cl_prompt,st_idx=st_idx)
        recons_q = F.normalize(recons_q, dim=1)
        
        st_idx = random.randint(0,1024-prompt_size-1)
        pc = self.encoder_q(im_pc+self.mix_prompt,small_prompt=self.cl_prompt,st_idx=st_idx)
        pc = F.normalize(pc, dim=1)

        # compute key features for  s1 and  s2  skeleton representations 
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            st_idx = random.randint(0,1024-prompt_size-1)
            k = self.encoder_k(im_k+self.base_prompt,small_prompt=self.cl_prompt,st_idx=st_idx)  # keys: NxC
            k = F.normalize(k, dim=1)
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

            kc = k * (1 - lamb) + k[randidx] * lamb
            kc = F.normalize(kc, dim=1)

        # MOCO
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        #TODO calculate the logits of mask
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        l_pos_mask = torch.einsum('nc,nc->n', [mask_q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_mask = torch.einsum('nc,ck->nk', [mask_q, self.queue.clone().detach()])

        l_pos_recons = torch.einsum('nc,nc->n', [recons_q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_recons = torch.einsum('nc,ck->nk', [recons_q, self.queue.clone().detach()])

        l_pos_mix = torch.einsum('nc,nc->n', [pc, kc]).unsqueeze(-1)
        l_neg_mix = torch.einsum('nc,ck->nk', [pc, self.queue.clone().detach()])

        lk_neg = torch.einsum('nc,ck->nk', [k, self.queue.clone().detach()])
        lk_neg_mix = torch.einsum('nc,ck->nk', [kc, self.queue.clone().detach()])
        
        # No Top-k operation
        lk_neg_topk = lk_neg
        topk_idx = torch.arange(0,lk_neg.shape[1]).cuda().unsqueeze(0).repeat(N,1)
        lk_neg_mix_topk = lk_neg_mix
        #from CMD
        loss_cmd = loss_kld(torch.gather(l_neg, -1, topk_idx) / self.student_T, lk_neg_topk / self.teacher_T) + loss_kld(torch.gather(l_neg_mix, -1, topk_idx) / self.student_T, lk_neg_mix_topk / self.teacher_T) + \
             loss_kld(torch.gather(l_neg_mask, -1, topk_idx) / self.student_T, lk_neg_topk / self.teacher_T) + loss_kld(torch.gather(l_neg_recons, -1, topk_idx) / self.student_T, lk_neg_topk / self.teacher_T)
        
        #  logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits_mask = torch.cat([l_pos_mask, l_neg_mask], dim=1)
        logits_recons = torch.cat([l_pos_recons, l_neg_recons], dim=1)
        logits_mix = torch.cat([l_pos_mix,l_neg_mix],dim=1)


        # apply temperature
        logits /= self.T
        logits_mask /= self.T
        logits_recons /= self.T
        logits_mix /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, logits_mix, logits_recons,logits_mask, labels, recons_data, loss_cmd * self.cmd_weight
    
class DEC(nn.Module):
    def __init__(self, input_size, frame,hidden_size):
        super(DEC, self).__init__()
        self.frame = frame
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.decoder = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True
        )
        self.reconstruction = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, input):
        self.decoder.flatten_parameters()
        X, _ = self.decoder(input)
        X = self.reconstruction(X)
        return X
        
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output