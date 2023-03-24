#from collections import OrderedDict
import torch
import torch.nn.functional as F

class MemoryBank:
    def __init__(self, max_size=128, threshold=(0.7,0.7), simfun='cos', batch_size = 16, feature_dim=256, device='cpu'):
        self.max_size=max_size
        self._mem = [None]*batch_size
        self._mask = [None]*batch_size  #  stored as (D,B)
        self._pos = [None]*batch_size
        # ('feat', torch.Size([64, 16, 256])),
        #  ('mask', torch.Size([16, 64])),
        #  ('pos', torch.Size([64, 16, 256]))
        self._labels = [None]*batch_size
        self.threshold=threshold
        self.simfun = simfun
        self.batch_size = batch_size
        self.feature_dim = feature_dim
        self.device = device

    def update(self, t, mask, pos,  label):
        t = t.detach()
        mask = mask.detach()
        pos = pos.detach()

        for i in range(self.batch_size):
            if self._mem[i] == None:
                self._mem[i] = torch.zeros(0,self.feature_dim,device=self.device)
                self._mask[i] = torch.ones(0, dtype=torch.bool,device=self.device)
                self._pos[i] = torch.ones(0, self.feature_dim, device=self.device)
                self._labels[i] = []

            self._mem[i],self._mask[i],self._pos[i],self._labels[i] = \
                self._update(t[...,i,:], mask[...,i,:], pos[...,i,:], label, self._mem[i],self._mask[i],self._pos[i],self._labels[i])

    def _update(self, new_feat, new_mask, new_pos, new_label, mem, mask, pos,label):

        new_feat = torch.masked_select(new_feat,new_mask.view(-1,1).logical_not()).view(-1,self.feature_dim)
        new_pos = torch.masked_select(new_pos, new_mask.view(-1, 1).logical_not()).view(-1, self.feature_dim)
        new_mask = torch.masked_select(new_mask, new_mask.logical_not())


        if new_label==None:
            new_label=''
        else:
            new_label = str(new_label)

        if mem.shape[0]>0:

            if self.simfun == 'cos':
                similarity = F.normalize(new_feat, dim=-1) @ F.normalize(mem, dim=-1).transpose(1,0) # (D, mem_size)  -1 to 1
            elif self.simfun == 'l2':
                similarity = (new_feat.view(-1,1,self.feature_dim) - mem.view(1,-1,self.feature_dim))**2 # 0 to +infinity
                similarity = (1-similarity)/(1+similarity)

            #(D, mem_size)
            maxsim_mem = torch.amax(similarity, dim=0) # (mem_size)
            maxsim_t = torch.amax(similarity, dim=1) # (D)

            # mem keep
            mask_keep = maxsim_mem<self.threshold[1] # len: (mem_size)
            mem = torch.masked_select(mem, mask_keep.view(-1,1)).view(-1,self.feature_dim)
            mask = torch.masked_select(mask, mask_keep)
            pos = torch.masked_select(pos, mask_keep.view(-1,1)).view(-1,self.feature_dim)
            label = [label[i] for i in range(len(label)) if mask_keep[i]]

            # new template
            mask_add = (maxsim_t<self.threshold[0]).logical_or(maxsim_t>=self.threshold[1] )
            add_feat = torch.masked_select(new_feat, mask_add.view(-1,1)).view(-1,self.feature_dim)
            add_mask = torch.masked_select(new_mask, mask_add)
            add_pos = torch.masked_select(new_pos, mask_add.view(-1,1)).view(-1,self.feature_dim)

            mem = torch.cat([mem,add_feat],dim=0)
            mask = torch.cat([mask,add_mask],dim=0)
            pos = torch.cat([pos, add_pos], dim=0)
            label = label + [new_label] * add_feat.shape[0]

            mem = mem[-self.max_size:,...]
            mem = mem.contiguous()
            mask = mask[-self.max_size:,...]
            mask = mask.contiguous()
            pos = pos[-self.max_size:, ...]
            pos = pos.contiguous()
            label = label[-self.max_size:]


        else:
            #new_feat, new_mask, new_pos, new_label
            mem = new_feat
            mask = new_mask
            pos = new_pos
            label = [new_label] * new_feat.shape[0]

            mem = mem[-self.max_size:, ...]
            mem = mem.contiguous()
            mask = mask[-self.max_size:, ...]
            mask = mask.contiguous()
            pos = pos[-self.max_size:, ...]
            pos = pos.contiguous()
            label = label[-self.max_size:]

        return mem, mask, pos, label


    def clear(self):
        self._mem = [None] * self.batch_size
        self._mask = [None] * self.batch_size  # stored as (D,B)
        self._pos = [None] * self.batch_size
        # ('feat', torch.Size([64, 16, 256])),
        #  ('mask', torch.Size([16, 64])),
        #  ('pos', torch.Size([64, 16, 256]))
        self._labels = [None] * self.batch_size

    def _pad(self, tensor, ret_size, pad_value):
        if tensor.dim() == 1:
            if ret_size[-1] != tensor.shape[-1]:
                tensor = F.pad(tensor,(0,ret_size[-1]-tensor.shape[-1]),value=pad_value)
        elif tensor.dim() == 2:
            if ret_size[-1] != tensor.shape[-1] or ret_size[-2] != tensor.shape[-2]:
                tensor = F.pad(tensor, (0, ret_size[-1]-tensor.shape[-1], 0, ret_size[-2]-tensor.shape[-2]), value=pad_value)
        return tensor




    def get(self, get_label=False):


        # ('feat', torch.Size([64, 16, 256])),
        #  ('mask', torch.Size([16, 64])),
        #  ('pos', torch.Size([64, 16, 256]))
        ret = {}
        ret['feat'] = torch.stack([ self._pad(f,(self.max_size,self.feature_dim),0.) for f in self._mem], dim=-2)
        ret['mask'] = torch.stack([self._pad(f, (self.max_size,), False) for f in self._mask], dim=-2)
        ret['pos'] = torch.stack([self._pad(f, (self.max_size, self.feature_dim), 0.) for f in self._pos], dim=-2)

        return ret
