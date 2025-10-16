from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

import matplotlib
matplotlib.use('agg')
from torch.utils.tensorboard.writer import SummaryWriter

from register import Register
register_models = Register()

from collections import OrderedDict
import clip

@register_models
class BaseIRModel(nn.Module):
    def __init__(self, hash_bit, backbone_frozen=True, device = 'cuda:0'):
        super().__init__()
        self.device = device
        
        print('clip32')
        self.cnn_model = clip.load('ViT-B/32', jit=False)[0]
        self.proj = nn.Linear(512, hash_bit)
        self.cnn_model.classifier = nn.Identity()

        if backbone_frozen:
            for param in self.cnn_model.parameters():
                param.requires_grad = False
            for param in self.cnn_model.classifier.parameters():
                param.requires_grad = False

        self.tanh_a = 1.0
        self.global_step = 0
    
    def get_feat(self, imgs):
        x = self.cnn_model.features(imgs)
        x = self.cnn_model.avgpool(x)
        x = torch.flatten(x, 1) 
        x = self.cnn_model.classifier[:-1](x)
        
        return x
    
    def get_code(self, imgs):
        feats = self.get_feat(imgs)
        h = self.proj(feats)
        b = torch.tanh(self.tanh_a*h)
        return feats, h, b

    def set_epoch(self, epoch):
        raise NotImplementedError

    def train_step(self):
        raise NotImplementedError

    def train_epoch_start(self):
        pass

    def train_epoch_end(self):
        pass

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]', strict: bool = True):
        return self.proj.load_state_dict(state_dict, strict)

    def required_state_dict(self):
        return self.proj.state_dict()
    
@register_models("cliph")
class CLIPH(BaseIRModel):
    def __init__(self, args):
        super().__init__(args.hash_bit, device=args.device)  
        self.T = args.T if hasattr(args, 'T') else 0.01
        self.th = args.th if hasattr(args, 'th') else 0.4
        self.neg_th = args.neg_th if hasattr(args, 'neg_th') else 0.2
        text = self._label_table(args.dataset)

        self.text = torch.cat([clip.tokenize(f"a photo of the {c}") for c in text]).to(args.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            self.text_features = self.cnn_model.encode_text(self.text)
            self.text_features = F.normalize(self.text_features, p=2, dim=-1)
            
        self.patch_features = None
        hook_handle = self.cnn_model.visual.transformer.register_forward_hook(self._forward_hook_vtransformer)
        
        if hasattr(args, 'lr'):
            self.optim = optim.SGD(
                [
                    {'params': self.proj.parameters(), 'lr': args.lr},
                ],
                lr=args.lr,
                momentum= 0.9,
                weight_decay=5e-4
            )

    def _forward_hook_vtransformer(self, module, input, output):
        out = output.permute(1, 0, 2)
        out = self.cnn_model.visual.ln_post(out[:, 1:, :])
        out = out @ self.cnn_model.visual.proj
        self.patch_features = out.detach().to(self.device, non_blocking=True)

    def _label_table(self, dataset):
        if dataset == 'flickr':
            file_path = 'data/flickr25k/class_name.txt'
        elif dataset == 'coco2014':
            file_path = 'data/coco2014/class_name80.txt'
        elif dataset == 'nuswide' or dataset == 'nuswide1':
            file_path = 'data/nuswide/class_name21.txt'
        else:
            raise NotImplementedError
        
        result = []
        with open(file_path, 'r') as f:
            for line in f:
                result.append(line.strip())
        
        return result
    
    def _sim_loss(self, img_features, codes):
        img_gcos = F.normalize(img_features, p=2, dim=-1) @ F.normalize(img_features, p=2, dim=-1).T
        code_gcos = F.normalize(codes, p=2, dim=-1) @ F.normalize(codes, p=2, dim=-1).T
        return F.mse_loss(img_gcos / 0.1, code_gcos / 0.1), img_gcos
    
    def _distribution_loss(self, g_features, l_features, codes):
        f_g_cos = g_features @ self.text_features.T
        f_l_cos = l_features @ self.text_features.T

        f_g_prob = F.softmax(f_g_cos / self.T, dim=-1)
        f_l_prob = F.softmax(f_l_cos / self.T, dim=-1)
        f_l_max_values, _ = torch.max(f_l_prob, dim=1)
        f_l_min_values, _ = torch.min(f_l_prob, dim=1)

        threshold = self.th
        f_l = torch.where(f_l_max_values > threshold, f_l_max_values, f_l_min_values)
        f_g = (f_l + f_g_prob) / 2 
        
        f_g_cor = f_g

        f_g_half_soft = torch.where(f_g_cor > threshold, f_g_cor, torch.zeros_like(f_g_cor))
        intersection_half_soft = f_g_half_soft @ f_g_half_soft.T
        
        sum_half_soft = f_g_half_soft.sum(dim=1)

        union_half_soft = sum_half_soft.unsqueeze(1) + sum_half_soft.unsqueeze(0) - intersection_half_soft
        
        iou = torch.where(union_half_soft > 0, intersection_half_soft / union_half_soft, torch.zeros_like(union_half_soft))      
        
        semantic_sim = torch.where(iou > self.neg_th, iou, torch.zeros_like(iou))

        hash_sim = F.normalize(codes, p=2, dim=-1) @ F.normalize(codes, p=2, dim=-1).T
        mse = F.mse_loss(semantic_sim / 0.1, hash_sim / 0.1)
        return mse, semantic_sim

    def train_step(self, data, tblog:SummaryWriter=None):
        self.optim.zero_grad()
        _, imgs, _ = data
        imgs = imgs.to(self.device, non_blocking=True)
        img_feats , _, img_hash = self.get_code(imgs)
        img_patch_f = self.patch_features.float()
        img_patch_f = F.normalize(img_patch_f, p=2, dim=-1)
        
        _distribution_loss, _ = self._distribution_loss(img_feats, img_patch_f, img_hash)
        loss = _distribution_loss

        loss.backward()
        self.optim.step()
        return {'loss': loss}
    
    def get_feat(self, imgs):  
        x = self.cnn_model.encode_image(imgs)  
        x = F.normalize(x, p=2, dim=-1)
        return x

    def get_code(self, imgs):
        feats = self.get_feat(imgs).float()
        h = self.proj(feats)
        b = torch.tanh(self.tanh_a * h)
        return feats, h, b
    
    def train_epoch_start(self):
        pass

    def train_epoch_end(self):
        pass

    def set_epoch(self, epoch):
        pass
    
    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]', strict: bool = True):
        return nn.Module.load_state_dict(self, state_dict, strict)
    
    def required_state_dict(self):
        return self.state_dict()

@register_models("umrch")
class UMRCH(CLIPH):
    def __init__(self, args):
        super().__init__(args)        

        self.temperature = args.temperature if hasattr(args, 'temperature') else 1
        self.alpha = args.alpha if hasattr(args, 'alpha') else 1
        self.beta = args.beta if hasattr(args, 'beta') else 1
        self.lr_strategy = args.lr_strategy if hasattr(args, 'lr_strategy') else None

        self.one = torch.ones(1, args.hash_bit).to(self.device, non_blocking=True)

        if hasattr(args, 'lr'):
            self.param_groups = [
                {'params': self.proj.parameters(), 'lr': args.lr},
            ]

            self.optim = optim.SGD(
                self.param_groups,
                lr=args.lr,
                momentum=0.9,
                weight_decay=5e-4
            )

            if self.lr_strategy is not None:
                self.scheduler = optim.lr_scheduler.OneCycleLR(
                    self.optim, 
                    max_lr=0.1,         
                    total_steps=100, 
                    pct_start=0.3,       
                    anneal_strategy='cos',  
                    final_div_factor=100   
                )
        
    def _contrastive_loss(self, codes, semantic_sim):
        
        N = codes.size(0) // 2 
        indices = torch.arange(N)
        soft_labels = semantic_sim
        soft_labels[indices, indices + N] = 1
        soft_labels[indices + N, indices] = 1
        
        mask = torch.where(soft_labels > 0, 
                           torch.ones_like(soft_labels), 
                           torch.zeros_like(soft_labels)
                           )
        
        similarity_matrix = F.normalize(codes, p=2, dim=-1) @ F.normalize(codes, p=2, dim=-1).T

        soft_pos_num = soft_labels.sum(dim=1)
        logits = similarity_matrix / self.temperature
        
        max_logits, _ = torch.max(logits, dim=1, keepdim=True)
        logits_stable = logits - max_logits

        exp_logits = torch.exp(logits_stable)
        prob = exp_logits

        prob_pos = prob * mask
        prob_neg = prob * (1 - mask)

        pos_num = mask.sum(dim=1)
        neg_num = 2*N - pos_num

        eps = 1e-8

        loss = -(torch.where(
            mask > 0, 
            soft_labels * torch.log((prob_pos + eps) / (prob_pos + prob_neg.sum(dim=1, keepdim=True) + eps)), 
            torch.zeros_like(prob)
            ).sum(dim=1) / (soft_pos_num + eps)).mean()
        
        return loss

    def train_step(self, data, tblog:SummaryWriter=None):
        self.optim.zero_grad()

        _, imgs1, _, imgs2 = data
        imgs1 = imgs1.to(self.device, non_blocking=True)
        imgs2 = imgs2.to(self.device, non_blocking=True)

        img1_feats , _, img1_hash = self.get_code(imgs1)
        img1_patch_f = self.patch_features.float()
        img1_patch_f = F.normalize(img1_patch_f, p=2, dim=-1)

        img2_feats , _, img2_hash = self.get_code(imgs2)
        img2_patch_f = self.patch_features.float()
        img2_patch_f = F.normalize(img2_patch_f, p=2, dim=-1)

        img_feats = torch.cat([img1_feats, img2_feats], dim=0)
        img_patch_f = torch.cat([img1_patch_f, img2_patch_f], dim=0)
        img_hash = torch.cat([img1_hash, img2_hash], dim=0)

        _distribution_loss, semantic_sim = self._distribution_loss(img_feats, img_patch_f, img_hash)
        _contrastive_loss = self._contrastive_loss(img_hash, semantic_sim)
        
        loss = self.alpha * _distribution_loss + self.beta * _contrastive_loss

        loss.backward()
        self.optim.step()
        return {'loss': loss, 'distribution_loss': _distribution_loss, 'contrastive_loss': _contrastive_loss}

    def get_feat(self, imgs):
        x = self.cnn_model.encode_image(imgs)  
        x = F.normalize(x, p=2, dim=-1)
        return x

    def get_code(self, imgs):
        feats = self.get_feat(imgs).float()
        h = self.proj(feats)
        b = torch.tanh(self.tanh_a * h)
        return feats, h, b

    def set_epoch(self, epoch):
        pass
 
    def train_epoch_start(self):
        pass
    
    def train_epoch_end(self):
        if self.lr_strategy is not None:
            self.scheduler.step()
        pass

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        return nn.Module.load_state_dict(self, state_dict, strict)

    def required_state_dict(self):
        return self.state_dict()
