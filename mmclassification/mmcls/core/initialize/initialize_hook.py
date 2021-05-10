import os.path as osp
import torch.nn as nn

from mmcv.runner import Hook
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle
import torch


class InitializeHook(Hook):

    def __init__(self, dataloader, interval=1, kmeans=True, by_epoch=True):
        #if not isinstance(dataloader, DataLoader):
        #    raise TypeError('dataloader must be a pytorch DataLoader, but got'
        #                    f' {type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.by_epoch = by_epoch
        self.kmeans = kmeans

    def before_train_epoch(self, runner):
        if runner.epoch == 0:
            if self.kmeans:
                from mmcls.apis import da_single_gpu_test
                features_s, mlp_features_s, results_s = da_single_gpu_test(runner.model, self.dataloader[0], bar_show=False, show=False)
                features_t, mlp_features_t, results_t = da_single_gpu_test(runner.model, self.dataloader[1], bar_show=False, show=False)
                label_s = torch.from_numpy(self.dataloader[0].dataset.get_gt_labels())
                label_t = torch.from_numpy(self.dataloader[1].dataset.get_gt_labels())
                mode = {'mode': 'cosine', 'norm':True}
                #runner.model.module.head.class_map.weight = nn.Parameter(self.initialize_class_map(features_s, label_s, features_t, label_t, mode))
                #runner.model.module.head.mlp_class_map.weight = nn.Parameter(self.initialize_class_map(mlp_features_s, label_s, mlp_features_t, label_t, mode))
                pseudo_label, _ = self.initialize_class_map(features_s, label_s, features_t, label_t, mode)
                runner.data_loader.dataset.update(pseudo_label)
                print('K-Means initializing psuedo Label done')
            else:
                from mmcls.apis import da_single_gpu_test
                _, _, results_t = da_single_gpu_test(runner.model, self.dataloader[1], bar_show=False, show=False)
                results_t = torch.from_numpy(np.vstack(results_t))
                _, pseudo_label = torch.max(results_t, dim=1)
                runner.data_loader.dataset.update(pseudo_label)
                print('Pseudo Label initialization done')

    """
    def after_train_epoch(self, runner):
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmcls.apis import da_single_gpu_test
        _, _, results_s = da_single_gpu_test(runner.model, self.dataloader[0], test_mode=self.test_mode, show=False)
        _, _, results_t = da_single_gpu_test(runner.model, self.dataloader[1], test_mode=self.test_mode, show=False)
        self.evaluate(runner, results_s, results_t)

    def after_train_iter(self, runner):
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from mmcls.apis import da_single_gpu_test
        runner.log_buffer.clear()
        #results_s = da_single_gpu_test(runner.model, self.dataloader[0], test_mode=self.test_mode, show=False)
        results_t = da_single_gpu_test(runner.model, self.dataloader[1], test_mode=self.test_mode, show=False)
        self.evaluate(runner, results_s, results_t)
    """
    def initialize_class_map(self, features_s, label_s, features_t, label_t, mode):
        center_s = self.calculate_center(features_s, label_s)
        pseudo_label, center_t = self.k_means(features=features_t, center=center_s, gt_label=label_s, label_t=label_t, **mode)
        return pseudo_label, center_t

    def k_means(self, mode, norm, features, center, gt_label, label_t):
        modes = {'cosine': self.cosine_dist, 'mse': self.mse_dist}
        dist_fn = modes.get(mode)
        center_now = center
        feat = features
        refs = torch.Tensor(range(31)).unsqueeze(1)
        gt_label_t = label_t
        gt_mask = (gt_label_t == refs).unsqueeze(2)
        gt_class_count = gt_mask.sum(dim=1)
    
        dist = dist_fn(feat, center_now, norm)
        _, pred_init = torch.min(dist, dim=1)
    
        cluster_iter = 0
    
        while True:
    
            dist = dist_fn(feat, center_now, norm)
            _, pred = torch.min(dist, dim=1)
            if cluster_iter == 0:
                result_init = pred == gt_label_t
                classwise_result_init = result_init.unsqueeze(0) * gt_mask.squeeze()
                acc_init = result_init.sum()/len(gt_label_t)
                classwise_acc_init = (classwise_result_init.sum(dim=1, keepdim=True)/gt_class_count).squeeze()
    
            if (cluster_iter != 0 and sum(pred == pred_init) == len(pred)) or cluster_iter == 100:
                result = pred == gt_label_t
                classwise_result = result.unsqueeze(0) * gt_mask.squeeze()
                acc = result.sum()/len(gt_label_t)
                classwise_acc = (classwise_result.sum(dim=1, keepdim=True)/gt_class_count).squeeze()
                print(acc)
                print(acc_init)
                #return acc_init, classwise_acc_init, acc, classwise_acc
            #if (cluster_iter != 0 and sum(pred == pred_init) == len(pred)) or cluster_iter == 100:
                return pred.to(torch.device('cuda')), center_now.to(torch.device('cuda'))
            mask = (pred == refs).unsqueeze(2)
            num_mask = mask.sum(dim=1)
            center_now = torch.sum(feat.unsqueeze(0) * mask, dim=1) / (num_mask + 10e-6)
            pred_init = pred
            cluster_iter += 1
    
    def calculate_center(self, feature, label):
        refs = torch.Tensor(range(31)).unsqueeze(1)
        mask = (label == refs).unsqueeze(2)
        num = mask.sum(dim=1)
        feat = feature.unsqueeze(0)
        center = torch.sum(feat * mask, dim=1) / (num + 10e-6)
    
        return center
    
    def cosine_dist(self, feature, center, norm):
        if norm:
            return - torch.matmul(feature/feature.norm(dim=1, keepdim=True), (center/center.norm(dim=1, keepdim=True)).T)
        else:
            return - torch.matmul(feature, center.T)
    
    
    def mse_dist(self, feature, center, norm):
        if norm:
            dist = (feature/feature.norm(dim=1, keepdim=True)).unsqueeze(1) - (center/center.norm(dim=1, keepdim=True)).unsqueeze(0)
        else:
            dist = feature.unsqueeze(1) - center.unsqueeze(0)
        return dist.pow(2).sum(2)
    
