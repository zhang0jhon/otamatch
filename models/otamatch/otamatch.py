import pickle
import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler
from collections import Counter
import os
import contextlib
from train_utils import AverageMeter

from .otamatch_utils import Get_Scalar, distributed_sinkhorn
from train_utils import ce_loss, wd_loss, EMA, Bn_Controller

import math
from sklearn.metrics import *
from copy import deepcopy


class OTAMatch:
    def __init__(self, net_builder, num_classes, ema_m, T, p_cutoff, lambda_u, \
                 hard_label=True, t_fn=None, p_fn=None, it=0, num_eval_iter=1000, tb_log=None, logger=None):
        """
        class OTAMatch contains setter of data_loader, optimizer, and model update methods.
        Args:
            net_builder: backbone network class (see net_builder in utils.py)
            num_classes: # of label classes 
            ema_m: momentum of exponential moving average for eval_model
            T: Temperature scaling parameter for output sharpening (only when hard_label = False)
            p_cutoff: confidence cutoff parameters for loss masking
            lambda_u: ratio of unsupervised loss to supervised loss
            hard_label: If True, consistency regularization use a hard pseudo label.
            it: initial iteration count
            num_eval_iter: freqeuncy of iteration (after 500,000 iters)
            tb_log: tensorboard writer (see train_utils.py)
            logger: logger (see utils.py)
        """

        super(OTAMatch, self).__init__()

        # momentum update param
        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m

        # create the encoders
        # network is builded only by num_classes,
        # other configs are covered in main.py

        self.model = net_builder(num_classes=num_classes)
        self.ema_model = None

        self.num_eval_iter = num_eval_iter
        self.t_fn = Get_Scalar(T)  # temperature params function
        self.p_fn = Get_Scalar(p_cutoff)  # confidence cutoff function
        self.lambda_u = lambda_u
        self.tb_log = tb_log
        self.use_hard_label = hard_label

        self.optimizer = None
        self.scheduler = None

        self.it = 0
        self.logger = logger
        self.print_fn = print if logger is None else logger.info

        self.bn_controller = Bn_Controller()

    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')

    def set_dset(self, dset):
        self.ulb_dset = dset

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, args, logger=None):

        ngpus_per_node = torch.cuda.device_count()

        # EMA Init
        self.model.train()
        self.ema = EMA(self.model, self.ema_m)
        self.ema.register()
        if args.resume == True:
            self.ema.load(self.ema_model)

        # p(y) based on the labeled examples seen during training
        dist_file_name = r"./data_statistics/" + args.dataset + '_' + str(args.num_labels) + '.json'
        if args.dataset.upper() == 'IMAGENET':
            p_target = None
        else:
            with open(dist_file_name, 'r') as f:
                p_target = json.loads(f.read())
                p_target = torch.tensor(p_target['distribution'])
                p_target = p_target.cuda(args.gpu)
            # print('p_target:', p_target)

        p_model = None

        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)

        start_batch.record()
        best_eval_acc, best_it = 0.0, 0

        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        # eval for once to verify if the checkpoint is loaded correctly
        if args.resume == True:
            eval_dict = self.evaluate(args=args)
            print(eval_dict)

        queue_length = len(self.ulb_dset)
        # queue_length -= queue_length % (args.batch_size * args.uratio * args.world_size)
        print("queue length: {}".format(queue_length))
        queue = torch.ones((queue_length, args.num_classes)).cuda(args.gpu) / args.num_classes
        # queue = F.log_softmax(queue, dim=-1)

        # selected_label = torch.ones((len(self.ulb_dset),), dtype=torch.long, ) * -1
        selected_label = torch.randint(0, args.num_classes, (len(self.ulb_dset),), dtype=torch.long)
        selected_label = selected_label.cuda(args.gpu)

        # classwise_acc = torch.zeros((args.num_classes,)).cuda(args.gpu)
        assign_weights = torch.ones((1, args.num_classes)).cuda(args.gpu)

        for (_, x_lb, y_lb), (x_ulb_idx, x_ulb_w, x_ulb_s) in zip(self.loader_dict['train_lb'],
                                                                  self.loader_dict['train_ulb']):
            # prevent the training iterations exceed args.num_train_iter
            if self.it > args.num_train_iter:
                break

            end_batch.record()
            torch.cuda.synchronize()
            start_run.record()

            num_lb = x_lb.shape[0]
            num_ulb = x_ulb_w.shape[0]
            assert num_ulb == x_ulb_s.shape[0]

            x_lb, x_ulb_w, x_ulb_s = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu), x_ulb_s.cuda(args.gpu)
            x_ulb_idx = x_ulb_idx.cuda(args.gpu)
            y_lb = y_lb.cuda(args.gpu)

            pseudo_counter = Counter(selected_label.tolist())
            if max(pseudo_counter.values()) < len(self.ulb_dset):  # not all(5w) -1
                for i in range(args.num_classes):
                    # assign_weights[:, i] = 1 - pseudo_counter[i] / float(len(self.ulb_dset))
                    assign_weights[:, i] = 1 if pseudo_counter[i]==0 else 1 + math.log(max(pseudo_counter.values()) / pseudo_counter[i])

            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))

            # inference and calculate sup/unsup losses
            with amp_cm():
                logits, embeddings = self.model(inputs)
                logits_x_lb = logits[:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
                embeddings_x_ulb = embeddings[num_lb:]
                # embeddings_x_ulb_w, embeddings_x_ulb_s = embeddings[num_lb:].chunk(2)
                sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')

                # hyper-params for update
                T = self.t_fn(self.it)
                p_cutoff = self.p_fn(self.it)

                # consistency regularization
                pseudo_label_w = torch.softmax(logits_x_ulb_w.detach(), dim=-1)
                max_probs_w, max_idx_w = torch.max(pseudo_label_w, dim=-1)

                #pseudo_label_s = torch.softmax(logits_x_ulb_s.detach(), dim=-1)
                #max_probs_s, max_idx_s = torch.max(pseudo_label_s, dim=-1)

                # ota
                with torch.no_grad():
                    pseudo_label = pseudo_label_w
                    max_probs, max_idx = torch.max(pseudo_label, dim=-1)

                    selected_label[x_ulb_idx] = max_idx

                    # update the queue
                    epsilon = (self.it / args.num_train_iter) ** args.gamma * args.epsilon_0 + 1. - args.epsilon_0 # convex gamma>1 concave gamma<1 linear gamma=1
                    queue[x_ulb_idx] = pseudo_label * epsilon + torch.ones_like(pseudo_label) * (1-epsilon) / args.num_classes
                    
                    # assignment
                    assignments = distributed_sinkhorn(queue, tau=args.tau)[x_ulb_idx]
                    if torch.isnan(assignments).any():
                        print("assignments contain nan, use pseudo label.")
                        assignments = pseudo_label

                    # assign_max_probs, assign_idx = torch.max(assignments, dim=-1)

                # todo: focal ota loss
                ota_loss = torch.sum(-assign_weights * assignments * F.log_softmax(logits_x_ulb_s, dim=-1), dim=-1)
                ota_loss = ota_loss.mean()

                # contrastive learning
                # contrast_loss = torch.zeros([1]).cuda(args.gpu)
                conf_mask = max_probs.ge(args.p_cutoff).float().repeat(1, 2)
                # conf_mask = max_probs.le(args.p_cutoff).float().repeat(1, 2)
                embeddings_logits = torch.div(torch.matmul(embeddings_x_ulb, embeddings_x_ulb.t()), args.temp) # embeddings.t()

                # supcontrast 
                labels = max_idx.contiguous().view(-1, 1)
                mask = torch.eq(labels, labels.t()).float().repeat(2, 2)
                # simclr 
                # mask = torch.eye(pseudo_label.shape[0], dtype=torch.float32).repeat(2, 2).cuda(args.gpu)

                # mask-out self-contrast cases
                logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(embeddings_logits.shape[0]).view(-1, 1).cuda(args.gpu), 0)
                # mask-out both high confidence cases
                both_high_conf_mask = conf_mask * conf_mask.t()
                logits_mask = logits_mask * both_high_conf_mask
                mask = mask * logits_mask #* conf_mask * conf_mask.t()
                    
                # compute log_prob
                exp_logits = torch.exp(embeddings_logits) * logits_mask #* conf_mask * conf_mask.t()
                log_prob = embeddings_logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)
                    
                # compute mean of log-likelihood over positive
                mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-10)
                contrast_loss = - mean_log_prob_pos.mean()

                total_loss = sup_loss + ota_loss + contrast_loss

            # parameter updates
            if args.amp:
                scaler.scale(total_loss).backward()
                if (args.clip > 0):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if (args.clip > 0):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                self.optimizer.step()

            self.scheduler.step()
            self.ema.update()
            self.model.zero_grad()

            end_run.record()
            torch.cuda.synchronize()

            # tensorboard_dict update
            tb_dict = {}
            tb_dict['train/sup_loss'] = sup_loss.detach()
            tb_dict['train/ota_loss'] = ota_loss.detach()
            tb_dict['train/contrast_loss'] = contrast_loss.detach()
            tb_dict['train/total_loss'] = total_loss.detach()
            tb_dict['train/mask_ratio'] = conf_mask.mean().detach()
            tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
            tb_dict['train/counter'] = np.array([pseudo_counter[i] for i in range(args.num_classes)])
            tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch) / 1000.
            tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.

            # Save model for each 10K steps and best model for each 1K steps
            if self.it % 10000 == 0:
                save_path = os.path.join(args.save_dir, args.save_name)
                if not args.multiprocessing_distributed or \
                        (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    self.save_model('latest_model.pth', save_path)

            if self.it % self.num_eval_iter == 0:
                eval_dict = self.evaluate(args=args)
                tb_dict.update(eval_dict)
                save_path = os.path.join(args.save_dir, args.save_name)
                if tb_dict['eval/top-1-acc'] > best_eval_acc:
                    best_eval_acc = tb_dict['eval/top-1-acc']
                    best_it = self.it
                self.print_fn(
                    f"{self.it} iteration, USE_EMA: {self.ema_m != 0}, {tb_dict}, BEST_EVAL_ACC: {best_eval_acc}, at {best_it} iters")

                if not args.multiprocessing_distributed or \
                        (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):

                    if self.it == best_it:
                        self.save_model('model_best.pth', save_path)
                    if not self.tb_log is None:
                        tb_dict.pop('train/counter')
                        self.tb_log.update(tb_dict, self.it)

            self.it += 1
            del tb_dict
            start_batch.record()
            if self.it > 0.8 * args.num_train_iter:
                self.num_eval_iter = 1000

        eval_dict = self.evaluate(args=args)
        eval_dict.update({'eval/best_acc': best_eval_acc, 'eval/best_it': best_it})
        return eval_dict

    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None):
        self.model.eval()
        self.ema.apply_shadow()
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_logits = []
        for _, x, y in eval_loader:
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            logits, _ = self.model(x)
            loss = F.cross_entropy(logits, y, reduction='mean')
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            y_logits.extend(torch.softmax(logits, dim=-1).cpu().tolist())
            total_loss += loss.detach() * num_batch
        top1 = accuracy_score(y_true, y_pred)
        top5 = top_k_accuracy_score(y_true, y_logits, k=5)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        F1 = f1_score(y_true, y_pred, average='macro')
        AUC = roc_auc_score(y_true, y_logits, multi_class='ovo')
        cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        self.ema.restore()
        self.model.train()
        return {'eval/loss': total_loss / total_num, 'eval/top-1-acc': top1, 'eval/top-5-acc': top5,
                'eval/precision': precision, 'eval/recall': recall, 'eval/F1': F1, 'eval/AUC': AUC}

    def save_model(self, save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        # copy EMA parameters to ema_model for saving with model as temp
        self.model.eval()
        self.ema.apply_shadow()
        ema_model = self.model.state_dict()
        self.ema.restore()
        self.model.train()

        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it,
                    'ema_model': ema_model},
                   save_filename)

        self.print_fn(f"model saved: {save_filename}")

    def load_model(self, load_path):
        checkpoint = torch.load(load_path)

        self.model.load_state_dict(checkpoint['model'])
        self.ema_model = deepcopy(self.model)
        self.ema_model.load_state_dict(checkpoint['ema_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.it = checkpoint['it']
        self.print_fn('model loaded')
    
    def load_pretrain(self, pretrain_path):
        self.print_fn("load checkpoint from {}".format(pretrain_path))
        state_dict = torch.load(pretrain_path)
        for k in list(state_dict.keys()):
            state_dict['module.'+k] = state_dict[k]
            del state_dict[k]
        msg = self.model.load_state_dict(state_dict, strict=False)
        self.print_fn(msg.missing_keys)
        self.ema_model = deepcopy(self.model)

    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]


if __name__ == "__main__":
    pass
