import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.models import Classifier, Signal_Recover
from models.models import masking2, segment_mask_v1,segment_mask_v3

from models.loss import *
from scipy.spatial.distance import cdist
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
from sklearn.metrics import accuracy_score
from einops import rearrange

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.cross_entropy = nn.CrossEntropyLoss()

    def update(self, *args, **kwargs):
        raise NotImplementedError

class TemSR(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super(TemSR, self).__init__(configs)


        # backbone.
        self.feature_extractor = backbone(configs)
        # classifier.
        self.classifier = Classifier(configs)

        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # recover module used in target domain.
        self.signal_recover = Signal_Recover(configs, hparams)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )


        self.recover_optimizer = torch.optim.Adam(
            self.signal_recover.parameters(),
            lr=hparams['learning_rate'],
            weight_decay=hparams['weight_decay']
        )

        # device
        self.device = device
        self.hparams = hparams

        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1)
        self.mmd_loss = CORAL()  # as hparameter .


    def pretrain(self, src_dataloader, avg_meter, logger):

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                self.pre_optimizer.zero_grad()

                # forward pass correct sequences
                seq_src_feat, src_feat = self.feature_extractor(src_x)   # [batch, num_nodes, feat_dim]

                # classifier predictions
                src_pred = self.classifier(src_feat)

                # normal cross entropy
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                src_cls_loss.backward()
                self.pre_optimizer.step()

                losses = {'cls_loss': src_cls_loss.detach().item()}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        src_only_model = deepcopy(self.network.state_dict())
        return src_only_model

    def update(self, trg_dataloader, avg_meter, logger):

        # defining best and last model
        best_src_risk = float('inf')
        best_model = self.network.state_dict()
        last_model = self.network.state_dict()

        # copy one fixed source domain feature extractor.
        src_feature_extractor = copy.deepcopy(self.feature_extractor)

        # freeze both classifier and source feature extractor
        for k, v in src_feature_extractor.named_parameters():
            v.requires_grad = False
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False

        src_like_epochs = self.hparams["src_like_epochs"]  # Number of epochs for src_like_entropy
        trg_disc_epochs = self.hparams["trg_disc_epochs"]  # Number of epochs for trg_disc_loss

        # sample_bank, entropy_bank = self.build_entropy_bank(trg_dataloader, self.signal_recover,
        #                                                     src_feature_extractor, self.classifier)
        num_sample = len(trg_dataloader.dataset)
        sample_bank = torch.zeros(num_sample, self.configs.input_channels, self.configs.sequence_len).to(self.device)
        entropy_bank = torch.ones(num_sample).to(self.device)

        # print(len(trg_dataloader))

        src_feature_extractor.eval()
        self.classifier.eval()
        self.signal_recover.train()
        self.feature_extractor.train()
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            for step, (trg_x, trg_y, trg_idx) in enumerate(trg_dataloader):
                print(step)

                trg_x = trg_x.float().to(self.device)

                self.optimizer.zero_grad()
                self.recover_optimizer.zero_grad()

                # masking and recover.
                masked_data, [mask, mask_indices] = masking2(trg_x, num_splits=self.hparams['num_splits'],
                                                             num_masked=min(self.hparams['num_masked'],self.hparams['num_splits']))

                recovered_data = self.signal_recover(masked_data)

                # recovered and original samples maximization

                _, recovered_feat = src_feature_extractor(recovered_data) # recovered features.
                trg_max_min_loss, recovered_data_entropy = Bank_info_max_anchor_min(src_feature_extractor, trg_x, recovered_data, recovered_feat,
                                                                                                              entropy_bank, sample_bank,
                                                                                                              self.classifier, self.hparams['anchor_percent'],
                                                                                                              self.hparams['CL_temp'], self.hparams['detach'])
                ### Update the bank
                entropy_bank[trg_idx] = recovered_data_entropy.detach().clone()
                sample_bank[trg_idx] = recovered_data.detach().clone()

                coeffi_anchor = self.hparams["trg_max_min_loss_wt"]*((epoch-1) / self.hparams["num_epochs"])

                raw_feat_seq, raw_feat = self.feature_extractor(trg_x)

                trg_disc_loss = self.mmd_loss(recovered_feat, raw_feat)  # discrepancy loss. (min)


                trg_pred = self.classifier(raw_feat)
                # Entropy loss for target
                trg_ent_loss = EntropyLoss(trg_pred) # entropy loss (min)
                # Computing Gent
                if self.hparams['Gent']:
                    softmax_out = nn.Softmax(dim=1)(trg_pred)
                    msoftmax = softmax_out.mean(dim=0)
                    trg_ent_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

                ## Entropy loss for source-like samples

                src_like_pred = self.classifier(recovered_feat)

                segmented_samples_ls = []
                segmented_sample_mask = segment_mask_v1(recovered_data, mask_indices,
                                                        num_segments=self.hparams['num_splits'],
                                                        num_removed=min(self.hparams['num_masked'],self.hparams['num_splits']))

                segmented_sample_first = segment_mask_v3(recovered_data, posi = 'first', padding=True,
                                                         num_segments=self.hparams['num_splits'],
                                                         num_removed=min(self.hparams['num_masked'],
                                                                         self.hparams['num_splits']))

                segmented_sample_last = segment_mask_v3(recovered_data, posi='last', padding=True,
                                                        num_segments=self.hparams['num_splits'],
                                                        num_removed=min(self.hparams['num_masked'],
                                                                        self.hparams['num_splits']))
                segmented_samples_ls.append(segmented_sample_mask)
                segmented_samples_ls.append(segmented_sample_first)
                segmented_samples_ls.append(segmented_sample_last)
                segmented_samples_ls = torch.cat(segmented_samples_ls, 0)
                _, segmented_feats = src_feature_extractor(segmented_samples_ls)
                segmented_pred = self.classifier(segmented_feats)
                segmented_all_pred = torch.cat((src_like_pred, segmented_pred),0)
                src_like_entropy = Temporal_EntropyLoss_v1(segmented_all_pred, src_like_pred.size(0))

                if self.hparams['Gent']:
                    softmax_out_src_like = nn.Softmax(dim=1)(src_like_pred)
                    msoftmax_src_like = softmax_out_src_like.mean(dim=0)
                    src_like_entropy -= torch.sum(-msoftmax_src_like * torch.log(msoftmax_src_like + 1e-5))
                total_cycle_epochs = src_like_epochs + trg_disc_epochs
                current_cycle_epoch = (epoch - 1) % total_cycle_epochs

                if current_cycle_epoch < src_like_epochs:
                    # Train with src_like_entropy for the specified epochs
                    loss = self.hparams['src_like_entropy_wt'] * src_like_entropy
                    loss += coeffi_anchor * trg_max_min_loss

                else:
                    # Train with trg_disc_loss for the next set of epochs
                    loss = self.hparams['disc_loss_wt'] * trg_disc_loss

                # Overall objective loss (include trg_ent_loss in every epoch)
                loss += self.hparams['ent_loss_wt'] * trg_ent_loss

                loss.backward()
                self.optimizer.step()
                self.recover_optimizer.step()

                losses = {'entropy_loss': trg_ent_loss.detach().item(), 'discrepancy_loss': trg_disc_loss.detach().item(),
                          'src_like_entropy_loss': src_like_entropy.detach().item(),'total_loss':loss.detach().item(),
                          'trg_max_min_loss': trg_max_min_loss.detach().item()}
                for key, val in losses.items():
                    avg_meter[key].update(val, src_like_pred.size(0))

            self.lr_scheduler.step()

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
            # last_model = deepcopy(self.network.state_dict(=))
        return last_model, best_model
