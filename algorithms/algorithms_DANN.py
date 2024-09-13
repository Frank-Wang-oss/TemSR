import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.models import Classifier, Signal_Recover
from models.models import masking2, segment_random_v1, segment_random_v2, segment_mask_v1,segment_mask_v2,segment_mask_v3
from models.loss import CrossEntropyLabelSmooth, EntropyLoss, MMD_loss, Temporal_EntropyLoss_v1, anchor_pull_loss, CORAL, AdversarialLoss
from scipy.spatial.distance import cdist
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
from sklearn.metrics import accuracy_score

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

# Baselines
class SFDA_DEMO(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super(SFDA_DEMO, self).__init__(configs)



        # backbone.
        self.feature_extractor = backbone(configs)
        # classifier.
        self.classifier = Classifier(configs)

        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # recover module used in target domain.
        self.signal_recover = Signal_Recover(configs)

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

        self.adversarial_loss = AdversarialLoss(configs.features_len * configs.final_out_channels).to(device)

        self.adver_optimizer = torch.optim.Adam(
            self.adversarial_loss.domain_classifier.parameters(),
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

        # obtain pseudo labels
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            for step, (trg_x, trg_y, trg_idx) in enumerate(trg_dataloader):

                trg_x = trg_x.float().to(self.device)

                self.optimizer.zero_grad()
                self.recover_optimizer.zero_grad()
                self.adver_optimizer.zero_grad()

                # masking and recover.
                masked_data, [mask, mask_indices] = masking2(trg_x, num_splits=self.configs.num_splits, num_masked=self.configs.num_masked)
                recovered_data = self.signal_recover(masked_data)


                # recovered and original samples maximization
                trg_max_min_loss = anchor_pull_loss(trg_x, recovered_data, src_feature_extractor, self.classifier,percent=self.configs.anchor_percent)

                recovered_feat_seq, recovered_feat = src_feature_extractor(recovered_data) # recovered features.

                raw_feat_seq, raw_feat = self.feature_extractor(trg_x)

                trg_disc_loss = self.adversarial_loss(recovered_feat, raw_feat)  # discrepancy loss. (min)


                trg_pred = self.classifier(raw_feat)
                # Entropy loss for target
                trg_ent_loss = EntropyLoss(trg_pred) # entropy loss (min)
                # Computing Gent
                if self.hparams['Gent']:
                    softmax_out = nn.Softmax(dim=1)(trg_pred)
                    msoftmax = softmax_out.mean(dim=0)
                    trg_ent_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

                # trg_prob = nn.Softmax(dim=1)(trg_pred)

                ## Entropy loss for source-like samples
                src_like_pred = self.classifier(recovered_feat)
                if self.hparams['regularization_loss_type'] == 'regularization_v1':

                    segmented_samples_ls = []
                    for _ in range(3):
                        segmented_samples = segment_random_v1(recovered_data,num_segments=self.configs.num_segments,
                                                              num_removed=self.configs.num_removed)
                        segmented_samples_ls.append(segmented_samples)
                    segmented_samples_ls = torch.cat(segmented_samples_ls,0)

                elif self.hparams['regularization_loss_type'] == 'regularization_v2':
                    segmented_samples_ls = []
                    for _ in range(3):
                        segmented_samples = segment_random_v2(recovered_data, num_segments=self.configs.num_segments,
                                                              num_removed=self.configs.num_removed)
                        segmented_samples_ls.append(segmented_samples)
                    segmented_samples_ls = torch.cat(segmented_samples_ls, 0)

                elif self.hparams['regularization_loss_type'] == 'regularization_v3':
                    segmented_samples_ls = []
                    segmented_sample_mask = segment_mask_v1(recovered_data, mask_indices,
                                                            num_segments=self.configs.num_segments,
                                                            num_removed=self.configs.num_removed)

                    segmented_sample_first = segment_mask_v3(recovered_data, posi = 'first', padding=True,
                                                             num_segments=self.configs.num_segments,
                                                             num_removed=self.configs.num_removed)

                    segmented_sample_last = segment_mask_v3(recovered_data, posi='last', padding=True,
                                                             num_segments=self.configs.num_segments,
                                                             num_removed=self.configs.num_removed)
                    segmented_samples_ls.append(segmented_sample_mask)
                    segmented_samples_ls.append(segmented_sample_first)
                    segmented_samples_ls.append(segmented_sample_last)
                    segmented_samples_ls = torch.cat(segmented_samples_ls, 0)
                elif self.hparams['regularization_loss_type'] == 'regularization_v4':
                    segmented_samples_ls = []
                    segmented_sample_mask = segment_mask_v2(recovered_data, mask_indices,
                                                            num_segments=self.configs.num_segments,
                                                            num_removed=self.configs.num_removed)

                    segmented_sample_first = segment_mask_v3(recovered_data, posi='first', padding=False,
                                                             num_segments=self.configs.num_segments,
                                                             num_removed=self.configs.num_removed)

                    segmented_sample_last = segment_mask_v3(recovered_data, posi='last', padding=False,
                                                            num_segments=self.configs.num_segments,
                                                            num_removed=self.configs.num_removed)
                    segmented_samples_ls.append(segmented_sample_mask)
                    segmented_samples_ls.append(segmented_sample_first)
                    segmented_samples_ls.append(segmented_sample_last)
                    segmented_samples_ls = torch.cat(segmented_samples_ls, 0)
                _, segmented_feats = src_feature_extractor(segmented_samples_ls)
                segmented_pred = self.classifier(segmented_feats)
                segmented_all_pred = torch.cat((src_like_pred, segmented_pred),0)
                src_like_entropy_diff, src_like_entropy = Temporal_EntropyLoss_v1(segmented_all_pred, src_like_pred.size(0))

                # Overall objective loss
                loss = self.hparams['ent_loss_wt'] * trg_ent_loss + \
                       self.hparams['disc_loss_wt'] * trg_disc_loss + \
                       self.hparams['trg_max_min_loss_wt'] * trg_max_min_loss + \
                       self.hparams['src_like_entropy_diff_wt'] * src_like_entropy_diff + \
                       self.hparams['src_like_entropy_wt'] * src_like_entropy
                # loss = self.hparams['disc_loss_wt'] * trg_disc_loss + \
                #                        self.hparams['trg_max_min_loss_wt'] * trg_max_min_loss + \
                #                        self.hparams['src_like_entropy_diff_wt'] * src_like_entropy_diff + \
                #                        self.hparams['src_like_entropy_wt'] * src_like_entropy
                loss.backward()
                self.optimizer.step()
                self.recover_optimizer.step()
                self.adver_optimizer.step()

                losses = {'entropy_loss': trg_ent_loss.detach().item(), 'discrepancy_loss': trg_disc_loss.detach().item(), 'trg_max_min_loss': trg_max_min_loss.detach().item(),
                          'src_like_entropy_diff_loss': src_like_entropy_diff.detach().item(), 'src_like_entropy_loss': src_like_entropy.detach().item(), 'total_loss':loss.detach().item()}
                for key, val in losses.items():
                    avg_meter[key].update(val, self.hparams['batch_size'])

            self.lr_scheduler.step()

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
            # last_model = deepcopy(self.network.state_dict())
        return last_model, best_model
