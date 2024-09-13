import itertools

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class
import math
from torch.nn.parameter import Parameter
import numpy as np

def get_backbone_class(backbone_name):
    """Return the algorithm class with the given name."""
    if backbone_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(backbone_name))
    return globals()[backbone_name]

### BACKBONE NETWORKS ###

### 1D-CNN ###
class CNN(nn.Module):
    def __init__(self, configs):
        super(CNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x_flat = self.adaptive_pool(x).view(x.shape[0], -1)
        return x, x_flat


class Classifier(nn.Module):
    def __init__(self, configs):
        super(Classifier, self).__init__()

        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

    def forward(self, x):
        predictions = self.logits(x)
        return predictions


class Signal_Recover(nn.Module):
    def __init__(self, configs, hparams):
        super(Signal_Recover, self).__init__()
        self.seq_length = configs.sequence_len
        self.num_channels = configs.input_channels
        self.hid_dim = configs.AR_hid_dim_raw
        # input size: batch_size, 9 channel, 128 seq_length
        self.rnn_encoder = nn.LSTM(input_size=self.num_channels, hidden_size=hparams['AR_hid_dim'], batch_first=True)
        self.rnn_decoder = nn.LSTM(input_size=hparams['AR_hid_dim'], hidden_size=self.hid_dim, batch_first=True)

    def forward(self, x):
        ### x size (bs, N, L)

        x = x.transpose(-1,-2)
        out, (h, c) = self.rnn_encoder(x)
        out, (h, c) = self.rnn_decoder(out)

        out = out.transpose(-1,-2)
        # take the last time step
        return out

def get_configs(dataset):
    dataset_class = get_dataset_class(dataset)
    hparams_class = get_hparams_class(dataset)
    return dataset_class(), hparams_class()

# temporal masking
def masking(x, num_splits=8, num_masked=4):
    # num_masked = int(masking_ratio * num_splits)
    patches = rearrange(x, 'a b (p l) -> a b p l', p=num_splits)
    masked_patches = patches.clone()  # deepcopy(patches)
    # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
    rand_indices = torch.rand(x.shape[1], num_splits).argsort(dim=-1)
    selected_indices = rand_indices[:, :num_masked]
    masks = []
    for i in range(masked_patches.shape[1]):
        masks.append(masked_patches[:, i, (selected_indices[i, :]), :])
        masked_patches[:, i, (selected_indices[i, :]), :] = 0
        # orig_patches[:, i, (selected_indices[i, :]), :] =
    mask = rearrange(torch.stack(masks), 'b a p l -> a b (p l)')
    masked_x = rearrange(masked_patches, 'a b p l -> a b (p l)', p=num_splits)

    return masked_x, mask
def masking2(x, num_splits=8, num_masked=4): ## also segment v1
    # Reshape input tensor to create patches
    ### Input size (bs, num_channels, time_length)
    patches = rearrange(x, 'b n (p l) -> b n p l', p=num_splits)
    masked_patches = patches.clone()

    # Generate random indices for masking
    rand_indices = torch.rand(x.shape[1], num_splits).argsort(dim=-1)
    selected_indices = rand_indices[:, :num_masked]
    # Create a mask tensor
    mask = torch.zeros_like(patches, dtype=torch.bool)

    # Create a batch index tensor
    batch_indices = torch.arange(x.shape[0]).unsqueeze(1).unsqueeze(2)

    # Apply the mask using advanced indexing
    mask[batch_indices, torch.arange(x.shape[1]).view(1, -1, 1), selected_indices.unsqueeze(0).expand(x.shape[0], -1, -1)] = True  # 32,1,1  1,6,1  32,6,1

    # Mask the selected patches
    masked_patches[mask] = 0

    # Reshape the masked patches and the mask back to the original shape
    mask = rearrange(mask, 'b n p l -> b n (p l)')
    masked_x = rearrange(masked_patches, 'b n p l -> b n (p l)', p=num_splits)

    return masked_x, [mask,rand_indices]


def masking3(x): ## also segment v1
    # Reshape input tensor to create patches
    ### Input size (bs, num_channels, time_length)
    mask = (torch.rand_like(x) > 0.3).float()

    masked_x = x * mask
    return masked_x

def segment_random_v1(x, num_segments=8, num_removed=2):
    # Reshape input tensor to create patches
    ### Input size (bs, num_channels, time_length)
    patches = rearrange(x, 'a b (p l) -> a b p l', p=num_segments)
    masked_patches = patches.clone()

    # Generate random indices for masking
    rand_indices = torch.rand(x.shape[1], num_segments).argsort(dim=-1)
    selected_indices = rand_indices[:, :num_removed]
    # Create a mask tensor
    mask = torch.zeros_like(patches, dtype=torch.bool)

    # Create a batch index tensor
    batch_indices = torch.arange(x.shape[0]).unsqueeze(1).unsqueeze(2)

    # Apply the mask using advanced indexing
    mask[batch_indices, torch.arange(x.shape[1]).view(1, -1, 1), selected_indices.unsqueeze(0).expand(x.shape[0], -1, -1)] = True  # 32,1,1  1,6,1  32,6,1

    # Mask the selected patches
    masked_patches[mask] = 0

    # Reshape the masked patches and the mask back to the original shape
    masked_x = rearrange(masked_patches, 'a b p l -> a b (p l)', p=num_segments)

    return masked_x

def segment_random_v2(x, num_segments=8, num_removed=2):
    # Reshape input tensor to create patches
    ### Input size (bs, num_channels, time_length)
    patches = rearrange(x, 'a b (p l) -> a b p l', p=num_segments)
    masked_patches = patches.clone()

    # Generate random indices for masking
    rand_indices = torch.rand(x.shape[1], num_segments).argsort(dim=-1)
    selected_indices_to_remove = rand_indices[:, :num_removed]


    # Create a mask tensor
    mask = torch.ones_like(patches, dtype=torch.bool)

    # Create a batch index tensor
    batch_indices = torch.arange(x.shape[0]).unsqueeze(1).unsqueeze(2)

    # Apply the mask using advanced indexing
    mask[batch_indices, torch.arange(x.shape[1]).view(1, -1, 1), selected_indices_to_remove.unsqueeze(0).expand(x.shape[0], -1, -1)] = False  # 32,1,1  1,6,1  32,6,1

    # Mask the selected patches
    masked_patches = masked_patches[mask].view(x.shape[0], x.shape[1], num_segments-num_removed, -1)
    # Reshape the masked patches and the mask back to the original shape
    masked_x = rearrange(masked_patches, 'a b p l -> a b (p l)', p=num_segments-num_removed)

    return masked_x

def segment_mask_v1(x, mask_indices, num_segments, num_removed=2):
    # Reshape input tensor to create patches
    ### Input size (bs, num_channels, time_length)
    patches = rearrange(x, 'a b (p l) -> a b p l', p=num_segments)


    masked_patches = patches.clone()

    # Generate random indices for masking
    selected_indices = mask_indices[:, -num_removed:]
    # Create a mask tensor
    mask = torch.zeros_like(patches, dtype=torch.bool)

    # Create a batch index tensor
    batch_indices = torch.arange(x.shape[0]).unsqueeze(1).unsqueeze(2)

    # Apply the mask using advanced indexing
    mask[batch_indices, torch.arange(x.shape[1]).view(1, -1, 1), selected_indices.unsqueeze(0).expand(x.shape[0], -1, -1)] = True  # 32,1,1  1,6,1  32,6,1

    # Mask the selected patches
    masked_patches[mask] = 0

    # Reshape the masked patches and the mask back to the original shape
    masked_x = rearrange(masked_patches, 'a b p l -> a b (p l)', p=num_segments)

    return masked_x


def segment_mask_v2(x, mask_indices, num_segments, num_removed=2):
    # Reshape input tensor to create patches
    ### Input size (bs, num_channels, time_length)
    patches = rearrange(x, 'a b (p l) -> a b p l', p=num_segments)


    masked_patches = patches.clone()

    # Generate random indices for masking
    selected_indices = mask_indices[:, -num_removed:]
    # Create a mask tensor
    mask = torch.ones_like(patches, dtype=torch.bool)

    # Create a batch index tensor
    batch_indices = torch.arange(x.shape[0]).unsqueeze(1).unsqueeze(2)

    # Apply the mask using advanced indexing
    mask[batch_indices, torch.arange(x.shape[1]).view(1, -1, 1), selected_indices.unsqueeze(0).expand(x.shape[0], -1, -1)] = False  # 32,1,1  1,6,1  32,6,1

    # Mask the selected patches
    masked_patches = masked_patches[mask].view(x.shape[0], x.shape[1], num_segments-num_removed, -1)
    # Reshape the masked patches and the mask back to the original shape
    masked_x = rearrange(masked_patches, 'a b p l -> a b (p l)', p=num_segments-num_removed)

    return masked_x

def segment_mask_v3(x, posi, padding, num_segments, num_removed=2):
    # Reshape input tensor to create patches
    ### Input size (bs, num_channels, time_length)
    patches = rearrange(x, 'a b (p l) -> a b p l', p=num_segments)


    masked_patches = patches.clone()

    if posi == 'first':
        if padding:
            masked_patches[:, :, :num_removed, :] = 0
        else:
            masked_patches = masked_patches[:,:,num_removed:,:]
    elif posi == 'last':
        if padding:
            masked_patches[:, :, -num_removed:, :] = 0
        else:
            masked_patches = masked_patches[:, :, :-num_removed, :]
    else:
        raise NotImplementedError('Not accurate position for segments')
    p = num_segments if padding else num_segments-num_removed
    masked_x = rearrange(masked_patches, 'a b p l -> a b (p l)', p=p)

    return masked_x


if __name__ == '__main__':
    net = nn.LSTM(1,8,batch_first=True)
    bs, num_channel, time_length = 1, 1, 8
    num_split = 8
    x = torch.rand(bs, num_channel, time_length)
    print(x)
    masked_x, [mask, mask_indices] = masking2(x,num_split)
    print(masked_x)
    masked_x,_ = net(masked_x)
    # segment_x = segment_mask_v1(masked_x, mask_indices,num_split)
    # print(segment_x.detach())
    # segment_x = segment_mask_v2(masked_x, mask_indices,num_split)
    # print(segment_x.detach())
    segment_x = segment_mask_v3(masked_x, 'last', True,num_split)
    print(segment_x)


















