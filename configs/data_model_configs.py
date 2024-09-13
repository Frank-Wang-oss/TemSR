import os
import torch


def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

class HAR():
    def __init__(self):
        super(HAR, self)
        self.scenarios = [("2", "11"), ("6", "23"), ("7", "13"), ("9", "18"), ("12", "16")]
        self.class_names = ['walk', 'upstairs', 'downstairs', 'sit', 'stand', 'lie']
        self.sequence_len = 128
        self.shuffle = True
        self.drop_last = False
        self.normalize = True

        # model configs
        self.input_channels = 9
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 6

        # features
        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1
        self.AR_hid_dim_raw = self.input_channels


        self.AR_hid_dim = 64

        # Temporal recovering (masking)
        self.num_splits = 8
        self.num_masked = 1

        # Temporal recovering (regularization)
        self.num_segments = 8
        self.num_removed = 1

        ## Anchor
        self.anchor_percent = 0.2
class EEG():
    def __init__(self):
        super(EEG, self).__init__()
        # data parameters
        self.scenarios = [("0", "11"), ("12", "5"), ("7", "18"), ("16", "1"), ("9", "14")]
        self.class_names = ['W', 'N1', 'N2', 'N3', 'REM']
        self.sequence_len = 3000
        self.shuffle = True
        self.drop_last = True
        self.normalize = True


        # model configs
        self.input_channels = 1
        self.kernel_size = 25
        self.stride = 6
        self.dropout = 0.2
        self.num_classes = 5

        # features
        self.mid_channels = 16
        self.final_out_channels = 8
        self.features_len = 1
        self.AR_hid_dim_raw = self.input_channels

        self.AR_hid_dim = 64

        # Temporal recovering (masking)
        self.num_splits = 8
        self.num_masked = 1

        # Temporal recovering (regularization)
        self.num_segments = 8
        self.num_removed = 1

        ## Anchor
        self.anchor_percent = 0.2
class FD():
    def __init__(self):
        super(FD, self).__init__()
        self.scenarios = [("0", "1"), ("1", "2"), ("3", "1"), ("1", "0"), ("2", "3")]
        self.class_names = ['Healthy', 'D1', 'D2']
        self.sequence_len = 5120
        self.shuffle = True
        self.drop_last = False
        self.normalize = True

        # Model configs
        self.input_channels = 1
        self.kernel_size = 32
        self.stride = 6
        self.dropout = 0.5
        self.num_classes = 3

        # features

        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1
        self.AR_hid_dim_raw = self.input_channels

        self.AR_hid_dim = 64

        # Temporal recovering (masking)
        self.num_splits = 8
        self.num_masked = 1

        # Temporal recovering (regularization)
        self.num_segments = 8
        self.num_removed = 1

        ## Anchor
        self.anchor_percent = 0.2