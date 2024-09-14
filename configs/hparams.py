def get_hparams_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

class HAR():
    def __init__(self):
        super(HAR, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 100,
            'lr_decay': 0.5
        }
        self.alg_hparams = {
            'TemSR':  {
                      'pre_learning_rate': 0.001, 'learning_rate': 0.0005, 'ent_loss_wt': 0.1,
                      'disc_loss_wt': 50, 'src_like_entropy_wt': 1.5, 'Gent': True, 'src_like_epochs': 3,
                      'trg_disc_epochs': 3, 'trg_max_min_loss_wt': 3.64, 'AR_hid_dim': 64, 'num_splits': 8,
                      'num_masked': 1, 'anchor_percent':0.3, 'CL_temp': 0.01, 'detach': True
                      }
        }

class EEG():
    def __init__(self):
        super(EEG, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5
        }

        self.alg_hparams = {
            'TemSR': {
                     'pre_learning_rate': 0.001, 'learning_rate': 0.00005, 'ent_loss_wt': 1,
                     'disc_loss_wt': 50, 'src_like_entropy_wt': 1, 'Gent': True, 'src_like_epochs': 3,
                     'trg_disc_epochs': 3,'trg_max_min_loss_wt': 2.09309,'AR_hid_dim': 64, 'num_splits': 8,
                     'num_masked': 1, 'anchor_percent': 0.3, 'CL_temp': 0.05, 'detach': True
                     }
        }

class FD():
    def __init__(self):
        super(FD, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5
        }
        self.alg_hparams = {
            'TemSR': {
                     'pre_learning_rate': 0.003, 'learning_rate': 0.000007, 'ent_loss_wt': 4.87809,
                     'disc_loss_wt': 50, 'src_like_entropy_wt': 3.45814, 'Gent': False, 'src_like_epochs': 3,
                     'trg_disc_epochs': 3, 'trg_max_min_loss_wt': 5.26417,'AR_hid_dim': 128, 'num_splits': 8,
                     'num_masked': 1,'anchor_percent': 0.3, 'CL_temp': 0.05, 'detach': False
                     }
        }

