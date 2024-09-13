
def get_sweep_hparams(dataset_name):
    """Return the dataset object (class, dict, etc.) with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]
HAR = {
'TemSR': {
                'pre_learning_rate':                {'values': [5e-3, 1e-3, 5e-4]},
                'learning_rate':                    {'values': [1e-3, 5e-4, 1e-4]},
                'ent_loss_wt':                      {'values': [1e-2, 1e-1, 1, 1.5, 2, 2.5, 3]},
                'disc_loss_wt':                     {'values': [10, 50, 100]},
                'src_like_entropy_wt':              {'values': [1e-1, 1, 1.5, 2, 2.5, 3]},
                'Gent':                             {'values': [True, False]},
                'trg_max_min_loss_wt':              {'distribution': 'uniform', 'min': 2, 'max': 7},
                'AR_hid_dim':                       {'values': [32, 64, 128, 164]},
                'num_splits':                       {'values': [8, 16]},
                'num_masked':                       {'values': [1, 2, 4]},
                'anchor_percent':                   {'values': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
                'CL_temp':                          {'values': [1, 1e-1, 5e-2, 1e-2]},
                'detach':                           {'values': [True, False]},
}
}

EEG = {
'TemSR': {
                'pre_learning_rate':                {'values': [3e-3, 1e-3]},
                'learning_rate':                    {'values': [1e-4, 5e-5, 1e-5, 7e-6]},
                'ent_loss_wt':                      {'values': [1e-1, 1, 1.5, 2, 2.5, 3]},
                'disc_loss_wt':                     {'values': [10, 50, 100]},
                'src_like_entropy_wt':              {'values': [1e-2, 1e-1, 1, 1.5, 2, 2.5, 3]},
                'Gent':                             {'values': [True]},
                'trg_max_min_loss_wt':              {'distribution': 'uniform', 'min': 2, 'max': 7},
                'AR_hid_dim':                       {'values': [64, 128, 164]},
                'num_splits':                       {'values': [8, 15, 30]},
                'num_masked':                       {'values': [1, 4, 8]},
                'anchor_percent':                   {'values': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
                'CL_temp':                          {'values': [1, 1e-1, 5e-2, 1e-2]},
                'detach':                           {'values': [True, False]},
}
}

FD = {
'SFDA_v10': {
                'pre_learning_rate':                {'values': [5e-3, 3e-3, 1e-3, 5e-4, 1e-4]},
                'learning_rate':                    {'values': [5e-4, 1e-4, 5e-5, 1e-5, 7e-6]},
                'ent_loss_wt':                      {'distribution': 'uniform', 'min': 0, 'max': 5},
                'disc_loss_wt':                     {'values': [1, 10, 50, 100]},
                'src_like_entropy_wt':              {'distribution': 'uniform', 'min': 0, 'max': 5},
                'Gent':                             {'values': [True, False]},
                'trg_max_min_loss_wt':              {'distribution': 'uniform', 'min': 2, 'max': 7},
                'AR_hid_dim':                       {'values': [32, 64, 128, 164]},
                'num_splits':                       {'values': [8, 16, 32]},
                'num_masked':                       {'values': [1, 4, 8]},
                'anchor_percent':                   {'values': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
                'CL_temp':                          {'values': [1, 1e-1, 5e-2, 1e-2]},
                'detach':                           {'values': [True, False]},
}

}
