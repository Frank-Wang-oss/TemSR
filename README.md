# Temporal Source Recovery for Time Series Unsupervised Domain Adaptation [[Paper]()] 


[//]: # (<p align="center">)

[//]: # (<img src="misc/temporal_adapt.PNG" width="900" class="center">)

[//]: # (</p>)

## Requirmenets:
- Python3
- Pytorch==1.10
- Numpy==1.22
- scikit-learn==1.4.1
- Pandas==1.3.5
- skorch==0.15.0 
- openpyxl==3.0.10 (for classification reports)
- Wandb=0.16.6 (for sweeps)

## Datasets

### Available Datasets
We used four public datasets in this study. We also provide the **preprocessed** versions as follows:
- [SSC](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/UD1IM9)
- [UCIHAR](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/0SYHTZ)
- [MFD](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/PU85XN)





## Training procedure

The experiments are organised in a hierarchical way such that:
- Several experiments are collected under one directory assigned by `--experiment_description`.
- Each experiment could have different trials, each is specified by `--run_description`.

### Training a model

To train a model:

```
python trainers/train.py  --experiment_description exp1  \
                --run_description run_1 \
                --da_method TemSR \
                --dataset HAR \
                --num_runs 3 \
```
### Launching a sweep
Sweeps here are deployed on [Wandb](https://wandb.ai/), which makes it easier for visualization, following the training progress, organizing sweeps, and collecting results.

```
python trainers/train.py  --experiment_description exp1_sweep  \
                --run_description sweep_over_lr \
                --da_method TemSR \
                --dataset HAR \
                --num_runs 3\
                --num_sweeps 50 \
                --sweep_project_wandb TemSR_HAR \
                --is_sweep True
```
Upon the run, you will find the running progress in the specified project page in wandb.




## Results
- Each run will have all the cross-domain scenarios results in the format `src_to_trg_run_x`, where `x`
is the run_id (you can have multiple runs by assigning `--num_runs` arg). 
- Under each directory, you will find the classification report, a log file, and the different risks scores. (you can also save the checkpoint by uncomment the code in train.py)
- By the end of the all the runs, you will find the overall average and std results in the run directory.



## Citation
If you found this work useful for you, please consider citing it (paper will be available soon).

[//]: # (```)

[//]: # (@inproceedings{mpau,)

[//]: # (  author = {Ragab, Mohamed and Eldele, Emadeldeen and Foo, Chuan-Sheng and Wu, Min and Li, Xiaoli and Chen, Zhenghua},)

[//]: # (  title = {Source-Free Domain Adaptation with Temporal Imputation for Time Series Data},)

[//]: # (  booktitle={29th SIGKDD Conference on Knowledge Discovery and Data Mining - Research Track},)

[//]: # (  year = {2023},)

[//]: # (  url={https://openreview.net/forum?id=v6GK0ijPW0B})

[//]: # (})

[//]: # (```)


