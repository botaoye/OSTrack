# OSTrack
The official implementation for the **ECCV 2022** paper [_Joint Feature Learning and Relation Modeling for Tracking: A One-Stream Framework_](https://arxiv.org/abs/2203.11991) 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/joint-feature-learning-and-relation-modeling/visual-object-tracking-on-lasot)](https://paperswithcode.com/sota/visual-object-tracking-on-got-10k?p=joint-feature-learning-and-relation-modeling)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/joint-feature-learning-and-relation-modeling/visual-object-tracking-on-got-10k)](https://paperswithcode.com/sota/visual-object-tracking-on-got-10k?p=joint-feature-learning-and-relation-modeling)

[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/joint-feature-learning-and-relation-modeling/visual-object-tracking-on-trackingnet&#41;]&#40;https://paperswithcode.com/sota/visual-object-tracking-on-trackingnet?p=joint-feature-learning-and-relation-modeling&#41;)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/joint-feature-learning-and-relation-modeling/visual-object-tracking-on-uav123)](https://paperswithcode.com/sota/visual-object-tracking-on-uav123?p=joint-feature-learning-and-relation-modeling)

<p align="center">
  <img width="85%" src="https://github.com/botaoye/OSTrack/blob/main/assets/arch.png" alt="Framework"/>
</p>


## :sunny: Highlights

### :star2: New One-stream Tracking Framework
OSTrack is a simple, neat, high-performance **one-stream tracking framework** for joint feature learning and relational modeling based on self-attention operators.
Without any additional temporal information, OSTrack achieves SOTA performance on multiple benchmarks. OSTrack can serve as a strong baseline for further research.

| Tracker     | GOT-10K (AO) | LaSOT (AUC) | TrackingNet (AUC) | UAV123(AUC) |
|:-----------:|:------------:|:-----------:|:-----------------:|:-----------:|
| OSTrack-384 | 73.7         | 71.1        | 83.9              | 70.7        |
| OSTrack-256 | 71.0         | 69.1        | 83.1              | 68.3        |


### :star2: Fast Training
OSTrack-256 can be trained in ~23 hours with 4*2080Ti (11GB of memory per GPU), which is much faster than recent SOTA transformer-based trackers. The fast training speed comes from:

1. While previous Siamese-stype trackers required separate feeding of the template and search region into the backbone at each iteration of training, OSTrack directly combines the template and search region. The tight and highly parallelized structure results in improved training and inference speed.
  
2. The proposed early candidate elimination (ECE) module significantly reduce memory and time consumption.
  
3. Pretrained Transformer weights enables faster convergence.

### :star2: Good performance-speed trade-off

[//]: # (![speed_vs_performance]&#40;https://github.com/botaoye/OSTrack/blob/main/assets/speed_vs_performance.png&#41;)
<p align="center">
  <img width="70%" src="https://github.com/botaoye/OSTrack/blob/main/assets/speed_vs_performance.png" alt="speed_vs_performance"/>
</p>

## Install the environment
**Option1**: Use the Anaconda
```
conda create -n ostrack python=3.8
conda activate ostrack
bash install.sh
```

**Option2**: Use the docker file

We provide the full docker file here.


## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Data Preparation
Put the tracking datasets in ./data. It should look like:
   ```
   ${PROJECT_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- images
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
   ```


## Training
Download pre-trained [MAE ViT-Base weights](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) and put it under `$PROJECT_ROOT$/pretrained_models` (different pretrained models can also be used, see [MAE](https://github.com/facebookresearch/mae) for more details).

```
python tracking/train.py --script ostrack --config vitb_256_mae_ce_32x4_ep300 --save_dir ./output --mode multiple --nproc_per_node 4 --use_wandb 1
```

Replace `--config` with the desired model config under `experiments/ostrack`. We use [wandb](https://github.com/wandb/client) to record detailed training logs, in case you don't want to use wandb, set `--use_wandb 0`.


## Evaluation
Download the model weights from [Google Drive](https://drive.google.com/drive/folders/1PS4inLS8bWNCecpYZ0W2fE5-A04DvTcd?usp=sharing) 

Put the downloaded weights on `$PROJECT_ROOT$/output/checkpoints/train/ostrack`

Change the corresponding values of `lib/test/evaluation/local.py` to the actual benchmark saving paths

Some testing examples:
- LaSOT or other off-line evaluated benchmarks (modify `--dataset` correspondingly)
```
python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset lasot --threads 16 --num_gpus 4
python tracking/analysis_results.py # need to modify tracker configs and names
```
- GOT10K-test
```
python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset got10k_test --threads 16 --num_gpus 4
python lib/test/utils/transform_got10k.py --tracker_name ostrack --cfg_name ostrack384_got10k_elimination_ep100
```
- TrackingNet
```
python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset trackingnet --threads 16 --num_gpus 4
python lib/test/utils/transform_trackingnet.py --tracker_name ostrack --cfg_name ostrack384_elimination_ep300
```

## Visualization or Debug 
[Visdom](https://github.com/fossasia/visdom) is used for visualization. 
1. Alive visdom in the sever by running `visdom`:

2. Simply set `--debug 1` during inference for visualization, e.g.:
```
python tracking/test.py ostrack ostrack384_elimination_ep300 --dataset vot22 --threads 1 --num_gpus 1 --debug 1
```
3. Open `http://localhost:8097` in your browser (remember to change the ip address and port according to the actual situation).

4. Then you can visualize the candidate elimination process.

![ECE_vis](https://github.com/botaoye/OSTrack/blob/main/assets/vis.png)


## Test FLOPs, and Speed
*Note:* The speeds reported in our paper were tested on a single RTX2080Ti GPU.

```
# Profiling vitb_256_mae_ce_32x4_ep300
python tracking/profile_model.py --script ostrack --config vitb_256_mae_ce_32x4_ep300
# Profiling vitb_384_mae_ce_32x4_ep300
python tracking/profile_model.py --script ostrack --config vitb_384_mae_ce_32x4_ep300
```


## Acknowledgments
* Thanks for the [STARK](https://github.com/researchmm/Stark) and [PyTracking](https://github.com/visionml/pytracking) library, which helps us to quickly implement our ideas.
* We use the implementation of the ViT from the [Timm](https://github.com/rwightman/pytorch-image-models) repo.  


## Citation
If our work is useful for your research, please consider cite:

```
@article{ye2022ostrack,
  title={Joint Feature Learning and Relation Modeling for Tracking: A One-Stream Framework},
  author={Ye, Botao and Chang, Hong and Ma, Bingpeng and Shan, Shiguang},
  journal={arXiv preprint arXiv:2203.11991},
  year={2022}
}
```
