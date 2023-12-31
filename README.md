# GAN_Architecture_Search_with_Predictors

## Dependencies
```bash
conda create --n autogan-dsp python=3.6
conda activate autogan-dsp

pip install torch==1.4.0 torchvision==0.5.0

pip install imageio scipy six numpy==1.18.1 python-dateutil==2.7.3 tensorboardX==1.6 tensorflow-gpu==1.13.1 tqdm==4.29.1
pip install pandas

```

## FID statistic files
You can download pre-caculated FID statisc files ( [Link](https://drive.google.com/drive/folders/1JGODyX1ekDzlhpWbeknir0OYv_TqMT9e?usp=sharing)) to `./search/fid_stat` and `./eval/fid_stat`. 

## Dataset
You can download ImageNet 32x32 data [Link] to `./eval/data`.

## Pre-trained model
We also provide pre-trained model of our discovered architectures [Link](https://drive.google.com/drive/folders/1yVCK0tWekuIi0fmPGSy235wqYGEVwwWF?usp=sharing) to `./eval/pre-trained`

## Search
```
cd search
sh exps/search.sh
```
discovered architectures stored in `./logs/{exp_name}/architectures.csv`

## Evaluation
You can replace `--arch` in .sh file to the other discovered architecture.
```
cd eval
sh exps/eval.sh
```
and test
```
sh exps/test_arch.sh
```

## Acknowledges
1. Inception Score code from [OpenAI's Improved GAN](https://github.com/openai/improved-gan/tree/master/inception_score) (official).
2. FID code and CIFAR-10 statistics file from [https://github.com/bioinf-jku/TTUR](https://github.com/bioinf-jku/TTUR) (official).
3. SAC code from [https://github.com/pranz24/pytorch-soft-actor-critic](https://github.com/pranz24/pytorch-soft-actor-critic).
4. our implementation incorporates code obtained from E2GAN https://github.com/Yuantian013/E2GAN, and Neural Predictor https://github.com/ultmaster/neuralpredictor.pytorch.
