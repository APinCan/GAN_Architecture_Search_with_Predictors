CUDA_VISIBLE_DEVICES=0 python3 test_arch.py \
--dataset imagenet32 \
--img_size 32 \
--bottom_width 4 \
--gen_model shared_gan \
--latent_dim 128 \
--gf_dim 256 \
--g_spectral_norm False \
--load_path logs/imagenet32-00001000010003-AutoGANDSP-Iter1_2023_10_17_14_15_13/Model/checkpoint_best.pth \
--arch 0 0 0 0 1 0 0 0 0 1 0 0 0 3 \
--exp_name test_imagenet32_00001000010003-autogandsp-1017141513Iter1-best1