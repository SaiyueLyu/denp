#!/usr/bin/env bash
cd ..

sigma=$1
steps=$2
reverse_seed=$3

CUDA_VISIBLE_DEVICES=0 python eval_certified_densepure.py \
--exp exp/imagenet \
--config configs/imagenet.yml \
-i imagenet-0 \
--domain imagenet \
--seed 0 \
--diffusion_type guided-ddpm \
--lp_norm L2 \
--outfile sqrt2/denp_$sigma-0 \
--sigma $sigma \
--N 10000 \
--N0 100 \
--certified_batch 120 \
--sample_id $(seq -s ' ' 0 8000 18000) \
--use_id \
--certify_mode purify \
--advanced_classifier beit \
--use_t_steps \
--num_t_steps $steps \
--save_predictions \
--predictions_path exp/imagenet/0- \
--reverse_seed $reverse_seed

