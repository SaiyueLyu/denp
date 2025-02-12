#!/usr/bin/env bash

for ind in {2..49}; do
    eai job new --preemptable --cpu 4 --gpu 1 --gpu-mem 32 --gpu-model-filter v100 --mem 16 -d snow.colab_public.data:/mnt/colab_public:rw -e HOME=/home/toolkit -d snow.home.saiyue_lyu:/mnt/home:rw -i registry.console.elementai.com/snow.interactive_toolkit/saiyue \
    -- /mnt/home/denp/bin/python /mnt/home/original/DensePure/eval_certified_densepure.py \
    --exp /mnt/home/original/DensePure/exp/imagenet \
    --config /mnt/home/original/DensePure/configs/imagenet.yml \
    -i /mnt/home/original/DensePure/terminal_logs/imagenet-$ind \
    --domain imagenet \
    --seed 0 \
    --diffusion_type guided-ddpm \
    --lp_norm L2 \
    --outfile /mnt/home/original/DensePure/new/denp_-$ind \
    --sigma 1.41421356237 \
    --N 10000 \
    --N0 100 \
    --certified_batch 20 \
    --sample_id $(seq -s ' ' 0 1000 49000) \
    --use_id \
    --certify_mode purify \
    --advanced_classifier beit \
    --use_t_steps \
    --num_t_steps 10 \
    --save_predictions \
    --predictions_path /mnt/home/original/DensePure/exp/imagenet/$ind- \
    --reverse_seed 1 \
    --id_index $ind
done
