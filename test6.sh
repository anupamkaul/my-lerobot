#!/bin/bash
lerobot-train \
  --policy.type=act \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --policy.device=mps \
  --batch_size=8 \
  --steps=500 \
  --save_freq=250 \
  --output_dir=outputs/train/my_second_policy \
  --policy.push_to_hub=False
