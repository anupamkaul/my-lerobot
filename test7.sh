#!/bin/bash
lerobot-eval \
  --policy.path=outputs/train/my_second_policy/checkpoints/last/pretrained_model \
  --env.type=aloha \
  --eval.batch_size=1 \
  --eval.n_episodes=2
