# Why Do We Need Weight Decay in Modern Deep Learning?

**Maksym Andriushchenko\*, Francesco D’Angelo\*, Aditya Varre, Nicolas Flammarion (EPFL)**

**NeurIPS 2024**

**Paper:** [https://arxiv.org/abs/2310.04415](https://arxiv.org/abs/2310.04415)
<p align="center"><img src="images/wd_summary_slide.png" width="900" /></p>

### Abstract
Weight decay is a broadly used technique for training state-of-the-art deep networks, including large language models. Despite its widespread usage, its role remains poorly understood. In this work, we highlight that the role of weight decay in modern deep learning is different from its regularization effect studied in classical learning theory. For overparameterized deep networks, we show how weight decay modifies the optimization dynamics enhancing the ever-present implicit regularization of SGD via the *loss stabilization mechanism*. In contrast, for underparameterized large language models trained with nearly online SGD, we describe how weight decay balances the *bias-variance tradeoff* in stochastic optimization leading to lower training loss. Moreover, we show that weight decay also prevents sudden loss divergences for `bfloat16` mixed-precision training which is a crucial tool for LLM training.  Overall, we present a unifying perspective from ResNets on vision tasks to LLMs: weight decay is never useful as an explicit regularizer but instead changes the training dynamics in a desirable way.






## Weight decay for overparameterized deep learning
### Env setup 
First you need to install all the required packages
```bash
cd overparameterized_nets
sudo apt-get update -y
sudo apt-get install libgl1 -y
conda env create -f environment.yml
```
### Scripts 
Here we list the main scripts to reproduce the figures in the overparameterized deep nets section 
#### Experiments in Figure 1:
```bash 
python train.py --model vgg16 --batch_size 128 --epochs 1000 --momentum 0.0 --lr 0.1 --lr_flow 1e-4 --first_decay 0.5 --wd 0.008 --exp_name vgg_fig1
python train.py --model vgg16 --batch_size 128 --epochs 1000 --momentum 0.0 --lr 0.1 --lr_flow 1e-4 --first_decay 0.5 --wd 0.0 --exp_name vgg_fig1
python train.py --model vgg16 --batch_size 128 --epochs 1000 --momentum 0.0 --lr 0.01 --lr_flow 1e-4 --first_decay 0.5 --wd 0.008 --exp_name vgg_fig1
python train.py --model vgg16 --batch_size 128 --epochs 1000 --momentum 0.0 --lr 0.01 --lr_flow 1e-4 --first_decay 0.5 --wd 0. --exp_name vgg_fig1
python train.py --model resnet18 --batch_norm --batch_size 128 --epochs 1000 --momentum 0.0 --lr 0.08 --lr_flow 1e-4 --first_decay 0.5 --wd 0.0125 --exp_name resnet18_fig1
python train.py --model resnet18 --batch_norm --batch_size 128 --epochs 1000 --momentum 0.0 --lr 0.08 --lr_flow 1e-4 --first_decay 0.5 --wd 0.0 --exp_name resnet18_fig1
python train.py --model resnet18 --batch_norm --batch_size 128 --epochs 1000 --momentum 0.0 --lr 0.001 --lr_flow 1e-4 --first_decay 0.5 --wd 0.0125 --exp_name resnet18_fig1
python train.py --model resnet18 --batch_norm --batch_size 128 --epochs 1000 --momentum 0.0 --lr 0.001 --lr_flow 1e-4 --first_decay 0.5 --wd 0.0 --exp_name resnet18_fig1
python train.py --model resnet34preact --dataset cifar100 --batch_norm --batch_size 128 --epochs 1000 --momentum 0.0 --lr 0.15 --lr_flow 1e-4 --first_decay 0.5 --wd 0.01 --exp_name resnet34_fig1
python train.py --model resnet34preact --dataset cifar100 --batch_norm --batch_size 128 --epochs 1000 --momentum 0.0 --lr 0.15 --lr_flow 1e-4 --first_decay 0.5 --wd 0.0 --exp_name resnet34_fig1
python train.py --model resnet34preact --dataset cifar100 --batch_norm --batch_size 128 --epochs 1000 --momentum 0.0 --lr 0.001 --lr_flow 1e-4 --first_decay 0.5 --wd 0.01 --exp_name resnet34_fig1
python train.py --model resnet34preact --dataset cifar100 --batch_norm --batch_size 128 --epochs 1000 --momentum 0.0 --lr 0.001 --lr_flow 1e-4 --first_decay 0.5 --wd 0.0 --exp_name resnet34_fig1
```
#### Experiments in Figure 2:
```bash
python traceh.py --model resnet18 --batch_norm --scale_inv --batch_size 128 --epochs 100 --momentum 0.0 --lr 0.001 --lr_flow 1e-4 --flow_every 2 --flow_steps 100 --radius 1.0 --wd 0.0 --exp_name fig2
python traceh.py --model resnet18 --batch_norm --scale_inv --batch_size 128 --epochs 100 --momentum 0.0 --lr 0.003 --lr_flow 1e-4 --flow_every 2 --flow_steps 100 --radius 1.0 --wd 0.0 --exp_name fig2
python traceh.py --model resnet18 --batch_norm --scale_inv --batch_size 128 --epochs 100 --momentum 0.0 --lr 0.005 --lr_flow 1e-4 --flow_every 2 --flow_steps 100 --radius 1.0 --wd 0.0 --exp_name fig2
```
#### Experiments in Figure 3:
```bash
python traceh.py --model resnet18 --batch_norm --batch_size 128 --epochs 100 --momentum 0.0 --lr 0.1 --lr_flow 1e-3 --flow_every 2 --flow_steps 100 --wd 0.015 --exp_name fig3
python traceh.py --model resnet18 --batch_norm --batch_size 128 --epochs 100 --momentum 0.0 --lr 0.1 --lr_flow 1e-3 --flow_every 2 --flow_steps 100 --wd 0.015 --exp_name fig3
python traceh.py --model resnet18 --batch_norm --batch_size 128 --epochs 100 --momentum 0.0 --lr 0.1 --lr_flow 1e-3 --flow_every 2 --flow_steps 100 --wd 0.015 --exp_name fig3
```




## Weight decay for large language models
The code is based on the amazing [NanoGPT repository](https://github.com/karpathy/nanoGPT/).


### Install
First you need to create a single-GPU virtual machine (e.g., on Google Cloud):
```bash
# create a virtual machine on google cloud
gcloud compute instances create weight-decay-vm \
    --project=weight-decay --zone=europe-west4-b \
    --image=pytorch-latest-gpu-v20230501 \
    --image-project=deeplearning-platform-release --machine-type=a2-highgpu-1g \
    --scopes=cloud-platform,storage-full --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd --metadata=install-nvidia-driver=True \
    --maintenance-policy=TERMINATE \
    --accelerator=type="nvidia-tesla-a100,count=1"
# ssh to the virtual machine
gcloud compute ssh --project weight-decay --zone europe-west4-b weight-decay-vm
```

Then follow the instructions from the original NanoGPT repository:
```bash
cd large_language_models
conda create -n py39 python=3.9 -y
conda activate py39
pip install numpy datasets tiktoken wandb tqdm ipdb torch transformers matplotlib seaborn

# data preparation
python data/shakespeare_char/prepare.py
python data/openwebtext/prepare.py  # takes ~1h

# now you can quickly try if installation was successful
python train.py config/train_shakespeare_char.py
```



### Scripts
Here we list the main scripts used to produce the figures in the paper.


Training of GPT-2-small models with `block_size=256` context length (to speed up experiments) with different weight decay and decaying vs. constant LRs:
```bash
# Cosine decay runs with different WD
python train.py config/train_gpt2_small_block256.py --batch_size=8 --gradient_accumulation_steps=32 --learning_rate=0.0006 --min_lr=0.00006 --max_iters=50000 --weight_decay=0.0 --wandb_run_name=owt_gpt2small_block256
python train.py config/train_gpt2_small_block256.py --batch_size=8 --gradient_accumulation_steps=32 --learning_rate=0.0006 --min_lr=0.00006 --max_iters=50000 --weight_decay=0.1 --wandb_run_name=owt_gpt2small_block256
python train.py config/train_gpt2_small_block256.py --batch_size=8 --gradient_accumulation_steps=32 --learning_rate=0.0006 --min_lr=0.00006 --max_iters=50000 --weight_decay=0.3 --wandb_run_name=owt_gpt2small_block256

# Const-LR runs with different WD
python train.py config/train_gpt2_small_block256.py --batch_size=8 --gradient_accumulation_steps=32 --learning_rate=0.0006 --min_lr=0.0006 --max_iters=50000 --weight_decay=0.0 --wandb_run_name=owt_gpt2small_block256
python train.py config/train_gpt2_small_block256.py --batch_size=8 --gradient_accumulation_steps=32 --learning_rate=0.0006 --min_lr=0.0006 --max_iters=50000 --weight_decay=0.1 --wandb_run_name=owt_gpt2small_block256
python train.py config/train_gpt2_small_block256.py --batch_size=8 --gradient_accumulation_steps=32 --learning_rate=0.0006 --min_lr=0.0006 --max_iters=50000 --weight_decay=0.3 --wandb_run_name=owt_gpt2small_block256
python train.py config/train_gpt2_small_block256.py --batch_size=8 --gradient_accumulation_steps=32 --learning_rate=0.0006 --min_lr=0.0006 --max_iters=50000 --weight_decay=0.6 --wandb_run_name=owt_gpt2small_block256
```


Example of fine-tuning with a tiny learning rate:
```bash
# Fine-tuning of GPT-2-small-block256 models
python train.py config/train_gpt2_small_block256.py --batch_size=4 --gradient_accumulation_steps=64 --learning_rate=0.00001 --min_lr=0.00001 --max_iters=10000 --wandb_run_name=owt_gpt2_small_block256_ft_lr0.0006_wd0   --init_from=resume --eval_examples=5000 --ckpt_path='/home/maksym/tml_wd/models_llm/2023-09-16_10-47-27.655-owt_gpt2small_block256-learning_rate=0.0006-min_lr=0.000060-weight_decay=0-n_embd=768-max_iters=50000-init_scale=0.02-iter=10000.pt'

# Fine-tuning the const-LR runs
python train.py config/train_gpt2_small_block256.py --batch_size=8 --gradient_accumulation_steps=32 --learning_rate=0.00001 --min_lr=0.00001 --max_iters=10000 --wandb_run_name=owt_gpt2_small_block256_ft_constlr0.0006_wd0   --init_from=resume --eval_examples=5000 --ckpt_path='/home/maksym/tml_wd/models_llm/2023-09-18_16-50-58.849-owt_gpt2small_block256-learning_rate=0.0006-min_lr=0.000600-weight_decay=0-n_embd=768-max_iters=50000-init_scale=0.02-iter=10000.pt'
```


`bfloat16` divergence experiments:
```bash
# bfloat16 divergence experiments
python train.py config/train_gpt2_small.py --batch_size=16 --gradient_accumulation_steps=16 --learning_rate=0.0006 --min_lr=0.00006 --max_iters=50000 --weight_decay=0.0 --dtype=bfloat16 --wandb_run_name=owt_gpt2small_high_lr_bfloat16 --random_seed=0 --out_dir=models
python train.py config/train_gpt2_small.py --batch_size=16 --gradient_accumulation_steps=16 --learning_rate=0.0006 --min_lr=0.00006 --max_iters=50000 --weight_decay=0.1 --dtype=bfloat16 --wandb_run_name=owt_gpt2small_high_lr_bfloat16 --random_seed=0 --out_dir=models
python train.py config/train_gpt2_small.py --batch_size=16 --gradient_accumulation_steps=16 --learning_rate=0.0006 --min_lr=0.00006 --max_iters=50000 --weight_decay=0.3 --dtype=bfloat16 --wandb_run_name=owt_gpt2small_high_lr_bfloat16 --random_seed=0 --out_dir=models
python train.py config/train_gpt2_small.py --batch_size=16 --gradient_accumulation_steps=16 --learning_rate=0.0006 --min_lr=0.00006 --max_iters=50000 --weight_decay=0.0 --dtype=float32  --wandb_run_name=owt_gpt2small_high_lr_float32  --random_seed=0 --out_dir=models
```


Experiments in the appendix:
```bash
# Penalizing all layers in WD
python train.py config/train_gpt2_small_block256.py --batch_size=8 --gradient_accumulation_steps=32 --learning_rate=0.0006 --min_lr=0.00006 --max_iters=50000 --weight_decay=0.003 --wd_substrings_include='wte wpe mlp attn lm_head ln' --wandb_run_name=owt_gpt2small_block256_wd_all --out_dir=models
python train.py config/train_gpt2_small_block256.py --batch_size=8 --gradient_accumulation_steps=32 --learning_rate=0.0006 --min_lr=0.00006 --max_iters=50000 --weight_decay=0.01  --wd_substrings_include='wte wpe mlp attn lm_head ln' --wandb_run_name=owt_gpt2small_block256_wd_all --out_dir=models
python train.py config/train_gpt2_small_block256.py --batch_size=8 --gradient_accumulation_steps=32 --learning_rate=0.0006 --min_lr=0.00006 --max_iters=50000 --weight_decay=0.03  --wd_substrings_include='wte wpe mlp attn lm_head ln' --wandb_run_name=owt_gpt2small_block256_wd_all --out_dir=models
python train.py config/train_gpt2_small_block256.py --batch_size=8 --gradient_accumulation_steps=32 --learning_rate=0.0006 --min_lr=0.00006 --max_iters=50000 --weight_decay=0.1   --wd_substrings_include='wte wpe mlp attn lm_head ln' --wandb_run_name=owt_gpt2small_block256_wd_all --out_dir=models
python train.py config/train_gpt2_small_block256.py --batch_size=32 --gradient_accumulation_steps=8 --learning_rate=0.0006 --min_lr=0.00006 --max_iters=50000 --weight_decay=0.15  --wd_substrings_include='wte wpe mlp attn lm_head ln' --wandb_run_name=owt_gpt2small_block256_wd_all --out_dir=models
python train.py config/train_gpt2_small_block256.py --batch_size=32 --gradient_accumulation_steps=8 --learning_rate=0.0006 --min_lr=0.00006 --max_iters=50000 --weight_decay=0.2   --wd_substrings_include='wte wpe mlp attn lm_head ln' --wandb_run_name=owt_gpt2small_block256_wd_all --out_dir=models
python train.py config/train_gpt2_small_block256.py --batch_size=32 --gradient_accumulation_steps=8 --learning_rate=0.0006 --min_lr=0.00006 --max_iters=50000 --weight_decay=0.3   --wd_substrings_include='wte wpe mlp attn lm_head ln' --wandb_run_name=owt_gpt2small_block256_wd_all

#  L2 regularization instead of AdamW (also without including LN params either)
python train.py config/train_gpt2_small_block256.py --batch_size=8 --gradient_accumulation_steps=32 --learning_rate=0.0006 --min_lr=0.00006 --max_iters=50000 --l2_reg=0.000001 --wandb_run_name=owt_gpt2small_block256_l2_reg --out_dir=models
python train.py config/train_gpt2_small_block256.py --batch_size=8 --gradient_accumulation_steps=32 --learning_rate=0.0006 --min_lr=0.00006 --max_iters=50000 --l2_reg=0.000003 --wandb_run_name=owt_gpt2small_block256_l2_reg --out_dir=models
python train.py config/train_gpt2_small_block256.py --batch_size=32 --gradient_accumulation_steps=8 --learning_rate=0.0006 --min_lr=0.00006 --max_iters=50000 --l2_reg=0.000005 --wandb_run_name=owt_gpt2small_block256_l2_reg --out_dir=models
python train.py config/train_gpt2_small_block256.py --batch_size=32 --gradient_accumulation_steps=8 --learning_rate=0.0006 --min_lr=0.00006 --max_iters=50000 --l2_reg=0.000007 --wandb_run_name=owt_gpt2small_block256_l2_reg --out_dir=models
python train.py config/train_gpt2_small_block256.py --batch_size=8 --gradient_accumulation_steps=32 --learning_rate=0.0006 --min_lr=0.00006 --max_iters=50000 --l2_reg=0.00001 --wandb_run_name=owt_gpt2small_block256_l2_reg --out_dir=models
python train.py config/train_gpt2_small_block256.py --batch_size=8 --gradient_accumulation_steps=32 --learning_rate=0.0006 --min_lr=0.00006 --max_iters=50000 --l2_reg=0.0001 --wandb_run_name=owt_gpt2small_block256_l2_reg --out_dir=models
python train.py config/train_gpt2_small_block256.py --batch_size=8 --gradient_accumulation_steps=32 --learning_rate=0.0006 --min_lr=0.00006 --max_iters=50000 --l2_reg=0.001 --wandb_run_name=owt_gpt2small_block256_l2_reg --out_dir=models

# SGD with momentum
python train.py config/train_gpt2_small_block256.py --batch_size=16 --gradient_accumulation_steps=16 --learning_rate=0.3 --max_iters=50000 --weight_decay=0.0     --opt_type=gdm --wandb_run_name=owt_gpt2small_block256_sgdm --out_dir=models
python train.py config/train_gpt2_small_block256.py --batch_size=16 --gradient_accumulation_steps=16 --learning_rate=0.3 --max_iters=50000 --weight_decay=0.00001 --opt_type=gdm --wandb_run_name=owt_gpt2small_block256_sgdm --out_dir=models
python train.py config/train_gpt2_small_block256.py --batch_size=16 --gradient_accumulation_steps=16 --learning_rate=0.3 --max_iters=50000 --weight_decay=0.00003 --opt_type=gdm --wandb_run_name=owt_gpt2small_block256_sgdm --out_dir=models
python train.py config/train_gpt2_small_block256.py --batch_size=16 --gradient_accumulation_steps=16 --learning_rate=0.3 --max_iters=50000 --weight_decay=0.0001  --opt_type=gdm --wandb_run_name=owt_gpt2small_block256_sgdm --out_dir=models

# Extra runs with a bit less aggressive decay
python train.py config/train_gpt2_small_block256.py --batch_size=32 --gradient_accumulation_steps=8 --learning_rate=0.0006 --min_lr=0.00006 --lr_decay_iters=55000  --max_iters=50000 --weight_decay=0.0 --wandb_run_name=owt_gpt2small_block256_slower_lr_decay
python train.py config/train_gpt2_small_block256.py --batch_size=32 --gradient_accumulation_steps=8 --learning_rate=0.0006 --min_lr=0.00006 --lr_decay_iters=60000  --max_iters=50000 --weight_decay=0.0 --wandb_run_name=owt_gpt2small_block256_slower_lr_decay --out_dir=models
python train.py config/train_gpt2_small_block256.py --batch_size=32 --gradient_accumulation_steps=8 --learning_rate=0.0006 --min_lr=0.00006 --lr_decay_iters=65000  --max_iters=50000 --weight_decay=0.0 --wandb_run_name=owt_gpt2small_block256_slower_lr_decay
```


