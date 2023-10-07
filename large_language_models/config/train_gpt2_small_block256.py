# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'llm'
wandb_run_name='owt'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
gradient_accumulation_steps = 40

# this makes total number of tokens be 300B
max_iters = 600000

# model
n_layer = 12  # default: 12
n_head = 12  # default: 12
n_embd = 768  # default: 768
block_size = 256  # default: 1024

# eval stuff
eval_interval = 200
eval_examples = 1000
log_interval = 10

# LR stuff
warmup_iters = 400 # default: 2000

# weight decay
weight_decay = 0.0
