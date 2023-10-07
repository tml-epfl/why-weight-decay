"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import copy
import os
import time
import math
import pickle
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT


# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = '/home/maksym/tml_wd/models_llm'
eval_interval = 2000
log_interval = 1
eval_examples = 1000
eval_only = False # if True, script exits right after the first eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
ckpt_path = ''
# wandb logging
wandb_log = True # disabled by default
wandb_project = 'llm'
wandb_run_name = 'owt'
# data: shakespeare: 1M of chars, owt: 9035582489 (~9B) of tokens
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# 9B tokens = 32 A100 days; 9035582489 / 32 / 24 / 60 = 196084 tokens = 1 min
# Example: 1024 context window * 256 batch size * 35000 iterations = 9,175,040,000 ≈ 9B tokens (1 epoch)
n_train_tokens = -1  
# model
random_seed = 0
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
learnable_ln = True
custom_init = False
init_scale = 0.02
weight_tying = True
force_save_ckpt = False # if True, will save a checkpoint every 2500 steps even if n_embd < 500
# adamw optimizer
opt_type = 'adam' # adam, gd, gdm
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 0.0
wd_substrings_include = 'wte wpe mlp attn lm_head' # apply WD only to params that contain one of these substrings
l2_reg = 0 # by default, we are not using it
beta1 = 0.9
beta2 = 0.95
dampening = 0 # for SGD+M only
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.
ema_coef = 0.99
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = None # should be ~= max_iters per Chinchilla
min_lr = None # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------
if min_lr is None:  # if min_lr is not specified in the config or as a cmd line argument
    min_lr = learning_rate / 10
if lr_decay_iters is None:
    lr_decay_iters = max_iters
wd_l2reg_param = weight_decay if weight_decay > 0.0 else l2_reg
wd_substrings_include = wd_substrings_include.split(' ')
cur_timestamp = str(datetime.now())[:-3].replace(' ', '_').replace(':', '-')
hps_str = f'{wandb_run_name}-learning_rate={learning_rate}-min_lr={min_lr:6f}-weight_decay={wd_l2reg_param}-n_embd={n_embd}-max_iters={max_iters}-init_scale={init_scale}-n_train_tokens={n_train_tokens}-seed={random_seed}'

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process and out_dir != '':
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(random_seed + seed_offset)  
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    if split == 'train' and n_train_tokens > 0:
        data = data[:n_train_tokens]
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num, grad_norm, clip_times, avg_adam_elr = 0, 0, 0, 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, learnable_ln=learnable_ln, vocab_size=None, dropout=dropout, 
                  custom_init=custom_init, init_scale=init_scale, weight_tying=weight_tying) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from in 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    max_iters += iter_num
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(opt_type, weight_decay, learning_rate, (beta1, beta2), device_type, wd_substrings_include, dampening)
if init_from == 'resume':
    checkpoint['optimizer']['param_groups'][0]['weight_decay'] = weight_decay
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

model_ema = copy.deepcopy(model).eval()

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(model):
    metrics = {}
    model.eval()
    for split in ['train', 'val']:
        eval_iters = int(math.ceil(eval_examples / batch_size))
        losses = torch.zeros(eval_iters)
        errs = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
            Y_pred = logits.view(-1, logits.size(-1)).max(1)[1]
            errs[k] = (Y_pred != Y.view(-1)).float().mean()
        metrics[split+'_loss'] = losses.mean()
        metrics[split+'_err'] = errs.mean()
    model.train()
    return metrics

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=hps_str, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
tstart = t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if (iter_num % eval_interval == 0 or iter_num == max_iters) and master_process:
        metrics = estimate_loss(model)
        metrics_ema = estimate_loss(model_ema)

        opt_state = optimizer.state_dict()['state']
        if iter_num > 0 and opt_type == 'adam':
            num_params = sum([np.prod(param_state['exp_avg_sq'].shape) for param_state in opt_state.values()])
            avg_adam_elr = sum([torch.sum(lr / (param_state['exp_avg_sq']**0.5 + optimizer.param_groups[0]['eps'])).item() for param_state in opt_state.values()]) / num_params

        print(f"step {iter_num}: loss {metrics['train_loss']:.3f}/{metrics_ema['train_loss']:.3f}, err {metrics['train_err']:.2%}/{metrics_ema['train_err']:.2%}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": metrics['train_loss'],
                "train/err": metrics['train_err'],
                "train/loss_ema": metrics_ema['train_loss'],
                "train/err_ema": metrics_ema['train_err'],
                "val/loss": metrics['val_loss'],
                "val/err": metrics['val_err'],
                "val/loss_ema": metrics_ema['val_loss'],
                "val/err_ema": metrics_ema['val_err'],
                "lr": lr,
                "avg_adam_elr": avg_adam_elr,
            })
        if metrics['val_loss'] < best_val_loss:
           best_val_loss = metrics['val_loss']
        if (iter_num % 5000 == 0 or iter_num == max_iters) and\
           (n_embd > 500 or force_save_ckpt) and\
           (init_from == 'scratch' or force_save_ckpt) and\
           out_dir != '': 
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, f'{cur_timestamp}-{hps_str}-iter={iter_num}.pt'))
    if iter_num == 0 and eval_only:
        break

    t0 = time.time()  # start timing this iteration     

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    loss_iter, err_iter = 0, 0
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            Y_pred = logits.view(-1, logits.size(-1)).max(1)[1]
            err = (Y_pred != Y.view(-1)).float().mean() / gradient_accumulation_steps
            loss_iter += loss.item()
            err_iter += err.item()
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    
    if np.isnan(loss_iter):
        print("loss is NaN, aborting training")
        break

    if l2_reg > 0:
        assert weight_decay == 0.0  # make sure we are using only l2_reg and not WD
        reg = l2_reg * model.weight_norm(wd_substrings_include)**2
        scaler.scale(reg).backward()
    
    grad_max_value = raw_model.max_grad_value()
    grad_layerwise_norms = {'grads_layerwise/' + '.'.join(pn.split('.')[2:-1]): n.grad.norm().item() for pn, n in model.named_parameters()}

    # clip the gradient
    if grad_clip > 0.0:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if grad_norm.item() > grad_clip:
            clip_times += 1
    else:
        grad_norm = raw_model.grad_norm()
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    model_ema.moving_average(model, ema_coef)

    if wandb_log and iter_num % 20 == 0:  # log every K steps so that the legends are properly displayed on hover
        layerwise_norms = model.compute_layerwise_norms()
        wandb.log({
            "iter": iter_num,
            "train_every_iter/loss": loss_iter,
            "train_every_iter/err": err_iter,
            "grads/scaler_scale": scaler.get_scale(),  
            "grads/grad_norm": grad_norm,
            "grads/grad_max_value": grad_max_value,
            "grads/clip_times": clip_times,
            "weights/weight_norm": raw_model.weight_norm().item(),
            "weights/max_param_value": raw_model.max_param_value().item(),
            **layerwise_norms,
            **grad_layerwise_norms,
        })
    if iter_num % log_interval == 0 and master_process:
        print(f"iter {iter_num}: loss {loss_iter:.3f}, err {err_iter:.2%}, iter time {time.time()-t0:.3f}, total time {(time.time()-tstart)/60:.2f}m")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()

