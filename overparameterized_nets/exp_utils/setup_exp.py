import os
import random
import string
from datetime import datetime
import logging
import numpy as np
import torch
import yaml
from prettyprinter import pprint
from torch.utils.tensorboard import SummaryWriter
from time import time
from aim import Run
import shutil


def set_exp(config):
    """This utils sets up all the experimental artifacts such as random seeds,
       log folders for results, tensorboard and aim loggers, saves a yaml file
       with all the hyperparams.

    Args:
        config: configuratore for the experiment

    Returns:
        aim_run: aim writer
        writer: tensorboard writer
        log_dir:
    """

    # TODO Checkout code version

    # Create experiments folders and artifacts
    experiment_name = config.exp_name
    # random_str = "".join(random.choices(
    #     string.ascii_letters + string.digits, k=10))
    # Create Aim writer
    aim_writer = Run(
        repo='/tmldata1/fdangelo/understanding-weight-decay/aim_paper_2', experiment=experiment_name)
    if config.date_time_dir is None:
        date_time_log_dir = datetime.now().strftime("%m-%d-%Y")
    else:
        date_time_log_dir = config.date_time_dir

    # subdirname = "lr_{}_epochs_{}_wd_{}_seed_{}_{}_{}".format(
    #     config.lr,
    #     config.epochs,
    #     config.wd,
    #     config.random_seed,
    #     datetime.now().strftime("%H%M%S"),
    #     random_str
    # )
    subdirname = aim_writer.hash
    log_dir = os.path.join(
        os.getcwd(),
        "out",
        experiment_name,
        date_time_log_dir,
        subdirname,
    )

    os.makedirs(log_dir)
    os.makedirs(os.path.join(log_dir, "models"))
    config.out_dir = log_dir
    print(f"Experiment folder created at {log_dir}")

    # Save parameters file
    configu = dict(config.__dict__)
    configu['hash'] = aim_writer.hash
    # Add Hparams to tensorbaord
    sorted_dict = {key: value for key, value in sorted(configu.items())}
    aim_writer['hparams'] = sorted_dict
    pprint(sorted_dict)

    dictionary_parameters = vars(config)

    # Deterministic computation.
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)

    # Ensure that runs are reproducible. Note, this slows down training!
    # https://pytorch.org/docs/stable/notes/randomness.html
    if config.det_run:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Select torch device.
    assert (hasattr(config, 'no_cuda') or hasattr(config, 'use_cuda'))
    assert (not hasattr(config, 'no_cuda') or not hasattr(config, 'use_cuda'))

    if hasattr(config, 'no_cuda'):
        use_cuda = not config.no_cuda and torch.cuda.is_available()
    else:
        use_cuda = config.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    with open(log_dir + "/" + "parameters.yml", "w") as yaml_file:
        yaml.dump(dictionary_parameters, stream=yaml_file,
                  default_flow_style=False)

    return device, aim_writer
