from datetime import datetime
import argparse
from data import get_dataset

dout_dir = (
    "./out/"
    + datetime.now().strftime("%Y-%m-%d")
    + "/run_"
    + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

default_args = dict(
    epochs=100,
    flow_steps=100,
    lr=0.03,
    random_seed=42,
    batch_size=128,
    save_every=100,
    out_dir=dout_dir,
    date_time_dir=None,
    exp_name="vgg_test",
    wd=0.0,
    use_cuda=True,
    det_run=True,
    loglevel_info=False,
    model='vgg16',
    model_width=64,  # default for VGG
    half_prec=False,
    no_data_augm=True,
    batch_norm=False,
    dataset='cifar10',
    momentum=0.9,
    lr_flow=1e-4,
    wd_flow=0.0,
    p_radius=None,
    first_decay=0.5,
    label_noise=0.0,
    label_noise_decay=0.5,
    margin_treshold=0.0,
    noise_scale=1.0,
    noise_batch_size=None,
    label_smoothing=0.0,
    lr_gamma_decay=1.0,
    wd_gamma_decay=1.0,
)


def configuration(args=None):
    parser = argparse.ArgumentParser(description="Exp vgg")

    # --------------------------------------------------------------------------------
    # TRAINING OPTIONS
    # --------------------------------------------------------------------------------
    tgroup = parser.add_argument_group("Training options")
    tgroup.add_argument(
        "--epochs",
        type=int,
        metavar="N",
        help="Number of training iters. " + "Default: %(default)s.",
    )
    tgroup.add_argument(
        "--flow_steps",
        type=int,
        metavar="N",
        help="Number of training iters. " + "Default: %(default)s.",
    )
    tgroup.add_argument(
        "--batch_size",
        type=int,
        metavar="N",
        help="Batch size. " + "Default: %(default)s.",
    )
    tgroup.add_argument(
        "--lr",
        type=float,
        help="Learning rate of optimizer. Default: " + "%(default)s.",
    )
    tgroup.add_argument(
        "--first_decay",
        type=float,
        help="Fraction of epochs when lr is first decayed: " + "%(default)s.",
    )
    tgroup.add_argument(
        "--lr_flow",
        type=float,
        help="Decaying the learning rate to this exact value: " +
        "%(default)s.",
    )
    tgroup.add_argument(
        "--wd_flow",
        type=float,
        help="Decaying the learning rate to this exact value: " +
        "%(default)s.",
    )
    tgroup.add_argument(
        "--p_radius",
        type=float,
        help="Radius for psgd: " + "%(default)s.",
    )
    tgroup.add_argument(
        "--lr_gamma_decay",
        type=float,
        help="Decaying factor of the learning rate: " + "%(default)s.",
    )
    tgroup.add_argument(
        "--wd_gamma_decay",
        type=float,
        help="Decaying factor of the weight decay: " + "%(default)s.",
    )
    tgroup.add_argument(
        "--wd",
        type=float,
        help="Weight decay of optimizer. Default: " + "%(default)s.",
    )
    tgroup.add_argument('--momentum', type=float,
                        help="Momentum of optimizer. Default: " + "%(default)s.",)

    # --------------------------------------------------------------------------------
    # MODEL OPTIONS
    # --------------------------------------------------------------------------------
    modelgroup = parser.add_argument_group("Model options")
    modelgroup.add_argument('--model', choices=['vit_basic', 'vit_exp', 'vgg16', 'resnet18', 'resnet18_plain', 'resnet18_gn', 'resnet_tiny',
                            'resnet_tiny_gn', 'resnet34', 'resnet34_plain', 'resnet34preact', 'resnet34_gn', 'wrn28', 'lenet', 'cnn', 'fc', 'linear'], type=str)
    modelgroup.add_argument('--model_width', type=int,
                            help='model width (# conv filters on the first layer for ResNets)')
    modelgroup.add_argument('--half_prec', action='store_true',
                            help='if enabled, runs everything as half precision [not recommended]')
    modelgroup.add_argument(
        '--batch_norm', action='store_true', help='if enabled, adds batch-norm layers')
    # --------------------------------------------------------------------------------
    # MISCELLANEOUS OPTIONS
    # --------------------------------------------------------------------------------
    mgroup = parser.add_argument_group("Miscellaneous options")
    mgroup.add_argument(
        "--random_seed",
        type=int,
        metavar="N",
        help="Random seed. Default: %(default)s.",
    )
    mgroup.add_argument(
        "--save_every",
        type=int,
        metavar="N",
        help="Numer of iterations for saving model. Default: %(default)s.",
    )
    mgroup.add_argument(
        "--out_dir", type=str, help="Where to store the outputs of this simulation."
    )
    mgroup.add_argument("--exp_name", type=str, help="Name of the experiment.")

    mgroup.add_argument("--date_time_dir", type=str,
                        help="Name of the date time dir.")

    # --------------------------------------------------------------------------------
    # EXPERIMENTS OPTIONS
    # --------------------------------------------------------------------------------
    expgroup = parser.add_argument_group("Experiments options")
    expgroup.add_argument(
        "--use_cuda",
        action="store_true",
        help="Use cuda",
    )
    expgroup.add_argument(
        "--det_run",
        action="store_true",
        help="Ensure deterministic behaviour",
    )
    expgroup.add_argument(
        "--loglevel_info",
        action="store_true",
        help="Info in the logger",
    )
    expgroup.add_argument('--no_data_augm', action='store_true')
    expgroup.add_argument(
        "--label_noise",
        type=float,
        help="Fraction of labels to flip: " + "%(default)s.",
    )
    expgroup.add_argument(
        "--noise_batch_size",
        type=int,
        metavar="N",
        help="Noise batch size. " + "Default: %(default)s.",
    )
    expgroup.add_argument(
        "--label_noise_decay",
        type=float,
        help="Fraction of labels to flip: " + "%(default)s.",
    )
    expgroup.add_argument(
        "--margin_treshold",
        type=float,
        help="Margin under which labels are flipped: " + "%(default)s.",
    )
    expgroup.add_argument(
        "--noise_scale",
        type=float,
        help="noise scale: " + "%(default)s.",
    )
    expgroup.add_argument(
        "--label_smoothing",
        type=float,
        help="noise scale: " + "%(default)s.",
    )
    # --------------------------------------------------------------------------------
    # DATA OPTIONS
    # --------------------------------------------------------------------------------
    datagroup = parser.add_argument_group("Dataset options")
    datagroup.add_argument(
        "--input_dim",
        type=int,
        help="Random seed. Default: %(default)s.",
    )
    datagroup.add_argument(
        '--dataset', choices=get_dataset.datasets_dict.keys(), type=str)

    parser.set_defaults(**default_args)

    if args is None:
        return parser.parse_args()
    else:
        return parser.parse_args(args=args)
