import argparse

arg_lists = []
parser = argparse.ArgumentParser(description="RAM")


def str2bool(v):
    return v.lower() in ("true", "1")


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# glimpse network params
glimpse_arg = add_argument_group("Glimpse Network Params")
glimpse_arg.add_argument(
    "--patch_size", type=int, default=8, help="size of extracted patch at highest res"
)
glimpse_arg.add_argument(
    "--glimpse_scale", type=int, default=1, help="scale of successive patches"
)
glimpse_arg.add_argument(
    "--num_patches", type=int, default=1, help="# of downscaled patches per glimpse"
)
glimpse_arg.add_argument(
    "--loc_hidden", type=int, default=128, help="hidden size of loc fc"
)
glimpse_arg.add_argument(
    "--glimpse_hidden", type=int, default=128, help="hidden size of glimpse fc"
)


# core network params
core_arg = add_argument_group("Core Network Params")
core_arg.add_argument(
    "--num_glimpses", type=int, default=6, help="# of glimpses, i.e. BPTT iterations"
)
core_arg.add_argument("--hidden_size", type=int, default=256, help="hidden size of rnn")


# reinforce params
reinforce_arg = add_argument_group("Reinforce Params")
reinforce_arg.add_argument(
    "--std", type=float, default=0.05, help="gaussian policy standard deviation"
)
reinforce_arg.add_argument(
    "--M", type=int, default=1, help="Monte Carlo sampling for valid and test sets"
)
reinforce_arg.add_argument(
    "--gamma", type=float, default=1, help="Discount factor for the DT-RAM REINFORCE reward"
)


# data params
data_arg = add_argument_group("Data Params")
data_arg.add_argument(
    "--valid_size",
    type=float,
    default=0.1,
    help="Proportion of training set used for validation",
)
data_arg.add_argument(
    "--batch_size", type=int, default=128, help="# of images in each batch of data"
)
data_arg.add_argument(
    "--num_workers",
    type=int,
    default=4,
    help="# of subprocesses to use for data loading",
)
data_arg.add_argument(
    "--shuffle",
    type=str2bool,
    default=True,
    help="Whether to shuffle the train and valid indices",
)
data_arg.add_argument(
    "--show_sample",
    type=str2bool,
    default=False,
    help="Whether to visualize a sample grid of the data",
)
data_arg.add_argument(
    "--num_classes",
    type=int,
    default=10,
    help="# of classes for classification in the dataset",
)
data_arg.add_argument(
    "--num_channels",
    type=int,
    default=1,
    help="# of color channels in the dataset images",
)


# training params
train_arg = add_argument_group("Training Params")
train_arg.add_argument(
    "--is_train", type=str2bool, default=True, help="Whether to train or test the model"
)
train_arg.add_argument(
    "--momentum", type=float, default=0.5, help="Nesterov momentum value"
)
train_arg.add_argument(
    "--epochs", type=int, default=200, help="# of epochs to train for"
)
train_arg.add_argument(
    "--init_lr", type=float, default=3e-4, help="Initial learning rate value"
)
train_arg.add_argument(
    "--lr_patience",
    type=int,
    default=20,
    help="Number of epochs to wait before reducing lr",
)
train_arg.add_argument(
    "--train_patience",
    type=int,
    default=50,
    help="Number of epochs to wait before stopping train",
)
train_arg.add_argument(
    "--train_dynamic", type=str2bool, default=False, help="Train dynamic model (DT-RAM) instead of classic RAM"
)


# other params
misc_arg = add_argument_group("Misc.")
misc_arg.add_argument(
    "--use_gpu", type=str2bool, default=True, help="Whether to run on the GPU"
)
misc_arg.add_argument(
    "--best",
    type=str2bool,
    default=True,
    help="Load best model or most recent for testing",
)
misc_arg.add_argument(
    "--random_seed", type=int, default=1, help="Seed to ensure reproducibility"
)
misc_arg.add_argument(
    "--data_dir", type=str, default="../data/mnist", help="Directory in which data is stored"
)
misc_arg.add_argument(
    "--ckpt_dir",
    type=str,
    default="./ckpt",
    help="Directory in which to save model checkpoints",
)
misc_arg.add_argument(
    "--logs_dir",
    type=str,
    default="./logs/",
    help="Directory in which Tensorboard logs wil be stored",
)
misc_arg.add_argument(
    "--use_tensorboard",
    type=str2bool,
    default=False,
    help="Whether to use tensorboard for visualization",
)
misc_arg.add_argument(
    "--resume",
    type=str2bool,
    default=False,
    help="Whether to resume training from checkpoint",
)
misc_arg.add_argument(
    "--resume_fresh",
    type=str2bool,
    default=False,
    help="Whether to resume training from checkpoint but restart epochs",
)
misc_arg.add_argument(
    "--ckpt_name",
    type=str,
    default="",
    help="Name of the checkpoint file to resume from",
)
misc_arg.add_argument(
    "--print_freq",
    type=int,
    default=10,
    help="How frequently to print training details",
)
misc_arg.add_argument(
    "--plot_freq", type=int, default=1, help="How frequently to plot glimpses"
)


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
