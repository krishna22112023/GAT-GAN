import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_metrics",type=str2bool,default=True)
    # -- Data params ---
    parser.add_argument("--dataset", type=str, default="motor")
    parser.add_argument("--normalize", type=str2bool, default=True)
    parser.add_argument('--seq_length',type=int,default=16)

    # -- Model params ---
    # No of blocks
    parser.add_argument("--n_blocks_gen",type=int,default=1)
    parser.add_argument("--n_blocks_disc", type=int, default=3)

    # 1D conv layer
    parser.add_argument("--kernel_size", type=int, default=7)
    # GAT layers
    parser.add_argument("--feat_gat_embed_dim", type=int, default=None)
    parser.add_argument("--time_gat_embed_dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0)

    # --- Train params ---
    parser.add_argument("--epochs", type=int, default=31)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--init_lr_gen", type=float, default=0.001)
    parser.add_argument("--init_lr_disc", type=float, default=0.001)
    parser.add_argument("--shuffle_dataset", type=str2bool, default=False)
    parser.add_argument("--use_cuda", type=str2bool, default=True)
    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument("--metric_iteration", type=int, default=10)
    parser.add_argument("--use_trained_params", type=str2bool, default=True)



    return parser
