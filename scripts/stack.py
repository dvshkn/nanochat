import argparse
import os

from nanochat.common import autodetect_device_type, compute_init, get_base_dir
from nanochat.checkpoint_manager import stack_checkpoint

device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
parser.add_argument("--g", type=int, default=2, help="growth factor")
parser.add_argument("--src-model-tag", type=str, default=None, help="source checkpoint directory name")
parser.add_argument("--dest-model-tag", type=str, default=None, help="destination checkpoint directory name")
args = parser.parse_args()

assert args.src_model_tag is not None and args.dest_model_tag is not None

base_dir = get_base_dir()
src_checkpoint_dir = os.path.join(base_dir, "base_checkpoints", args.src_model_tag)
dest_checkpoint_dir = os.path.join(base_dir, "base_checkpoints", args.dest_model_tag)

stack_checkpoint(src_checkpoint_dir, dest_checkpoint_dir, device, g=args.g)
