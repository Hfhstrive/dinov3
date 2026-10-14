"""Continue self-supervised pretraining on a custom unlabeled image dataset.

Use torchrun --standalone --nproc_per_node=1 (or 4) -m dinov3.train.custom.
"""

import argparse
import math
import os
from pathlib import Path
import sys
import tempfile

from omegaconf import OmegaConf

from dinov3.configs import get_default_config
from dinov3.data.datasets import UnlabeledImages


MODELS = {
    "vitb16": "vit_base",
    "vits16": "vit_small",
    "vits16plus": "vit_small_plus",
    "vitl16": "vit_large",
    "vith16plus": "vit_huge2_plus",
    "convnext_tiny": "convnext_tiny",
    "convnext_small": "convnext_small",
    "convnext_base": "convnext_base",
    "convnext_large": "convnext_large",
}


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Image directory; nested folders are supported")
    parser.add_argument("--manifest", help="Optional UTF-8 list of relative or absolute image paths")
    parser.add_argument("--model", choices=MODELS, default="vitb16")
    parser.add_argument("--weights", help="Explicit official backbone .pth (overrides weights-root)")
    parser.add_argument("--weights-root", default="/media/inno/Pretrained/DINOV3")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8, help="Images per GPU, before multi-crop expansion")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--smoke-test", action="store_true", help="Short real training, saving checkpoints and exports")
    parser.add_argument("--steps", type=int, default=5, help="Number of optimizer steps for smoke-test")
    parser.add_argument("--dry-run", action="store_true", help="Check paths and print config without using the GPU")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("opts", nargs="*", help="Optional config overrides: crops.global_crops_size=256")
    return parser


def make_config(args):
    if args.batch_size < 2 or args.epochs < 1 or args.steps < 1 or args.num_workers < 0:
        raise ValueError("Require batch-size >= 2, epochs/steps >= 1 and num-workers >= 0.")
    root = Path(args.data_root).expanduser().resolve()
    manifest = str(Path(args.manifest).expanduser().resolve()) if args.manifest else None
    for value in (str(root), manifest):
        if value and ":" in value:
            raise ValueError("Dataset paths containing ':' are not supported by the dataset-string format.")
    dataset = UnlabeledImages(root=root, extra=manifest)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if len(dataset) < args.batch_size * world_size:
        raise ValueError("Dataset must contain at least one global batch of images.")
    if args.weights:
        weights = Path(args.weights).expanduser().resolve()
    else:
        matches = sorted(Path(args.weights_root).expanduser().rglob(f"dinov3_{args.model}_pretrain_lvd1689m-*.pth"))
        if len(matches) != 1:
            raise ValueError(f"Expected one LVD-1689M weight file for {args.model}, found {len(matches)}; use --weights.")
        weights = matches[0].resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)

    cfg = OmegaConf.merge(get_default_config(), OmegaConf.load(Path(__file__).parents[1] / "configs/train/custom.yaml"))
    cfg.student.arch = MODELS[args.model]
    cfg.student.pretrained_weights = str(weights)
    if args.model in {"vits16plus", "vith16plus"}:
        cfg.student.ffn_layer = "swiglu"
        cfg.student.ffn_ratio = 6.0
    cfg.train.dataset_path = f"UnlabeledImages:root={root}" + (f":extra={manifest}" if manifest else "")
    cfg.train.output_dir = str(Path(args.output_dir).expanduser().resolve())
    cfg.train.batch_size_per_gpu = args.batch_size
    cfg.train.num_workers = args.num_workers
    cfg.optim.epochs = args.epochs
    # Official epochs count a fixed number of iterations; here make them approximate full data passes.
    cfg.train.OFFICIAL_EPOCH_LENGTH = math.ceil(len(dataset) / (args.batch_size * world_size))
    if args.model.startswith("convnext"):
        cfg.student.n_storage_tokens = 0
        cfg.student.mask_k_bias = False
        cfg.student.patch_size = 32  # Native last-stage feature stride, not ViT's 16.
        cfg.ibot.loss_weight = 0.0
        cfg.ibot.mask_sample_probability = 0.0
        cfg.optim.layerwise_decay = 1.0
    if args.smoke_test:
        cfg.optim.epochs = 1
        cfg.train.OFFICIAL_EPOCH_LENGTH = args.steps
        cfg.optim.warmup_epochs = 0
        cfg.teacher.warmup_teacher_temp_epochs = 0
        cfg.crops.local_crops_number = 2
        cfg.checkpointing.period = max(1, args.steps // 2)
        cfg.evaluation.eval_period_iterations = 0
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    if cfg.train.batch_size_per_gpu != args.batch_size:
        raise ValueError("Use --batch-size to change batch size so the epoch length is calculated correctly.")
    if cfg.multidistillation.enabled or cfg.distillation.enabled:
        raise ValueError("This entry point is for custom-dataset SSL adaptation; distillation is a separate training stage.")
    if args.model.startswith("convnext") and cfg.ibot.loss_weight != 0:
        raise ValueError("ConvNeXt adaptation requires ibot.loss_weight=0.")
    if cfg.train.sharded_eval_checkpoint:
        raise ValueError("Use consolidated eval checkpoints for standalone backbone export.")
    if cfg.ibot.loss_weight > 0 and cfg.ibot.mask_sample_probability <= 0:
        raise ValueError("iBOT requires a positive mask_sample_probability.")
    if cfg.ibot.loss_weight > 0 and (
        int(2 * args.batch_size * cfg.ibot.mask_sample_probability) < 1 or cfg.crops.global_crops_size < 64
    ):
        raise ValueError("iBOT needs at least one masked crop per GPU and global crop size >= 64.")
    crop_stride = 32 if args.model.startswith("convnext") else 16
    for size in (cfg.crops.global_crops_size, cfg.crops.local_crops_size):
        if not isinstance(size, int) or size < crop_stride or size % crop_stride:
            raise ValueError(f"Crop sizes must be positive multiples of {crop_stride} for {args.model}.")
    if not args.dry_run:
        previous_config = Path(cfg.train.output_dir) / "config.yaml"
        checkpoints = Path(cfg.train.output_dir) / "ckpt"
        if checkpoints.is_dir() and any(checkpoints.iterdir()):
            if args.no_resume:
                raise ValueError("Output directory already contains checkpoints; choose a new --output-dir for a fresh run.")
            if previous_config.is_file():
                previous = OmegaConf.load(previous_config)
                for key in (
                    "student.arch", "student.n_storage_tokens", "student.norm_layer",
                    "student.patch_size", "student.ffn_layer", "student.ffn_ratio",
                    "dino.head_n_prototypes", "ibot.head_n_prototypes",
                    "train.dataset_path",
                ):
                    if OmegaConf.select(previous, key) != OmegaConf.select(cfg, key):
                        raise ValueError(f"Resume configuration changed {key}; choose a new --output-dir.")
    print(
        f"Custom dataset: {len(dataset):,} images | {args.model} | {world_size} GPU(s) | "
        f"global batch {args.batch_size * world_size} | "
        f"{cfg.train.OFFICIAL_EPOCH_LENGTH} iterations/epoch | "
        f"{cfg.optim.epochs * cfg.train.OFFICIAL_EPOCH_LENGTH} total steps",
        flush=True,
    )
    return cfg


def main():
    args = get_parser().parse_args()
    cfg = make_config(args)
    if args.dry_run:
        print(OmegaConf.to_yaml(cfg))
        return
    # Each rank gets a private temporary config; the official entry point persists
    # the effective config in output_dir/config.yaml and handles resume/FSDP.
    with tempfile.TemporaryDirectory(prefix="dinov3_custom_") as temp_dir:
        config_path = Path(temp_dir) / "config.yaml"
        OmegaConf.save(cfg, config_path)
        sys.argv = [
            "dinov3.train.train", "--config-file", str(config_path),
            "--output-dir", cfg.train.output_dir, "--seed", str(args.seed),
        ] + (["--no-resume"] if args.no_resume else [])
        from dinov3.train.train import main as train_main

        try:
            train_main()
        finally:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
