import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from dinov3.checkpointer.backbone import load_pretrained_backbone
from dinov3.data import make_dataset
from dinov3.hub import backbones
from dinov3.layers import SwiGLUFFN
from dinov3.models import build_model
from dinov3.train.custom import get_parser, make_config
from dinov3.train.train import build_schedulers
from dinov3.train.ssl_meta_arch import SSLMetaArch


class CustomDatasetTests(unittest.TestCase):
    def test_plus_models_match_official_architectures_and_find_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for index in range(2):
                Image.new("RGB", (64, 64)).save(root / f"{index}.png")
            weights_root = root / "pretrained"
            weights_root.mkdir()
            for name in ("vits16plus", "vith16plus"):
                with self.subTest(model=name), patch.dict(os.environ, {"WORLD_SIZE": "1"}):
                    weights_path = weights_root / f"dinov3_{name}_pretrain_lvd1689m-testhash.pth"
                    torch.save({}, weights_path)
                    args = get_parser().parse_args([
                        "--data-root", str(root), "--weights-root", str(weights_root), "--model", name,
                        "--output-dir", str(root / name), "--batch-size", "2", "--dry-run",
                    ])
                    cfg = make_config(args)
                    self.assertEqual(cfg.student.ffn_layer, "swiglu")
                    self.assertEqual(cfg.student.ffn_ratio, 6)
                    self.assertEqual(cfg.student.pretrained_weights, str(weights_path))
                    # Meta tensors allow comparing the full H+ architecture without allocating its weights.
                    with torch.device("meta"):
                        configured, _ = build_model(cfg.student, only_teacher=True, device="meta")
                        official = getattr(backbones, f"dinov3_{name}")(pretrained=False)
                    self.assertEqual(
                        {key: tuple(tensor.shape) for key, tensor in configured.state_dict().items()},
                        {key: tuple(tensor.shape) for key, tensor in official.state_dict().items()},
                    )
                    self.assertIsInstance(configured.blocks[0].mlp, SwiGLUFFN)
                    self.assertEqual(configured.norm.eps, official.norm.eps)
                    self.assertEqual(configured.rope_embed.dtype, official.rope_embed.dtype)
                    self.assertEqual(configured.n_storage_tokens, official.n_storage_tokens)
                    self.assertEqual(configured.num_heads, official.num_heads)

    def test_dino_only_forward_does_not_call_ibot_head(self):
        class Backbone(torch.nn.Module):
            def forward(self, images, masks, is_training):
                self_masks = masks
                self_test.assertEqual(self_masks, [None, None])
                return [
                    {
                        "x_norm_clstoken": torch.ones(len(image), 4),
                        "x_storage_tokens": torch.empty(len(image), 0, 4),
                        "x_norm_patchtokens": torch.ones(len(image), 49, 4),
                    }
                    for image in images
                ]

        class ForbiddenHead(torch.nn.Module):
            def forward(self, images):
                raise AssertionError("Disabled iBOT head must not run")

        self_test = self
        model = SSLMetaArch.__new__(SSLMetaArch)
        torch.nn.Module.__init__(model)
        model.ibot_loss_weight = 0
        model.is_distillation_enabled = False
        model.student = torch.nn.ModuleDict({
            "backbone": Backbone(), "dino_head": torch.nn.Linear(4, 3), "ibot_head": ForbiddenHead(),
        })
        global_output, local_output = model.get_student_output(
            global_crops=torch.zeros(2, 2, 3, 224, 224),
            local_crops=torch.zeros(2, 2, 3, 96, 96), upperbound=0,
            masks=torch.zeros(4, 49, dtype=torch.bool), mask_indices_list=torch.empty(0, dtype=torch.long),
        )
        self.assertIsNone(global_output["masked_patch_after_head"])
        self.assertIsNone(global_output["masked_patch_pre_head"])
        self.assertEqual(tuple(local_output["cls_after_head"].shape), (2, 2, 3))

    def test_nested_images_and_manifest_without_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            nested = root / "检查 A"
            nested.mkdir()
            image_path = nested / "frame=1.PNG"
            Image.new("RGB", (64, 64), (120, 30, 60)).save(image_path)
            (root / "annotations.json").write_text("{}")
            dataset = make_dataset(dataset_str=f"UnlabeledImages:root={root}")
            image, target = dataset[0]
            self.assertEqual(len(dataset), 1)
            self.assertEqual((image.mode, image.size, target), ("RGB", (64, 64), 0))
            manifest = root / "train.txt"
            manifest.write_text("检查 A/frame=1.PNG\n\n", encoding="utf-8")
            listed = make_dataset(dataset_str=f"UnlabeledImages:root={root}:extra={manifest}")
            self.assertEqual(listed[0][0].tobytes(), image.tobytes())
            manifest.write_text(str(image_path) + "\n", encoding="utf-8")
            self.assertEqual(make_dataset(dataset_str=f"UnlabeledImages:root={root}:extra={manifest}")[0][0].size, (64, 64))

    def test_weights_load_strictly_and_teacher_heads_are_excluded(self):
        with tempfile.TemporaryDirectory() as folder:
            original = torch.nn.Linear(3, 2)
            path = Path(folder) / "weights.pth"
            torch.save(original.state_dict(), path)
            restored = torch.nn.Linear(3, 2)
            load_pretrained_backbone(restored, path)
            torch.testing.assert_close(restored.weight, original.weight)
            teacher_state = {"backbone." + key: tensor for key, tensor in original.state_dict().items()}
            teacher_state["dino_head.weight"] = torch.ones(7, 2)
            torch.save({"teacher": teacher_state}, path)
            load_pretrained_backbone(restored, path)
            torch.testing.assert_close(restored.bias, original.bias)
            with self.assertRaises(RuntimeError):
                load_pretrained_backbone(torch.nn.Linear(3, 4), path)

    def test_epoch_length_and_fractional_warmup_on_four_ranks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for index in range(33):
                Image.new("RGB", (64, 64)).save(root / f"{index}.png")
            weights = root / "weights.pth"
            torch.save({}, weights)
            args = get_parser().parse_args([
                "--data-root", str(root), "--weights", str(weights),
                "--output-dir", str(root / "output"), "--batch-size", "2",
            ])
            with patch.dict(os.environ, {"WORLD_SIZE": "4"}):
                cfg = make_config(args)
            self.assertEqual(cfg.train.OFFICIAL_EPOCH_LENGTH, 5)
            schedules = build_schedulers(cfg)
            self.assertEqual(len(schedules[0].schedule), 50)
            self.assertEqual(schedules[3][0], cfg.teacher.teacher_temp)

    def test_convnext_disables_ibot_and_checks_crop_stride(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for index in range(2):
                Image.new("RGB", (64, 64)).save(root / f"{index}.png")
            weights = root / "weights.pth"
            torch.save({}, weights)
            args = get_parser().parse_args([
                "--data-root", str(root), "--weights", str(weights), "--model", "convnext_tiny",
                "--output-dir", str(root / "output"), "--batch-size", "2", "--smoke-test",
            ])
            with patch.dict(os.environ, {"WORLD_SIZE": "1"}):
                cfg = make_config(args)
                self.assertEqual(cfg.student.patch_size, 32)
                self.assertEqual(cfg.ibot.loss_weight, 0)
                self.assertEqual(cfg.ibot.mask_sample_probability, 0)
                args.opts = ["crops.local_crops_size=80"]
                with self.assertRaises(ValueError):
                    make_config(args)


if __name__ == "__main__":
    unittest.main()
