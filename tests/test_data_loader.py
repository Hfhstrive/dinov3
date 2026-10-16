import gc
import multiprocessing
import unittest
import warnings

from omegaconf import OmegaConf

from dinov3.data.loaders import make_data_loader
from dinov3.train.train import configure_training_gc


class GarbageCollectionProbe:
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return gc.isenabled()


class DataLoaderTests(unittest.TestCase):
    def test_training_gc_policy_is_inherited_by_forked_data_workers(self):
        if multiprocessing.get_start_method() != "fork":
            self.skipTest("This regression concerns Linux fork workers inheriting disabled cyclic GC.")
        was_enabled = gc.isenabled()
        try:
            for enabled in (False, True):
                with self.subTest(gc_enabled=enabled):
                    cfg = OmegaConf.create({"train": {"gc_enabled": enabled}})
                    configure_training_gc(cfg)
                    loader = make_data_loader(
                        dataset=GarbageCollectionProbe(), batch_size=2, num_workers=1,
                        prefetch_factor=1, sampler_type=None, shuffle=False, collate_fn=list,
                    )
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message=".*pin_memory.*no accelerator.*")
                        batches = list(loader)
                    self.assertEqual(batches, [[enabled, enabled], [enabled, enabled]])
        finally:
            gc.enable() if was_enabled else gc.disable()

    def test_prefetch_batches_and_zero_worker_loading(self):
        dataset = list(range(6))
        for workers in (0, 1):
            with self.subTest(workers=workers):
                loader = make_data_loader(
                    dataset=dataset, batch_size=2, num_workers=workers,
                    prefetch_factor=1, sampler_type=None, shuffle=False, collate_fn=list,
                )
                self.assertEqual(loader.prefetch_factor, 1 if workers else None)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*pin_memory.*no accelerator.*")
                    batches = list(loader)
                self.assertEqual(batches, [[0, 1], [2, 3], [4, 5]])

    def test_invalid_prefetch_is_rejected_before_workers_start(self):
        dataset = list(range(2))
        for prefetch in (0, -1, 1.5):
            with self.subTest(prefetch=prefetch), self.assertRaisesRegex(ValueError, "positive integer"):
                make_data_loader(dataset=dataset, batch_size=2, num_workers=1, prefetch_factor=prefetch)


if __name__ == "__main__":
    unittest.main()
