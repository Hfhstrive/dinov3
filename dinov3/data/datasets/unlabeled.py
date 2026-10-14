from pathlib import Path

from .extended import ExtendedVisionDataset


class UnlabeledImages(ExtendedVisionDataset):
    """Images under any directory tree, or paths listed in a UTF-8 manifest.

    No class directories or annotation files are required. Relative manifest
    entries are resolved against root; extra is the optional manifest path.
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(self, root, extra=None, **kwargs):
        super().__init__(root=str(Path(root).expanduser()), **kwargs)
        root_path = Path(self.root)
        if not root_path.is_dir():
            raise FileNotFoundError(f"Image root does not exist: {root_path}")
        if extra:
            manifest = Path(extra).expanduser()
            if not manifest.is_absolute():
                manifest = root_path / manifest
            with manifest.open(encoding="utf-8") as handle:
                self.paths = [line.strip() for line in handle if line.strip()]
        else:
            self.paths = sorted(
                str(path.relative_to(root_path))
                for path in root_path.rglob("*")
                if path.is_file() and path.suffix.lower() in self.EXTENSIONS
            )
        if not self.paths:
            raise ValueError(f"No images found under {root_path}")

    def get_image_data(self, index):
        path = Path(self.paths[index])
        if not path.is_absolute():
            path = Path(self.root) / path
        try:
            return path.read_bytes()
        except OSError as exc:
            raise RuntimeError(f"Cannot read image: {path}") from exc

    def get_target(self, index):
        return 0  # Placeholder; the SSL loader discards targets.

    def __len__(self):
        return len(self.paths)
