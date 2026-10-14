import contextlib
import io
import logging
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

from dinov3.logging import _LevelColoredFormatter


class ColoredLoggingTests(unittest.TestCase):
    def setUp(self):
        self.formatter = _LevelColoredFormatter("%(levelname)s %(message)s")

    def test_logging_config_does_not_emit_handler_error(self):
        config = OmegaConf.create({"student": {"arch": "vit_huge2_plus"}, "optim": {"epochs": 20}})
        stream = io.StringIO()
        errors = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(self.formatter)
        logger = logging.Logger("test_config_logging")
        logger.addHandler(handler)
        try:
            with contextlib.redirect_stderr(errors):
                logger.info(config)
            self.assertEqual(errors.getvalue(), "")
            self.assertIn(str(config), stream.getvalue())
        finally:
            handler.close()

    def test_formatting_preserves_message_objects_and_arguments(self):
        for message, args in (
            (42, ()),
            ({"arch": "vit_base"}, ()),
            ("%s", ("ready",)),
            ("step %d loss %.3f", (12, 0.1234)),
            ("count %(count)d", ({"count": 3},)),
            ("100% complete", ()),
        ):
            with self.subTest(message=message):
                record = logging.LogRecord("test", logging.INFO, __file__, 1, message, args, None)
                original_args = record.args
                rendered = record.getMessage()
                with patch("dinov3.logging.colored", side_effect=lambda text, **kwargs: f"<color>{text}</color>"):
                    formatted = self.formatter.format(record)
                self.assertEqual(formatted, f"<color>INFO </color>{rendered}")
                self.assertIs(record.msg, message)
                self.assertIs(record.args, original_args)

    def test_exception_traceback_is_preserved(self):
        try:
            raise ValueError("example failure")
        except ValueError:
            import sys

            record = logging.LogRecord("test", logging.ERROR, __file__, 1, "operation failed", (), sys.exc_info())
        formatted = self.formatter.format(record)
        self.assertIn("operation failed", formatted)
        self.assertIn("Traceback (most recent call last):", formatted)
        self.assertIn("ValueError: example failure", formatted)


if __name__ == "__main__":
    unittest.main()
