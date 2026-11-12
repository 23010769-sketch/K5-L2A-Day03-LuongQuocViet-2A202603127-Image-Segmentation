"""Exercise local UI review in temporary storage, without network or learner artifacts."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from automation.annotation_review import ReviewStore
from automation.checkpoint_loader import load_checkpoints
from automation.config import ROOT


def button(app, label):
    return next(widget for widget in app.button if widget.label == label)


class LocalUiTests(unittest.TestCase):
    def test_checkpoint_switch_and_manual_review(self):
        with tempfile.TemporaryDirectory() as directory:
            def temporary_store(cp):
                return ReviewStore(cp, Path(directory))
            with patch("automation.annotation_review.ReviewStore", side_effect=temporary_store):
                app = AppTest.from_file(str(ROOT / "automation/app.py"), default_timeout=30).run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                button(app, "Tạo mask vẽ tay mới").click().run()
                button(app, "Áp dụng các nét trong bảng").click().run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                app.text_area[0].set_value("Synthetic UI test only; native pixel at origin.").run()
                app.checkbox[0].check().run()
                button(app, "Chấp nhận local").click().run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                cp = load_checkpoints()["cp1_holes"]
                self.assertTrue(ReviewStore(cp, Path(directory)).load()[0].approved)
                for name in load_checkpoints():
                    next(widget for widget in app.selectbox if widget.label == "Checkpoint").select(name).run()
                    self.assertFalse(app.exception, name)
                    self.assertFalse(app.error, name)


if __name__ == "__main__":
    unittest.main()
