"""Synthetic fixtures are tests, never learner annotations or submission exports."""
from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

from automation.annotation_review import ReviewStore, decide, new_item, revise
from automation.checkpoint_loader import load_checkpoints
from automation.config import ROOT, Settings
from automation.cvat_client import CvatClient, build_shapes, label_mapping
from automation.export_validator import decode_coco, inspect_pixels
from automation.mask_validator import binary_mask, validate_mask, validate_scene
from automation.segmentation_engine import grabcut, segformer, strokes


class CheckpointTests(unittest.TestCase):
    def test_repo_contract_and_paths(self):
        checkpoints = load_checkpoints()
        self.assertEqual(len(checkpoints), 6)
        self.assertEqual(checkpoints["cp4_curb"].labels, ("road", "sidewalk"))
        self.assertIn("traffic sign", checkpoints["cp3_thin"].labels)
        self.assertEqual(checkpoints["cp2_slice"].images[0].name, "000000017627.jpg")
        for cp in checkpoints.values():
            self.assertTrue(cp.images[0].is_file())
            self.assertIn(cp.export_format, cp.rule)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.cp = load_checkpoints()["cp1_holes"]
        self.mask = np.zeros((20, 30), dtype=bool)
        self.mask[2:18, 2:28] = True
        self.mask[5:9, 5:9] = False

    def test_holes_preserved_and_flagged(self):
        before = self.mask.copy()
        result = validate_mask(self.mask, "car", self.cp, self.mask.shape)
        self.assertEqual(result.metrics["holes"], 1)
        np.testing.assert_array_equal(self.mask, before)

    def test_empty_unknown_label_and_dimensions(self):
        self.assertTrue(validate_mask(np.zeros((20, 30), bool), "car", self.cp, (20, 30)).errors)
        self.assertTrue(validate_mask(self.mask, "curb", self.cp, (20, 30)).errors)
        self.assertTrue(validate_mask(self.mask, "car", self.cp, (10, 30)).errors)
        with self.assertRaises(ValueError):
            binary_mask(np.full((20, 30), 100), (20, 30))

    def test_occlusion_does_not_split_instance(self):
        cp = load_checkpoints()["cp5_occlusion"]
        mask = np.zeros((20, 30), bool)
        mask[1:5, 1:5] = True
        mask[10:14, 10:14] = True
        result = validate_mask(mask, "car", cp, mask.shape)
        self.assertEqual(result.metrics["components"], 2)
        self.assertFalse(result.errors)

    def test_overlap_and_roi_are_scoped(self):
        cp = load_checkpoints()["cp4_curb"]
        result = validate_scene([("road", self.mask), ("sidewalk", self.mask)], cp, self.mask.shape)
        self.assertTrue(result.errors)
        result = validate_scene([("car", self.mask)], self.cp, self.mask.shape, np.ones(self.mask.shape, bool))
        self.assertFalse(result.errors)
        self.assertEqual(result.metrics["uncovered_required_pixels"], int((~self.mask).sum()))

    def test_native_one_pixel_stroke_and_erase(self):
        blank = np.zeros((20, 30), bool)
        edits = [dict(x1=3, y1=2, x2=3, y2=12, radius=1, operation="add")]
        painted = strokes(blank, edits)
        self.assertEqual(int(painted.sum()), 11)
        self.assertFalse(strokes(painted, [dict(edits[0], operation="erase")]).any())
        with self.assertRaises(ValueError):
            strokes(blank, [dict(edits[0], x1=-1)])

    def test_cpu_grabcut_real_algorithm(self):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[20:45, 20:45] = (220, 220, 220)
        mask = grabcut(image, (10, 10, 55, 55))
        self.assertTrue(mask[30, 30])
        self.assertFalse(mask[0, 0])
        self.assertEqual(mask.shape, image.shape[:2])

    def test_models_require_policy_before_loading(self):
        with self.assertRaises(PermissionError):
            segformer(np.zeros((10, 10, 3)), self.cp, Path("not-a-model"), Settings())


class ReviewAndApiTests(unittest.TestCase):
    def setUp(self):
        self.cp = load_checkpoints()["cp1_holes"]
        with Image.open(self.cp.images[0]) as image:
            self.shape = (image.height, image.width)
        self.mask = np.zeros(self.shape, bool)
        self.mask[10:40, 10:40] = True
        self.mask[20:25, 20:25] = False
        self.mask[60:70, 60:70] = True
        self.item = new_item(self.cp, "car", self.mask, "synthetic unit test only")
        self.accepted = decide(self.item, "accepted", "Synthetic test review, not learner work")
        self.labels = [dict(id=100 + index * 7, name=label, type="any", attributes=[])
                       for index, label in enumerate(self.cp.labels)]
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)

    def test_review_roundtrip_and_edit_invalidates_approval(self):
        store = ReviewStore(self.cp, self.directory)
        store.save(self.accepted)
        loaded = store.load()[0]
        self.assertTrue(loaded.approved)
        np.testing.assert_array_equal(loaded.mask(self.shape), self.mask)
        edited = revise(loaded, self.cp, np.zeros(self.shape, bool), "car")
        self.assertFalse(edited.approved)
        store.save(edited)
        self.assertEqual(store.load()[0].decision, "pending")
        self.assertFalse(replace(loaded, label="bus").approved)
        with self.assertRaises(ValueError):
            decide(self.item, "accepted", "")

    def test_cvat_payload_roundtrip_hole_and_disconnected_parts(self):
        from cvat_sdk.api_client import models
        from cvat_sdk.api_client.model_utils import to_json
        shapes = build_shapes(self.cp, [self.accepted], self.labels)
        payload = models.PatchedLabeledDataRequest(shapes=[models.LabeledShapeRequest(**shapes[0])])
        shape = to_json(payload)["shapes"][0]
        self.assertEqual(shape["label_id"], label_mapping(self.cp, self.labels)["car"])
        points = list(map(int, shape["points"]))
        x1, y1, x2, y2 = points[-4:]
        runs = points[:-4]
        decoded = np.repeat(np.arange(len(runs)) % 2, runs).reshape(y2-y1+1, x2-x1+1)
        np.testing.assert_array_equal(decoded.astype(bool), self.mask[y1:y2+1, x1:x2+1])
        self.assertEqual(len(shapes), 1)  # disconnected parts retain one instance

    def test_payload_rejects_stale_pending_bad_mapping(self):
        for item in [self.item, replace(self.accepted, image_sha256="stale")]:
            with self.assertRaises(ValueError):
                build_shapes(self.cp, [item], self.labels)
        with self.assertRaises(ValueError):
            label_mapping(self.cp, self.labels[:-1])
        with self.assertRaises(ValueError):
            build_shapes(self.cp, [], self.labels)

    def test_policy_and_confirmation_fail_before_network(self):
        with patch("cvat_sdk.Client") as sdk:
            with self.assertRaises(PermissionError):
                with CvatClient(Settings()):
                    pass
            sdk.assert_not_called()
        with self.assertRaises(PermissionError):
            CvatClient(Settings()).upload(1, self.cp, [self.accepted], "")
        settings = Settings(api_allowed=True, write_allowed=True, annotation_read_allowed=True)
        with self.assertRaises(PermissionError):
            CvatClient(settings).upload(1, self.cp, [self.accepted], "WRONG")
        for host in ["http://example.com", "https://user:secret@example.com", "https://app.cvat.ai/path"]:
            with self.assertRaises(ValueError):
                Settings(host=host, token="dummy", api_allowed=True).require_api()

    def fake_client(self, *, existing=False, fail=False):
        annotations = {"version": 0, "shapes": [{"id": 9}] if existing else [], "tags": [], "tracks": []}
        task = SimpleNamespace(id=42, name=self.cp.name, status="annotation",
                               get_labels=lambda: self.labels,
                               get_meta=lambda: {"frames": [{"name": self.cp.images[0].name,
                                                              "width": self.shape[1], "height": self.shape[0]}]},
                               get_jobs=lambda: [], get_annotations=lambda: copy.deepcopy(annotations))
        writes = []
        def create(**kwargs):
            from cvat_sdk.api_client.model_utils import to_json
            writes.append(kwargs)
            if fail:
                raise TimeoutError("Synthetic ambiguous transport failure")
            shapes = to_json(kwargs["patched_labeled_data_request"])["shapes"]
            for index, shape in enumerate(shapes):
                shape["id"] = 1000 + index
            annotations["shapes"] = shapes
            return copy.deepcopy(annotations), None
        settings = Settings(token="dummy", api_allowed=True, write_allowed=True, annotation_read_allowed=True)
        client = CvatClient(settings, self.directory)
        client._client = SimpleNamespace(tasks=SimpleNamespace(retrieve=lambda task_id: task),
                                         api_client=SimpleNamespace(tasks_api=SimpleNamespace(partial_update_annotations=create)))
        return client, task, writes

    def test_additive_write_and_readback_with_fake_server(self):
        client, _, writes = self.fake_client()
        result = client.upload(42, self.cp, [self.accepted], "UPLOAD cp1_holes TASK 42 MASKS 1")
        self.assertEqual(result["state"], "verified")
        self.assertEqual(writes[0]["action"], "create")
        self.assertEqual(result["annotation_ids"], [1000])

    def test_never_write_into_existing_task(self):
        client, _, writes = self.fake_client(existing=True)
        with self.assertRaises(ValueError):
            client.upload(42, self.cp, [self.accepted], "UPLOAD cp1_holes TASK 42 MASKS 1")
        self.assertFalse(writes)

    def test_ambiguous_write_blocks_retry(self):
        client, _, writes = self.fake_client(fail=True)
        with self.assertRaises(TimeoutError):
            client.upload(42, self.cp, [self.accepted], "UPLOAD cp1_holes TASK 42 MASKS 1")
        with self.assertRaises(FileExistsError):
            client.upload(42, self.cp, [self.accepted], "UPLOAD cp1_holes TASK 42 MASKS 1")
        self.assertEqual(len(writes), 1)

    def test_mismatched_task_or_frame_blocks_metadata(self):
        client, task, _ = self.fake_client()
        task.name = "hard_panoptic"
        with self.assertRaises(ValueError):
            client.inspect(42, self.cp)
        task.name = self.cp.name
        task.get_meta = lambda: {"frames": [{"name": "wrong.jpg", "width": self.shape[1], "height": self.shape[0]}]}
        with self.assertRaises(ValueError):
            client.inspect(42, self.cp)


class ExportTests(unittest.TestCase):
    def test_coco_compressed_uncompressed_and_invalid(self):
        rng = np.random.default_rng(42)
        for mask in [rng.random((21, 37)) > .7, np.zeros((21, 37), bool), np.ones((21, 37), bool)]:
            encoded = coco_mask.encode(np.asfortranarray(mask, dtype=np.uint8))
            encoded["counts"] = encoded["counts"].decode("ascii")
            np.testing.assert_array_equal(decode_coco(encoded, mask.shape), mask)
        np.testing.assert_array_equal(decode_coco({"size": [2, 3], "counts": [0, 6]}, (2, 3)), np.ones((2, 3), bool))
        for seg in [{"size": [2, 3], "counts": [5]}, {"size": [3, 2], "counts": [6]},
                    {"size": [2, 3], "counts": "~"}, [[0, 0, float("nan"), 1, 1, 1]]]:
            with self.assertRaises(ValueError):
                decode_coco(seg, (2, 3))

    def test_deep_qc_detects_empty_mask_despite_structural_ok(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "data/checkpoints/cp4_curb"
            (base / "images").mkdir(parents=True)
            (root / "data/manifest.json").write_text(json.dumps({"tasks": {"cp4_curb": {"type": "semantic", "path": "checkpoints/cp4_curb"}}}))
            (base / "classes.json").write_text(json.dumps({"classes": ["road", "sidewalk"]}))
            Image.new("RGB", (10, 10)).save(base / "images/test.jpg")
            png = io.BytesIO()
            Image.new("RGB", (10, 10)).save(png, format="PNG")
            archive = root / "cp4_curb.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("labelmap.txt", "background:0,0,0::\nroad:1,2,3::\nsidewalk:4,5,6::")
                stream.writestr("SegmentationClass/test.png", png.getvalue())
            result = inspect_pixels("cp4_curb", archive, root)
            self.assertTrue(any("không có pixel" in message for message in result["errors"]))


if __name__ == "__main__":
    unittest.main()
