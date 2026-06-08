"""Tests for DatasetConfig and the YAML study-config loader."""
from __future__ import annotations

import pytest

from data.annotations import ShapeType, validate_shape_size_range
from data.dataset_config import DatasetConfig, build_cpu_dataset


class TestSizeValidation:
    def test_min_above_half_image_raises(self):
        # cap = 64//2 = 32; min 40 can't fit -> clear error, not a randrange crash.
        with pytest.raises(ValueError):
            validate_shape_size_range((64, 64), (40, 60))

    def test_max_above_half_image_warns(self):
        # cap = 256//2 = 128; max 200 is silently capped -> warn instead.
        with pytest.warns(UserWarning):
            validate_shape_size_range((256, 256), (20, 200))

    def test_in_range_is_silent(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_shape_size_range((256, 256), (20, 128))  # 128 == cap, fine


class TestDatasetConfig:
    def test_from_dict_coerces_and_validates(self):
        dc = DatasetConfig.from_dict({
            "num_shapes_range": [1, 3],          # list -> tuple
            "shape_types": ["rectangle", "circle"],  # lower -> upper
            "rotate_shapes": True,
        })
        assert dc.num_shapes_range == (1, 3)
        assert dc.shape_types == ("RECTANGLE", "CIRCLE")
        assert dc.rotate_shapes is True

    def test_from_dict_rejects_unknown_keys(self):
        with pytest.raises(ValueError):
            DatasetConfig.from_dict({"rotate": True})  # typo for rotate_shapes

    def test_from_dict_rejects_bad_shape_name(self):
        with pytest.raises(ValueError):
            DatasetConfig.from_dict({"shape_types": ["pentagon"]})

    def test_from_dict_rejects_bad_background_and_outline(self):
        with pytest.raises(ValueError):
            DatasetConfig.from_dict({"background": "gradient"})
        with pytest.raises(ValueError):
            DatasetConfig.from_dict({"shape_outline": "dotted"})

    def test_null_override_inherits_rather_than_clobbering(self):
        # A null in YAML (-> None) means "not provided": keep the inherited value.
        train = DatasetConfig(num_shapes_range=(2, 4))
        val = train.merged({"num_shapes_range": None, "rotate_shapes": True})
        assert val.num_shapes_range == (2, 4)  # inherited, not None
        assert val.rotate_shapes is True

    def test_val_inherits_train_and_overrides(self):
        train = DatasetConfig(rotate_shapes=False, shape_size_range=(20, 128),
                              background="solid")
        val = train.merged({"rotate_shapes": True})
        # Only rotate_shapes changed; everything else inherited from train.
        assert val.rotate_shapes is True
        assert val.shape_size_range == (20, 128)
        assert val.background == "solid"
        # train is untouched.
        assert train.rotate_shapes is False

    def test_default_for_task(self):
        single = DatasetConfig.default_for_task("heatmap")
        assert single.num_shapes_range == (1, 1)
        assert single.shape_size_range == (20, 128)
        assert single.rotate_shapes is True
        multi = DatasetConfig.default_for_task("multi", num_shapes_range=(0, 5))
        assert multi.num_shapes_range == (0, 5)
        assert multi.rotate_shapes is False

    def test_enum_views(self):
        dc = DatasetConfig(shape_types=("CIRCLE",), background="texture", shape_outline="thin")
        assert dc.shape_type_enums() == (ShapeType.CIRCLE,)
        assert dc.background_enum().name == "TEXTURE"
        assert dc.outline_enum().name == "THIN"


class TestBuildDataset:
    def test_build_cpu_dataset_honors_config(self):
        from torchvision import transforms
        dc = DatasetConfig(num_shapes_range=(2, 2), shape_size_range=(15, 25),
                           rotate_shapes=False, max_overlap=0.3)
        ds = build_cpu_dataset(dc, num_images=4, image_size=(64, 64), seed=0,
                               transform=transforms.ToTensor())
        assert ds.num_shapes_range == (2, 2)
        assert ds.shape_size_range == (15, 25)
        assert ds.rotate_shapes is False
        assert ds.max_overlap == 0.3
        img, anns = ds[0]
        assert img.shape == (3, 64, 64)


class TestLoadStudyConfig:
    def test_yaml_train_val_inheritance(self, tmp_path):
        from scripts.run_training import load_run_config
        cfg_path = tmp_path / "study.yaml"
        cfg_path.write_text(
            "task: heatmap\n"
            "epochs: 7\n"
            "train:\n"
            "  rotate_shapes: false\n"
            "val:\n"
            "  rotate_shapes: true\n"
        )
        rc = load_run_config(str(cfg_path))
        assert rc.task == "heatmap"
        assert rc.epochs == 7
        # heatmap task default (1,1) is inherited even though the config omits it.
        assert rc.train.num_shapes_range == (1, 1)
        assert rc.train.rotate_shapes is False
        # val inherits everything from train, overriding only rotation.
        assert rc.val.rotate_shapes is True
        assert rc.val.num_shapes_range == (1, 1)

    def test_yaml_rejects_unknown_top_level_key(self, tmp_path):
        from scripts.run_training import load_run_config
        cfg_path = tmp_path / "bad.yaml"
        cfg_path.write_text("task: single\nepocs: 5\n")  # typo
        with pytest.raises(ValueError):
            load_run_config(str(cfg_path))
