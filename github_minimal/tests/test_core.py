import unittest

import numpy as np
import torch

import main as entry
import preprocessing
from chronos_adapter import ChronosAdapter
from chronos_grouping import build_feature_groups
from evaluate import evaluate_metrics
from experiment_config import DEFAULT_EXPERIMENT_MODELS, SUPPORTED_EXPERIMENT_MODELS
from model_utils import forward_batch, match_target_shape
from train import CombinedLoss, create_model


class CoreTests(unittest.TestCase):
    def test_main_registry_and_defaults(self):
        self.assertEqual(SUPPORTED_EXPERIMENT_MODELS, DEFAULT_EXPERIMENT_MODELS)
        self.assertEqual(len(DEFAULT_EXPERIMENT_MODELS), 9)
        config = entry.build_config(entry.build_parser().parse_args([]))
        self.assertEqual((config["seq_len"], config["pred_len"]), (168, 24))
        self.assertEqual(config["epochs"], 15)
        self.assertEqual(config["train_window_stride"], 1)
        self.assertEqual(config["recursive_stride"], 24)
        self.assertEqual(config["chronos_learning_rate"], 1e-5)

    def test_sequential_groups_cover_all_features(self):
        groups = build_feature_groups([f"device_{index}" for index in range(366)], 32)
        self.assertEqual(len(groups), 12)
        self.assertEqual([index for group in groups for index in group], list(range(366)))
        self.assertLessEqual(max(map(len, groups)), 32)

    def test_ordered_windows_preserve_target_mask(self):
        values = np.arange(20, dtype=np.float32).reshape(10, 2)
        mask = np.ones_like(values, dtype=bool)
        mask[5, 1] = False
        dataset = preprocessing.MultivariateTransformerDataset(
            values, seq_len=3, pred_len=2, target_mask=mask,
        )
        context, target, target_mask = dataset[2]
        np.testing.assert_array_equal(context.numpy(), values[2:5])
        np.testing.assert_array_equal(target.numpy(), values[5:7])
        self.assertFalse(target_mask[0, 1])

    def test_loss_and_metrics_exclude_missing_targets(self):
        prediction = torch.tensor([[[1.0, 999.0]]])
        target = torch.tensor([[[1.0, 0.0]]])
        mask = torch.tensor([[[True, False]]])
        self.assertEqual(CombinedLoss()(prediction, target, mask).item(), 0)
        metrics = evaluate_metrics(prediction.numpy(), np.where(mask.numpy(), target.numpy(), np.nan))
        self.assertEqual(metrics["mae"], 0)

    def test_chronos_uses_requested_median_not_point_return(self):
        class FakePipeline:
            def predict_quantiles(self, inputs, prediction_length, **kwargs):
                self.quantiles = kwargs["quantile_levels"]
                forecasts = [torch.full((len(item), prediction_length, 1), 0.5) for item in inputs]
                return forecasts, torch.full((1, prediction_length), 999.0)

        adapter = ChronosAdapter.__new__(ChronosAdapter)
        adapter.seq_len = 3
        adapter.pred_len = 2
        adapter.feature_groups = ((0, 1), (2, 3))
        adapter.inference_batch_size = 16
        pipeline = FakePipeline()
        result = adapter._predict_chunked(torch.zeros(1, 3, 4), pipeline=pipeline)
        self.assertEqual(pipeline.quantiles, [0.5])
        torch.testing.assert_close(result, torch.full((1, 2, 4), 0.5))

    def test_all_seven_non_chronos_model_interfaces(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        config = {"seq_len": 168, "pred_len": 24, "input_dim": 8, "dropout": 0.1}
        inputs = {
            "x_enc": torch.randn(2, 168, 8),
            "x_mark_enc": torch.zeros(2, 168, 4),
            "x_dec": torch.zeros(2, 108, 8),
            "x_mark_dec": torch.zeros(2, 108, 4),
        }
        target = torch.zeros(2, 24, 8)
        for name in ("MLP", "RNN", "GRU", "LSTM", "TCN", "iTransformer", "Autoformer"):
            with self.subTest(model=name):
                model = create_model(name, config, device="cpu").eval()
                with torch.no_grad():
                    prediction = match_target_shape(forward_batch(model, inputs, "cpu", name), target)
                self.assertEqual(tuple(prediction.shape), (2, 24, 8))
                self.assertTrue(torch.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
