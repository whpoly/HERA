"""Training history logger — records per-epoch metrics to CSV files."""

import csv
import math
import os


class TrainingLogger:
    """Records training metrics (train MAE, train MSE, val MAE, etc.) to a CSV file
    so the full training process can be reviewed or plotted later.

    One CSV file is created per (model, dataset, mode, seed) combination.
    """

    HEADER = ["epoch", "train_mae", "train_mse", "val_mae", "best_val_mae", "lr", "test_mae"]

    @staticmethod
    def filepath_for(log_dir, seed):
        return os.path.join(log_dir, f"seed{seed}_history.csv")

    @classmethod
    def completed_test_mae(cls, log_dir, seed):
        """Return the completed test MAE for a split, or None if incomplete."""
        filepath = cls.filepath_for(log_dir, seed)
        if not os.path.isfile(filepath):
            return None

        completed_mae = None
        try:
            with open(filepath, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("epoch") != "TEST":
                        continue
                    value = row.get("test_mae") or row.get("lr")
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(value):
                        completed_mae = value
        except (OSError, csv.Error):
            return None

        return completed_mae

    def __init__(self, log_dir, model_name, dataset_name, mode, seed):
        os.makedirs(log_dir, exist_ok=True)
        self.filepath = self.filepath_for(log_dir, seed)
        self.rows = []

    def log(self, epoch, train_mae, train_mse, val_mae, best_val_mae, lr):
        """Append one epoch's metrics."""
        self.rows.append({
            "epoch": epoch,
            "train_mae": f"{train_mae:.6f}",
            "train_mse": f"{train_mse:.6f}",
            "val_mae": f"{val_mae:.6f}",
            "best_val_mae": f"{best_val_mae:.6f}",
            "lr": f"{lr:.2e}",
            "test_mae": "",
        })
        # Write full file each time (atomic, safe for interruption)
        self._flush()

    def log_test_result(self, test_mae):
        """Append a final summary row with the test MAE."""
        self.rows.append({
            "epoch": "TEST",
            "train_mae": "",
            "train_mse": "",
            "val_mae": "",
            "best_val_mae": "",
            "lr": "",
            "test_mae": f"{test_mae:.6f}",
        })
        self._flush()

    def _flush(self):
        with open(self.filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.HEADER)
            writer.writeheader()
            writer.writerows(self.rows)
