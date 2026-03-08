"""Training history logger — records per-epoch metrics to CSV files."""

import csv
import os


class TrainingLogger:
    """Records training metrics (train MAE, train MSE, val MAE, etc.) to a CSV file
    so the full training process can be reviewed or plotted later.

    One CSV file is created per (model, dataset, mode, seed) combination.
    """

    HEADER = ["epoch", "train_mae", "train_mse", "val_mae", "best_val_mae", "lr"]

    def __init__(self, log_dir, model_name, dataset_name, mode, seed):
        os.makedirs(log_dir, exist_ok=True)
        filename = f"seed{seed}_history.csv"
        self.filepath = os.path.join(log_dir, filename)
        self.rows = []

        # If resuming, load existing rows
        if os.path.isfile(self.filepath):
            with open(self.filepath, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.rows.append(row)

    def log(self, epoch, train_mae, train_mse, val_mae, best_val_mae, lr):
        """Append one epoch's metrics."""
        self.rows.append({
            "epoch": epoch,
            "train_mae": f"{train_mae:.6f}",
            "train_mse": f"{train_mse:.6f}",
            "val_mae": f"{val_mae:.6f}",
            "best_val_mae": f"{best_val_mae:.6f}",
            "lr": f"{lr:.2e}",
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
            "lr": f"{test_mae:.6f}",
        })
        self._flush()

    def _flush(self):
        with open(self.filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.HEADER)
            writer.writeheader()
            writer.writerows(self.rows)
