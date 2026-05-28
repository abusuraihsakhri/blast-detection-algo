"""
Dry-run script to verify the WarmStartTrainer pipeline.
Copies only 5 images per dataset to test the full pipeline flow
without waiting hours for 21,000 images to merge and train.
"""

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training_pipeline import WarmStartTrainer
from database import init_db

if __name__ == "__main__":
    print(f"Running dry-run WarmStart training pipeline with max_samples=5...")
    init_db()
    trainer = WarmStartTrainer()
    trainer.train(max_samples=5)
    print("Dry-run complete.")
