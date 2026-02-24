"""
Configuration file for the Hub-Ripple MIA project.
Centralizes dataset paths and common settings.
"""

import os

# Get the project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset path - using local data folder for persistent storage
DATASET_PATH = os.path.join(PROJECT_ROOT, 'data')

# Common experiment settings
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 200
DEFAULT_HIDDEN_CHANNELS = 16
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_WEIGHT_DECAY = 5e-4
DEFAULT_DROPOUT = 0.5

# Dataset names
CORA_DATASET = 'Cora'
CITESEER_DATASET = 'CiteSeer'
PUBMED_DATASET = 'PubMed'

def get_dataset_path(dataset_name=CORA_DATASET):
    """
    Get the path to a specific dataset.

    Args:
        dataset_name: Name of the dataset (default: Cora)

    Returns:
        Absolute path to the dataset directory
    """
    return DATASET_PATH

# Print configuration when imported
if __name__ == "__main__":
    print("=" * 60)
    print("Hub-Ripple MIA Project Configuration")
    print("=" * 60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Dataset Path: {DATASET_PATH}")
    print(f"Default Seed: {DEFAULT_SEED}")
    print(f"Default Epochs: {DEFAULT_EPOCHS}")
    print("=" * 60)
