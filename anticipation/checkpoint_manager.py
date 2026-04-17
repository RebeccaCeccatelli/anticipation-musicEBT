"""
Checkpoint management for resuming interrupted tokenization jobs.
"""

import json
import os
from datetime import datetime
from typing import Dict, Optional


def get_checkpoint_path(token_dir: str, split_name: str) -> str:
    """Get the checkpoint file path for a given split."""
    return os.path.join(token_dir, f'.checkpoint-{split_name}.json')


def save_checkpoint(
    token_dir: str, 
    split_name: str, 
    file_index: int, 
    total_files: int,
    processed_count: int
) -> None:
    """
    Save checkpoint state for resuming interrupted tokenization.
    
    Args:
        token_dir: Directory containing tokenized output
        split_name: Split name (train, test, validation)
        file_index: Current file index (next file to process)
        total_files: Total number of files in this split
        processed_count: Number of files successfully processed so far
    """
    checkpoint_path = get_checkpoint_path(token_dir, split_name)
    
    state = {
        'split_name': split_name,
        'file_index': file_index,
        'total_files': total_files,
        'processed_count': processed_count,
        'timestamp': datetime.now().isoformat(),
        'percent_complete': round(100.0 * file_index / total_files, 2)
    }
    
    os.makedirs(token_dir, exist_ok=True)
    with open(checkpoint_path, 'w') as f:
        json.dump(state, f, indent=2)


def load_checkpoint(token_dir: str, split_name: str) -> Optional[Dict]:
    """
    Load checkpoint state for resuming interrupted tokenization.
    
    Args:
        token_dir: Directory containing tokenized output
        split_name: Split name (train, test, validation)
    
    Returns:
        Dictionary with checkpoint state, or None if no checkpoint exists
    """
    checkpoint_path = get_checkpoint_path(token_dir, split_name)
    
    if not os.path.exists(checkpoint_path):
        return None
    
    try:
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        print(f"⚠️ Warning: Checkpoint file corrupted or unreadable: {checkpoint_path}")
        return None


def delete_checkpoint(token_dir: str, split_name: str) -> None:
    """Delete checkpoint file after successful completion."""
    checkpoint_path = get_checkpoint_path(token_dir, split_name)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
