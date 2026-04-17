import os
import sys
from argparse import ArgumentParser
from multiprocessing import Pool, RLock
from glob import glob
from pathlib import Path
from datetime import datetime

import wandb
from tqdm import tqdm

from path_utils import get_dataset_path
from dataloaders.constants import DatasetType
from anticipation.config import *
from anticipation.tokenize import tokenize, tokenize_ia
from anticipation.checkpoint_manager import save_checkpoint, load_checkpoint, delete_checkpoint

def log_print(*args, **kwargs):
    """Print with immediate flush to ensure logs appear in output."""
    print(*args, **kwargs)
    sys.stdout.flush()

def main(args):
    # Initialize wandb for logging with proper settings
    dataset_name = "giga-midi"
    vanilla_mode = args.vanilla if hasattr(args, 'vanilla') else False
    
    project_formatted = f"Giga-Midi-Tokenization"
    splits_str = args.split if args.split else "all-splits"
    
    # Build run name: anticipation-vanilla-SPLIT-TIME or anticipation-SPLIT-TIME
    if vanilla_mode:
        run_name = f"anticipation-vanilla-{splits_str}-{datetime.now().strftime('%m%d-%H%M')}"
        vocab_mode = "vanilla"
    else:
        run_name = f"anticipation-{splits_str}-{datetime.now().strftime('%m%d-%H%M')}"
        vocab_mode = "anticipation"
    
    wandb.init(
        project=project_formatted,
        name=run_name,
        settings=wandb.Settings(console="wrap_raw"),
        config={
            "dataset": dataset_name,
            "pipeline": vocab_mode,
            "splits": args.split if args.split else ["train", "test", "validation"],
            "augment": args.augment,
            "interarrival": True,  # GigaMIDI always uses interarrival
            "use_vanilla": vanilla_mode
        }
    )
    
    # 1. Setup Paths
    root_dir = get_dataset_path("giga-midi")
    midi_dir = os.path.join(root_dir, "midi")
    
    # Use separate folders for vanilla vs non-vanilla anticipation
    folder_name = "anticipation-vanilla" if vanilla_mode else "anticipation"
    token_dir = os.path.join(root_dir, "tokens", folder_name)
    
    os.makedirs(token_dir, exist_ok=True)

    encoding = 'interarrival' if args.interarrival else 'arrival'
    log_print(f'=== Anticipation Tokenization: GigaMIDI ===')
    log_print(f'  Encoding: {encoding}')
    log_print(f'  Source MIDI: {midi_dir}')
    log_print(f'  Output Tokens: {token_dir}')

    # 2. Determine which splits to process
    if args.split:
        # Process only specified split
        split_names = [args.split]
    else:
        # Process all splits
        split_names = ['train', 'test', 'validation']
    
    split_paths = [os.path.join(midi_dir, s) for s in split_names]

    # 3. Recursive Discovery
    log_print("🔍 Searching for .compound.txt files...")
    all_files = []
    for p in split_paths:
        if not os.path.exists(p):
            log_print(f"⚠️ Warning: Split path {p} not found. Skipping.")
            all_files.append([])
            continue
        found = sorted(glob(f'{p}/**/*.compound.txt', recursive=True))  # Sort for determinism
        all_files.append(found)

    # 4. Filter Tasks (Skip Logic with Checkpoint Support)
    all_outputs = [os.path.join(token_dir, f'tokenized-events-{s}.txt') for s in split_names]
    all_augments = [args.augment if s == 'train' else 1 for s in split_names]
    
    tasks = []
    for i, (f_list, out_path, aug) in enumerate(zip(all_files, all_outputs, all_augments)):
        split_name = split_names[i]
        
        # Check for existing output file
        output_exists = os.path.exists(out_path) and os.path.getsize(out_path) > 0
        
        # Load checkpoint if resuming
        checkpoint = load_checkpoint(token_dir, split_name) if args.resume else None
        
        if output_exists and not checkpoint:
            # Already complete, skip
            log_print(f"⏩ Split '{split_name}' already exists at {os.path.basename(out_path)}. Skipping.")
        elif checkpoint and args.resume:
            # Resume from checkpoint
            start_idx = checkpoint['file_index']
            total_files = len(f_list)
            percent = checkpoint['percent_complete']
            log_print(f"📋 Split '{split_name}' resuming from file {start_idx}/{total_files} ({percent}%)")
            log_print(f"   Checkpoint from: {checkpoint['timestamp']}")
            
            if start_idx >= total_files:
                log_print(f"   Checkpoint complete! All files already processed.")
                continue
            
            # Slice file list to continue from checkpoint
            files_to_process = f_list[start_idx:]
            log_print(f"📦 Split '{split_name}' added to queue ({len(files_to_process)} files remaining).")
            tasks.append((files_to_process, out_path, aug, i, start_idx, split_name, True))
        elif not f_list:
            log_print(f"❓ Split '{split_name}' has no input files. Skipping.")
        else:
            # Fresh start
            log_print(f"📦 Split '{split_name}' added to queue ({len(f_list)} files).")
            tasks.append((f_list, out_path, aug, i, 0, split_name, False))

    if not tasks:
        log_print("✨ All splits are already up to date!")
        return

    # 5. Parallel Execution
    log_print(f"🚀 Processing {len(tasks)} split(s) with {PREPROC_WORKERS} workers...")
    func = tokenize_ia if args.interarrival else tokenize
    
    with Pool(processes=PREPROC_WORKERS, initargs=(RLock(),), initializer=tqdm.set_lock) as pool:
        results = pool.starmap(func, tasks)

    # 6. Post-Processing: Clean up checkpoints for completed splits and concatenate part files
    for i, task in enumerate(tasks):
        split_name = task[5]
        output_path = task[1]
        
        # Remove checkpoint marker since processing completed
        delete_checkpoint(token_dir, split_name)
        
        result = results[i]
        if result:
            seqcount = result[0] if isinstance(result, tuple) else 0
            log_print(f"✨ {split_name.capitalize()} split complete ({seqcount} sequences written)")
            wandb.log({f"{split_name}_sequences_written": seqcount})
    
    log_print(f'\n✨ Tokenization complete. Files located in: {token_dir}')
    wandb.finish()

if __name__ == '__main__':
    parser = ArgumentParser(description='Tokenizes GigaMIDI after preprocessing (with checkpoint recovery)')
    parser.add_argument('-k', '--augment', type=int, default=1)
    parser.add_argument('-i', '--interarrival', action='store_true')
    parser.add_argument('--vanilla', action='store_true',
                       help='Use vanilla anticipation tokenizer (no control block)')
    parser.add_argument('--split', type=str, default=None, 
                       choices=['train', 'test', 'validation'],
                       help='Process only this split (default: process all splits)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint if previous job timed out (auto-detected for specified split)')
    main(parser.parse_args())