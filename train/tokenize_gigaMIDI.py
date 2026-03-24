import os
import sys
from argparse import ArgumentParser
from multiprocessing import Pool, RLock
from glob import glob
from pathlib import Path

from tqdm import tqdm

from path_utils import get_dataset_path
from dataloaders.constants import DatasetType
from anticipation.config import *
from anticipation.tokenize import tokenize, tokenize_ia

def main(args):
    # 1. Setup Paths
    root_dir = get_dataset_path("giga-midi")
    midi_dir = os.path.join(root_dir, "midi")
    token_dir = os.path.join(root_dir, "tokens", "anticipation")
    
    os.makedirs(token_dir, exist_ok=True)

    encoding = 'interarrival' if args.interarrival else 'arrival'
    print(f'=== Anticipation Tokenization: GigaMIDI ===')
    print(f'  Encoding: {encoding}')
    print(f'  Source MIDI: {midi_dir}')
    print(f'  Output Tokens: {token_dir}')

    # 2. Match actual folder names (lowercase)
    split_names = ['train', 'test', 'validation']
    split_paths = [os.path.join(midi_dir, s) for s in split_names]

    # 3. Recursive Discovery
    print("🔍 Searching for .compound.txt files...")
    all_files = []
    for p in split_paths:
        if not os.path.exists(p):
            print(f"⚠️ Warning: Split path {p} not found. Skipping.")
            all_files.append([])
            continue
        found = glob(f'{p}/**/*.compound.txt', recursive=True)
        all_files.append(found)

    # 4. Filter Tasks (The Skip Logic)
    all_outputs = [os.path.join(token_dir, f'tokenized-events-{s}.txt') for s in split_names]
    all_augments = [args.augment if s == 'train' else 1 for s in split_names]
    
    tasks = []
    for i, (f_list, out_path, aug) in enumerate(zip(all_files, all_outputs, all_augments)):
        split_name = split_names[i]
        
        # Logic: Skip if file exists and is not 0 bytes
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"⏩ Split '{split_name}' already exists at {os.path.basename(out_path)}. Skipping.")
        elif not f_list:
            print(f"❓ Split '{split_name}' has no input files. Skipping.")
        else:
            print(f"📦 Split '{split_name}' added to queue ({len(f_list)} files).")
            # Tuple order: (files, output_path, augment_factor, split_index_for_tqdm)
            tasks.append((f_list, out_path, aug, i))

    if not tasks:
        print("✨ All splits are already up to date!")
        return

    # 5. Parallel Execution
    print(f"🚀 Processing {len(tasks)} split(s) with {PREPROC_WORKERS} workers...")
    func = tokenize_ia if args.interarrival else tokenize
    
    with Pool(processes=PREPROC_WORKERS, initargs=(RLock(),), initializer=tqdm.set_lock) as pool:
        results = pool.starmap(func, tasks)

    print(f'\n✨ Tokenization complete. Files located in: {token_dir}')

if __name__ == '__main__':
    parser = ArgumentParser(description='Tokenizes GigaMIDI after preprocessing')
    parser.add_argument('-k', '--augment', type=int, default=1)
    parser.add_argument('-i', '--interarrival', action='store_true')
    main(parser.parse_args())