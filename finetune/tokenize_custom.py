import os
from argparse import ArgumentParser
from multiprocessing import Pool, RLock
from glob import glob
from tqdm import tqdm

from anticipation.config import *
from anticipation.tokenize import tokenize, tokenize_ia


def main(args):
    # Paths must be provided - anticipation module is independent
    midi_dir = args.midi_dir
    token_dir = args.token_dir
    
    if not midi_dir or not token_dir:
        raise ValueError("midi_dir and token_dir are required parameters")

    print(f'Reading Preprocessed files from: {midi_dir}')
    print(f'Saving Final Tokens to: {token_dir}')

    split_names = ['Train', 'Test', 'Validation']
    split_paths = [os.path.join(midi_dir, s) for s in split_names]

    # We look for .compound.txt files inside the MIDI/split folders
    files = [glob(f'{p}/*.compound.txt') for p in split_paths]

    # We save the aggregated .txt token files inside the TOKENS folder
    outputs = [os.path.join(token_dir, f'tokenized-events-{s}.txt') for s in split_names]

    augment = [args.augment if s == 'Train' else 1 for s in split_names]

    func = tokenize_ia if args.interarrival else tokenize
    with Pool(processes=PREPROC_WORKERS, initargs=(RLock(),), initializer=tqdm.set_lock) as pool:
        results = pool.starmap(func, zip(files, outputs, augment, range(len(split_names))))

    print(f'Tokenization complete. Output at: {token_dir}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="jordan-progrock-dataset")
    parser.add_argument('--midi_dir', type=str, required=True, help="Path to MIDI directory with Train/Test/Validation subfolders")
    parser.add_argument('--token_dir', type=str, required=True, help="Path to output token directory")
    parser.add_argument('-k', '--augment', type=int, default=1)
    parser.add_argument('-i', '--interarrival', action='store_true')
    main(parser.parse_args())