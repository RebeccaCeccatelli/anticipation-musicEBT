# Anticipatory Music Transformer (music-EBT fork)

This repository is the anticipation tokenizer used by [music-EBT](https://github.com/RebeccaCeccatelli/music_EBT) (TONIC), where it is a git submodule at `data/mus/symbolic/tokenization/anticipation`. We use it to tokenize GigaMIDI into arrival-time anticipation sequences for training Energy-Based Transformers and AR baselines.

**Main contribution:** this fork extends the anticipation library to the large-scale [GigaMIDI](https://huggingface.co/datasets/Metacreation/GigaMIDI) dataset with an end-to-end tokenization pipeline that can resume from checkpoints after cluster time-outs, and it adds an optional "vanilla" vocabulary without anticipation controls.

## Fork lineage

1. **[jthickstun/anticipation](https://github.com/jthickstun/anticipation)** is the original implementation of the [Anticipatory Music Transformer](https://arxiv.org/abs/2306.08620) by Thickstun, Hall, Donahue and Liang. Its README is kept below.
2. **[lancelotblanchard/anticipation](https://github.com/lancelotblanchard/anticipation)** by Lancelot Blanchard is the fork we forked from. It added the piano-roll visualiser, the drum-track preprocessing option, `tokenize-custom.py` and other changes. They are listed under "Changes inherited from lancelotblanchard/anticipation".
3. **This fork** (RebeccaCeccatelli/anticipation-musicEBT) adds the changes below, starting from upstream commit `78c8fd3`.

## Additions in this fork

### GigaMIDI tokenization pipeline
- **`train/tokenize_gigaMIDI.py`** tokenizes a preprocessed GigaMIDI tree (`<root>/midi/{train,test,validation}`) into `<root>/tokens/anticipation[-vanilla]/`. Options:
  - `--root_dir` (defaults to `$CUSTOM_STORAGE_PATH/giga-midi`)
  - `-k/--augment` (1 = plain AR, 10 = AMT-style anticipation with control tokens). Since `efcb168` this applies to **all splits**, not just train.
  - `-i/--interarrival`, `--split` to process one split, `--resume` to continue from a checkpoint, and `--vanilla`
- Input files are **sorted** before processing, so the file order stays the same when a job is resumed.

### Checkpoint and resume for long cluster jobs
- **`anticipation/checkpoint_manager.py`** provides `save_checkpoint`, `load_checkpoint` and `delete_checkpoint`, which write `.checkpoint-<split>.json` files in the token directory.
- `tokenize()` and `tokenize_ia()` in `anticipation/tokenize.py` take new `start_idx`, `split_name` and `token_dir` arguments. They save a checkpoint every ~5% of files and append to the output file when resuming. This lets a tokenization job survive the 12-hour SLURM limit. See **[CHECKPOINT_README.md](CHECKPOINT_README.md)** for the full workflow. The auto-resubmission driver is in the music-EBT repo (`job_scripts/mus/tokenization/auto_submit_gigamidi.sh`).

### "Vanilla" vocabulary (no anticipation control block)
- `anticipation/vocab.py` was renamed to **`vocab_ant.py`**.
- **`anticipation/vocab_vanilla.py`** is the same event vocabulary but without the control block (ATIME/ADUR/ANOTE) and without the `ANTICIPATE` token, so the vocabulary is about half the size. It is meant for plain sequential generation.
- **`anticipation/vocab_selector.py`** picks the vocabulary at import time. It uses `vocab_ant` by default and `vocab_vanilla` when `ANTICIPATION_VANILLA=true` is set. All modules (`convert`, `ops`, `sample`, `tokenize`, `visuals`, `humaneval/*`, `scripts/*`, `tests/*`) now import from `vocab_selector`.
- Note: `--vanilla` in `tokenize_gigaMIDI.py` only changes the output folder and the W&B run name. You must also export `ANTICIPATION_VANILLA=true` to get the vanilla vocabulary.

### Experiment tracking
- [Weights & Biases](https://wandb.ai) progress logging (project `Giga-Midi-Tokenization`) in `train/midi_preprocess.py`, `train/tokenize_gigaMIDI.py` and `anticipation/tokenize.py`. Only worker 0 logs, throttled to every 10 files. If W&B fails to start, preprocessing carries on without it.

### Packaging and independence from music-EBT
- Added `__init__.py` files so the repository can be imported as a package from music-EBT.
- Renamed scripts so they can be imported as modules: `train/midi-preprocess.py` → `train/midi_preprocess.py`, `train/tokenize-lakh.py` → `train/tokenize_lakh.py`, `finetune/tokenize-custom.py` → `finetune/tokenize_custom.py`.
- `finetune/tokenize_custom.py` now **requires** `--midi_dir` and `--token_dir`, and `tokenize_gigaMIDI.py` takes `--root_dir`. No paths point into music-EBT any more, so this repository does not depend on it.
- `requirements.txt` now gives minimum versions for `transformers` and `tqdm` instead of pinning exact versions.

### Usage (GigaMIDI)
```
# 1. MIDI -> compound tokens (writes <file>.compound.txt next to each MIDI)
python train/midi_preprocess.py <root>/midi

# 2. compound -> anticipation sequences (AMT-style, augment x10)
python train/tokenize_gigaMIDI.py --root_dir <root> -k 10 [--split train] [--resume]

# Vanilla vocabulary instead
ANTICIPATION_VANILLA=true python train/tokenize_gigaMIDI.py --root_dir <root> --vanilla
```

## Changes inherited from lancelotblanchard/anticipation

- Commit dc9a77c: Add new visualisation tool `visuals.show_piano_roll(tokens)`.
- Commit 2d79000: Update `ops.max_time` to optionally include duration through `include_duration` argument.
- Commit 45dffe8: Add `retrieveCheckpointAndBuildOnnx.sh` script for use with AWS Sagemaker.
- Commit 8c93d2a: Update `--add-kick-drum` to `add-drum` to add drum track instead of just kick.
- Commits 363e3ac, cd93efc: Add `--add-kick-drum` argument in `midi-preprocess.py`.
- Commit ccfbeb6: Add `tokenize-custom.py` in `finetune`.
- Commit 3b3d414: Add `removeControls` option to `events_to_compound` function in `convert.py`. 
- Commit ffb0c22: Remove the check for long tracks in `tokenize.py`.
- Commit 69600b0: Update `sample.py` to prevent the generation of REST tokens instead of time and durations.

A branch `harmonizer-version` was created upstream for use in another project. It implements an `active_instruments` filter in the generation process. That branch is not part of this fork.

### How to prepare a custom dataset for fine-tuning

1. Create 3 MIDI files for Train, Validation, and Test, and put them in a directory following this structure:
```
├── {DATASET_NAME}
│   ├── Train
│   │   ├── {DATASET_NAME}_train.mid
│   ├── Validation
│   │   ├── {DATASET_NAME}_validation.mid
│   ├── Test
│   │   ├── {DATASET_NAME}_test.mid
```

2. In `./train`, run `python3 midi_preprocess.py {DATASET_DIR}`. The argument `--add-drum` adds a drum track (kick on every bar, hi-hat on every beat).

3. In `./finetune`, run `python3 tokenize_custom.py --midi_dir {DATASET_DIR} --token_dir {OUTPUT_DIR} -k 10`.

## Original README (jthickstun/anticipation)

Implementation of the methods described in [Anticipatory Music Transformer](https://arxiv.org/abs/2306.08620).

by [__John Thickstun__](https://johnthickstun.com/), [__David Hall__](http://dlwh.org/), [__Chris Donahue__](https://chrisdonahue.com/), and [__Percy Liang__](https://cs.stanford.edu/~pliang/).

-------------------------------------------------------------------------------------

This repository provides the code for creating anticipatory training datasets, and for sampling from models trained with anticipation. It does _not_ contain code for training these models: you may use the preprocessed datasets constructed here as input to your favorite codebase for training autoregressive transformer models. We used the [Levanter](https://github.com/stanford-crfm/levanter) codebase to train models, and include instructions [here](train) for training an Anticipatory Music Transformer with Levanter.

For additional detail about this work, see the [paper](https://arxiv.org/abs/2306.08620). You may also be interested in this [blog post](https://crfm.stanford.edu/2023/06/16/anticipatory-music-transformer.html).

Pretrained models are hosted by the Center for Research on Foundation Models (CRFM) on the [HuggingFace Hub](https://huggingface.co/stanford-crfm). 

This project is licensed under the terms of the Apache License, Version 2.0.

Begin by installing the anticipation package (from the root of this repository).

```
pip install .
```

## Software Dependencies

Run the following command to install dependencies.

```
pip install -r requirements.txt
```

## Generating Music with an Anticipatory Music Transformer

See the [Colab](https://colab.research.google.com/drive/1HCQDtGFwROpHRqcmZbV0byqbxDb74YGu?usp=sharing) notebook for interactive examples of music generation using the Anticipatory Music Transformer.

Load a pretrained model using the HuggingFace Transformers package, e.g.:

```
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained('stanford-crfm/music-medium-800k').cuda()
```

Sample from this model using the custom `generate` function implementated by the anticipation package. You can convert generated event tokens to midi using the `events_to_midi` function:

```
from anticipation.sample import generate
from anticipation.convert import events_to_midi

length = 10 # time in seconds
events = generate(model, start_time=0, end_time=length, top_p=.98)
mid = events_to_midi(events)
mid.save('generated.mid')
```

Load your own MIDI and tokenize it using the `midi_to_events` function.

```
from anticipation.convert import midi_to_events

events = midi_to_events('examples/strawberry.mid')
```

To isolate a segment of a longer stream of events, use the `ops` library to clip the stream and translate the clipped segment to time zero. To isolate a melodic line, use `extract_instruments`:

```
from anticipation import ops
from anticipation.tokenize import extract_instruments

events = ops.clip(events, 41, 41+20)
events = ops.translate(events, -ops.min_time(events, seconds=False))

events, melody = extract_instruments(events, [53])
```

To generate an accompaniment to an isolated melody, call the `generate` function using the melody as control inputs. Recombine the generated accompaniment with the melody controls using `ops.combine`:

```
history = ops.clip(events, 0, 5, clip_duration=False)
accompaniment = generate(model, 5, 20, inputs=history, controls=melody, top_p=.98)
completed_events = ops.combine(accompaniment, melody)
mid = events_to_midi(completed_events)
mid.save('generated.mid')
```

See the [Colab](https://colab.research.google.com/drive/1HCQDtGFwROpHRqcmZbV0byqbxDb74YGu?usp=sharing) notebook for additional examples of infilling control using the Anticipatory Music Transformer.

## Training an Anticipatory Music Transformer

See the [train](train) directory for instructions on preprocessing the Lakh MIDI dataset and using [Levanter](https://github.com/stanford-crfm/levanter) to train an Anticipatory Music Transformer.

## Reproducing the Human Evaluation Procedure

See the [humaneval](humaneval) directory for instructions on reproducing data used for the human evaluation results reported in the paper.

-------------------------------------------------------------------------------------

To reference this work, please cite

```bib
@article{thickstun2023anticipatory,
  title={Anticipatory Music Transformer},
  author={Thickstun, John and Hall, David and Donahue, Chris and Liang, Percy},
  journal={arXiv preprint arXiv:2306.08620},
  year={2023}
}
```
