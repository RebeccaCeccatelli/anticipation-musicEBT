# GigaMIDI Anticipation Tokenization with Checkpoint Recovery

## Overview

The tokenization pipeline now supports **Option B: Fine-grained checkpoint tracking** with **automatic job resubmission**. This allows long-running tokenization jobs to continue automatically across multiple SLURM submissions, handling the 12-hour cluster time limit seamlessly.

## How It Works

### 1. Checkpoint System (Option B)

Instead of processing all files in a single job, the system now:

- **Saves state every ~5% of files processed**: When processing ~170k GigaMIDI files, checkpoints are saved every ~8,500 files
- **Stores exact file index and timestamp**: Checkpoint files (`.checkpoint-{split}.json`) contain:
  - Current file index: "We've processed files 0-8500, next file to process is 8501"
  - Total files in split
  - Timestamp of last checkpoint
  - Percent progress
- **Resumes deterministically**: Files are sorted before processing to ensure consistent ordering across job submissions

### 2. Automatic Job Resubmission

The `auto_submit_gigamidi.py` script:
1. **Submits initial job** for each split sequentially
2. **Monitors job status** every 30 seconds with `sacct`
3. **Detects timeout** (job reaches 12-hour limit)
4. **Checks checkpoint** to confirm progress was saved
5. **Resubmits job with `--resume` flag** to continue from exact checkpoint
6. **Repeats until completion** (up to 5 attempts per split by default)

### Example Timeline

**Train split (136k files) with vanilla tokenization:**

```
Job 1 (0-00:12:00):  Processes files 0→8,500   (TIMEOUT)
  ✓ Checkpoint saved: "next file is 8,501"

Job 2 (0-00:12:00):  Processes files 8,501→17,000  (TIMEOUT)
  ✓ Checkpoint saved: "next file is 17,001"

Job 3 (0-00:12:00):  Processes files 17,001→25,500  (TIMEOUT)
  ✓ Checkpoint saved: "next file is 25,501"

... continues until ...

Job 16:  Processes files 128,000→136,984  (COMPLETED)
  ✓ Checkpoint deleted
```

Each job independently continues from the exact file index, so you can safely (re)run jobs without worrying about duplicates or missing files.

## Usage

### Quick Start: Auto-submit All Splits

```bash
cd /home/rebcecca/music-EBT

# Non-vanilla (arrival-time) tokenization
bash job_scripts/mus/tokenization/auto_submit_gigamidi.sh

# Vanilla tokenization  
bash job_scripts/mus/tokenization/auto_submit_gigamidi.sh --vanilla
```

The script will:
- Submit jobs for train, test, and validation sequentially
- Monitor each job and automatically resubmit on timeout
- Print progress updates and final summary
- Exit with status 0 if all splits complete, 1 if any fail

### Advanced Usage

```bash
# Process only train split with vanilla
bash job_scripts/mus/tokenization/auto_submit_gigamidi.sh --vanilla --splits train

# Process train and test only (skip validation)
bash job_scripts/mus/tokenization/auto_submit_gigamidi.sh --vanilla --splits train test

# Increase max resume attempts to 10
bash job_scripts/mus/tokenization/auto_submit_gigamidi.sh --max-attempts 10

# Check status more frequently (every 15 seconds instead of 30)
bash job_scripts/mus/tokenization/auto_submit_gigamidi.sh --poll-interval 15
```

### Manual Job Submission (if needed)

```bash
# Submit just the train split (non-vanilla)
sbatch custom.sh --split train

# Resume train split from checkpoint
sbatch custom.sh --split train --resume

# Process vanilla variant
sbatch custom.sh --split train --vanilla --resume
```

## How Checkpoint Files Work

### Checkpoint File Format

Location: `/home/rebcecca/orcd/pool/music_datasets/GigaMIDI/tokens/anticipation/.checkpoint-{split}.json`

Example content:
```json
{
  "split_name": "train",
  "file_index": 8501,
  "total_files": 136984,
  "processed_count": 8500,
  "timestamp": "2026-04-16T15:32:45.123456",
  "percent_complete": 6.21
}
```

### Checkpoint Lifecycle

1. **Created**: After processing ~5% of files (automatic)
2. **Updated**: Every ~5% of files (automatic)
3. **Deleted**: When split successfully completes
4. **Persisted**: If job times out (preserved for resume)

### Manual Checkpoint Management

```python
from tokenizer_utils import load_checkpoint, delete_checkpoint, save_checkpoint

# Check current checkpoint
checkpoint = load_checkpoint(token_dir, 'train')
if checkpoint:
    print(f"Train split at {checkpoint['percent_complete']}% complete")

# Manually delete checkpoint (forces restart on next run)
delete_checkpoint(token_dir, 'train')

# Manually create checkpoint (for advanced use cases)
save_checkpoint(token_dir, 'train', file_index=50000, 
                total_files=136984, processed_count=49999)
```

## Monitoring & Debugging

### Monitor Running Jobs

```bash
# See all GigaMIDI tokenization jobs
squeue -u $USER | grep tokenize

# Detailed status
sacct -u $USER | grep gigamidi

# Live status of specific job
watch 'sacct -j 12345 -o State --noheader'
```

### View Progress

```bash
# Check checkpoint progress
cat /home/rebcecca/orcd/pool/music_datasets/GigaMIDI/tokens/anticipation/.checkpoint-train.json

# Monitor output file size (should grow as tokenization progresses)
watch 'ls -lh /home/rebcecca/orcd/pool/music_datasets/GigaMIDI/tokens/anticipation/tokenized-events-*.txt'

# Watch logs (if using WandB)
# Find in WandB dashboard under "Music-EBT" project
```

### Troubleshooting

**Problem**: Job times out but no checkpoint found
```bash
# Possible cause: Validation phase took too long
# Check job logs for error messages
# Re-run with no resume flag (will restart from beginning)
sbatch custom.sh --split train
```

**Problem**: Stuck on resumed job timeouts
```bash
# Delete checkpoint and restart fresh
rm /home/rebcecca/orcd/pool/music_datasets/GigaMIDI/tokens/anticipation/.checkpoint-train.json
sbatch custom.sh --split train
```

**Problem**: Output file incomplete after timeout
```bash
# This is safe! The file is continuously appended to.
# Resumed jobs will continue from exact file_index checkpoint.
# No duplicates will be written.
sbatch custom.sh --split train --resume
```

## Performance Expectations

### Timing Breakdown (GigaMIDI Vanilla Tokenization)

- **Validation phase**: ~10 hours (one-time, shared across jobs)
- **Preprocessing**: ~1 hour (one-time, creates .compound.txt files)
- **Tokenization rate**: ~1 minute per 1,000 files

### Estimated Jobs Needed

**Train split (136,984 files):**
- At 1 min/1000 files = ~137 minutes per job
- With 12-hour limit: 4-5 jobs total
- Total wall-clock time: ~48-60 hours (multiple jobs run in parallel across your allocation)

**Test split (17,044 files):**
- ~17 minutes per job
- Usually completes in 1 job

**Validation split (17,044 files):**
- ~17 minutes per job
- Usually completes in 1 job

## Architecture Details

### Code Changes

1. **`tokenizer_utils.py`** - New functions:
   - `save_checkpoint()`: Save progress state
   - `load_checkpoint()`: Load progress state
   - `delete_checkpoint()`: Clean up after completion
   - `get_checkpoint_path()`: Get checkpoint file location

2. **`anticipation/train/tokenize_gigaMIDI.py`** - New features:
   - `--split` argument: Process only specific split
   - `--resume` flag: Continue from checkpoint
   - Sorted file lists for deterministic ordering
   - Passes checkpoint info to tokenization functions

3. **`anticipation/anticipation/tokenize.py`** - Modified functions:
   - `tokenize()` and `tokenize_ia()` now accept:
     - `start_idx`: Starting file index for resumption
     - `split_name`: For checkpoint identification
     - `token_dir`: For checkpoint file path
   - Checkpoint saved every ~5% of files
   - Tracking of "actual_file_index" for correct resume point

4. **`anticipation/train/auto_submit_gigamidi.py`** - New script:
   - Finds and runs SLURM submission script
   - Monitors job status via `sacct`
   - Detects and handles timeouts
   - Resubmits with `--resume` flag
   - Provides progress reporting

### Design Rationale: Option B vs Option A

**Option B (Chosen) - Fine-grained State Tracking:**
- ✅ Exact resume point: "continue from file 8,501" vs "try next part file"
- ✅ No re-processing: Each file processed exactly once
- ✅ Deterministic: Sorted file lists ensure consistent ordering
- ✅ Auto-resume: Automatic job resubmission via monitoring script
- ⚠️  Slightly more complex implementation

**Option A (Simple Part Files) - Not Chosen:**
- ✅ Simpler to implement
- ✅ Clear file segmentation (.part0, .part1, etc.)
- ❌ Requires manual monitoring
- ❌ Manual concatenation after all parts complete
- ❌ Less efficient: Might not use full 12-hour window

## Next Steps

### To fully deploy:

1. **Test**: Run test split to verify checkpoint/resume behavior:
   ```bash
   cd /home/rebcecca/music-EBT
   bash job_scripts/mus/tokenization/auto_submit_gigamidi.sh --vanilla --splits test
   ```

2. **Monitor**: Watch in a separate terminal:
   ```bash
   # Check checkpoint progress
   cat /home/rebcecca/orcd/pool/music_datasets/GigaMIDI/tokens/anticipation/.checkpoint-test.json
   
   # Or watch in real-time
   watch 'cat /home/rebcecca/orcd/pool/music_datasets/GigaMIDI/tokens/anticipation/.checkpoint-test.json'
   ```

3. **Scale**: Once confident, deploy for train split:
   ```bash
   cd /home/rebcecca/music-EBT
   bash job_scripts/mus/tokenization/auto_submit_gigamidi.sh --vanilla --splits train
   ```

4. **Monitor in background**: Consider running auto-submission in a `screen` or `tmux` session for long-running deployments

## File Locations

- **Auto-submission script**: `job_scripts/mus/tokenization/auto_submit_gigamidi.sh`
- **Checkpoint utilities**: `data/mus/symbolic/tokenization/tokenizer_utils.py`
- **Checkpoint files**: `~/.../GigaMIDI/tokens/anticipation/.checkpoint-{split}.json`
- **Output files**: `~/.../GigaMIDI/tokens/anticipation/tokenized-events-{split}.txt`

## Questions?

If you encounter issues or have questions:
1. Check checkpoint file for progress: `cat .checkpoint-{split}.json`
2. Monitor logs: `tail -f job_output.log`
3. Check SLURM status: `sacct -j {job_id}`
4. Review this documentation for troubleshooting section
