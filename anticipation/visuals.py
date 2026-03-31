"""
Utilities for inspecting encoded music data.
"""

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import anticipation.ops as ops
from anticipation.config import *
from anticipation.vocab_selector import *

def visualize(tokens, output, selected=None):
    #colors = ['white', 'silver', 'red', 'sienna', 'darkorange', 'gold', 'yellow', 'palegreen', 'seagreen', 'cyan',
    #          'dodgerblue', 'slategray', 'navy', 'mediumpurple', 'mediumorchid', 'magenta', 'lightpink']
    colors = ['white', '#426aa0', '#b26789', '#de9283', '#eac29f', 'silver', 'red', 'sienna', 'darkorange', 'gold', 'yellow', 'palegreen', 'seagreen', 'cyan', 'dodgerblue', 'slategray', 'navy']

    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
 
    max_time = ops.max_time(tokens, seconds=False)
    grid = np.zeros([max_time, MAX_PITCH])
    instruments = list(sorted(list(ops.get_instruments(tokens).keys())))
    if 128 in instruments:
        instruments.remove(128)
 
    for j, (tm, dur, note) in enumerate(zip(tokens[0::3],tokens[1::3],tokens[2::3])):
        if note == SEPARATOR:
            assert tm == SEPARATOR and dur == SEPARATOR
            print(j, 'SEPARATOR')
            continue

        if note == REST:
            continue

        assert note < CONTROL_OFFSET

        tm = tm - TIME_OFFSET
        dur = dur - DUR_OFFSET
        note = note - NOTE_OFFSET
        instr = note//2**7
        pitch = note - (2**7)*instr

        if instr == 128: # drums
            continue     # we don't visualize this

        if selected and instr not in selected:
            continue

        grid[tm:tm+dur, pitch] = 1+instruments.index(instr)

    plt.clf()
    plt.axis('off')
    cmap = matplotlib.colors.ListedColormap(colors)
    bounds = list(range(MAX_TRACK_INSTR)) + [16]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    plt.imshow(np.flipud(grid.T), aspect=16, cmap=cmap, norm=norm, interpolation='none')

    patches = [matplotlib.patches.Patch(color=colors[i+1], label=f"{instruments[i]}")
               for i in range(len(instruments))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

    plt.tight_layout()
    plt.savefig(output)

def show_piano_roll(tokens, save_path=None, bar_length=200):
    """
    Create a piano roll representation of the tokens
    Args:
        tokens (list): List of MIDI tokens.
        save_path (str): Path to save the piano roll image. If None, does not save.
        bar_length (int): Length of bars in the piano roll.
    """

    # Collect list of instruments (control instruments will be 128+)
    instrs = list(set([(n - NOTE_OFFSET) // 128 if n < CONTROL_OFFSET else 128 + ((n - ANOTE_OFFSET) // 128) for n in tokens[2::3]]))
    # print(instrs)

    # Create a piano roll
    roll = np.zeros((len(instrs), 128, int(ops.max_time(tokens, seconds=False, include_duration=True))), dtype=np.uint8)

    for t, d, n in zip(tokens[0::3], tokens[1::3], tokens[2::3]):
        if n == REST:
            continue
        # Calculate the instrument index
        i = (n - NOTE_OFFSET) // 128 if n < CONTROL_OFFSET else 128 + ((n - ANOTE_OFFSET) // 128)

        # Find the index of the instrument in the list
        idx = instrs.index(i)

        # If the note is a control signal, adjust the time and duration
        if n < CONTROL_OFFSET:
            note = (n - NOTE_OFFSET) % 128
        else:
            t = t - CONTROL_OFFSET
            d = d - CONTROL_OFFSET
            note = (n - ANOTE_OFFSET) % 128

        roll[idx, note, t:t + (d - DUR_OFFSET)] = 1


    plt.figure(figsize=(12, 6))
    colors = matplotlib.colormaps.get_cmap('tab10')  # Or 'tab20', 'hsv', etc.

    instrs_drawn = set()
    for i, instr in enumerate(instrs):
        # Find all active notes for this instrument
        for note in range(128):
            times = np.where(roll[i, note] == 1)[0]
            if len(times) > 0:
                plt.scatter(times, [note] * len(times), s=1, color=colors(i), label=('CTRL: ' if instr >= 128 else '') + f'Instr {instr%128}' if instr not in instrs_drawn else "", alpha=0.5)
                instrs_drawn.add(instr)
    plt.xlabel('Time (steps)')
    plt.ylabel('MIDI Note')
    plt.title('Piano Roll')
    plt.legend(loc='upper right', markerscale=5, fontsize=8, frameon=False)
    plt.ylim(0, 127)
    plt.xlim(0, roll.shape[2])
    plt.tight_layout()
    # Create bar lines
    for i in range(0, roll.shape[2], bar_length):
        plt.axvline(x=i, color='gray', linestyle='--', linewidth=0.5)
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()