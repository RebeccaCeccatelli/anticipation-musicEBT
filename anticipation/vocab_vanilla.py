"""
Simplified vocabulary without anticipation control block.

Compared to vocab.py:
- Removes the entire control block (ATIME, ADUR, ANOTE tokens)
- Reduces vocabulary size by ~50%
- Still supports arrival-time and interarrival-time encodings
- Suitable for standard sequential music generation

This is the "vanilla" version without anticipatory infilling capability.
"""

# training sequence vocab

from anticipation.config import *

# the event block (same as original)
EVENT_OFFSET = 0
TIME_OFFSET = EVENT_OFFSET
DUR_OFFSET = TIME_OFFSET + MAX_TIME
NOTE_OFFSET = DUR_OFFSET + MAX_DUR
REST = NOTE_OFFSET + MAX_NOTE

# the special block (before control block for logical organization)
SPECIAL_OFFSET = NOTE_OFFSET + MAX_NOTE + 1
SEPARATOR = SPECIAL_OFFSET
AUTOREGRESS = SPECIAL_OFFSET + 1
VOCAB_SIZE = AUTOREGRESS + 1  # No control block in vanilla

# the control block (unreachable in vanilla - defined for ops.py compatibility)
# In vanilla, VOCAB_SIZE < CONTROL_OFFSET, so no tokens will ever reach the control branch
CONTROL_OFFSET = VOCAB_SIZE
ATIME_OFFSET = CONTROL_OFFSET + 0
ADUR_OFFSET = ATIME_OFFSET + MAX_TIME
ANOTE_OFFSET = ADUR_OFFSET + MAX_DUR

# interarrival-time (MIDI-like) vocab (same as original)
MIDI_TIME_OFFSET = 0
MIDI_START_OFFSET = MIDI_TIME_OFFSET + MAX_INTERARRIVAL
MIDI_END_OFFSET = MIDI_START_OFFSET + MAX_NOTE
MIDI_SEPARATOR = MIDI_END_OFFSET + MAX_NOTE
MIDI_VOCAB_SIZE = MIDI_SEPARATOR + 1

if __name__ == '__main__':
    print('Vanilla Arrival-Time Training Sequence Format (NO ANTICIPATION):')
    print('Event Offset: ', EVENT_OFFSET)
    print('  -> time offset :', TIME_OFFSET)
    print('  -> duration offset :', DUR_OFFSET)
    print('  -> note offset :', NOTE_OFFSET)
    print('  -> rest token: ', REST)
    print('Special Token Offset: ', SPECIAL_OFFSET)
    print('  -> separator token: ', SEPARATOR)
    print('  -> autoregression flag: ', AUTOREGRESS)
    print('Vanilla Arrival Encoding Vocabulary Size: ', VOCAB_SIZE)
    print('  (Original VOCAB_SIZE would be ~2x larger due to control block)')
    print('')
    print('Interarrival-Time Training Sequence Format:')
    print('  -> time offset: ', MIDI_TIME_OFFSET)
    print('  -> note-on offset: ', MIDI_START_OFFSET)
    print('  -> note-off offset: ', MIDI_END_OFFSET)
    print('  -> separator token: ', MIDI_SEPARATOR)
    print('Interarrival Encoding Vocabulary Size: ', MIDI_VOCAB_SIZE)
