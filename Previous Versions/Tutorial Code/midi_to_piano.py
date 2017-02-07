import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from mido import MidiFile


def to_piano_roll(midi):
    """Convert MIDI file to a 2D NumPy ndarray (notes, timesteps)."""
    notes = 88
    tempo = 500000  # Assume same default tempo and 4/4 for all MIDI files.
    seconds_per_beat = tempo / 1000000.0
    seconds_per_tick = seconds_per_beat / midi.ticks_per_beat
    velocities = np.zeros(notes)
    sequence = []
    for m in midi:
        ticks = int(np.round(m.time / seconds_per_tick))
        ls = [velocities.copy()] * ticks
        sequence.extend(ls)
        if m.type == 'note_on':
            velocities[m.note] = m.velocity
        elif m.type == 'note_off':
            velocities[m.note] = 0
        else:
            continue
    piano_roll = np.array(sequence).T
    return piano_roll

piano_roll = to_piano_roll(MidiFile(r'.\Bach_BWV849-02_001_20090916-SMD.mid'))
print(piano_roll.shape)
librosa.display.specshow(piano_roll)
plt.title('Piano Roll')
plt.tight_layout()
plt.show()