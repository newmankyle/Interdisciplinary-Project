from pretty_midi import *

import numpy as np
from scipy import optimize
from scipy import stats

import librosa.display
from librosa import cqt
from librosa import resample
import matplotlib.pyplot as plt


def constantThreshold(input, thresh):
    output = np.zeros((128, input.shape[1]))
    count = 0
    thresh = np.amax(input) * thresh
    for (i,j), value in np.ndenumerate(input):
        if (value > 0):
            #print(math.floor((i)/3) + 36)
            newI=math.floor((i))
            output.itemset((newI,j), 1)
        else:
            newI=math.floor((i))
            output.itemset((newI,j), 0)
    #print(count)
    return output
def convert_roll_to_binary(pianoInput):
    output = np.zeros((88, pianoInput.shape[1]))
    for (i,j), value in np.ndenumerate(pianoInput):
        if (i >= 24 and i <= 96 and value == 1):
            newI= i - 24
            output.itemset((newI,j), 1)
    return output
    
def printMidi(binaryInput):
    print()
    for (i,j), value in np.ndenumerate(binaryInput):
        if(i < 24 and value == 1):
            print(librosa.midi_to_note(i), i)
    
def main():
    # Load MIDI file into PrettyMIDI object
    midi_data = pretty_midi.PrettyMIDI(r'.\Bach_BWV849-02_001_20090916-SMD.mid')
    
    # audio_path = r".\Bach_BWV849-02_001_20090916-SMD.wav" #tried SAARLAND's MP3 and my own conversion. 100, and 58 frame difference.
    audio_path = r".\Bach_BWV849-02_001_20090916-SMD.wav"
    x, sr = librosa.load(audio_path, sr = 44100)
    y = librosa.resample(x, sr, 16000)
    
    audio_data = librosa.cqt(y, sr=sr, hop_length=512, n_bins= 36*7, bins_per_octave=36,  real=True)
    audio_data = np.abs(audio_data)
    zscoreC = stats.zscore(audio_data, axis=1) # zscore returns the standardized cqt for input.
    # Print an empirical estimate of its global tempo
    
    # cqt is sampled at a frame rate of 31.25 frames/second. This yields 32 ms long frames.
    # We sample the midi at the same rate in order to achieve the same frame rates as the cqt.
    times = np.zeros((128, zscoreC.shape[1]))
    piano_roll = midi_data.get_piano_roll(fs=31.25)
    
    binary = constantThreshold(piano_roll, 0.1)
    #print(binary.shape)
    
    #printMidi(binary)
    
    binary1 = convert_roll_to_binary(binary)
    print(binary1.shape)
    print(audio_data.shape)
    print(zscoreC.shape[1] - binary1.shape[1])
    
    # librosa.display.specshow(binary1, x_axis='time', y_axis='cqt_note')
    # plt.title('Piano Roll')
    # plt.tight_layout()
    # plt.show()
    
    # displays the standardized CQT
    # librosa.display.specshow(librosa.logamplitude(zscoreC**2, ref_power=np.max),
              # sr=sr, x_axis='time', y_axis='cqt_note')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Constant-Q power spectrum')
    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()