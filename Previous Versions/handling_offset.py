from pretty_midi import *

import pickle

import numpy as np
from scipy import optimize
from scipy import stats

import librosa.display
from librosa import cqt
from librosa import resample

import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')
#%matplotlib inline ##I KNOW this is illegal in python, but I've seen it in two different examples...

#import IPython.display

#from sknn.mlp import Classifier, Layer
#import network

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation
from keras.optimizers import SGD

import math
# frame-base evaluations compare the transcribed output to the MIDI accompaniment.

def reportStats(output, realOut):
    TP = 0
    FP = 0
    FN = 0
    
    for (i,j), value in np.ndenumerate(output):
        realValue = realOut.item((i,j))
        if (value == 1 and realValue == 1):
            TP = TP + 1
        elif (value == 0 and realValue == 1):
            FN = FN + 1
        elif (value == 1 and realValue == 0):
            FP = FP + 1
        else:
            continue
    
    p=(TP / (TP+FP))
    r=(TP / (TP+FN))
    a=(TP / (TP+FP+FN))
    
    print("precision: ", p)
    print("recall: ", r)
    print("accuracy: ", a)
    
    if (p*r*a == 0):
        f = 0
    else:
        f=((2*p*r)/(p+r))
    print("f-measure: ", f)

def printRoll(binaryRoll):
    librosa.display.specshow(binaryRoll, x_axis='time', y_axis='cqt_note')
    plt.title('Piano Roll')
    plt.tight_layout()
    plt.show()

def printCQT(cqt):
    # displays the standardized CQT
    librosa.display.specshow(librosa.logamplitude(cqt**2, ref_power=np.max),
              sr=sr, x_axis='time', y_axis='cqt_note')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Constant-Q power spectrum')
    plt.tight_layout()
    plt.show()


def convert_roll_to_binary(pianoInput):
    output = np.zeros((88, pianoInput.shape[1]))
    for (i,j), value in np.ndenumerate(pianoInput):
        if (i >= 24 and i <= 96):
            if (value > 0):
                newI= i - 24
                output.itemset((newI,j), 1)
            else:
                newI= i - 24
                output.itemset((newI,j), 0)
            
    return output

def constantThreshold(input, thresh):
    output = np.zeros((88, input.shape[1]))
    count = 0
    thresh = np.amax(input) * thresh
    for (i,j), value in np.ndenumerate(input):
        if (value > thresh):
            #print(math.floor((i)/3) + 36)
            newI=math.floor((i)/3)
            output.itemset((newI,j), 1)
        else:
            newI=math.floor((i)/3)
            output.itemset((newI,j), 0)
    #print(count)
    return output
    
def main():
    
    
    # audio_path = r".\Bach_BWV849-02_001_20090916-SMD.wav" #tried SAARLAND's MP3 and my own conversion. 100, and 58 frame difference.
    audio_path = r".\Bach_BWV849-02_001_20090916-SMD.wav"
    x, sr = librosa.load(audio_path, sr = 44100)
    y = librosa.resample(x, sr, 16000)
    
    audio_data = np.abs(librosa.cqt(y, sr=sr, hop_length=512, n_bins= 36*7, bins_per_octave=36,  real=True))
    #audio_data = np.abs(audio_data)
    zscoreC = stats.zscore(audio_data, axis=1) # zscore returns the standardized cqt for input.
    zscoreC = np.delete(zscoreC,-1,1)
    print("printing audio first column info: ", np.max(zscoreC[0,:]))
    
    # Load MIDI file into PrettyMIDI object
    midi_data = pretty_midi.PrettyMIDI(r'.\Bach_BWV849-02_001_20090916-SMD.mid')
    # cqt is sampled at a frame rate of 31.25 frames/second. This yields 32 ms long frames.
    # We sample the midi at the same rate in order to achieve the same frame rates as the cqt.
    piano_roll = midi_data.get_piano_roll(fs=31.25)
    binary = convert_roll_to_binary(piano_roll)
    print("printing midi first column info: ", np.max(binary[0,:]))
    # print()   
    
    #testBinary=constantThreshold(zscoreC, 0)
    # print(np.max(testBinary))
    # print()
    #reportStats(testBinary, binary)
    
    
    

if __name__ == "__main__":
    main()
    