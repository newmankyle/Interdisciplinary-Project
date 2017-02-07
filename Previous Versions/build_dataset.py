from pretty_midi import *

import pickle
import glob

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

def load_dataset():
    data=np.ndarray(shape=[0,252])
    ground_truth=np.ndarray(shape=[0,88])
    
    try:
        data = pickle.load(open("dataset.p", "rb"))
        ground_truth = pickle.load(open("groundTruth.p", "rb"))
        print("unpickling worked: ", data.shape, " " , ground_truth.shape)
    except pickle.UnpicklingError as e:
        print(e)
        return
    except FileNotFoundError as f:
        print(f)
        print("one or both pickle files were not found. Trying to build dataset from scratch:")
        
        #reinitialize data containers to avoid reconcatenation
        data=np.ndarray(shape=[0,252])
        ground_truth=np.ndarray(shape=[0,88])
        
        audio_path = '.\Dataset\AudioData\*.wav'
        midi_path = '.\Dataset\MidiData\*.mid'
        
        audio_files = glob.glob(audio_path)
        midi_files = glob.glob(midi_path)
        
        for Afile, Mfile in zip(audio_files, midi_files):
            # audio_path = r".\Bach_BWV849-02_001_20090916-SMD.wav" #tried SAARLAND's MP3 and my own conversion. 100, and 58 frame difference.
            #audio_path = r".\WavFiles\Bach_BWV849-02_001_20090916-SMD.wav"
            x, sr = librosa.load(Afile, sr = 16000)
            
            audio_data = np.abs(librosa.cqt(x, sr=sr, hop_length=512, n_bins= 36*7, bins_per_octave=36,  real=True))
            audio_data = stats.zscore(audio_data, axis=1) # zscore returns the standardized cqt for input.
        
            # Load MIDI file into PrettyMIDI object
            midi_data = pretty_midi.PrettyMIDI(Mfile)
    
            # cqt is sampled at a frame rate of 31.25 frames/second. This yields 32 ms long frames.
            # We sample the midi at the same rate in order to achieve the same frame rates as the cqt.
            piano_roll = midi_data.get_piano_roll(fs=31.25)
            midi_data = convert_roll_to_binary(piano_roll)
    
        
            if (audio_data.shape[1] > midi_data.shape[1]):
                audio_data.resize((audio_data.shape[0], midi_data.shape[1]))
                #print("audio_data was too long. Reshaped to: ", audio_data.shape)
            elif (midi_data.shape[1] > audio_data.shape[1]):
                midi_data.resize((midi_data.shape[0], audio_data.shape[1]))
                #print("midi_data was too long. Reshaped to: ", midi_data.shape)
       
            audio_data = np.swapaxes(audio_data, 0,1)
            midi_data = np.swapaxes(midi_data, 0,1)
        
            data = np.concatenate((data, audio_data), axis=0)
            ground_truth = np.concatenate((ground_truth, midi_data), axis=0)
            #print("current shape of audio: ", data.shape)

        print("trying to pickle.")
        pickle.dump(data, open("dataset.p", "wb"))
        pickle.dump(ground_truth, open("groundTruth.p", "wb"))
        
    return data, ground_truth
    
def main():
    
    
    data, ground_truth = load_dataset()
    # testBinary=constantThreshold(audio_data, 0)

    # reportStats(testBinary, midi_data)
    

    
    # model = Sequential()
    # model.add(Dense(125, input_dim=252))
    # model.add(Activation('sigmoid'))
    # model.add(Dense(88))
    # model.add(Activation('sigmoid'))
    # model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
    # model.fit(data, train_y, nb_epoch=10, batch_size=100)
    
    

if __name__ == "__main__":
    main()
    