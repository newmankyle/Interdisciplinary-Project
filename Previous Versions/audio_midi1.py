import numpy as np
from scipy import optimize
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')
#%matplotlib inline ##I KNOW this is illegal in python, but I've seen it twice...

import IPython.display

# Librosa for audio
from librosa import cqt
# And the display module for visualization
import librosa.display

from music21 import note
from music21 import audioSearch
from music21 import midi

import math
# frame-base evaluations compare the transcribed output to the MIDI accompaniment.

# downsample from 44.1 kHz to 16 kHz
# 7 octave CQT's w/ 36 bins.
# hop size is 512 samples.
# creates a 252 dimensional vector

## ACOUSTIC MODEL INPUT: takes the above vector, computes the mean and standard deviation,
## then subtracts the mean from the vector, and divides by the standard deviation.
    # Mean: sum(X)/count(X)
    # Standard Deviation: sqrt( mean( [Xi - mean(X)]**2 ) )
    
## ACOUSTIC MODEL OUTPUT: 88 units corresponding to the 88 output pitches.
## the output of the final layer is transformed by a sigmoid function.

## CONVOLUTION NEURAL NETWORK PARAMETERS: 
    # window size {3,5,7,9}     ws = (7)
    # conv layers {1,2,3,4} (2)
    # filters/layer {10,25,50,75,100}   n1=n2=(50)
    # fully connected layers {1,2,3}    Lfc = (2)
    # hidden units in fully connected layers {200,500,1000}     h1 = 1000, h2 = 200
    
    # conv activation functions: sigmoid prime aka. hyperbolic tangent function.
    # fully connected layer activations: sigmoid function.
    
    # pooling over the frequency axis
    # pooling size P = (1,3)
    
    # dropout w/ rate 0.5 applied to convolutional AND fully connected layers
    
    # window shape. choose from: {(3,3), (3,5), (5,5), (3,25), (5,25), (3,75), (5,75)} w1 = (5,25), w2 = (3,5)
    
    # trained with Stochastic Gradient Descent. Batch Size: 256
    # initial learning rate of 0.01. Constant momentum rate 0.9
    # Stop training if validation error doesn't decrease after 20 iterations.

## LANGUAGE MODEL INPUT: Takes the 88 dimensional binary vectors from the AM and sample the MIDI provided
## sample the MIDI transcriptions of the training data at the rate 32 ms.
## creates sequences of 88 dimensional binary vectors (which correspond to A0-C8 on the piano).

## RNN-NADE PARAMETERS:
    # trained with SGD w/ sequences of lenght 100
    # recurrent units: Hrrn = {50,100,150,200,250,300} (200)
    # hidden units for NADE = {50.100,150,200,250,300} (150)
    # initial learning rate 0.001. linearly reduced to 0 over 1000 iterations.
    # constant momentum rate of 0.9
    
## Post-Processing paramaters.
    # For the AM output, any probability below the threshold value is 0'd, and any above is set to 1.
    # Individual pitch HMM's
    
# pip uninstall matplotlib
# git clone https://github.com/matplotlib/matplotlib.git
# cd matplotlib
# python setup.py install
# pip install --upgrade matplotlib

# pip install IPython
# pip install pygame #?
# pip install music21

## NEXT:
    # Come up with a basic transcription machine
    # directly transfrom the cqt into midi
    # have output statistics
    # have a constant threshold
    # string together harmonics. C's in each frame-base
    # grid search of threshold

## QUESTIONS:
    # What does standardization do? It just seems to up the power.
    # is the standardization supposed to compute normalized values?
    # should I be standardizing along the y axis or x axis?
    # Try ftp connection at Uvic. Shaw seems to block ftp connections. (inbound port 21)

def printPitches(pitches):
    print(pitches.shape, librosa.hz_to_note(np.amax(pitches)))
    for (i,j), x in np.ndenumerate(pitches):
        if (x > 0):
            print(librosa.hz_to_note(x))
    
def threshold(cqt, thresh):
    output = np.zeros((88, cqt.shape[1]))
    count = 0
    thresh = np.amax(cqt) * thresh
    for (i,j), value in np.ndenumerate(cqt):
        if (value > thresh):
            print(math.floor((i)/3) + 36)
            newI=math.floor((i)/3)
            output.itemset((newI,j), 1)
        else:
            newI=math.floor((i)/3)
            output.itemset((newI,j), 0)
    #print(count)
    return output

def printMidi(binaryInput):
    print()
    for (i,j), value in np.ndenumerate(binaryInput):
        if(value>0):
            print(librosa.midi_to_note(i+24))
    
def main():

    #loads and downsamples the audio to 16kHz as per the article
    audio_path = r"C:\Users\Kyle\workspace\CSC497\Project\UMAPiano-DB-A4-NO-M.wav"
    x, sr = librosa.load(audio_path, sr = 16000)
    
    p = audioSearch.transcriber.monophonicStreamFromFile(audio_path) # returns a stream containing notes
    #p.show('text')
    p = midi.translate.streamToMidiFile(p) # converts a stream to a midi file
    
    # computes the cqt of the audio file. In this case A4 on the piano.
    # returns np.ndarray [shape=(n_bins, t), dtype=np.complex or np.float]
    C = librosa.cqt(x, sr=sr, hop_length=512, n_bins= 36*7, bins_per_octave=36,  real=True)
    C = np.abs(C)
    # print(C.shape)
    # # displays the CQT of the input waveform
    # librosa.display.specshow(librosa.logamplitude(C**2, ref_power=np.max), sr=sr, x_axis='time', y_axis='cqt_note')    
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Constant-Q power spectrum')
    # plt.tight_layout()
    # plt.show()
    
    #print("C: ", C)                
    #mean = C.mean(axis=0).reshape(-1, 1) #compute the mean of the cqt, but keep an empty second dimension
    #mean = C.mean(axis=0).transpose
    #standDev = np.ndarray.std(C, axis=0, dtype=complex).transpose
    
    #finalC = (C - mean) / standDev  # identical to zscore. zscore is faster.
    zscoreC = stats.zscore(C, axis=1) # zscore returns the standardized cqt for input.
    binaryInput = threshold(zscoreC, 0.9)
    
    printMidi(binaryInput)
    
    #pitches, magnitudes = librosa.piptrack(y=x, sr=sr,  threshold=0.99, fmin=librosa.note_to_hz('A0'), fmax=librosa.note_to_hz('C8'))
    #printPitches(pitches)
    
    # print(zscoreC.shape)
    # # displays the standardized CQT
    # librosa.display.specshow(librosa.logamplitude(zscoreC**2, ref_power=np.max),
                # sr=sr, x_axis='time', y_axis='cqt_note')
    
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Constant-Q power spectrum')
    # plt.tight_layout()
    # plt.show()

    # #print()
    # #print("finalC: ", finalC)
    # print()
    # print("zscore: ", np.abs(zscoreC))
    
    
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Constant-Q power spectrum')
    # plt.tight_layout()
    
    # plt.show()

if __name__ == "__main__":
    main()