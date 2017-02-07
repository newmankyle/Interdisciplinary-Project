###
# Author: Kyle Newman   V00781162
# Date: December 22nd, 2016
#
# Description: Automatic Music Transcription framework designed for the CSC 497 Interdisciplinary project.
#       This is the second most up-to-date version for the project, the main difference being that this
#       generates its test set from the method get_tester() which produces the audio-midi pair for the first
#       file in the Dataset. Later it was deemed more efficient and accurate to split dataset for training and
#       validation, so get_tester() fell out of favour.
#
# Input: No command line commands. Must either have the folder 'Dataset' provided in the directory, or the pickle files
#       'dataset.p' and 'groundTruth.p' in order to train the NN. The dataset is inputted into the NN in
#       form (n_samples, n_features). The Dataset provided is around 67,000 samples long.
#
# Output: The Keras NN log, as well as evaluation information from either reportStats()
#        or model.evaluate() or both.
#
###

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

from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, Adadelta, Adam, Adamax, Nadam
from keras.callbacks import EarlyStopping

import math

# frame-base evaluations compare the transcribed output to the MIDI accompaniment.
# input: a test-output and the corresponding ground-truth. Both inputs must be numpy
#       arrays, and both must be the same shape.
# ouput: computes the precision, recall, accuracy, and fmeasure by comparing inputs
#       cell-by-cell. 
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

# Librosa commands needed to print a piano roll. Must be in the form (128, n_samples).
#   Since the NN output is 88 dimensions, we need to convert it to its original roll form
#   to use this method accurately.
def printRoll(binaryRoll):
    librosa.display.specshow(binaryRoll, x_axis='time', y_axis='cqt_note')
    plt.title('Piano Roll')
    plt.tight_layout()
    plt.show()

# Librosa commands needed to print a CQT.
def printCQT(cqt):
    # displays the standardized CQT
    librosa.display.specshow(librosa.logamplitude(cqt**2, ref_power=np.max),
              sr=sr, x_axis='time', y_axis='cqt_note')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Constant-Q power spectrum')
    plt.tight_layout()
    plt.show()

# Method for converting the pretty_midi piano roll into an 88 dimensional representation.
# input: a piano roll of the shape (128, n_samples)
# output: a binary array of the form (88, n_samples)
def convert_roll_to_binary(pianoInput):
    output = np.zeros((88, pianoInput.shape[1]))
    for (i,j), value in np.ndenumerate(pianoInput):
        if (i >= 12 and i <= 84): #C1 - C7
            if (value > 0):
                newI= i - 12 + 3 #index 0 is A0, index 3 is C1
                output.itemset((newI,j), 1)
            else:
                newI= i - 12 + 3
                output.itemset((newI,j), 0)
            
    return output

# Method for converting the 88 dimensional midi representation back into the pretty_midi format.
#    Librosa piano roll figure requires the input to be of the form (128, n_samples), so a
#    deconversion method was necessary.
# input: NN output of the form (88, n_samples)
# output: pretty_midi numpy array of the form (128, n_samples)    
def convert_binary_to_roll(pianoInput):
    output = np.zeros((128, pianoInput.shape[1]))
    for (i,j), value in np.ndenumerate(pianoInput):
        if (value > 0):
            newI= i + 12 - 3
            output.itemset((newI,j), 1)
        else:
            newI= i + 12 - 3
            output.itemset((newI,j), 0)
            
    return output

# A basic thresholding method for converting the CQT array into an 88 dimensional output
#   without using machine learning.
# input: A CQT array for conversion. Thresh: a decimal value between 0 and 1. Represents
#   the tolerance for the thresholding as a fraction of the MAX value in the CQT.
# output: an output representation of the form (88, n_samples)
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

# This method is the post processing method for thresholding the NN output. A separate method
#       was deemed necessary as constantThreshold() transformed the input shape as well as
#       thresholded the system.
# input: a NN output of the form (n_samples, 88). thresh: Thresh: a decimal value between 0 and 1. Represents
#   the tolerance for the thresholding as a fraction of the MAX value in the NN output.
# output: an output representation of the form (n_samples, 88)
def constantThreshold1(input, thresh):
    output = np.zeros((input.shape[0], 88))
    count = 0
    thresh = np.amax(input) * thresh
    for (i,j), value in np.ndenumerate(input):
        if (value > thresh):
            #print(math.floor((i)/3) + 36)
            newJ=math.floor((i)/1)
            output.itemset((i,newJ), 1)
        else:
            newJ=math.floor((i)/1)
            output.itemset((i,newJ), 0)
    #print(count)
    return output

# A method for producing the dataset for the NN. The dataset is produced in one of two ways: we look for
#   the files 'dataset.p' (pickle file for the audio) and 'groundTruth.p' (pickle file for the midi). If 
#   they aren't present we look for the folder '.\Dataset\' and use glob to search for all files wav and 
#   midi files in each of the subdirectories. Once the dataset is compiled, they are saved as pickle files
#   to significantly speed up the next running of the program (Generating the dataset from sratch takes 
#   around 5 - 10 minutes as Librosa's CQT is quite slow and numpy.enurmerate is similarly a slow process for
#   large piano rolls. Thus pickle files are a necessity for testing).
# input: none.
# output: returns the audio-midi pair datasets 'data' (audio) and 'ground_truth' (midi).    
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

# a test dataset generator. Not included in the main program as we instead split the dataset
# input: none, but the first audio-midi file pair must be included in the same folder as this program.
# output: test_audio and test_midi are the audio-midi pair as numpy arrays (n_samples, 252) and (n_samples, 88)
#   respectively.    
def get_tester():
    audio_path = r".\Bach_BWV849-02_001_20090916-SMD.wav"
    x, sr = librosa.load(audio_path, sr = 44100)
    y = librosa.resample(x, sr, 16000)
    
    audio_data = np.abs(librosa.cqt(y, sr=sr, hop_length=512, n_bins= 36*7, bins_per_octave=36,  real=True))

    audio_data = stats.zscore(audio_data, axis=1) # zscore returns the standardized cqt for input.
    
    # Load MIDI file into PrettyMIDI object
    midi_data = pretty_midi.PrettyMIDI(r'.\Bach_BWV849-02_001_20090916-SMD.mid')
    # cqt is sampled at a frame rate of 31.25 frames/second. This yields 32 ms long frames.
    # We sample the midi at the same rate in order to achieve the same frame rates as the cqt.
    piano_roll = midi_data.get_piano_roll(fs=31.25)
    binary = convert_roll_to_binary(piano_roll)
    
    if (audio_data.shape[1] > binary.shape[1]):
        audio_data.resize((audio_data.shape[0], binary.shape[1]))
        #print("audio_data was too long. Reshaped to: ", audio_data.shape)
    elif (binary.shape[1] > audio_data.shape[1]):
        binary.resize((binary.shape[0], audio_data.shape[1]))
        #print("binary was too long. Reshaped to: ", binary.shape)
    
    test_audio = np.swapaxes(audio_data, 0,1)
    test_midi = np.swapaxes(binary, 0,1)
    
    return test_audio, test_midi
 
def main():
    
    print("getting datasets")
    data, ground_truth = load_dataset()
    # testBinary=constantThreshold(audio_data, 0)

    # reportStats(testBinary, midi_data)
    
    n_samples = data.shape[0]
    
    print()
    print("building the model")
    model = Sequential()
    
    #model.add(Dropout(0.3, input_shape=(252,)))
    model.add(Dense(125, input_dim=252, activation='sigmoid'))
    #model.add(Activation('sigmoid'))
    model.add(Dropout(0.3))
    model.add(Dense(125, activation='sigmoid'))
    #model.add(Activation('sigmoid'))
    #model.add(Dropout(0.3))
    model.add(Dense(output_dim=88, activation='sigmoid'))
    #model.add(Activation('sigmoid'))
	
    #sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    #adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    #adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #adamax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='auto')
    model.compile(loss='categorical_crossentropy', optimizer=nadam, metrics=['precision', 'recall', 'accuracy', 'fmeasure'])
    
    print()
    print("model compiled, starting training")
    model.fit(data[:n_samples/2], ground_truth[:n_samples / 2], nb_epoch=200, verbose=2, batch_size=100, validation_split=0.2, callbacks=[early_stopping])
	
    print()
    print("model trained, starting prediction")
    test_audio, test_midi = get_tester()
    test_out = model.predict(test_audio, batch_size=100)
    #test_out = np.swapaxes(test_out, 0, 1)
    test_out = constantThreshold1(test_out, 0.001)

    print(test_out.shape, test_midi.shape)
    print()
    reportStats(test_out, test_midi)
    binary1 = np.swapaxes(test_out, 0, 1)
    binary1 = convert_binary_to_roll(binary1)
    print(binary1.shape)
    printRoll(binary1)
    score=model.evaluate(test_audio, test_midi, batch_size=100)
    print("Keras score was: ", score)
    
    

if __name__ == "__main__":
    main()
    