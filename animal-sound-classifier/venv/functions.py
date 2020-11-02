import pyaudio
import wave
import sys
import time
import struct
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import pandas as pd
from warnings import simplefilter


# Helper functions that are used to implement other methods

#Read the audio file and play
def playAudio(file_name):
    # Open the sound file
    wf = wave.open(file_name, 'r')

    # Create an interface to PortAudio
    p = pyaudio.PyAudio()

    def callback(in_data, frame_count, time_info, status):
        data = wf.readframes(frame_count)
        return (data, pyaudio.paContinue)

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                    stream_callback=callback)

    stream.start_stream()

    while stream.is_active():
        time.sleep(0.1)

    stream.stop_stream()
    stream.close()
    wf.close()

    p.terminate()

#Read the audio file and plot
def plotAudio(file_name):
    spf = wave.open(file_name, 'r')
    # Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'Int16')
    fs = spf.getframerate()

    Time = np.linspace(0, len(signal) / fs, num=len(signal))

    plt.figure(1)
    plt.title(file_name.split('/')[len(file_name.split('/'))-1])
    plt.plot(Time, signal)
    plt.xlabel('sec')
    plt.ylabel('Hz')
    plt.savefig('frequency.png')

# Helper functions that are used to implement other methods

#Audio-processing helper functions
pth = os.getcwd() + "/"
path = pth + "bird-dir/bird-types.txt"
bird_names = open(path, "r")
bird_names = [x[:-1] for x in list(bird_names)]

def initiate_birds():
    birds = []
    for each in bird_names:
        birds.append(each.replace('\n',''))
    birds.sort()
    return birds

def initiate_libr(birds):
    libr = dict()
    for x in birds:
        libr.update({x: list()})
    initiate_libr_helper(libr)
    return libr

def initiate_libr_helper(libr):
    bird_names = open(path, "r")
    for n in bird_names:
        path_t = pth + "bird-dir/" + n.replace('\n','') + ".txt"
        name_is = open(path_t, "r")
        for t in name_is:
            libr.update(name=libr[n.replace('\n','')].append(t.replace('\n','')))
        libr.pop('name')
    return libr


# Functions for the confusion Matrix
def precision(label, confusion_matrix):
    simplefilter(action='ignore', category=RuntimeError)
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label] / col.sum()


def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()


def precision_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_precisions = 0
    for label in range(rows):
        sum_of_precisions += precision(label, confusion_matrix)
    return sum_of_precisions / rows


def recall_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_recalls = 0
    for label in range(columns):
        sum_of_recalls += recall(label, confusion_matrix)
    return sum_of_recalls / columns

def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements



def preprocess():
    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)

    SAMPLE_RATE = 44100

    #Initials
    path = os.getcwd() + "/"
    song = os.getcwd() + "/bird-sounds/"
    bird_path = path + "bird-dir/bird-types.txt"
    bird_names = open(bird_path, "r")

    #Birds List
    birds = initiate_birds()

    #Birds and their songs Dictionary
    libr = initiate_libr(birds)

    train = pd.read_csv(f'{path}train.csv')
    test = pd.read_csv(f'{path}test.csv')

    train = pd.DataFrame(train)
    test = pd.DataFrame(test)


    def get_mfcc(name, path):
        b, _ = librosa.core.load(name, sr = SAMPLE_RATE)
        assert _ == SAMPLE_RATE
        gmm = librosa.feature.mfcc(b, sr = SAMPLE_RATE, n_mfcc=20)
        return pd.Series(np.hstack((np.mean(gmm, axis=1), np.std(gmm, axis=1))))


    train_data = pd.DataFrame()

    train_data['song'] = train['song']
    test_data = pd.DataFrame()
    test_data['song'] = test['song']
    train_data = train_data['song'].apply(get_mfcc, path=train['song'][0:])
    print('done loading train mfcc')
    test_data = test_data['song'].apply(get_mfcc, path=test['song'][0:])
    print('done loading test mfcc')

    train_data['bird'] = train['bird']
    test_data['bird'] = np.zeros((len(test['song'])))


    X = train_data.drop('bird', axis=1)
    feature_names = list(X.columns)
    X = X.values
    num_class = len(birds)
    c2i = {}
    i2c = {}
    for i, c in enumerate(birds):
        c2i[c] = i
        i2c[i] = c
    y = np.array([c2i[x] for x in train_data['bird'][0:]])

    return X,y


# For Unlabelled dataset
def pre_process_unlabelled():
    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)

    SAMPLE_RATE = 16000

    #Audio-processing helper functions
    pth = os.getcwd() + "/"
    path = pth + "cats_dogs/"

    data = pd.read_csv('train_test_split.csv', index_col=None)
    data = data.drop(columns=['Index'])

    # Cat set
    train_cat = data['train_cat'].values
    test_cat = data['test_cat'].values

    # Dog set
    train_dog = data['train_dog'].values
    test_dog = data['test_dog'].values

    X_train = list()
    X_test = list()

    for x in train_dog:
        X_train.append(path+str(x))

    for y in train_cat:
        X_train.append(path+str(y))

    for x in test_dog:
        X_test.append(path+str(x))

    for y in test_cat:
        X_test.append(path+str(y))

    for test in X_test:
        X_train.append(test)

    def get_mfcc(name, path):
        b, _ = librosa.core.load(name, sr=SAMPLE_RATE)
        assert _ == SAMPLE_RATE
        gmm = librosa.feature.mfcc(b, sr=SAMPLE_RATE, n_mfcc=20)
        return pd.Series(np.hstack((np.mean(gmm, axis=1), np.std(gmm, axis=1))))

    X = list()
    # MFCC Feature Extraction
    for x in range(len(X_train)):
        X.append(list(get_mfcc(X_train[0],'')))

    X = pd.DataFrame(X)

    return X.values

def list_view():
    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)

    SAMPLE_RATE = 16000

    # Audio-processing helper functions
    pth = os.getcwd() + "/"
    path = pth + "cats_dogs/"

    data = pd.read_csv('train_test_split.csv', index_col=None)
    data = data.drop(columns=['Index'])

    # Cat set
    train_cat = data['train_cat'].values
    test_cat = data['test_cat'].values

    # Dog set
    train_dog = data['train_dog'].values
    test_dog = data['test_dog'].values

    X_train = list()
    X_test = list()
    names1 = list()
    names2 = list()
    for x in train_dog:
        X_train.append(path + str(x))
        names1.append(str(x))

    for y in train_cat:
        X_train.append(path + str(y))
        names1.append(str(x))


    for x in test_dog:
        X_test.append(path + str(x))
        names2.append(str(x))


    for y in test_cat:
        X_test.append(path + str(y))
        names2.append(str(x))


    for test in X_test:
        X_train.append(test)

    for a in names2:
        names1.append(a)

    return X_train, names1


