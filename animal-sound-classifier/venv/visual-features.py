import os
import matplotlib.pyplot as plt
import librosa
import librosa.display as dsp
import sklearn

from functions import plotAudio
from functions import initiate_birds
from functions import initiate_libr


#Initials
path = os.getcwd() + "/"
song = os.getcwd() + "/bird-sounds/"
bird_path = path + "bird-dir/bird-types.txt"
bird_names = open(bird_path, "r")

#Birds List
birds = initiate_birds()

#Birds and their songs Dictionary
libr = initiate_libr(birds)


#Play ond plot bird songs counter # of times

paths = []
counter = 707

for t in libr.keys():
    for x in libr[t]:
        if counter == 0:
            break
        pth = song + t +"/" + x
        paths.append(pth)
        counter -= 1

# In order to see the visual features
# of selected song change "HERE" -> paths["HERE"] until 706

#From here to
x, sr = librosa.load(paths[222])

librosa.load(paths[222])

#Set title of graphs
title = paths[222].split('/')[len(paths[222].split('/'))-1]

plotAudio(paths[222])
#Here


X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))

plt.figure(figsize=(14, 5))
dsp.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.title(title)
plt.show()


#If to pring log of frequencies
dsp.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()
plt.title(title)
plt.show()

# Zooming in
n0 = 9000
n1 = 9100
plt.figure(figsize=(14, 5))
plt.plot(x[n0:n1])
plt.grid()
plt.title(title)
plt.show()

#Zero Crossings
zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)
print(f'Zero Crossings: {sum(zero_crossings)}')

#Spectral Centroid
spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
spectral_centroids.shape
# Computing the time variable for visualization
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)

# Normalising the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

#Plotting the Spectral Centroid along the waveform
dsp.waveplot(x, sr=sr, alpha=0.4)

plt.plot(t, normalize(spectral_centroids), color='r')
plt.title(title)
plt.show()

#Spectral Rolloff
#specified percentage of the total spectral energy, e.g. 85%, lies.
spectral_rolloff = librosa.feature.spectral_rolloff(x, sr=sr)[0]
dsp.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='r')
plt.title(title)
plt.show()

#MFCC
mfccs = librosa.feature.mfcc(x, sr=sr)

#Displaying  the MFCCs:
dsp.specshow(mfccs, sr=sr, x_axis='time')
plt.title(title)
plt.show()





