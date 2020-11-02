import csv
import os

from functions import initiate_birds
from functions import initiate_libr

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
#Define train and test dicts

ones = open('train.csv', 'w').close()
zeros = open('test.csv', 'w').close()

train = dict()
test = dict()

for st in libr.keys():
    train.update({st: []})
    test.update({st: []})

#Init train and test dicts
for l in libr.keys():
    counter = 0
    for x in libr[l]:
        if counter >= round(len(libr[l]) * 0.85):
            test.update(l=test[l].append(x))
            counter += 1
        else:
            train.update(l=train[l].append(x))
            counter +=1

train.pop('l')
test.pop('l')

#x = open('train.csv', 'w+').close()

#Create train.csv
row = ['bird', 'song']
with open('train.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(row)


for t in train.keys():
    for val in train[t]:
        file = open('train.csv', 'a', newline='')
        with file:
            imp = t[:] + "~~" + song + t + "/" + val
            writer = csv.writer(file)
            writer.writerow(imp.split("~~"))
csvFile.close()

#Create test,csv
with open('test.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(row)


for t in test.keys():
    for val in test[t]:
        file = open('test.csv', 'a', newline='')
        with file:
            imp = t[:] + "~~" + song + t + "/" + val
            writer = csv.writer(file)
            writer.writerow(imp.split("~~"))
csvFile.close()






