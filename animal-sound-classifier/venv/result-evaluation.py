### Helper database connection functions will be implemented in here.
### This module will be implemented when GUI and network package of the project is ready.
from subprocess import call

# Visual Results are printed here

# Written Results are printed here
def call_visual_result():
    call(["open", "RF-CM.png"])
    call(["open", "SVM-CM.png"])
    call(["open", "DBSCAN.png"])
    call(["open", "K-Means.png"])

# Sends output to GUI
def send_to_gui():
    print('Data will sent to gui')

# Random Forest Classification Results will be forwarded
def rf_result(image, lst):
    call(["open", image])
    print(lst)
    send_to_gui()


# SVM Classification Results will be forwarded
def svm_result(image, lst):
    call(["open", image])
    print(lst)
    send_to_gui()


# DBSCAN Classification Results will be forwarded
def dbscan_result(image, lst):
    call(["open", image])
    print(lst)
    send_to_gui()


# K-Means Classification Results will be forwarded
def kmeans_result(image, lst):
    call(["open", image])
    print(lst)
    send_to_gui()




