#%% MODULE BEGINS
module_name = 'PA2'
'''
Version: <***>
Description:
<***>
Authors:
<***>
Date Created : <***>
Date Last Updated: <***>
Doc:
<***>
Notes:
<***>
'''
#
#%% IMPORTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# if __name__ == "__main__":
#     import os
#     os.chdir(r"C:\Users\melof\OneDrive\Documents\GitHub\cmpsML_SpaceSomethingorOther")

#custom imports
#other imports
from copy import deepcopy as dpcpy
import scipy.signal as signal
from matplotlib import pyplot as plt
# import mne
import numpy as np
import pandas as pd
import seaborn as sns
import pickle as pckl
#
#%% USER INTERFACE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pathSoIRoot = 'INPUT\\stream'
pathSoi = f'{pathSoIRoot}\\'
soi_file = '1_132_bk_pic.pckl'

#Load SoI object
with open(f'{pathSoi}{soi_file}', 'rb') as fp:
       soi = pckl.load(fp)

#
#%% CONSTANTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% CONFIGURATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% INITIALIZATIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% DECLARATIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Global declarations Start Here
#Class definitions Start Here
#Function definitions Start Here
#%% MAIN CODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main code start here
def main():
    '''Channel Index: 
    M1: 19
    P7: 14
    P3: 6   '''

    sample_freq = soi['info']['eeg_info']['effective_srate']
    
    m1Stream = soi['series'][19]
    m1tStamp = soi['tStamp']
    m1tStamp -= m1tStamp[0]
    m1Label = soi['info']['eeg_info']['channels'][19]['label']

    p7Stream = soi['series'][14]
    p7tStamp = soi['tStamp']
    p7tStamp -= p7tStamp[0]
    p7Label = soi['info']['eeg_info']['channels'][14]['label']

    p3Stream = soi['series'][6]
    p3tStamp = soi['tStamp']
    p3tStamp -= p3tStamp[0]
    p3Label = soi['info']['eeg_info']['channels'][6]['label']

    # Plot streams
    plt.figure(figsize=(12, 6))

    plt.subplot(3, 1, 1)
    plt.plot(m1tStamp, m1Stream)
    plt.title(m1Label)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 2)
    plt.plot(p7tStamp, p7Stream)
    plt.title(p7Label)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 3)
    plt.plot(p3tStamp, p3Stream)
    plt.title(p3Label)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()
    
    #Apply notch filter
    # sample_freq = 250
    notch_freq = [60, 120, 180, 240]
    for freq in notch_freq:
        b_notch, a_notch = signal.iirnotch(w0=freq, Q=50, fs=sample_freq)
        m1Stream_notched = signal.filtfilt(b_notch, a_notch, m1Stream)
        p7Stream_notched = signal.filtfilt(b_notch, a_notch, p7Stream)
        p3Stream_notched = signal.filtfilt(b_notch, a_notch, p3Stream)
    
    #Apply impedance filter
    impedance = [124, 126]
    b_imp, a_imp = signal.butter(N=4, Wn=[impedance[0] / (sample_freq/2), impedance[1] / (sample_freq / 2)], btype='bandstop')
    m1Stream_impeded = signal.filtfilt(b_imp, a_imp, m1Stream_notched)
    p7Stream_impeded = signal.filtfilt(b_imp, a_imp, p7Stream_notched)
    p3Stream_impeded = signal.filtfilt(b_imp, a_imp, p3Stream_notched)
    
    #Apply bandpass filter
    bandpass = [0.5, 32]
    b_bandpass, a_bandpass = signal.butter(N=4, Wn=[bandpass[0] / (sample_freq/2), bandpass[1]/(sample_freq/2)],btype='bandpass')
    m1Stream_bandpass = signal.filtfilt(b_bandpass, a_bandpass, m1Stream_impeded)
    p7Stream_bandpass = signal.filtfilt(b_bandpass, a_bandpass, p7Stream_impeded)
    p3Stream_bandpass = signal.filtfilt(b_bandpass, a_bandpass, p3Stream_impeded)
    
    # Plot original and filtered signals
    plt.figure(figsize=(12, 6))

    plt.subplot(3, 1, 1)
    plt.plot(m1tStamp, m1Stream, label='Original')
    plt.plot(m1tStamp, m1Stream_notched, label='Filtered')
    plt.title(m1Label)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(p7tStamp, p7Stream, label='Original')
    plt.plot(p7tStamp, p7Stream_impeded, label='Filtered')
    plt.title(p7Label)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(p3tStamp, p3Stream, label='Original')
    plt.plot(p3tStamp, p3Stream_bandpass, label='Filtered')
    plt.title(p3Label)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()
    plt.show()
       
#
#%% SELF-RUN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    print(f"\"{module_name}\" module begins.")
    main()
#TEST Code
