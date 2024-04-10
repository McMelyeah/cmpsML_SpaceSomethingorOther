#%% MODULE BEGINS
module_name = 'PA3'
'''
Version: <***>
Description:
<***>
Authors: Brooks Schafer, Melinda McElveen
Date Created : <***>
Date Last Updated: <***>
Doc:
<***>
Notes:
<***>
'''
#%% IMPORTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    import os
    # os.chdir(r"C:\Users\brook\OneDrive\Documents\GitHub\cmpsML_SpaceSomethingorOther\CODE")
    os.chdir(r"C:\Users\melof\OneDrive\Documents\GitHub\cmpsML_SpaceSomethingorOther\CODE")

#custom imports
#other imports
from copy import deepcopy as dpcpy

from matplotlib import pyplot as plt
import scipy.signal as signal
import numpy as np
import pandas as pd
import seaborn as sns
import pickle as pckl
from scipy.stats import kurtosis, skew
#
#%% USER INTERFACE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
chLabel = input("Enter a channel label(ex, M1):").upper()
l = input("Enter number of windows per stream(ex, 100):")
#
#%% CONSTANTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% CONFIGURATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% INITIALIZATIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
windowStats = pd.DataFrame(columns=['sb', 'se', 'streamID', 'windowID', 'mean', 'std', 'kur', 'skew'])
features = pd.DataFrame(columns=['sb', 'se', 'streamID', 'f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'classLabel'])
#%% DECLARATIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Global declarations Start Here
#Class definitions Start Here
#Function definitions Start Here
def getIndex(chLabel):
    pathSoIRoot = 'INPUT\\DataSmall\\sb1\\se1'
    pathSoi = f'{pathSoIRoot}\\'
    soi_file = '1_1_bk_pic.pckl'
    #Load SoI objectM1
    with open(f'{pathSoi}{soi_file}', 'rb') as fp:
        soi = pckl.load(fp)

    #Finding channel index
    for i in range(len(soi['info']['eeg_info']['channels'])):
        if soi['info']['eeg_info']['channels'][i]['label'][0] == chLabel:
            chIndex = i
            print("Index found!\n")

    return chIndex
    
def applyFilters(stream, sampleFreq):
    #Apply notch filter
    notch_freq = [60, 120, 180, 240]
    for freq in notch_freq:
        b_notch, a_notch = signal.iirnotch(w0=freq, Q=50, fs=sampleFreq)
        stream = signal.filtfilt(b_notch, a_notch, stream)
    
    #Apply impedance filter
    impedance = [124, 126]
    b_imp, a_imp = signal.butter(N=4, Wn=[impedance[0] / (sampleFreq/2), impedance[1] / (sampleFreq / 2)], btype='bandstop')
    stream = signal.filtfilt(b_imp, a_imp, stream)
    
    #Apply bandpass filter
    bandpass = [0.5, 32]
    b_bandpass, a_bandpass = signal.butter(N=4, Wn=[bandpass[0] / (sampleFreq/2), bandpass[1] / (sampleFreq/2)], btype='bandpass')
    stream = signal.filtfilt(b_bandpass, a_bandpass, stream)

    return stream

def getWindowStats(stream, stream_id, window_size=1000, overlap=0.5):
    print("Length of stream:", len(stream))
    print("Stream:", stream[:10])
    features = []
    num_samples = len(stream)
    window_size = min(window_size, num_samples)
    step_size = int(window_size * (1 - overlap))
    if num_samples < window_size:
        return pd.DataFrame()
    for start in range(0, num_samples - window_size +1, step_size):
        end = start + window_size
        window = stream[start:end]
        mean = np.mean(window)
        std = np.std(window)
        kur = kurtosis(window)
        skewedness = skew(window)
        features.append({
            'sb': start,
            'se': end,
            'streamID': stream_id,
            'windowID': len(features) +1,
            'mean': mean,
            'std': std,
            'kur': kur,
            'skew': skewedness
        })
    features_df = pd.DataFrame(features)
    return features_df
    
    

#
#%% MAIN CODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main code start here
def main():
    chIndex = getIndex(chLabel)
    
    for sb in ['sb1', 'sb2']:
        for se in ['se1', 'se2']:
            pathSoIRoot = 'INPUT\\DataSmall\\' + sb + '\\' + se
            files = os.listdir(pathSoIRoot)
            for file in files:
                pathSoi = f'{pathSoIRoot}\\'
                soi_file = file
                #Load SoI object
                with open(f'{pathSoi}{soi_file}', 'rb') as fp:
                    soi = pckl.load(fp)
                
                #Do stuff with stream
                filteredStream = applyFilters(soi['series'][chIndex], soi['info']['eeg_info']['effective_srate'])
                filteredStream -= np.mean(filteredStream)
                window_stats = getWindowStats(filteredStream, soi_file)
                
                print(window_stats.head())
                
                plt.figure(figsize=(8,6))
                plt.scatter(window_stats['mean'], window_stats['std'], alpha=0.5)
                plt.xlabel('Mean')
                plt.ylabel('Standard Deviation')
                plt.title('Scatterplot of Mean vs. Standard Deviation')
                plt.grid(True)
                plt.show()


#             
#%% SELF-RUN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    print(f"\"{module_name}\" module begins.")
    main()
#