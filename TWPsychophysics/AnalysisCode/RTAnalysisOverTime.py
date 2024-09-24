# An analysis script that calculates the wave speed and trigger rates over time
# To be used to analyse data from "Experiment name"
# Pauses happened every 5 presentations
# 
# Created 17/09/2024 RJH
# 
# To do-
# Create list with each block in linear order for each contrast level
# Currently very messy need to clean up...




import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats


# Get participant details and parameters
PartInitials = "RH"
Conditions = [0.9, 0.75, 0.6]

FilePrefix = "TW_" + PartInitials + "*"
DataLocation = '/home/rowanhuxley/Documents/Data_Various/BinRiv/PsychophysicsGeneral/data/'
SearchTxt = DataLocation + FilePrefix

# %%
# Travel time analysis

# Create list of all file names
AllFileNames =  []
for File in glob.glob(SearchTxt):
    AllFileNames.append(File)
    
# Create list of all data rows
SepSeries = []

for File in AllFileNames:
    df = pd.read_excel(File)
    df = df.reset_index()  # make sure indexes pair with number of rows
    for index, row in df.iterrows():
        row.pop("index")
        row.pop("ContrastLevel_mean")
        row.pop("ContrastLevel_std")
        TravelTimeMean = row.pop("TravelTime_mean")
        TravelTimeSd = row.pop("TravelTime_std")
        SepSeries.append(row)


# Create lists of each block
CLGrouped = []
BPGrouped = []
RTGrouped = []
for x in SepSeries:
    # Contrast Level
    CLSeries = x.iloc[1:18]
    CLList = CLSeries.to_list()
    CLGrouped.append(CLList)
    # Button Press
    BPSeries = x.iloc[18:35]
    BPList = BPSeries.to_list()
    BPGrouped.append(BPList)
    # Response Time
    RTSeries = x.iloc[35:52]
    RTList = RTSeries.to_list()
    RTGrouped.append(RTList)

# Remove trials where no wave was triggered
for x in range(len(RTGrouped)):
    for y in range(len(RTGrouped[x])):
        if BPGrouped[x][y] == "['right']":
            RTGrouped[x][y] = np.nan


# Convert all of the lists to arrays
RTGroupedArray = np.array(RTGrouped)

# Remove very clear outliers
Outliers = RTGroupedArray[RTGroupedArray > 5]
print('Outliers Removed=')
print(Outliers)
RTGroupedArray[RTGroupedArray > 3] = np.nan

# Calculate mean and stdev for each trial
AllDataAv = np.nanmean(RTGroupedArray, axis=0)
AllDataStd = np.nanstd(RTGroupedArray, axis=0)


# %%
# Reaction time anaysis

RTFilePrefix = "RT_" + PartInitials + "*"

RTSearchTxt = DataLocation + RTFilePrefix


# Create list of all file names
RTAllFileNames =  []
for File in glob.glob(RTSearchTxt):
    RTAllFileNames.append(File)

# Create list of all reaction times
RTAll = []
for File in RTAllFileNames:
    df = pd.read_excel(File)
    df = df.reset_index()  # make sure indexes pair with number of rows
    for index, row in df.iterrows():
        row.pop("index")
        row.pop("n")
        row.pop("RT_mean")
        row.pop("RT_std")
        for x in range(row.size):
            RTAll.append(float(row.iloc[x]))

# Get mean and std 
RTMean = float(np.mean(RTAll))

# # Subtract RT from all TW values
# TW9 = [x - RTMean for x in TW9]
# TW75 = [x - RTMean for x in TW75]
# TW6 = [x - RTMean for x in TW6]   
    
# AllDataTrue = [TW9, TW75, TW6]
AllDataAvRT = [x - RTMean for x in AllDataAv]

# %%
# Trigger rates over time

# encode lefts and rights as 1s and 0s
TmpList=[]
BPBinarised = []
for x in BPGrouped:
    for y in x:
        if y == "['right']":
            TmpList.append(1)
        elif y == "['left']":
            TmpList.append(0)
    BPBinarised.append(TmpList)
    TmpList = []
    
# Convert all of the lists to arrays
BPArray = np.array(BPBinarised)

# calculate % trigger at each trial number
BPCount = np.count_nonzero(BPArray, axis=0)

# Make initatied wave # into percentage
for x in range(len(BPCount)):
    BPCount[x] = BPCount[x]/45*100


# %%
# Plotting

fig, ax = plt.subplots(3)

fig.tight_layout()
fig.set_figheight(8)

Time = np.linspace(1, 17, 17)


# Plot wave speed over time

#ax.plot(Time, AllDataAvRT)
ax[0].errorbar(Time, AllDataAvRT, AllDataStd, linestyle='None', marker='.')
#ax[0].xticks(Time)
#ax[0].set_xlabel('Trial Number')
ax[0].set_ylabel('Wave travel time (s)')


# Fit and plot linear regression to wave speed over time

slope, intercept, r, p, std_err = stats.linregress(Time, AllDataAvRT)

def FitLinReg(x):
  return slope * x + intercept

mymodel = list(map(FitLinReg, Time))

ax[1].scatter(Time, AllDataAvRT)
ax[1].plot(Time, mymodel, color='orange')
ax[1].set_ylabel('Wave travel time (s)')
ax[1].text(-.95, 1.92, 'A', fontsize=16, fontweight='bold', va='top', ha='right')

# Plot trigger rates over time

ax[2].scatter(Time, BPCount, color='C0')
ax[2].set_xlabel('Presentation number')
ax[2].set_ylabel('Trigger rate (%)')
ax[2].set_ylim(0,100)
ax[2].text(-.95, 120, 'B', fontsize=16, fontweight='bold', va='top', ha='right')

#ax[2].set_title("Trigger rates at each trial", fontsize=13)

slope, intercept, r, p, std_err = stats.linregress(Time, BPCount)

mymodel = list(map(FitLinReg, Time))


ax[2].plot(Time, mymodel, color='orange')

