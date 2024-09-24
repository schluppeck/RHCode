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


# Seperate each real column to seperate series
ContLevel = pd.Series()
ButtonPress = pd.Series()
ReactionTime = pd.Series()
for x in SepSeries:
    ContLevel = pd.concat([ContLevel, x.iloc[1:18]])
    ButtonPress = pd.concat([ButtonPress, x.iloc[18:35]])
    ReactionTime = pd.concat([ReactionTime, x.iloc[35:52]])


# Create list of reactiontimes that were true 
ReactionTimeTrue = []
ContLevelsTrue = []
for x in range(ReactionTime.size):
    if ButtonPress.iloc[x] == "['right']":
        ReactionTimeTrue.append(ReactionTime.iloc[x])
        ContLevelsTrue.append(round(ContLevel.iloc[x], 1))
        
# Split out reaction times based on condition
TW9 = []
TW75 = []
TW6 = []
for x in range(len(ReactionTimeTrue)):
    if ContLevelsTrue[x] == 0.9:
        TW9.append(ReactionTimeTrue[x])
    if ContLevelsTrue[x] == 0.8:
        TW75.append(ReactionTimeTrue[x])
    if ContLevelsTrue[x] == 0.6:
        TW6.append(ReactionTimeTrue[x])


# %%
# Reaction time anaysis

RTPrefix = "RT_" + PartInitials + "*"

DataLocation = '/home/rowanhuxley/Documents/Data_Various/BinRiv/PsychophysicsGeneral/data/'
RTSearchTxt = DataLocation + RTPrefix


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
RTMean = float(np.nanmean(RTAll))

# Subtract RT from all TW values
TW9 = [x - RTMean for x in TW9]
TW75 = [x - RTMean for x in TW75]
TW6 = [x - RTMean for x in TW6]   
    
AllDataTrue = [TW9, TW75, TW6]

# %%
# Plotting and ANOVA


fig, ax = plt.subplots(2)

ax[0].boxplot(AllDataTrue, tick_labels=Conditions)
ax[0].set_xlabel('Contrast level')
ax[0].set_ylabel('Reaction time (s)')

#TW9.sort()
counts, bins = np.histogram(TW75)
#plt.stairs(counts, bins)

Time = np.linspace(1, 158, 158)
#ax[1].stairs(counts, bins)

z =stats.probplot(TW6, plot=plt)



# Test for normality
print(stats.shapiro(TW9))
#print(stats.shapiro(TW75))
print(stats.shapiro(TW6))

print(stats.ttest_rel(TW9, TW75))
print(stats.ttest_rel(TW75, TW6))
print(stats.ttest_rel(TW6, TW9))
# F, p = f_oneway(TW9, TW75, TW6)
# print(F)
# print(p)