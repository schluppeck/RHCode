# A temp file where I move the data into a tidy format
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math as math
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Get participant details and parameters
PartInitials = "RH"
Conditions = [0.9, 0.75, 0.6]

FilePrefix = "TW_" + PartInitials + "*"

DataLocation = '/home/rowanhuxley/Documents/Data_Various/BinRiv/PsychophysicsGeneral/data/'
SearchTxt = DataLocation + FilePrefix

# %% Travel time analysis

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
ContLevel = pd.Series(name = "Contrast Level")
ButtonPress = pd.Series(name = "Button_Pressed")
ResponseTime = pd.Series(name = "Response_Time")
for x in SepSeries:
    ContLevel = pd.concat([ContLevel, x.iloc[1:18]])
    ButtonPress = pd.concat([ButtonPress, x.iloc[18:35]])
    ResponseTime = pd.concat([ResponseTime, x.iloc[35:52]])

## Create dataframe with each row being a trial
# Issues with duplicate labels so have to change series to np array
ContLevel = ContLevel.to_numpy()
ButtonPress = ButtonPress.to_numpy()
ResponseTime = ResponseTime.to_numpy()

# Create 1d Numpy array of participant intials
PartInits = np.full(len(ContLevel), PartInitials)

# Recombine data into tidy format
ColumnLabels = ["Participant_Id", "Contrast_Level", "Button_Pressed", "Response_Time"]
AllData = pd.DataFrame([PartInits, ContLevel, ButtonPress, ResponseTime], index = ColumnLabels)
AllData = AllData.transpose()


# Create list of response times that were true at each contrast level
TW9 = []
TW75 = []
TW6 = []




ReactionTimeTrue = []
ContLevelsTrue = []
for x in range(ResponseTime.size):
    if ButtonPress.iloc[x] == "['right']":
        ReactionTimeTrue.append(ResponseTime.iloc[x])
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
RTSE = stats.sem(RTAll)

# Calculating propagated error
TW9SE = stats.sem(TW9)
TW75SE = stats.sem(TW75)
TW6SE = stats.sem(TW6)

TW9SEP = math.sqrt((TW9SE ** 2) + (RTSE ** 2))
TW75SEP = math.sqrt((TW75SE ** 2) + (RTSE ** 2))
TW6SEP = math.sqrt((TW6SE ** 2) + (RTSE ** 2))

# Subtract RT from all TW values
TW9 = [x - RTMean for x in TW9]
TW75 = [x - RTMean for x in TW75]
TW6 = [x - RTMean for x in TW6]   

AllDataTrue = [TW9, TW75, TW6]


print(np.std(RTAll))

# %%

# Descriptives
print(np.mean(TW9))
print(np.std(TW9))
print(np.mean(TW75))
print(np.std(TW75))
print(np.mean(TW6))
print(np.std(TW6))
# Plotting

Time = np.linspace(1, 158, 158)


#%% Chi squared

# Count number of trigger waves (.count isnt working so had to do this...)
ButtonPressTrue=[]
for x in range(ButtonPress.size):
    if ButtonPress.iloc[x] == "['right']":
        ButtonPressTrue.append(1)
    else:
        ButtonPressTrue.append(0)


ContListFull = ContLevel.to_list()

BP9 = []
BP75 = []
BP6 = []
#split into conditions
for x in range(len(ButtonPressTrue)):
    if ContListFull[x] == 0.899999976158:
        BP9.append(ButtonPressTrue[x])
    if ContListFull[x] == 0.75:
        BP75.append(ButtonPressTrue[x])
    if ContListFull[x] == 0.600000023842:
        BP6.append(ButtonPressTrue[x])

BP9Sum = sum(BP9)
BP75Sum = sum(BP75)
BP6Sum = sum(BP6)

ButtonTrueArray = np.array((BP9Sum,BP75Sum,BP6Sum))


print('ChiSquared')
print(stats.chisquare(ButtonTrueArray))
# %%

#HISTOGRAM
fig, ax = plt.subplots(1, 3, figsize=(15,1))

fig.tight_layout()
fig.set_figheight(8)

counts, bins = np.histogram(TW6, bins=20)
ax[0].stairs(counts, bins, baseline=1)
ax[0].set_xlabel('Response time (s)', fontsize=12)
ax[0].set_ylabel('Frequency', fontsize=12)
ax[0].set_title('0.6', fontsize=12)

counts, bins = np.histogram(TW75, bins=20)
ax[1].stairs(counts, bins, baseline=1)
ax[1].set_xlabel('Response time (s)', fontsize=12)
ax[1].set_title('0.75', fontsize=12)

counts, bins = np.histogram(TW9, bins=20)
ax[2].stairs(counts, bins, baseline=1)
ax[2].set_xlabel('Response time (s)', fontsize=12)
ax[2].set_title('0.9', fontsize=12)
ax[2].text(-5.3, 26, 'A', fontsize=16, fontweight='bold', va='top', ha='right')
#ax[1] =stats.probplot(TW6, plot=plt)

# Q-Q PLOT AUTOMATIC
fig, ax = plt.subplots(1, 3, figsize=(15,1))

fig.tight_layout()
fig.set_figheight(8)

stats.probplot(TW9, plot=ax[0])
ax[0].set_title('0.9%', fontsize=12)
ax[0].tick_params(axis='x', labelsize=10)
ax[0].set_xlabel('Theoretical quantiles', fontsize=12)
ax[0].set_ylabel('Ordered values', fontsize=12)
ax[0].set_ylim(0,2.5)

stats.probplot(TW75, plot=ax[1])
ax[1].set_title('0.75%', fontsize=12)
ax[1].tick_params(axis='x', labelsize=10)
ax[1].set_xlabel('Theoretical quantiles', fontsize=12)
ax[1].set_ylabel('', fontsize=12)
ax[1].set_ylim(0,2.5)

stats.probplot(TW6, plot=ax[2])
ax[2].set_title('0.6%', fontsize=12)
ax[2].tick_params(axis='x', labelsize=10)
ax[2].set_xlabel('Theoretical quantiles', fontsize=12)
ax[2].set_ylabel('', fontsize=12)
ax[2].set_ylim(0,2.5)

fig.text(0, 0.83, 'B', fontsize=16, fontweight='bold', va='top', ha='right')


# Q-Q PLOT MANUAL
f, a = plt.subplots(1,3, figsize=(15,1))
f.tight_layout()
f.set_figheight(8)

v=np.linspace(-2,2,num=50)
n=np.linspace(0,2,num=50)

# 0.6
b = stats.probplot(TW6)
z = b[0]
x = z[0]
c = z[1]

a[0].plot(v,n, linewidth=3,color='tab:orange') # Linear
a[0].scatter(x,c) #q-q
a[0].set_title('0.6', fontsize=12)
a[0].set_xlabel('Expected values', fontsize=12)

# 0.75
b = stats.probplot(TW75)
z = b[0]
x = z[0]
c = z[1]

a[1].plot(v,n, linewidth=3,color='tab:orange') # Linear
a[1].scatter(x,c) #q-q
a[1].set_title('0.75', fontsize=12)
a[1].set_xlabel('Expected values', fontsize=12)

# 0.9
b = stats.probplot(TW9)
z = b[0]
x = z[0]
c = z[1]

a[2].plot(v,n, linewidth=3,color='tab:orange') # Linear
a[2].scatter(x,c) #q-q
a[2].set_title('0.9', fontsize=12)
a[2].set_xlabel('Expected values', fontsize=12)

a[0].set_ylabel('Observed values', fontsize=12)

f.text(0, 0.83, 'B', fontsize=15, fontweight='bold', va='top', ha='right')



# Test for normality
print(stats.shapiro(TW9))
print(stats.shapiro(TW75))
print(stats.shapiro(RTAll))



#BOX PLOT
fig2, ax2 = plt.subplots(1)
ax2.boxplot(AllDataTrue,  meanline=True)
ax2.set_xticklabels(Conditions)
ax2.set_xlabel('Contrast level')
ax2.set_ylabel('Wave travel time (s)')





# ANOVA ----- NEED TO INSTALL STATSMODELS MODULE
# Create data frame of data
#AllCond = pd.DataFrame(TW6,TW75,TW9)
#print(AnovaRM(AllCond))

# exporting data to run repeated measures ANOVA as running it here is a nightmare with current setup
import csv
with open('Data.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for x in TW6:
        spamwriter.writerow([x])     
    for x in TW75:
        spamwriter.writerow([x])
    for x in TW9:
        spamwriter.writerow([x])     
                            
    
