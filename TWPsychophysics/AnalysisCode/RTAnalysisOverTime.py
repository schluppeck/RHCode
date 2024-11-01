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
from scipy import signal
from sklearn.metrics import mean_squared_error

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
        if BPGrouped[x][y] == "['left']":
            RTGrouped[x][y] = np.nan


# Convert all of the lists to arrays
RTGroupedArray = np.array(RTGrouped)

# Remove 3sd outliers
Outliers = RTGroupedArray[RTGroupedArray > 3.58]
print('Outliers Removed=')
print(Outliers)
RTGroupedArray[RTGroupedArray > 3.58] = np.nan

# Split runs by conditions
TW9 = np.array(RTGrouped[0::3])
TW75 = np.array(RTGrouped[1::3])
TW6 = np.array(RTGrouped[2::3])

# Seeing if the weird trends are due to noise
#TW9 = TW9[0:7:]
#TW75 = TW75[0:7:]
#TW6 = TW6[0:7:]

# Average across trials
TW9AV = np.nanmean(TW9, axis=0)
TW75AV = np.nanmean(TW75, axis=0)
TW6AV = np.nanmean(TW6, axis=0)

# Calculate mean and stdev for each trial
AllDataAv = np.nanmean(RTGroupedArray, axis=0)
AllDataStd = np.nanstd(RTGroupedArray, axis=0)

AAMean =np.nanmean(AllDataAv, axis=0)
AASD =np.nanmean(AllDataStd, axis=0)
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
TW9AV = [x - RTMean for x in TW9AV]
TW75AV = [x - RTMean for x in TW75AV]
TW6AV = [x - RTMean for x in TW6AV]   
    

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

BP9 =BPArray[0::3]
BP75 =BPArray[1::3]
BP6 =BPArray[2::3]

BP9 = np.count_nonzero(BP9, axis=0)
BP75 = np.count_nonzero(BP75, axis=0)
BP6 = np.count_nonzero(BP6, axis=0)

# calculate % trigger at each trial number
BPCount = np.count_nonzero(BPArray, axis=0)

# Make initatied wave # into percentage
for x in range(len(BPCount)):
    BPCount[x] = BPCount[x]/45*100


for x in range(len(BP9)):
    BP9[x] = BP9[x]/15*100

for x in range(len(BP75)):
    BP75[x] = BP75[x]/15*100

for x in range(len(BP6)):
    BP6[x] = BP6[x]/15*100

# %%
# PLOTTING

fig, ax = plt.subplots(2)

fig.tight_layout()
fig.set_figheight(6)

Time = np.linspace(1, 17, 17)

# WAVE SPEED

# Create standard error 
RTGroupTrans = RTGroupedArray.transpose()

Errors = []
Tmp = []
for x in RTGroupTrans:
    for y in x:
        if y < 100:
            Tmp.append(y)
    CurStdErr = np.std(Tmp)
    Errors.append(CurStdErr)
    Tmp = []
# Make appropreate for dot size
ErrorsDots = [i * 1.7  for i in Errors]
ErrorsDotsSquared = np.power(ErrorsDots,4)

Base = np.linspace(10,10,len(Errors))
DotSizes = []
for x in range(len(Errors)):
    n = Base[x]/ErrorsDotsSquared[x]
    DotSizes.append(n)
# Fit and plot linear regression to wave speed over time

slope, intercept, r, p, std_err = stats.linregress(Time, AllDataAvRT)

def FitLinReg(x):
  return slope * x + intercept

mymodel = list(map(FitLinReg, Time))

ax[0].scatter(Time, AllDataAvRT, DotSizes)
ax[0].plot(Time, mymodel, color='orange')
ax[0].set_ylabel('Wave travel time (s)')
ax[0].set_ylim(0.9, 1.3)
ax[0].text(-.95, 1.36, 'A', fontsize=16, fontweight='bold', va='top', ha='right')


# WAVE SPEED

ax[1].scatter(Time, BPCount, color='C0')
ax[1].set_xlabel('Presentation number')
ax[1].set_ylabel('Trigger rate (%)')
ax[1].set_ylim(0,100)
ax[1].text(-.95, 114, 'B', fontsize=16, fontweight='bold', va='top', ha='right')

#Calculate regression and plot it

slope, intercept, r, p, std_err = stats.linregress(Time, BPCount)
mymodel = list(map(FitLinReg, Time))

ax[1].plot(Time, mymodel, color='orange')



# EACH CONDITION


f, a = plt.subplots(2,3, figsize=(15,1))
f.tight_layout()
f.set_figheight(4)
a[1,0].text(-1.4, 230, 'A', fontsize=16, fontweight='bold', va='top', ha='right')
f.subplots_adjust(bottom=-0.1, right=1, top=1) # for some reason that right adjust fixes xticks
# 0.9

# Create standard error 
TW9Trans = TW9.transpose()

Errors = []
Tmp = []
for x in TW9Trans:
    for y in x:
        if y < 100:
            Tmp.append(y)
    CurStdErr = np.std(Tmp)
    Errors.append(CurStdErr)
    Tmp = []
# Make appropreate for dot size
ErrorsDots = [i * 1  for i in Errors]
ErrorsDotsSquared = np.power(ErrorsDots,1)

Base = np.linspace(10,10,len(Errors))
DotSizes = []
for x in range(len(Errors)):
    n = Base[x]/ErrorsDotsSquared[x]
    DotSizes.append(n)


# Fit and plot linear regression to wave speed over time

slope, intercept, r, p, std_err = stats.linregress(Time, TW9AV)
print('slope, intercept, r, p, std_err')
print('TW9 regression fit')
print(slope, intercept, r, p, std_err)


mymodel = list(map(FitLinReg, Time))

a[0,0].scatter(Time, TW9AV, DotSizes)
a[0,0].plot(Time, mymodel, color='orange')
a[0,0].set_ylabel('Wave travel time (s)')
a[0,0].set_ylim(0.8, 1.55)
a[0,0].set_title('0.9')



# 0.75

# Create standard error 
TW75Trans = TW75.transpose()

Errors = []
Tmp = []
for x in TW75Trans:
    for y in x:
        if y < 100:
            Tmp.append(y)
    CurStdErr = np.std(Tmp)
    Errors.append(CurStdErr)
    Tmp = []
# Make appropreate for dot size
ErrorsDots = [i * 1 for i in Errors]
ErrorsDotsSquared = np.power(ErrorsDots,1.5)

Base = np.linspace(10,10,len(Errors))
DotSizes = []
for x in range(len(Errors)):
    n = Base[x]/ErrorsDotsSquared[x]
    DotSizes.append(n)


# Fit and plot linear regression to wave speed over time

slope, intercept, r, p, std_err = stats.linregress(Time, TW75AV)
print('TW75 regression fit')
print(slope, intercept, r, p, std_err)

mymodel = list(map(FitLinReg, Time))

a[0,1].scatter(Time, TW75AV, DotSizes)
a[0,1].plot(Time, mymodel, color='orange')
a[0,1].set_ylim(0.8, 1.55)
a[0,1].set_title('0.75')



# 0.6

# Create standard error 
TW6Trans = TW6.transpose()

Errors = []
Tmp = []
for x in TW6Trans:
    for y in x:
        if y < 100:
            Tmp.append(y)
    CurStdErr = np.std(Tmp)
    Errors.append(CurStdErr)
    Tmp = []
# Make appropreate for dot size
ErrorsDots = [i * 1  for i in Errors]
ErrorsDotsSquared = np.power(ErrorsDots,1)

Base = np.linspace(10,10,len(Errors))
DotSizes = []
for x in range(len(Errors)):
    n = Base[x]/ErrorsDotsSquared[x]
    DotSizes.append(n)


# Fit and plot linear regression to wave speed over time

slope, intercept, r, p, std_err = stats.linregress(Time, TW6AV)
print('TW6 regression fit')
print(slope, intercept, r, p, std_err)


mymodel = list(map(FitLinReg, Time))

a[0,2].scatter(Time, TW6AV, DotSizes)
a[0,2].plot(Time, mymodel, color='orange')
a[0,2].set_ylim(0.8, 1.55)
a[0,2].set_title('0.6')




# TRIGGER RATE

# 0.9
a[1,0].scatter(Time, BP9, color='C0')
a[1,0].set_xlabel('Presentation number')
a[1,0].set_ylabel('Trigger rate (%)')
a[1,0].set_ylim(0,100)
a[1,0].text(-1.6, 114, 'B', fontsize=16, fontweight='bold', va='top', ha='right')

#Calculate regression and plot it

slope, intercept, r, p, std_err = stats.linregress(Time, BP9)
mymodel = list(map(FitLinReg, Time))
print('BP9 regression fit')
print(slope, intercept, r, p, std_err)

a[1,0].plot(Time, mymodel, color='orange')


# 0.75
a[1,1].scatter(Time, BP75, color='C0')
a[1,1].set_xlabel('Presentation number')

a[1,1].set_ylim(0,100)

#Calculate regression and plot it

slope, intercept, r, p, std_err = stats.linregress(Time, BP75)
mymodel = list(map(FitLinReg, Time))
print('BP75 regression fit')
print(slope, intercept, r, p, std_err)

a[1,1].plot(Time, mymodel, color='orange')

# 0.6
a[1,2].scatter(Time, BP6, color='C0')
a[1,2].set_xlabel('Presentation number')

a[1,2].set_ylim(0,100)

#Calculate regression and plot it

slope, intercept, r, p, std_err = stats.linregress(Time, BP6)
mymodel = list(map(FitLinReg, Time))

print('BP6 regression fit')
print(slope, intercept, r, p, std_err)

a[1,2].plot(Time, mymodel, color='orange')
#a[1,2].set_xticks()


# %% Polynomial regression 


# perform regression
          
def polyregress(xdata,ydata,degree):
  return np.polynomial.polynomial.polyfit(xdata,ydata,degree)

def evaluate(x,coeffs):
  y = 0
  m = 1
  for c in coeffs:
    y += c * m
    m *= x
  return y

# interpolation function

def ntrp(x,xa,xb,ya,yb): return (x-xa) * (yb-ya) / (xb-xa) + ya


# Group TWs into one list to then iterate over
GroupedTW = [TW9AV, TW75AV, TW6AV]

# Create subplots
f2, a2 = plt.subplots(2, 3, figsize=(17,6))
f2.tight_layout()
CycleNum = 0
Names=["0.9", "0.75", "0.6"]

for CurContLevel in GroupedTW:

# Create coefficients for the given contrast level
    
    coeffs = polyregress(Time, CurContLevel, 10)
    if CycleNum == 1:
        coeffs = polyregress(Time, CurContLevel, 12)
    # generate regression curve datax
    xplot = []
    yplot = []
    plotpoints = 17
    xmin = min(Time)
    xmax = max(Time)

    for n in range(plotpoints+1):
        x = ntrp(n,0,plotpoints,xmin,xmax)
        y = evaluate(x,coeffs)
        xplot += [x]
        yplot += [y]

    a2[0, CycleNum].scatter(Time,CurContLevel,s=32)
    a2[0, CycleNum].plot(xplot, yplot, color="orange")
    a2[0, CycleNum].set_ylim(0.65,1.55)
    a2[0, CycleNum].set_title(f"{Names[CycleNum]}")
    CycleNum += 1
    yplot.pop(0)
    print(mean_squared_error(TW9AV, yplot))

a2[0,0].set_ylabel('Mean response time (s)')

## Sawtooth function!

t = np.linspace(1, 17, 17)
Sig = signal.sawtooth(0.05 * np.pi * 8 * t)
Sig = Sig=Sig/4.8
#a2[0,0].plot(t,Sig+1.1, color="orange")




# FITTING TO TRIGGER RATE 

GroupedBP = [BP9, BP75, BP6]
CycleNum = 0
for CurContLevel in GroupedBP:

# Create coefficients for the given contrast level
    
    coeffs = polyregress(Time, CurContLevel,6)
    if CycleNum == 0:
        coeffs = polyregress(Time, CurContLevel, 5)
    # generate regression curve datax
    xplot = []
    yplot = []
    plotpoints = 17
    xmin = min(Time)
    xmax = max(Time)

    for n in range(plotpoints+1):
        x = ntrp(n,0,plotpoints,xmin,xmax)
        y = evaluate(x,coeffs)
        xplot += [x]
        yplot += [y]

    a2[1, CycleNum].scatter(Time,CurContLevel,s=32)
    a2[1, CycleNum].plot(xplot, yplot, color="orange")
    #a2[1, CycleNum].set_ylim(0.65,1.55)
    CycleNum += 1
    yplot.pop(0)
    print(mean_squared_error(BP6, yplot))

a2[1,0].set_ylabel('Trigger rate (%)')





