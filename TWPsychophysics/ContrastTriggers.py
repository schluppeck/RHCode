#Import modules
from psychopy import *
#import numpy as num
from scipy import *
import time, copy 
from datetime import datetime
from numpy.random import shuffle

#--------------------------------------
#              Initialisation
#--------------------------------------

#Experiment params
NumTrials = 1

# Get Date and start time
now = datetime.now()
Date = now.strftime('%d%m%y_%H%M')

#present a dialogue box for changing params
params = {'Observer':''}
paramsDlg2 = gui.DlgFromDict(params, title='Travelling Waves Basic', fixed=['date'])
 
ConditionList = [0.9, 0.75, 0.6] 
 
Exp = data.TrialHandler(ConditionList,NumTrials, method='random', dataTypes=None, extraInfo=None,seed=None,originPath=None)


#Setup window
resX=1680
resY=1050
winL = visual.Window(size=(resX,resY), monitor='testMonitor', units='pix', bitsMode=None, fullscr=True, allowGUI=False, color=0.0,screen=0)
winR = visual.Window(size=(resX,resY), monitor='testMonitor', units='pix', bitsMode=None, fullscr=True, allowGUI=False, color=0.0,screen=1)

# Are these needed???
winL.setGamma([1.688,1.688,1.688])
winR.setGamma([1.841,1.841,1.841])

# Clock
Clock = core.Clock()

#--------------------------------------
#            Create stimlui 
#--------------------------------------

#Create stimlui initial stimuli
RadL = visual.RadialStim(winL,size=resY-350,angularCycles=25, color=-1,angularRes=35,units="pix", radialCycles = 0, contrast = 0.7)
maskL = visual.RadialStim(winL,color=[0,0,0],size=(resY-350)*0.745,angularCycles=25,angularRes=35,units="pix")
ConcR = visual.RadialStim(winR,size=resY-350,angularCycles=0, color=-1,angularRes=35,units="pix", radialCycles= 8, ori=300, contrast = 0.3)
maskR = visual.RadialStim(winR,color=[0,0,0],size=(resY-350)*0.76,angularCycles=25,angularRes=35,units="pix")

# End location

EndLocL2 = visual.PatchStim(winL, tex='None', units='pix', pos=[-140,-420], size=(7,60), color=[-1,-1,-1], ori=200)
EndLocL1 = visual.PatchStim(winL, tex='None', units='pix', pos=[0,-440], size=(7,60), color=[-1,-1,-1], ori=0)
EndLocR2 = visual.PatchStim(winR, tex='None', units='pix', pos=[-140,-420], size=(7,60), color=[-1,-1,-1], ori=200)
EndLocR1 = visual.PatchStim(winR, tex='None', units='pix', pos=[0,-440], size=(7,60), color=[-1,-1,-1], ori=0)

#Create fixation and fusion lock
#Dot
fixationL = visual.PatchStim(winL, texRes=512, tex='None', mask="circle",units='pix',rgb=-1, pos=[0,0], size=(20.0,20.0))
fixationR = visual.PatchStim(winR, texRes=512, tex='None', mask="circle",units='pix',rgb=-1, pos=[0,0], size=(20.0,20.0))

#Spokes
# to the right window    
BarTop = visual.PatchStim(winR, tex='none', units='pix', rgb=1.0, pos=(0,85),ori=0, size=(15,100))
BarLeft = visual.PatchStim(winR, tex='none', units='pix', rgb=1.0, pos=(-85,0),ori=0, size=(100,15))

# to the left window    
BarBottom = visual.PatchStim(winL, tex='none', units='pix', rgb=-1.0, pos=(-0,-85),ori=0, size=(15,100))
BarRight = visual.PatchStim(winL, tex='none', units='pix', rgb=-1.0, pos=(85,0),ori=0, size=(100,15))

#Fusion lock
LockSize=512
array = zeros([LockSize,LockSize])
for n in range(0, LockSize+1, 32):
    array[n:16+n,0:16]=1
    array[n-16:16+n-16,0:16]=-1
    array[n:16+n,LockSize-16:LockSize]=-1
    array[n-16:16+n-16,LockSize-16:LockSize]=1
    
    array[0:16,n:16+n]=1
    array[0:16, n-16:16+n-16]=-1
    array[LockSize-16:LockSize,n:16+n]=-1
    array[LockSize-16:LockSize, n-16:16+n-16]=1

fusionL = visual.PatchStim(winL, tex=array, 
    size=(1050,1050), units='pix',
    interpolate=False,
    autoLog=True)
fusionR = visual.PatchStim(winR, tex=array, 
    size=(1050,1050), units='pix',
    interpolate=False,
    autoLog=True) 

Fixation = [fixationL, fixationR, BarBottom, BarTop, BarLeft, BarRight, EndLocL1, EndLocL2, EndLocR1, EndLocR2]

# Break stimuli

BreakStimL = visual.RadialStim(winL,size=resY-350,angularCycles=0, color=1,angularRes=35,units="pix", radialCycles = 0, contrast = 1)
BreakStimR = visual.RadialStim(winR,size=resY-350,angularCycles=0, color=1,angularRes=35,units="pix", radialCycles = 0, contrast = 1)

#--------------------------------------
#                 Messages
#--------------------------------------

# Inscructions = 

# Check stimuli is fusing
FixationMsgL = visual.TextStim(winL, 'Please ensure stimuli is fusing. \nPress the Right arrow when the wave reaches the desitination point\nPress the Left arrow if the wave is not triggered\nPress any button to begin', pos=(0,250),  
    flipHoriz=True, height=40, wrapWidth=1000)
FixationMsgR = visual.TextStim(winR, 'Please ensure stimuli is fusing. \nPress the Right arrow when the wave reaches the desitination point\nPress the Left arrow if the wave is not triggered\nPress any button to begin', pos=(0,250),  
    flipHoriz=True, height=40, wrapWidth=1000)

#PauseMsgL = visual.TextStim(winL, 'Break in experiment \nPress any button to continue',  flipHoriz=True, height=40, wrapWidth=1000)
#PauseMsgR = visual.TextStim(winR, 'Break in experiment \nPress any button to continue',  flipHoriz=True, height=40, wrapWidth=1000)

EndMsgL = visual.TextStim(winL, 'Experiment Over \nThanks for Participating!',  flipHoriz=True, height=40, wrapWidth=1000)
EndMsgR = visual.TextStim(winR, 'Experiment Over \nThanks for Participating!',  flipHoriz=True, height=40, wrapWidth=1000)

#--------------------------------------
#           Presentation Loop
#--------------------------------------

#Check fusion
fusionL.draw()
fusionR.draw()
for x in Fixation:
    x.draw()
FixationMsgL.draw()
FixationMsgR.draw()
winL.flip()
winR.flip()
event.waitKeys()

BreakNum = 0


for y in Exp:
# Create trigger with new contrast
    triggerR = visual.RadialStim(winR,size=resY-350,angularCycles=0, color=-1,angularRes=35,units="pix", radialCycles = 8, 
        visibleWedge = (330,360), contrast = y)

    fusionL.draw()
    fusionR.draw()
    for x in Fixation:
        x.draw()
    winL.flip()
    winR.flip()
    time.sleep(2)

#    Flash suppression
    fusionL.draw()
    fusionR.draw()
    ConcR.draw()
    maskR.draw()
    for x in Fixation:
        x.draw()

    winL.flip()
    winR.flip()
    time.sleep(1)

#   Present stimuli
    fusionL.draw()
    fusionR.draw()
    RadL.draw()
    maskL.draw()
    ConcR.draw()
    maskR.draw()
    for x in Fixation:
        x.draw()
    
    winL.flip()
    winR.flip()
    time.sleep(0.5)

#   Present trigger
    fusionL.draw()
    fusionR.draw()
    RadL.draw()
    maskL.draw()
    ConcR.draw()
    triggerR.draw()
    maskR.draw()
    
    for x in Fixation:
        x.draw()

    Clock.reset() # Start timing
    
    winL.flip()
    winR.flip()
    time.sleep(0.25)

# Remove Trigger and wait for response
    fusionL.draw()
    fusionR.draw()
    RadL.draw()
    maskL.draw()
    ConcR.draw()
    maskR.draw()

    for x in Fixation:
        x.draw()
    winL.flip()
    winR.flip()

    Keys = event.waitKeys()
# Add data from trial
    RespTime = Clock.getTime()
    Exp.addData('TravelTime', RespTime)
    Exp.addData('ContrastLevel', y)
    Exp.addData('KeyPressed', Keys)
#    Clear Screen
    for x in Fixation:
        x.draw()
    winL.flip()
    winR.flip()
    BreakNum +=1
    if BreakNum == -1:
        fusionL.draw()
        fusionR.draw()
        BreakStimL.draw()
        BreakStimR.draw()
        maskL.draw()
        maskR.draw()
        for x in Fixation:
            x.draw()
        winL.flip()
        winR.flip()
        BreakNum = 0
        time.sleep(5)

#--------------------------------------
#              End/Cleanup
#--------------------------------------

#Display end msg
EndMsgL.draw()
EndMsgR.draw()
winL.flip()
winR.flip()
event.waitKeys()

#Output reaction times in csv

FileName = './data/TW_' + params['Observer'] +  '_' + Date
Exp.saveAsExcel(FileName)
#Close screen
winL.close()
winR.close()
