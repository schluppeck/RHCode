import glob
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
from scipy import stats

# ds 2024-10-18
# python documentation in proper style

# Get participant details and parameters
PartInitials = "RH"
FilePrefix = "TW_" + PartInitials + "*.csv"

# add GLOB code for next step, but for starters with n=1 files

DataLocation = f"../Data/{PartInitials}Data/"
SearchTxt = DataLocation + FilePrefix

# Create list of all file names
AllFileNames =  []
for File in glob.glob(SearchTxt):
    AllFileNames.append(File)

df = pd.read_csv(AllFileNames[0])
df = df.reset_index()  # make sure indexes pair with number of rows

# there must be a vectorised version of this
# for now loop through and check against string and replacei with simplified
for index, _ in df.iterrows():
  df.loc[index,"contrast"] = round(df.loc[index,"contrast"], ndigits = 3)
  if "right in df.loc[index, "direction"]:
    df.loc[index,"direction"] = "right"
  else:
    df.loc[index,"direction"] = "left"


df.groupby(['direction','contrast']).mean()
