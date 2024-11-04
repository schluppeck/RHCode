import numpy as np
import pandas as pd

# pip install sdv
import sdv 
from sdv.datasets.local import load_csvs
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

import matplotlib.pyplot as plt

# testing out simulated data
#
#  load CSV
#  https://docs.sdv.dev/sdv/single-table-data/data-preparation/loading-data
#
# denis schluppeck 
# 2024-11-04


# assume that my_folder contains a CSV file named 'guests.csv'
datasets = load_csvs(
    folder_name='./',
    read_csv_parameters={
        'skipinitialspace': True
    })

# the data is available under the file name
mydata = datasets['TW_RH_060824_1536_tidy']

metadata = Metadata.detect_from_dataframe(
    data=mydata,
    table_name='table')

metadata.validate()
metadata.save_to_json("metadata.json")
# Step 1: Create the synthesizer
synthesizer = GaussianCopulaSynthesizer(metadata)

# Step 2: Train the synthesizer
synthesizer.fit(data)

# Step 3: Generate synthetic data
synthetic_data = synthesizer.sample(num_rows=1000)

plt.style.use('seaborn-white')


# plot:
fig, ax = plt.subplots()

x = synthetic_data['time']
ax.hist(x, bins=30, linewidth=0.5, edgecolor="white")

plt.show()

# count left and right

d = synthetic_data['direction']
# not the best
# sum(d == "left")

# more pythonic / pandas
d.value_counts()