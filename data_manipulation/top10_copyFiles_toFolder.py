"""This file collects the top 10 genres and uses the filename to put them all in one folder"""

import pandas as pd
import os
import os.path
from pathlib import Path
import shutil

df = pd.read_csv(r'train_info.csv')
df = df.drop(['artist','style','date'], axis=1)
print(df)
# Top 10 of the genres
print(df['genre'].value_counts().head(10))
list = []
for val, cnt in df['genre'].value_counts().head(10).iteritems():
    list.append(val)
    print('Genre', val, 'was found', cnt, 'times')
print(f"List: {list}")

# Filter dataframe to only top10 genre
df = df[df['genre'].isin(list)]
df = df.reset_index(drop=True)
#df.to_csv('top10.csv', index=False)
print(len(df.index))
# Grab filename and put it into a list
col_list = df.filename.values.tolist()

# Copy the file and put it in new folder
print(os.getcwd())
OldFolder= Path(r"train")
NewFolder= Path(r"top10_train")

r"""
for file_name in col_list:
    shutil.copy(OldFolder / file_name, NewFolder )
"""

