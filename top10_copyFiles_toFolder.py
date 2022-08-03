import pandas as pd
import os
import os.path
from pathlib import Path
import shutil

df = pd.read_csv(r'train_info.csv')
df = df.drop(['artist','style','date'], axis=1)
print(df)
r"""
print(f"List of Genres: {df['genre'].unique()}")
print(f"Total number of Genres: {len(df['genre'].unique())} ")
print(f"List of Style: {df['style'].unique()}")
print(f"Total number of Style: {len(df['style'].unique())} ")
"""
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

##########################################################################
# Moving genre to separate subfolders

df = df[df['genre'] == 'symbolic painting']
df = df.reset_index(drop=True)
col_list = df.filename.values.tolist()

r"""
# Make the genres directories
os.mkdir("Top10_Genres")
for genre in list:
    path = ("Top10_Genres/" + genre)
    os.mkdir(path)
"""

# Copy the file and put it in new folder
OldFolder= Path(r"train")
NewFolder= Path(r"Top10_Genres\symbolic painting")

# Uncomment to either copy the files or move the files

for file_name in col_list:
    shutil.copy(OldFolder / file_name, NewFolder )
    #shutil.move(OldFolder / file_name, NewFolder)
