"""Dataset: https://www.kaggle.com/c/painter-by-numbers
Using annotated csv file to separate img files to their own directories """

import pandas as pd 
import os
import os.path
from pathlib import Path
import time 
import shutil

# Initiate start time to calculate runtime 
start_time = time.time()

df = pd.read_csv(r'dataset\Painter by Numbers\all_data_info.csv')
df = df.drop(['pixelsx','pixelsy','size_bytes','source','artist_group'], axis=1)
print(df)

print(f"List of Genres: {df['genre'].unique()}")
print(f"Total number of Genres: {len(df['genre'].unique())} ")

# Top 10 of the genres
print(df['genre'].value_counts().head(10))
list = []
for val, cnt in df['genre'].value_counts().head(10).iteritems():
    list.append(val)
    print('Genre', val, 'was found', cnt, 'times')
print(f"List: {list}")

# We will remove paintings that is not in the train dataset
df = df[df.in_train != False]
print(df)

# Filter dataframe to only 'portrait' genre
df = df[df.genre == 'portrait']
df = df.reset_index(drop=True)
print(df)

# Grab filename and put it into a list
col_list = df.new_filename.values.tolist()

# Uncomment to make genres directories
r"""
# Make the genres directories
os.mkdir("Genres")
for genre in list:
    path = ("Genres/" + genre)
    os.mkdir(path)
"""

# Copy the file and put it in new folder
OldFolder= Path(r"dataset\Painter by Numbers\train")
NewFolder= Path(r"Genres\portrait")

# Uncomment to either copy the files or move the files
r"""
for file_name in col_list:
    shutil.copy(OldFolder / file_name, NewFolder )
    #shutil.move(OldFolder / file_name, NewFolder)
"""

# Print program runtime
elapsed = (time.time() - start_time)
time = time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))
print("Process finished --- %s --- " % (time))
