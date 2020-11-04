from sklearn.model_selection import train_test_split
import os
from os import listdir
from os.path import isfile, join
import shutil

# folder = "/home/koosk/data/images/adipocyte/dataset/20x-nuclei_thresholded-tiled"
folder = "/home/koosk/data/images/adipocyte/dataset/20x-tiles/C01"

random_state = 2475
train_ratio = 0.7
validation_ratio = 0.2
test_ratio = 0.10


onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]

indices = range(len(onlyfiles))
x_train, x_test, y_train, y_test = train_test_split(indices, indices, test_size=1 - train_ratio, random_state=random_state)

# test is now test_ratio% of the initial data set
# validation is now validation_ratio% of the initial data set
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=random_state)

print(x_train)
print(x_val)
print(x_test)

print(len(x_train))
print(len(x_val))
print(len(x_test))

train_folder = join(folder, "train")
test_folder = join(folder, "test")
val_folder = join(folder, "val")
os.makedirs(train_folder, exist_ok=False)
os.makedirs(test_folder, exist_ok=False)
os.makedirs(val_folder, exist_ok=False)

for idx in range(len(onlyfiles)):
    dst = []
    if idx in x_train:
        dst = train_folder
    elif idx in x_val:
        dst = val_folder
    else:
        dst = test_folder

    dst = join(dst, onlyfiles[idx])
    shutil.copy(join(folder, onlyfiles[idx]), dst)

