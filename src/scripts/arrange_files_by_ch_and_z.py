import os
import shutil
from os import listdir
from os.path import isfile, join

rename = False  # rename to ...A01Z01...

# folder = "/home/koosk/data/images/adipocyte/dataset/60x-tiles"
folder = "/home/koosk/data/images/adipocyte/dataset/20x"


patterns = ["Z01C01", "Z01C02", "Z01C03", "Z01C04", "Z02C04", "Z03C04", "Z04C04",
            "Z05C04", "Z06C04", "Z07C04"]
for dir in patterns:
    os.makedirs(join(folder, dir))
files = [f for f in listdir(folder) if isfile(join(folder, f))]

for file in files:
    pattern_to_use = None
    for pattern in patterns:
        if pattern in file:
            pattern_to_use = pattern
    assert pattern_to_use is not None
    out_file = file
    if rename:
        out_file = out_file.replace("Z01C02", "Z01C01")
        out_file = out_file.replace("Z01C03", "Z01C01")
        out_file = out_file.replace("Z01C04", "Z01C01")
        out_file = out_file.replace("Z02C04", "Z01C01")
        out_file = out_file.replace("Z03C04", "Z01C01")
        out_file = out_file.replace("Z04C04", "Z01C01")
        out_file = out_file.replace("Z05C04", "Z01C01")
        out_file = out_file.replace("Z06C04", "Z01C01")
        out_file = out_file.replace("Z07C04", "Z01C01")

        out_file = out_file.replace("L01A02", "L01A01")
        out_file = out_file.replace("L01A03", "L01A01")
        out_file = out_file.replace("L01A04", "L01A01")

    dst = join(folder, pattern_to_use, out_file)
    # shutil.move(join(folder, file), dst)
    shutil.copy(join(folder, file), dst)

