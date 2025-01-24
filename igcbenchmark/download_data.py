
"""
Download and export images from the NSD dataset.

Author: Pierre Lelievre
"""

import os
import h5py
from PIL import Image
from subprocess import run


NSD_AWS = 's3://natural-scenes-dataset'
AWS_CMD_SYNC = 'aws s3 sync'
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


# Utils


def create_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    return path


def get_raw_dir():
    return create_dir(os.path.join(DATA_DIR, 'raw'))


def get_img_dir():
    return create_dir(os.path.join(DATA_DIR, 'images'))


# Create data directory


create_dir(DATA_DIR)


# Download stimuli images


folder_nsd = f'{NSD_AWS}/nsddata_stimuli/stimuli/nsd'
copy_cmd = (
    f"{AWS_CMD_SYNC} {folder_nsd} {get_raw_dir()} --exclude '*' "
    '--include nsd_stimuli.hdf5')
run(copy_cmd, shell=True, check=True)


# Export images


stimuli_file = h5py.File(os.path.join(get_raw_dir(), 'nsd_stimuli.hdf5'), 'r')
stimuli_data = stimuli_file['imgBrick']
target_dir = get_img_dir()
for i, img in enumerate(stimuli_data):
    pil_img = Image.fromarray(img, 'RGB')
    pil_img.save(os.path.join(target_dir, f'nsd_{i:05d}.png'), 'PNG')
    if (i+1)%1000 == 0:
        print(f'{i+1:05d} images done')
stimuli_file.close()
