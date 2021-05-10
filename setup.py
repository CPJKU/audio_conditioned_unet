import os

import setuptools
import zipfile

from setuptools import setup


setup(
    name='audio_conditioned_unet',
    version='0.1dev',
    description='Audio-Conditioned U-Net for Position Estimation in Full Sheet Images',
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: MusicInformationRetrieval",
    ],
    author='Florian Henkel',
)

# extract sound font
SOUND_FONT_PATH = "./audio_conditioned_unet/sound_fonts/grand-piano-YDP-20160804.sf2"
if not os.path.exists(SOUND_FONT_PATH):
    """
    This extracts the zip-compressed soundfont used in the game.
    (Required due to a file size limitation of github.)
    """
    print("Extracting soundfont file %s ..." % SOUND_FONT_PATH)
    zip_ref = zipfile.ZipFile(SOUND_FONT_PATH + ".zip", 'r')
    zip_ref.extractall(os.path.dirname(SOUND_FONT_PATH))
    zip_ref.close()

# extract data
DATA_PATH = "./data/nottingham_npz.zip"
if os.path.exists(DATA_PATH):
    print("Extracting data %s ..." % DATA_PATH)
    zip_ref = zipfile.ZipFile(DATA_PATH, 'r',  zipfile.ZIP_DEFLATED)
    zip_ref.extractall(os.path.dirname(DATA_PATH))
    zip_ref.close()
