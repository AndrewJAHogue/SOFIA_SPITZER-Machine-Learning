
from astropy.nddata import Cutout2D
from photutils.background import Background2D, MedianBackground
from photutils.utils import circular_footprint
from photutils.segmentation import detect_threshold, detect_sources
from astropy.stats import SigmaClip
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import SqrtStretch
import matplotlib.pyplot as plt

from astropy.stats import sigma_clipped_stats

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.neighbors import KNeighborsRegressor

import joblib

import numpy as np

spits_cutout = joblib.load('./datasets/spits_cutout_x2828_y5085.joblib')

spits_coords = joblib.load('./datasets/coords_spits_cutout_x2828_y5085.joblib')

spits_cutout_reshaped = spits_cutout.reshape(1, -1)[0]

CUTOUT_SIZE = (50, 50)

def maskBkgAndSources(input_data):
    sigma_clip = SigmaClip(sigma=3.)

    bkg_estimator = MedianBackground()

    bkg = Background2D(input_data, CUTOUT_SIZE, filter_size=(
        3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)


    print(f'{bkg.background_median = }\n{bkg.background_rms_median = }')


    input_data_masked = input_data - bkg.background

    input_data_double_masked = input_data_masked - source_mask

    return input_data_double_masked
    