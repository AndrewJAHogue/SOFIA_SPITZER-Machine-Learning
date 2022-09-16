# %% [markdown]
# # Imports

# %%
import matplotlib.pyplot as plt

from astropy.stats import sigma_clipped_stats
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
import joblib
import numpy as np
import pickle


spits_cutout = joblib.load('./datasets/spits_cutout_x2828_y5085.joblib')
spits_coords = joblib.load('./datasets/coords_spits_cutout_x2828_y5085.joblib')
spits_cutout_reshaped = spits_cutout.reshape(1,-1)[0]

import numpy as np

import matplotlib.pyplot as plt

from astropy.visualization import SqrtStretch

from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.aperture import CircularAperture
from photutils.detection import DAOStarFinder

# %%
def cartesionGrid(full_data, cutout_data):
    from sklearn.utils import extmath

    y1, x1 = np.where(full_data == cutout_data[0, 0])
    y1 = y1[0]
    x1 = x1[0]
    first_corner = y1, x1

    diag_corner = np.add(first_corner, cutout_data.shape)
    coords_x = np.array(range(first_corner[1], diag_corner[1]))
    coords_y = np.array(range(first_corner[0], diag_corner[0]))

    return extmath.cartesian([coords_x, coords_y]) # gives every tuple combination of x and y values; i.e. all points

# def get_spits_data():
#     from astropy.io import fits
#     from computer_path import FullMaps

#     return fits.getdata(FullMaps.Spitzer())
def get_spits_data():
    return joblib.load('./datasets/spits_data.joblib')


def cartesionGrid_spits(cutout_data):
    return cartesionGrid(get_spits_data(), cutout_data)
                    

# %% [markdown]
# # Creating mask

# %%
mean, median, std = sigma_clipped_stats(spits_cutout, sigma=3.0)  
print((mean, median, std))

# %%
daofind = DAOStarFinder(fwhm=4.0, threshold=5.*std)
sources = daofind(spits_cutout - median)
for col in sources.colnames:
    sources[ col ].info.format = '%.8g'
sources

# %%
positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
apertures = CircularAperture(positions, r=4.)
norm = ImageNormalize(stretch=SqrtStretch())


# %% [markdown]
# # Background Masking
# Let's get rid of the background noise

# %%
norm = ImageNormalize(stretch=SqrtStretch())

# %%
from astropy.stats import SigmaClip
from photutils.segmentation import detect_threshold, detect_sources
from photutils.utils import circular_footprint

# %% [markdown]
# ## creating the source mask

# %%
sigma_clip = SigmaClip(sigma=10.90, maxiters=10)
threshold = detect_threshold(spits_cutout, nsigma=6.250, sigma_clip=sigma_clip)
segment_img = detect_sources(spits_cutout, threshold, npixels=5)
footprint = circular_footprint(radius=1)
mask = segment_img.make_source_mask(footprint=footprint)
mean, median, std = sigma_clipped_stats(spits_cutout, sigma=5.0, mask=mask)
print(f'{mean, median, std = }')

# %% [markdown]
# ## Creating a Background2D

# %%
from photutils.background import Background2D, MedianBackground
sigma_clip = SigmaClip(sigma=3.)
bkg_estimator = MedianBackground()
bkg = Background2D(spits_cutout, (50,50), filter_size=(3,3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)


# %%
spits_cutout_masked = spits_cutout - bkg.background
spits_double_masked = spits_cutout_masked - mask

# Look's like background was successfully masked

# %%
sources = daofind(spits_cutout_masked - median)
for col in sources.colnames:
    sources[ col ].info.format = '%.8g'
sources

# %% [markdown]
# # Actual individual cutouts

# %%
from astropy.nddata import Cutout2D
import pandas as pd
x = sources['xcentroid']
y = sources['ycentroid']
points = list(zip(x, y))

cutouts = []
coords =  []
for p_index, point in enumerate( points ):
    # if p_index > 10:
    c = Cutout2D(spits_cutout_masked, point, (50, 50)).data
    if c.shape == (50, 50):
        cutouts.append(c) 

cutouts = np.array(cutouts)
print(f'{cutouts.shape = }')
# %%
