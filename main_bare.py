# %% [markdown]

# # Imports


# %%

from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.utils.validation import check_random_state
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.svm import SVR
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

    # gives every tuple combination of x and y values; i.e. all points
    return extmath.cartesian([coords_x, coords_y])


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
def mask_with_daofind():
    daofind = DAOStarFinder(fwhm=4.0, threshold=5.*std)
    sources = daofind(spits_cutout - median)
    for col in sources.colnames:
        sources[col].info.format = '%.8g'

    return sources, daofind, col


sources, daofind, col = mask_with_daofind()
# %%

positions = np.transpose((sources['xcentroid'], sources['ycentroid']))

apertures = CircularAperture(positions, r=4.)

norm = ImageNormalize(stretch=SqrtStretch())


plt.imshow(spits_cutout, cmap='Greys', origin='lower',
           norm=norm, interpolation='nearest')

apertures.plot(color='blue', lw=1.5, alpha=0.5)


# %% [markdown]

# # Background Masking

# Let's get rid of the background noise


# %%

norm = ImageNormalize(stretch=SqrtStretch())

plt.imshow(spits_cutout, norm=norm, origin='lower',
           cmap='Greys_r', interpolation='nearest')


# %%


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


sigma_clip = SigmaClip(sigma=3.)

bkg_estimator = MedianBackground()

bkg = Background2D(spits_cutout, (50, 50), filter_size=(
    3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)


# %%

print(f'{bkg.background_median = }\n{bkg.background_rms_median = }')


# %%

spits_cutout_masked = spits_cutout - bkg.background

spits_double_masked = spits_cutout_masked - mask


# %%

# %matplotlib widget

plt.subplot(131)

plt.imshow(spits_cutout_masked, norm=norm, origin='lower',
           cmap='Greys_r', interpolation='nearest')

plt.subplot(132)

plt.imshow(spits_double_masked, norm=norm, origin='lower',
           cmap='Greys_r', interpolation='nearest')

plt.subplot(133)

plt.tight_layout()

plt.imshow(spits_cutout)


# %% [markdown]

# Look's like background was successfully masked


# %%

sources = daofind(spits_cutout_masked - median)

for col in sources.colnames:

    sources[col].info.format = '%.8g'
sources


# %% [markdown]

# # Actual individual cutouts


# %%


x = sources['xcentroid']

y = sources['ycentroid']

points = list(zip(x, y))


cutouts = []

coords = []

for p_index, point in enumerate(points):

    if p_index > 10:

        c = Cutout2D(spits_cutout_masked, point, (50, 50)).data

        coord = cartesionGrid_spits(
            Cutout2D(spits_cutout, point, (50, 50)).data)
        cutouts.append(c)


cutouts = np.array(cutouts)


# %%

# %matplotlib inline

target_cutout = cutouts[0]

plt.imshow(target_cutout, norm=norm, origin='lower',
           cmap='Greys_r', interpolation='nearest')


# %%


# gscv to tune

params = {"n_components": np.arange(20)}

grid = GridSearchCV(PCA(), params)
grid.fit(target_cutout)

n_comp = grid.best_params_['n_components']


print(f'{ grid.best_params_ = } ')


# fit data to model

pca = PCA(whiten=False, n_components=n_comp)
data = pca.fit_transform(target_cutout)


params = {"bandwidth": np.logspace(-1, 1, 20)}

grid = GridSearchCV(KernelDensity(), params)
grid.fit(data)


kde = grid.best_estimator_


new_data = kde.sample(44, random_state=0)

new_data = pca.inverse_transform(new_data)


# %%

# %matplotlib inline

plt.imshow(new_data)


# %% [markdown]

# # Multiple cutouts


# %% [markdown]

# ## Setting up data


# %%


x = sources['xcentroid']

y = sources['ycentroid']

points = list(zip(x, y))


cutouts = []

cutouts_1d = []

cutouts_coords = []

for p_index, point in enumerate(points):

    if p_index > 10:

        c = Cutout2D(spits_cutout_masked, point, (50, 50)).data

        coord = cartesionGrid_spits(
            Cutout2D(spits_cutout, point, (50, 50)).data)

        cutouts.append(c)

        cutouts_1d.append(c.flatten())
        cutouts_coords.append(coord)


cutouts_coords = np.array(cutouts_coords)

cutouts = np.array(cutouts)

cutouts_1d = np.array(cutouts_1d)


print(f'{spits_coords.shape = }')

print(f'{cutouts_coords.shape = }')

print(f'{cutouts_coords.reshape(-1,2).shape = }')


cutouts_coords = cutouts_coords.reshape(-1, 2)

print(f'{cutouts_coords.shape = }')


# %% [markdown]

# ## Masking cutouts


# %% [markdown]

# Just want to mask the peaks of the stars, to simulate the supersaturated data of other Spitzer points


# %%

masked_cutouts = []

masked_cutouts_1d = []

for c in cutouts:

    sigma_clip = SigmaClip(sigma=7., maxiters=100)

    threshold = detect_threshold(c, nsigma=3.990, sigma_clip=sigma_clip)

    segment_img = detect_sources(c, threshold, npixels=5)

    footprint = circular_footprint(radius=1)

    mask = segment_img.make_source_mask(footprint=footprint)

    mean, median, std = sigma_clipped_stats(c, sigma=5.0, mask=mask)

    # print(f'{mean, median, std = }')

    m = c - mask - median

    masked_cutouts.append(m)

    masked_cutouts_1d.append(m.flatten())


masked_cutouts = np.array(masked_cutouts)

masked_cutouts_1d = np.array(masked_cutouts_1d)


# %matplotlib inline

plt.subplot(121)

plt.imshow(cutouts[0], norm=norm, origin='lower',
           cmap='Greys_r', interpolation='nearest')

plt.subplot(122)

plt.imshow(masked_cutouts[5], norm=norm, origin='lower',
           cmap='Greys_r', interpolation='nearest')


# %%


print(f'training data size {masked_cutouts.shape}')

print('65 (50, 50) masked_cutouts ')

print(f'training data size {masked_cutouts_1d.shape}')


# %% [markdown]

# Just checking the cutouts


# %%

col_size = 11  # limit so we don't get spammed with hundreds of images

for plot_index, c in enumerate(masked_cutouts):

    plot_index += 1

    if plot_index < col_size:

        plt.subplot(1, col_size, plot_index)

        plt.imshow(c, norm=norm, origin='lower',
                   cmap='Greys_r', interpolation='nearest')


# %%

n_comp = 1


# fit data to model

pca = PCA(whiten=False, n_components=n_comp)

full_data = pca.fit_transform(cutouts_1d)


params = {"bandwidth": np.logspace(-1, 1, 20)}

grid = GridSearchCV(KernelDensity(), params)
grid.fit(data)


kde = grid.best_estimator_


new_data = kde.sample(44, random_state=0)

new_data = pca.inverse_transform(new_data)


# %%

plt.subplot(121)

plt.imshow(new_data[0].reshape(50, 50))

plt.subplot(122)

plt.imshow(cutouts[0])


# %% [markdown]

# Can predict new cutouts pretty good. But we need to use coordinates, don't we


# %% [markdown]

# ### A handy rescale method

# ```

# rescale(data_pred, (2500 / data_pred.size), preserve_range=True, mode='constant', cval=np.NaN).reshape(50,50)

# ```


# %%


x_train, x_test, data_train, data_test = train_test_split(
    masked_cutouts_1d, cutouts_1d, random_state=0, shuffle=False)


# %%

# Import the necessary modules and libraries


# Create a random dataset

rng = np.random.RandomState(1)

X = np.sort(5 * rng.rand(80, 1), axis=0)

y = np.sin(X).ravel()

y[::5] += 3 * (0.5 - rng.rand(16))


# Fit regression model

regr_1 = DecisionTreeRegressor(max_depth=2)

regr_2 = DecisionTreeRegressor(max_depth=5)

regr_1.fit(X, y)

regr_2.fit(X, y)


# Predict

X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]

y_1 = regr_1.predict(X_test)

y_2 = regr_2.predict(X_test)


# Plot the results
plt.figure()

plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")

plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)

plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)

plt.xlabel("data")

plt.ylabel("target")

plt.title("Decision Tree Regression")
plt.legend()

plt.show()


# %%


# Load the faces datasets

data, targets = fetch_olivetti_faces(return_X_y=True)


train = data[targets < 30]

test = data[targets >= 30]  # Test on independent people


# Test on a subset of people

n_faces = 5

rng = check_random_state(4)

face_ids = rng.randint(test.shape[0], size=(n_faces,))

test = test[face_ids, :]


n_pixels = data.shape[1]

# Upper half of the faces

X_train = train[:, : (n_pixels + 1) // 2]

# Lower half of the faces

y_train = train[:, n_pixels // 2:]

X_test = test[:, : (n_pixels + 1) // 2]

y_test = test[:, n_pixels // 2:]


# Fit estimators

ESTIMATORS = {

    "Extra trees": ExtraTreesRegressor(

        n_estimators=10, max_features=32, random_state=0
    ),

    "K-nn": KNeighborsRegressor(),

    "Linear regression": LinearRegression(),

    "Ridge": RidgeCV(),

}


y_test_predict = dict()

for name, estimator in ESTIMATORS.items():

    estimator.fit(X_train, y_train)

    y_test_predict[name] = estimator.predict(X_test)


# Plot the completed faces

image_shape = (64, 64)


n_cols = 1 + len(ESTIMATORS)

plt.figure(figsize=(2.0 * n_cols, 2.26 * n_faces))

plt.suptitle("Face completion with multi-output estimators", size=16)


for i in range(n_faces):

    true_face = np.hstack((X_test[i], y_test[i]))

    if i:

        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1)

    else:

        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1, title="true faces")

    sub.axis("off")

    sub.imshow(

        true_face.reshape(image_shape), cmap=plt.cm.gray, interpolation="nearest"
    )

    for j, est in enumerate(sorted(ESTIMATORS)):

        completed_face = np.hstack((X_test[i], y_test_predict[est][i]))

        if i:

            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j)

        else:

            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j, title=est)

        sub.axis("off")

        sub.imshow(
            completed_face.reshape(image_shape),

            cmap=plt.cm.gray,

            interpolation="nearest",
        )


plt.show()


# %%

kde = KNeighborsRegressor()

kde.fit(X_train, y_train)


# %%

X_train.shape, y_train.shape


# %% [markdown]

# # KNN


# %%

input_train, input_test, output_train, output_test = train_test_split(
    masked_cutouts_1d, cutouts_1d, test_size=0.2)


# %%
input_test.shape, output_test.shape


# %%

kde = KNeighborsRegressor()

kde.fit(input_train, output_train)


kde_pred = kde.predict(input_test)


# %%

pred_img = kde_pred[0].reshape(50, 50)

plt.subplot(121)

plt.imshow(pred_img)

plt.subplot(122)

plt.imshow(x_train[2].reshape(50, 50))
