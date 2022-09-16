# %% [markdown]


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


spits_cutout = joblib.load("./datasets/spits_cutout_x2828_y5085.joblib")
spits_coords = joblib.load("./datasets/coords_spits_cutout_x2828_y5085.joblib")
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


def get_spits_data():

    return joblib.load("./datasets/spits_data.joblib")


def cartesionGrid_spits(cutout_data):

    return cartesionGrid(get_spits_data(), cutout_data)


# %% [markdown]
# # Creating mask
# %%

mean, median, std = sigma_clipped_stats(spits_cutout, sigma=3.0)
print((mean, median, std))


# %%


def mask_with_dao(input_data):
    mean, median, std = sigma_clipped_stats(input_data, sigma=3.0)

    d = DAOStarFinder(fwhm=10.0, threshold=5.0 * std)

    s = d(input_data - median)

    for col in s.colnames:

        s[col].info.format = "%.8g"

    return s, d


sources, daofind = mask_with_dao(spits_cutout)


# %%


print(f"Number of stars found = {len(sources)  } ")


# %%


positions = np.transpose((sources["xcentroid"], sources["ycentroid"]))
apertures = CircularAperture(positions, r=4.0)
norm = ImageNormalize(stretch=SqrtStretch())

plt.imshow(spits_cutout, cmap="Greys", origin="lower", norm=norm, interpolation="nearest")
apertures.plot(color="blue", lw=1.5, alpha=0.5)


# %% [markdown]


# # Background Masking


# Let's get rid of the background noise


# %%
norm = ImageNormalize(stretch=SqrtStretch())
plt.imshow(spits_cutout, norm=norm, origin="lower", cmap="Greys_r", interpolation="nearest")

# %% [markdown]
# ## creating the source mask
# %%
SIGMA = 10.90


def mask_sources(input_data, S):

    sigma_clip = SigmaClip(sigma=S, maxiters=10)
    threshold = detect_threshold(input_data, nsigma=6.250, sigma_clip=sigma_clip)
    segment_img = detect_sources(input_data, threshold, npixels=5)
    footprint = circular_footprint(radius=1)
    mask = segment_img.make_source_mask(footprint=footprint)
    return mask


source_mask = mask_sources(spits_cutout, SIGMA)
mean, median, std = sigma_clipped_stats(spits_cutout, sigma=5.0, mask=source_mask)
print(f"{mean, median, std = }")
# %% [markdown]
# ## Creating a Background2D
# %%
sigma_clip = SigmaClip(sigma=3.0)
bkg_estimator = MedianBackground()
bkg = Background2D(
    spits_cutout, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator,
)
# %%
print(f"{bkg.background_median = }\n{bkg.background_rms_median = }")
# %%
spits_cutout_masked = spits_cutout - bkg.background
spits_double_masked = spits_cutout_masked - source_mask
sources = daofind(spits_cutout_masked - median)
for col in sources.colnames:
    sources[col].info.format = "%.8g"
# %% [markdown]
# # Actual individual cutouts
# %%
def createCutoutsList(sources_list, input_data):

    from astropy.nddata import Cutout2D

    x = sources_list["xcentroid"]

    y = sources_list["ycentroid"]

    points = list(zip(x, y))

    cutouts = []

    for p_index, point in enumerate(points):

        # if p_index > 10:

        # some points were of a different size

        # and numpy was throwing a fit that they werern't all (50,50)

        c = Cutout2D(input_data, point, (50, 50)).data
        if c.shape >= (50, 50):
            cutouts.append(c)

    cutouts = np.array(cutouts)

    return cutouts


# masking each cutout
def maskSources(input_cutouts_list):
    masked_cutouts = []
    masked_cutouts_1d = []
    for i, c in enumerate(input_cutouts_list):

        sigma_clip = SigmaClip(sigma=7.0, maxiters=10)

        threshold = detect_threshold(c, nsigma=3.990, sigma_clip=sigma_clip)

        segment_img = detect_sources(c, threshold, npixels=10)

        footprint = circular_footprint(radius=1)

        # mask = np.array([[]])
        try:
            mask = segment_img.make_source_mask(footprint=footprint)

        except AttributeError:

            print(f"No sources were found for the cutout at index {i}")

        try:
            mean, median, std = sigma_clipped_stats(c, sigma=5.0, mask=mask)
        except UnboundLocalError:
            mask = 0
            mean, median, std = sigma_clipped_stats(c, sigma=5.0, mask=mask)
        # print(f'{mean, median, std = }')

        m = c - mask - median
        masked_cutouts.append(m)

    masked_cutouts = np.array(masked_cutouts)

    return masked_cutouts


cutouts = createCutoutsList(sources, spits_cutout_masked)

masked_cutouts = maskSources(cutouts)

masked_cutouts_1d = masked_cutouts.reshape(masked_cutouts.shape[0], 2500)

# %%
# training the model

model_params = {"test_size": 0.2}


cutouts_1d = cutouts.reshape(cutouts.shape[0], 2500)

if masked_cutouts_1d.shape[0] == cutouts_1d.shape[0]:
    input_train, input_test, output_train, output_test = train_test_split(
        masked_cutouts_1d, cutouts_1d, test_size=0.2
    )

##
# save the training data
with open('./knn_training_data_masked.joblib', 'wb') as f:
    joblib.dump(masked_cutouts, f)
with open('./knn_training_data.joblib', 'wb') as f:
    joblib.dump(cutouts_1d, f)
# %%
input_test.shape, output_test.shape
# %%
knn = KNeighborsRegressor()
knn.fit(input_train, output_train)
knn_pred = knn.predict(input_test)
# %%
def compareResult(input_predicted):
    pred_img = input_predicted[0].reshape(50, 50)

    plt.subplot(121)

    plt.title("Predicted")

    plt.imshow(pred_img)

    plt.subplot(122)

    plt.title("Training Data")

    plt.imshow(input_train[0].reshape(50, 50))


compareResult(knn_pred)

with open('knn.joblib', 'wb') as f:
    joblib.dump(knn, f)

# %%


def compareResults(input_predicted_list):
    shape_colls = input_predicted_list.shape[0]
    n_colls = shape_colls // 2
    if shape_colls % 2 != 0:
        n_colls += 1

    for plot_index, cutout in enumerate(input_predicted_list):
        plot_index += 1
        plt.subplot(1, n_colls, plot_index)

    pred_img = input_predicted[0].reshape(50, 50)

    plt.subplot(121)

    plt.title("Predicted")

    plt.imshow(pred_img)

    plt.subplot(122)

    plt.title("Training Data")

    plt.imshow(input_train[0].reshape(50, 50))


# %%
def plot_gallery(images, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)))
        plt.xticks(())
        plt.yticks(())


# %%
plot_gallery(knn_pred, 50, 50)

# %%
plot_gallery(input_train, 50, 50)
# %%
knn.score(input_test, output_test)
