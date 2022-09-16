## imports
# %%


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
from py import process
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
import joblib
import numpy as np
from ajh_utils import computer_path, lineplots

CUTOUT_SIZE = 50

# %%


def get_region_cutouts( x, y, h, w,):
    from ajh_utils import computer_path
    from astropy.nddata import Cutout2D

    spits_data = computer_path.Star_Datasets.get_spits_data()
    sofia_data = computer_path.Star_Datasets.get_sofia_data()

    spits_cutout = Cutout2D(spits_data, (x, y), (h, w)).data
    sofia_cutout = Cutout2D(sofia_data, (x, y), (h, w)).data

    return sofia_cutout, spits_cutout


## coords
x, y = 261, 5486
# region size
h, w = 2000, 2000

sofia_region, spits_cutout = get_region_cutouts(x, y, h, w)
# %%

# sofia and spits region cutouts
plt.subplot(121)
plt.imshow(sofia_region)
plt.subplot(122)
plt.imshow(spits_cutout)
# %%
# first, let's compare our old model

# load the knn pickle
import joblib

knn = joblib.load(f"./knn.joblib")
print(f"The KNN model expects a 2d list with each sample of length {knn.n_features_in_}")
# %%
# lets get a saturated star cutout from spits
spits_data = computer_path.Star_Datasets.get_spits_data()
sofia_data = computer_path.Star_Datasets.get_sofia_data()
# %%
## let's mask the background noise of the region first


def mask_sources(input_data, S):

    sigma_clip = SigmaClip(sigma=S, maxiters=10)
    threshold = detect_threshold(input_data, nsigma=6.250, sigma_clip=sigma_clip)
    segment_img = detect_sources(input_data, threshold, npixels=5)
    footprint = circular_footprint(radius=1)
    mask = segment_img.make_source_mask(footprint=footprint)
    return mask


def maskBackground(input_data, CUTOUT_SIZE=CUTOUT_SIZE, sigma=6.0):
    sigma_clip = SigmaClip(sigma=sigma)

    bkg_estimator = MedianBackground()

    bkg = Background2D(
        input_data, CUTOUT_SIZE, filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator
    )

    print(f"{bkg.background_median = }\n{bkg.background_rms_median = }")

    input_data_masked = input_data - bkg.background

    # source_mask = mask_sources(input_data, sigma)
    input_data_double_masked = input_data_masked

    return input_data_double_masked


# def getSourcesList(input_data, bkg, source_mask, median):
#     sigma = 6.0
#     sigma_clip = SigmaClip(sigma=sigma)

#     bkg_estimator = MedianBackground()

#     bkg = Background2D(
#         input_data, CUTOUT_SIZE, filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator
#     )

#     input_data_masked = input_data - bkg.background
#     # input_double_masked = input_data_masked - source_mask
#     sources = daofind(input_data_masked - median)
#     for col in sources.colnames:
#         sources[col].info.format = "%.8g"

#     return sources


def getSourcesList(input_data, sigma=3.0, fwhm=10., threshold=5.):
    mean, median, std = sigma_clipped_stats(input_data, sigma=sigma)
    d = DAOStarFinder(fwhm=fwhm, threshold=threshold * std)
    s = d(input_data - median)
    for col in s.colnames:
        s[col].info.format = "%.8g"
    
    print(f'Found {len(s)} stars')

    return s, d


# %%
##
x1, y1 = 464, 5335
single_spits_star = Cutout2D(spits_data, (x1, y1), CUTOUT_SIZE).data
single_spits_bkg_masked = maskBackground(single_spits_star, CUTOUT_SIZE)
plt.subplot(121)
plt.imshow(single_spits_bkg_masked, cmap="Greys", origin="lower", interpolation="nearest")
plt.subplot(122)
plt.imshow(single_spits_star, cmap="Greys", origin="lower", interpolation="nearest")

# %%
## have to impute the nan values
from sklearn.impute import KNNImputer

imputer = KNNImputer(missing_values=np.NaN, n_neighbors=40)
imputed_single_cutout = imputer.fit_transform(single_spits_star)

plt.imshow(imputed_single_cutout)
# %%
## lets change the parameters of the model
params = knn.get_params()
knn.set_params(n_neighbors=3)
# %%

## alright, lets have the old knn predict the result
single_knn_predicted = knn.predict([imputed_single_cutout.flatten()])

plt.subplot(131)
plt.title("Spitzer")
plt.imshow(single_spits_star)
plt.subplot(132)
plt.title("Predicted")
plt.imshow(single_knn_predicted[0].reshape(50, 50))
plt.subplot(133)
plt.title("Sofia")
plt.imshow(Cutout2D(sofia_data, (x1, y1), CUTOUT_SIZE).data)

## so it looks very mismatched but it actually matches pretty well with SOFIA

# %%
## let's do a linecut of them

from ajh_utils.lineplots import GetNthRow

xs, y_org = GetNthRow(single_spits_star, 25)
plt.plot(xs, y_org, label="Spitzer")
xs, y_pred = GetNthRow(single_knn_predicted[0].reshape(50, 50), 25)
plt.plot(xs, y_pred, label="Predicted")
xs, y_sofia = GetNthRow(Cutout2D(sofia_data, (x1, y1), CUTOUT_SIZE).data, 25)
# plt.plot(xs, y_sofia, label='Sofia')
plt.legend()


# %%
## train on other cutouts of spits

# make noise estimators
SIGMA = 3
# SIGMA = 10.90
source_mask = mask_sources(spits_cutout, SIGMA)
mean, median, std = sigma_clipped_stats(spits_cutout, sigma=5.0, mask=source_mask)
sigma_clip = SigmaClip(sigma=SIGMA)

bkg_estimator = MedianBackground()

bkg = Background2D(
    spits_cutout, CUTOUT_SIZE, filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator
)

# get list of all sources
spits_sources, daofind = getSourcesList(spits_cutout)
print(f"Number of stars found = {len(spits_sources)  } ")

# create a cutout at all these sources


def createCutoutsList(sources_list, input_data, cutout_size = (50,50)):
    from astropy.nddata import Cutout2D

    x = sources_list["xcentroid"]
    y = sources_list["ycentroid"]
    points = list(zip(x, y))
    cutouts = []

    for p_index, point in enumerate(points):
        # if p_index > 10:
        # some points were of a different size
        # and numpy was throwing a fit that they werern't all (50,50)
        c = Cutout2D(input_data, point, cutout_size).data
        if c.shape >= cutout_size:
            cutouts.append(c)

    cutouts = np.array(cutouts)

    return cutouts


cutouts = createCutoutsList(spits_sources, spits_cutout)

# %%
def plot_gallery(images, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)

        try:
            plt.imshow(images[i].reshape((h, w)))
        except IndexError:
            pass

        plt.xticks(())
        plt.yticks(())


plot_gallery(cutouts, 50, 50)


# %%
## mask the cutouts  
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

masked_cutouts = maskSources(cutouts)
# reshape them into a 2d list
masked_cutouts_1d = masked_cutouts.reshape(masked_cutouts.shape[0], 2500)
# reshape the original list into a 2d list as well
cutouts_1d = cutouts.reshape(cutouts.shape[0], CUTOUT_SIZE**2)


# %%

imputer = KNNImputer(missing_values=np.NaN, n_neighbors=40)
imputed_masked_cutouts_1d = imputer.fit_transform(masked_cutouts_1d)
imputed_cutouts_1d = imputer.fit_transform(cutouts_1d)


# %%
## make and fit the model
split_params = {"test_size": 0.2}
input_train, input_test, output_train, output_test = train_test_split(
   imputed_masked_cutouts_1d, imputed_cutouts_1d, test_size=0.2
)

knn = KNeighborsRegressor()
knn.fit(input_train, output_train)

predictions = knn.predict(input_test)



# %%

def compareResults(images_predicted, images_test, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.title('Predicted')
        plt.imshow(images_predicted[i].reshape((h, w)))
        # plt.subplot(n_row + 1, n_col, i)
        # plt.title('Original')
        # plt.imshow(images_test[i].reshape((h, w)))
        plt.xticks(())
        plt.yticks(())

compareResults(predictions, output_test, 50, 50)

# %%
knn.score(input_test, output_test)

## doesn't look that great tbh


# %%
# let's try including the older training data set
old_training_data = joblib.load('./knn_training_data.joblib')
old_training_masked_data = joblib.load('./knn_training_data_masked.joblib')

# concat the training data
all_cutouts = np.append(cutouts, old_training_data)
all_cutouts = all_cutouts.reshape(-1, 50, 50)

## go back over and mask it
masked_cutouts = maskSources(all_cutouts)
# reshape them into a 2d list
masked_cutouts_1d = masked_cutouts.reshape(masked_cutouts.shape[0], 2500)
# reshape the original list into a 2d list as well
cutouts_1d = np.array([])
cutouts_1d = all_cutouts.reshape(all_cutouts.shape[0], CUTOUT_SIZE**2)


# %%

imputer = KNNImputer(missing_values=np.NaN)
imputed_masked_cutouts_1d = imputer.fit_transform(masked_cutouts_1d)
imputed_cutouts_1d = imputer.fit_transform(cutouts_1d)


# %%
## make and fit the model
split_params = {"test_size": 0.2}
input_train, input_test, output_train, output_test = train_test_split(
   imputed_masked_cutouts_1d, imputed_cutouts_1d, test_size=0.2
)

knn = KNeighborsRegressor()
knn.fit(input_train, output_train)

predictions = knn.predict(input_test)


# %%

plot_gallery(predictions, 50, 50)
knn.score(input_test, output_test)
# %%


test_pred = knn.predict([ imputed_single_cutout.flatten() ])
test_img = test_pred.reshape(50,50)

plt.subplot(131)
plt.title("Spitzer")
plt.imshow(single_spits_star)
plt.subplot(132)
plt.title("Predicted")
plt.imshow(test_img)
plt.subplot(133)
plt.title("Sofia")
plt.imshow(Cutout2D(sofia_data, (x1, y1), CUTOUT_SIZE).data)


# %%

xs, y_org = GetNthRow(single_spits_star, CUTOUT_SIZE//2)
plt.plot(xs, y_org, label="Spitzer")
xs, y_pred = GetNthRow(test_img, CUTOUT_SIZE//2)
plt.plot(xs, y_pred, label="Predicted")
plt.legend()

## looks better than before, but still not quite there  

# %%
## we haven't even started using the SOFIA data at all
CUTOUT_SIZE = 50

# build a new region of sofia, with some nicer sources
# new_region_x = 282
# new_region_y = 5512
# new_region_size = 650
# sofia_region = Cutout2D(sofia_data, (new_region_x, new_region_y), new_region_size).data
# get cutouts of all the sources
sofia_sources, sofia_dao = getSourcesList(sofia_region, fwhm=6,sigma=4.25, threshold=7.)
sofia_cutouts = createCutoutsList(sofia_sources, sofia_region, (CUTOUT_SIZE, CUTOUT_SIZE))

spits_withSofiaSources_cutouts = createCutoutsList(sofia_sources, spits_cutout, (CUTOUT_SIZE, CUTOUT_SIZE))
spits_h = spits_withSofiaSources_cutouts.shape[1]
spits_withSofiaSources_cutouts = spits_withSofiaSources_cutouts.reshape(-1, spits_h**2)

def processData(input_data, sigma=3.):
    """
    Description:
        Mask the background noise with a sigma of 6 and impute any nan values 
    Arguments:
        input_data: Must be a 2D array
    Returns:
        Numpy array
    """    
    # masking the background
    masked = maskBackground(input_data, CUTOUT_SIZE, sigma)

    # imputed nan values
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(missing_values=np.NaN, n_neighbors=40)
    imputed_data = imputer.fit_transform(masked)

    return imputed_data

# process spits cutoutts
processed_spits = processData(spits_withSofiaSources_cutouts, 6.)

# flatten the cutouts into 2d list
sofia_h = sofia_cutouts.shape[1]
training_sofia_cutouts = sofia_cutouts.reshape(-1, sofia_h**2)
# process the sofia cutouts
processed_sofia = processData(training_sofia_cutouts, 4.25)


# %%


# split up the training data
spits_train, spits_test, sofia_train, sofia_test = train_test_split(
    processed_spits, processed_sofia
)

## train the model on just the proccessed_spits and the corresponding sofia cutouts
knn.fit(spits_train, sofia_train)


# %%
# predict 
pred = knn.predict(spits_test)
pred = pred.reshape(-1, CUTOUT_SIZE, CUTOUT_SIZE)


# %%
## lets plot these results
plot_gallery(pred, CUTOUT_SIZE, CUTOUT_SIZE)
# %%
## lets plot the originals 
plot_gallery(sofia_cutouts, CUTOUT_SIZE, CUTOUT_SIZE)

# %%
## lets plot the originals 
plot_gallery(spits_withSofiaSources_cutouts, CUTOUT_SIZE, CUTOUT_SIZE)

## WOW
## THESE look awful


# %%
# lets combine these cutouts with the rest of the training data
# the old cutouts
old_masked = mask_sources(old_training_data, SIGMA)
old_input = maskBackground(old_masked)

old_output = old_training_data

#combine the old data with the new training sets
all_input = np.append(old_input, processed_spits)
all_input = old_input.reshape(-1, 2500)

all_output = np.append(old_output, processed_sofia)
all_output = old_output.reshape(-1, 2500)

# split the data up
spits_train, spits_test, sofia_train, sofia_test = train_test_split(
   all_output, all_input 
)



# %%
# train the model
knn.set_params(n_neighbors=2)
knn.fit(spits_train, sofia_train)

all_pred = knn.predict(spits_test)
print(f'{knn.score(spits_test, sofia_test) = }')
# %%

plot_gallery(all_pred, 50,50, 8, 4)

# %%
import time
current_time = time.strftime("day-%d-%m_time-%H-%M")
# with open(f'./{current_time}_training_data.jbl', 'wb') as f:
    # joblib.dump(all_output, f)

# current_time = time.strftime("day-%m-%d_time-%H-%M")
# with open(f'./{current_time}_testing_data.jbl', 'wb') as f:
#     joblib.dump(all_input, f)