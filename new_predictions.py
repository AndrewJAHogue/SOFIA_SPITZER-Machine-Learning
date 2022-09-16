# %%
# imports
import numpy as np
from handy_dandy import rms

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import joblib

from spits_sofia_training import CUTOUT_SIZE

def rms(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse) 



training_data = joblib.load('./12-09_13-25_training_data.jbl')
testing_data = joblib.load('./day-12-09_time-13-28_testing_data.jbl')

# %%
# load new region to try and predict from
from ajh_utils.computer_path import Star_Datasets

spits_data = Star_Datasets.get_spits_data()
sofia_data = Star_Datasets.get_sofia_data()

from astropy.nddata import Cutout2D
input_pred = Cutout2D(spits_data, ( 373, 5122 ), (50, 50)).data
plt.subplot(121)
plt.imshow(input_pred)

output_pred_test = Cutout2D(sofia_data, ( 373, 5122 ), (50, 50)).data
plt.subplot(122)
plt.imshow(output_pred_test)


# %%
# split the data
input_train, input_test, output_train, output_test = train_test_split(
    training_data, testing_data, test_size=0.2
    )

rcv = RidgeCV()
rcv.fit(input_train, output_train)

# %%
# input data to predict off of needs to be proccessed
CUTOUT_SIZE = 50
def maskBackground(input_data, CUTOUT_SIZE=CUTOUT_SIZE, sigma=6.0):
    from astropy.stats import SigmaClip
    from photutils.background import Background2D, MedianBackground
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

processed = maskBackground(input_pred, 50, 1.9)
plt.imshow(processed.reshape(50,50))

# %%
# predict new results
input_pred = input_pred.reshape(1, -1)

prediction = rcv.predict(processed.reshape(1,-1))
print( f'RMS error = {rms(prediction, output_pred_test.reshape(1,-1))  }' )

prediction = prediction.reshape(50, 50)

plt.imshow(prediction)

# %%
# look at some linecuts
from ajh_utils.lineplots import SingleLinePlot, GetNthRow

# SingleLinePlot(25, 25, data=prediction.reshape(50,50))
# SingleLinePlot(25, 25, data=output_pred_test)


x, y_pred = GetNthRow(prediction, 25)
x, y_sofia = GetNthRow(output_pred_test, 25)
x, y_spits = GetNthRow(processed.reshape(50,50), 25)

plt.plot(x, y_spits, label='Spitzer')
plt.plot(x, y_sofia, label='Sofia')
plt.plot(x, y_pred,  label='Prediction')
plt.plot(x, y_pred / 10,  label='Prediction / 10')
plt.legend()
# the prediction is wayyy to big 

# %%
# lets add this last cutout to the training set
new_training_data = np.append(training_data, input_pred)
new_training_data = new_training_data.reshape(-1, 2500)

new_testing_data = np.append(testing_data, output_pred_test)
new_testing_data = new_testing_data.reshape(-1, 2500)


# %%
rcv.fit(new_training_data, new_testing_data)

pred_test = rcv.predict(input_pred)
plt.imshow(pred_test.reshape(50,50))