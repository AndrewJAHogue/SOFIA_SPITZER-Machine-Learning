# %%
# imports
from matplotlib.pyplot import plot, title
import numpy as np
from sklearn.utils.validation import check_random_state

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import joblib

def rms(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse) 



training_data = joblib.load('./12-09_13-25_training_data.jbl')
testing_data = joblib.load('./day-12-09_time-13-28_testing_data.jbl')

# %%
# split the data
input_train, input_test, output_train, output_test = train_test_split(training_data, testing_data, test_size=0.2)

# setup estimators
ESTIMATORS = {
    "Extra trees": ExtraTreesRegressor(
        n_estimators=100, max_features=32, random_state=0
    ),
    "K-nn": KNeighborsRegressor(weights='distance', n_neighbors=17),
    "Linear regression": LinearRegression(),
    "Ridge": RidgeCV(),
}

# fit loop
all_predictions = dict()
for name, est in ESTIMATORS.items():
    est.fit(input_train, output_train)
    all_predictions[name] = est.predict(input_test)

    

from ajh_utils.lineplots import plot_gallery, compare_results

for est in all_predictions.keys():
    print(f'{est} rms = {rms(output_test, all_predictions[est])}')



# %%
##  imagegrid

# setup samples of predictions
c1 = all_predictions['Linear regression'][:4]
c2 = all_predictions['Ridge'][:4]
c3 = all_predictions['K-nn'][:4]
c4 = all_predictions['Extra trees'][:4]

plot_gallery(output_test[:4], 50, 50, 1, 4)
compare_results([c1, c2, c3, c4], 50, 50, 1, 4)


# %%
# r_2 scorings

from sklearn.metrics import r2_score

# r2_score(all_predictions['K-nn'][0], output_test[0])

r2Scores = []
for y, row in enumerate( output_test ):
    score = r2_score(all_predictions['Ridge'][y], row)
    r2Scores.append(score)

mean_score = np.mean(r2Scores)
print(f'Ridge: {mean_score = }')


# %%
from ajh_utils.lineplots import GetNthRow

x, y_input = GetNthRow(output_test[1].reshape(50,50), 25)
x, y_pred = GetNthRow(c3[1].reshape(50,50), 25)

plt.plot(x, y_input, label='Original')
# plt.plot(x, y_pred, label='Predicted')
plt.legend()
