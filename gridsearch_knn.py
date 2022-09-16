# %%
# imports
from matplotlib.pyplot import plot, title
import numpy as np
from sklearn.utils.validation import check_random_state

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import joblib

training_data = joblib.load('./12-09_13-25_training_data.jbl')
testing_data = joblib.load('./day-12-09_time-13-28_testing_data.jbl')


# %%
# grid search for knn parameters
param_grid = [ { 'n_neighbors': list( range(1, 100) ), 'weights': ['uniform', 'distance']} ]

knn = KNeighborsRegressor(n_jobs=-1)
grid = GridSearchCV(knn, param_grid)

grid_search = grid.fit(training_data, testing_data)
