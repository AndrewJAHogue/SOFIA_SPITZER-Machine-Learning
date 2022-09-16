

def get_region_cutouts( x, y, h, w,):
    """Loads the SOFIA and Spitzer datasets and creates a cutout centered at (x,y) of size (h, w) 

    Args:
        x (int): x coordinate to center cutout on
        y (int): y coordinate to center cutout on
        h (int): heighth of returned cutouts
        w (int): width of returned cutouts

    Returns:
        Returns two cutouts of equal size, centered on the same (x, y). Returns a cutout from the SOFIA dataset and a cutout from the Spitzer dataset; tuple-like
    """    
    from ajh_utils import computer_path, lineplots
    from astropy.nddata import Cutout2D

    spits_data = computer_path.Star_Datasets.get_spits_data()
    sofia_data = computer_path.Star_Datasets.get_sofia_data()

    spits_cutout = Cutout2D(spits_data, (x, y), (h, w)).data
    sofia_cutout = Cutout2D(sofia_data, (x, y), (h, w)).data

    return sofia_cutout, spits_cutout

    
def rms(y_true, y_pred):
    from numpy import sqrt
    """Calculate rms error of predictions

    Args:
        y_true (array-like): predictions
        y_pred (array-like): testing data

    Returns:
        ndarray  
    """    
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_true, y_pred)
    return sqrt(mse) 

