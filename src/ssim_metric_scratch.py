import numpy as np

def ssim_scratch(y_true, y_pred):
    x = y_true
    y = y_pred
    ave_x = np.mean(x)
    ave_y = np.mean(y)
    var_x = np.var(x)
    var_y = np.var(y)
    covariance = np.mean(x*y) - ave_x*ave_y
    c1 = 0.01**4
    c2 = 0.03**4
    ssim = (2*ave_y*ave_x+c1)*(2*covariance+c2)
    ssim = ssim/((pow(ave_x,2)+pow(ave_y,2)+c1) * (var_x+var_y+c2))
    return (ssim)