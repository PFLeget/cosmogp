from test_gp_algorithm import generate_data_test, cosmogp_gp
import numpy as np


x, y, kernel, det_kernel = generate_data_test(10,7,
                                              kernel_amplitude=1.,correlation_length=1.,
                                              white_noise=0,seed=1)



ynew = np.zeros_like(y)
resids = np.zeros_like(y)
stds = np.zeros_like(y)


for i in range(len(y)):
    ynew[i], resids[i], stds[i], time_excution = cosmogp_gp(x[i], y[i], xpredict=None, kernel='rbf1d',
                                                            hyperparameter_default=[1.,1.],
                                                            search_hyperparameter=False)
