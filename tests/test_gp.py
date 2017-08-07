"""Pedagogic code to see how works gp."""

import numpy as np
import pylab as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import cPickle
import cosmogp
import copy

def gp(x,y,y_err,xnew,sigma,l,mean=None,time_mean=None,diff=None):
    """
    Compute interpolation and error using gaussian process.
    """
    gp = cosmogp.gaussian_process(y, x, y_err=y_err,
                                  Mean_Y=mean, Time_mean=time_mean,
                                  kernel='RBF1D')

    gp.hyperparameters = [sigma,l]

    if mean is not None and diff is not None:
        gp.substract_Mean(diff=diff)

    gp.get_prediction(new_binning=xnew,svd_method=False)

    return gp.Prediction[0], np.sqrt(np.diag(gp.covariance_matrix[0]))


def gaussian_process_interactif_snia():
    """
    interactive plot for SNIa interpolation.
    """
    sig0 = 0.27
    l0 = 8.77

    dic = cPickle.load(open('data_snia.pkl'))
    snia_y = dic['y']
    snia_y_err = dic['y_err']
    snia_time = dic['time']
    snia_mean = dic['mean']
    snia_mean_time = dic['mean_time']
    snia_diff = dic['diff']

    predict, std = gp(snia_time, snia_y, np.sqrt(snia_y_err**2+0.03**2),
                      np.linspace(-12,42,60),sig0,l0,mean=snia_mean,time_mean=snia_mean_time,
                      diff=[snia_diff])

    fig = plt.figure(figsize=(12,12))
    ax = fig.gca()
    plt.subplot()
    plt.subplots_adjust(left=0.25, right=0.99, bottom=0.25)

    plt.scatter(snia_time,snia_y-snia_diff,c='r',s=50)
    plt.errorbar(snia_time,snia_y-snia_diff,yerr=snia_y_err,linestyle='',elinewidth=3,ecolor='r')

    plt.plot(np.linspace(-12,42,60), predict,'b',label='Prediction')
    plt.fill_between(np.linspace(-12,42,60), predict - std,
                     predict + std, color='b', alpha=0.7)

    plt.xlabel('Time (days)',fontsize='20')
    plt.ylabel('-2.5 log(flux) + cst.',fontsize='20')
    plt.ylim(-20,-15)
    plt.xlim(-12,42)
    plt.gca().invert_yaxis()

    axsigma = plt.axes([0.25, 0.1, 0.65, 0.03])
    axlength = plt.axes([0.25, 0.15, 0.65, 0.03])
    sigma = Slider(axsigma, r'kernel amplitude ($\sigma$)', 0.1, 1., valinit=sig0)
    length = Slider(axlength, r'correlation length ($l$)', 1., 20.0, valinit=l0)

    def update(val):

        ax.cla()
        plt.subplot(1,1,1)
        predict, std = gp(snia_time, snia_y, np.sqrt(snia_y_err**2+0.03**2),
                          np.linspace(-12,42,60),sigma.val,length.val,mean=snia_mean,time_mean=snia_mean_time,
                          diff=[snia_diff])

        plt.scatter(snia_time,snia_y-snia_diff,c='r',s=50)
        plt.errorbar(snia_time,snia_y-snia_diff,yerr=snia_y_err,linestyle='',elinewidth=3,ecolor='r')

        plt.plot(np.linspace(-12,42,60), predict ,'b',label='Prediction')
        plt.fill_between(np.linspace(-12,42,60), predict - std,
                         predict + std, color='b', alpha=0.7)

        plt.xlabel('Time (days)',fontsize='20')
        plt.ylabel('-2.5 log(flux) + cst.',fontsize='20')
        plt.ylim(-20,-15)
        plt.xlim(-12,42)
        plt.gca().invert_yaxis()


    sigma.on_changed(update)
    length.on_changed(update)

    plt.show()


def gaussian_process_interactif_gp():
    """
    interactive plot for gp interpolation.

    data are generated from a rbf kernel and
    are fit with the same kernel. its possible to
    change hyperparameter value to see how looks like
    interpolation.
    """
    n_object = 100 # number of object to fit
    total_point = 0
    a = 0.2

    all_grid = []
    all_y = []
    all_y_err = []
    Y = []

    np.random.seed(1)

    for i in range(n_object): # generate object not on the same grid of observation
        n_point = int(np.random.uniform(60,60)) # random number of points for one object
        total_point += n_point
        grid = np.linspace(-10,40,n_point) # fixed grid where gp will be generated
        k = cosmogp.rbf_kernel_1d(grid,np.array([0.5,2])) # build the kernel with fixed hyperparameter

        y = np.random.multivariate_normal(np.zeros_like(grid), k+((a*a)*np.eye(len(k)))) # generation of gaussian process
        all_grid.append(grid)
        all_y.append(y)
        for i in range(len(y)):
            Y.append(y[i])
            all_y_err.append(a*np.ones(len(k)))

    sig0 = 0.57
    l0 = 2.28

    snia_y = all_y[0]
    snia_y_err = all_y_err[0]
    snia_time = all_grid[0]
    snia_mean = np.zeros_like(np.linspace(-20,50,60))
    snia_mean_time = np.linspace(-20,50,60)
    snia_diff = None

    predict, std = gp(snia_time, snia_y, snia_y_err,
                      np.linspace(-20,50,100),sig0,l0,mean=None,time_mean=None,
                      diff=snia_diff)

    fig = plt.figure(figsize=(12,12))
    ax = fig.gca()
    plt.subplot()
    plt.subplots_adjust(left=0.25, right=0.99, bottom=0.25)

    plt.scatter(snia_time,snia_y,c='r',s=50)
    plt.errorbar(snia_time,snia_y,yerr=snia_y_err,linestyle='',elinewidth=3,ecolor='r')

    plt.plot(np.linspace(-20,50,100), predict,'b',label='Prediction')
    plt.fill_between(np.linspace(-20,50,100), predict - std,
                     predict + std, color='b', alpha=0.7)

    plt.xlabel('X',fontsize='20')
    plt.ylabel('Y',fontsize='20')
    plt.ylim(-2,2)
    plt.xlim(-20,50)

    axsigma = plt.axes([0.25, 0.1, 0.65, 0.03])
    axlength = plt.axes([0.25, 0.15, 0.65, 0.03])
    sigma = Slider(axsigma, r'kernel amplitude ($\sigma$)', 0.01, 2., valinit=sig0)
    length = Slider(axlength, r'correlation length ($l$)', 0.01, 20.0, valinit=l0)

    def update(val):

        ax.cla()
        plt.subplot(1,1,1)
        predict, std = gp(snia_time, snia_y, snia_y_err,
                          np.linspace(-20,50,100),sigma.val,length.val,mean=None,time_mean=None,
                          diff=snia_diff)

        plt.scatter(snia_time,snia_y,c='r',s=50)
        plt.errorbar(snia_time,snia_y,yerr=snia_y_err,linestyle='',elinewidth=3,ecolor='r')

        plt.plot(np.linspace(-20,50,100), predict ,'b',label='Prediction')
        plt.fill_between(np.linspace(-20,50,100), predict - std,
                         predict + std, color='b', alpha=0.7)

        plt.xlabel('X',fontsize='20')
        plt.ylabel('Y',fontsize='20')
        plt.ylim(-2,2)
        plt.xlim(-20,50)


    sigma.on_changed(update)
    length.on_changed(update)
    plt.show()


if __name__=='__main__':

    gaussian_process_interactif_gp()
