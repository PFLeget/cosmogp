"""Pedagogic code to see how works gp."""

import numpy as np
import pylab as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import cPickle
import cosmogp
import copy
import treecorr
import scipy.optimize as op

def add_subplot_axes(ax,rect,axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax
                                                                        

def gp(x,y,y_err,xnew,sigma,l,mean=None,time_mean=None,diff=None):
    """
    Compute interpolation and error using gaussian process.
    """
    gp = cosmogp.gaussian_process(y, x, y_err=y_err,
                                  Mean_Y=mean, diff = diff, Time_mean=time_mean,
                                  kernel='RBF1D')

    gp.hyperparameters = [sigma,l]

    #if mean is not None and diff is not None:
    #    gp.substract_Mean(diff=diff)

    gp.get_prediction(new_binning=xnew,svd_method=False)

    return gp.Prediction[0], np.sqrt(np.diag(gp.covariance_matrix[0])), gp.kernel_matrix[0]


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

def gaussian_process_movie_gp(total=600,sub=100):
    """
    interactive plot for gp interpolation.

    data are generated from a rbf kernel and
    are fit with the same kernel. its possible to
    change hyperparameter value to see how looks like
    interpolation.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.animation as manimation

    #plt.style.use('dark_background')
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='gp_interp_example', artist='Matplotlib',
                    comment='interp_random_data')
    writer = FFMpegWriter(fps=24, metadata=metadata, bitrate=6000)
    Name_mp4="gp_interp_example.mp4"
                            
                            
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


    fig = plt.figure(figsize=(8,8))
    ax = fig.gca()
    P1=fig.add_subplot(1,1,1)
    plt.subplots_adjust(left=0.12, right=0.98, bottom=0.09,top=0.99)

    SIGMA = np.ones(total)*0.5
    SIGMA[:sub] = np.linspace(0.5,1,sub)
    SIGMA[sub:2*sub] = np.linspace(1,0.01,sub)
    SIGMA[2*sub:3*sub] = np.linspace(0.01,0.5,sub)
    
    
    L = np.ones_like(SIGMA)*2.
    L[4*sub:5*sub] = np.linspace(2,10,sub)
    L[5*sub:6*sub] = np.linspace(10,0.01,sub)
    L[6*sub:7*sub] = np.linspace(0.01,2,sub)

    subylim = np.ones_like(SIGMA)
    subylim[:3*sub] = 1.
    subylim[int(3.5*sub):4*sub] = np.linspace(1.,0.27,sub/2)
    subylim[4*sub:7*sub] = 0.27
    
    with writer.saving(fig, Name_mp4, 250):
        for i in range(len(L)):
            print i 
            if i!=0:
                ax.cla()
                plt.subplot(1,1,1)

            predict, std, Kernel = gp(snia_time, snia_y, snia_y_err,
                                      np.linspace(-20,50,100),SIGMA[i],L[i],mean=None,time_mean=None,
                                      diff=snia_diff)

            plt.scatter(snia_time,snia_y,c='k',s=70,label='data',lw=0,zorder=0)
            plt.errorbar(snia_time,snia_y,yerr=snia_y_err,linestyle='',elinewidth=3,ecolor='k',zorder=0)

            plt.plot(np.linspace(-20,50,100), predict ,'r',lw=4,label='gp interpolation')
            plt.fill_between(np.linspace(-20,50,100), predict - std,
                             predict + std, color='r', alpha=0.7)

            plt.xlabel('X',fontsize='20')
            plt.ylabel('Y',fontsize='20')
            plt.ylim(-2,2)
            plt.xlim(-20,50)
            if i<4*sub:
                color_sig = 'red'
                color_l = 'k'
                size_sig = 20
                size_l = 16
            else:
                color_sig = 'k'
                color_l = 'red'
                size_sig = 16
                size_l = 20
            
            plt.text(-15,-1.25,r'kernel amplitude ($\sigma=%.2f$)'%(SIGMA[i]),fontsize=size_sig,color=color_sig)
            plt.text(-15,-1.5,r'correlation length ($l=%.2f$)'%(L[i]),fontsize=size_l,color=color_l)
            plt.legend()

            subpos = [0.73,0.08,0.25,0.25]
            subax1 = add_subplot_axes(P1,subpos)
            #subax1.patch.set_facecolor('black')
            dist_exp = np.linspace(0,10,100)
            corr = SIGMA[i]**2 * np.exp(-0.5*(dist_exp/L[i])**2)
            subax1.plot(dist_exp,corr,'k',lw=4)
            subax1.set_xlim(0,10)
            subax1.set_ylim(0,subylim[i])
            subax1.set_xlabel('X',fontsize=16)
            subax1.set_ylabel(r'$\xi$(X)',fontsize=16)
            #subax1.imshow(Kernel,vmin=0.,vmax=2.,interpolation='nearest')
            #subax1.text(47,12,r'$\xi$',fontsize=18)

            writer.grab_frame()

    plt.show()


def plot_interactif_2D_GP():

    n_point = 300
    a = 0.2 # known noise

    x1 = np.random.uniform(-10,10,n_point) # x1 coordinate
    x2 = np.random.uniform(-10,10,n_point) # x2 coordinate

    X = np.zeros((n_point,2)) # fixed grid where gp will be generated

    X[:,0] = x1
    X[:,1] = x2

    K = cosmogp.rbf_kernel_2d(X,np.array([0.5,1.,1.,0.])) # build the kernel with fixed hyperparameter

    Y = np.random.multivariate_normal(np.zeros(n_point), K + ((a*a)*np.eye(len(K)))) # generation of gaussian process
    print 'FOR PF: ', np.std(Y) 
    Y_err = np.ones(np.shape(Y))*a

    gp = cosmogp.gaussian_process(Y,X,y_err=Y_err,kernel='RBF2D')
    gp.hyperparameters = np.array([np.sqrt(0.5),1.,1.,0.])

    xy = np.linspace(-10,10,50)
    XX,YY = np.meshgrid(xy,xy)
    xx = XX.ravel()
    yy = YY.ravel()
    Cooord = np.array([xx,yy]).T
    
    gp.get_prediction(new_binning=Cooord,svd_method=False)

    fig = plt.figure(figsize=(12,7))
    ax = fig.gca()
    plt.subplots_adjust(left=0.1, right=0.99, bottom=0.25)
    plt.subplot(1,2,1)
    
    #plt.pcolor(XX,YY,gp.Prediction[0].reshape(np.shape(XX)))
    plt.imshow(gp.Prediction[0].reshape(np.shape(XX)),vmin= -2,vmax=2)
    plt.colorbar()
    #plt.ylim(-10,10)
    #plt.xlim(-10,10)

    print 'FOR PF 2: ', np.std(gp.Prediction[0])
    plt.subplot(1,2,2)
    

    size_x = np.max(xx) - np.min(yy)
    size_y = np.max(yy) - np.min(xx)
    rho = float(len(xx)) / (size_x * size_y)
    MIN = np.sqrt(1./rho)
    MAX = np.sqrt(size_x**2 + size_y**2)/2.
                    
    xx = np.random.uniform(-10,10,50**2)
    yy = np.random.uniform(-10,10,50**2)
    Coord = np.array([xx,yy]).T    
    gp.get_prediction(new_binning=Coord,svd_method=False)

    cat = treecorr.Catalog(x=Coord[:,0], y=Coord[:,1], k=(gp.Prediction[0]-np.mean(gp.Prediction[0])))
    kk = treecorr.KKCorrelation(min_sep=MIN, max_sep=MAX, nbins=40)
    kk.process(cat)

    distance = np.exp(kk.logr)
    
    def kernel(sigma,L):
        pcf = sigma**2 * np.exp(-0.5*((distance/L)**2))
        return pcf

    def chi2_exp(param):
        residual = kk.xi - kernel(param[0],param[1])
        return np.sum(residual**2)

    best_param_exp = op.fmin(chi2_exp,[1e-2,0.1],disp=False)
    xi_plus_best_exp = kernel(best_param_exp[0],best_param_exp[1])
    plt.scatter(distance,kk.xi)
    plt.plot(distance,xi_plus_best_exp)
    plt.xscale('log')

    axsigma = plt.axes([0.25, 0.1, 0.65, 0.03])
    axlength = plt.axes([0.25, 0.15, 0.65, 0.03])
    sigma = Slider(axsigma, r'kernel amplitude ($\sigma$)', 0.01, 2., valinit=np.sqrt(0.5))
    length = Slider(axlength, r'correlation length ($l$)', 0.01, 20.0, valinit=1.)
                    
    def update(val):

        ax.cla()
        plt.subplot(1,2,1)

        gp.hyperparameters = np.array([np.sqrt(sigma.val),length.val,length.val,0.])
        gp.get_prediction(new_binning=Cooord,svd_method=False)
        
        plt.imshow(gp.Prediction[0].reshape(np.shape(XX)),vmin= -2,vmax=2)

        plt.subplot(1,2,2)
        plt.cla()
        gp.get_prediction(new_binning=Coord,svd_method=False)

        cat = treecorr.Catalog(x=Coord[:,0], y=Coord[:,1], k=(gp.Prediction[0]-np.mean(gp.Prediction[0])))
        kk = treecorr.KKCorrelation(min_sep=MIN, max_sep=MAX, nbins=40)
        kk.process(cat)


        distance = np.exp(kk.logr)
    
        def kernel(sigma,L):
            pcf = sigma**2 * np.exp(-0.5*((distance/L)**2))
            return pcf
        
        def chi2_exp(param):
            residual = kk.xi - kernel(param[0],param[1])
            return np.sum(residual**2)
        
               
        best_param_exp = op.fmin(chi2_exp,[1e-2,0.1],disp=False)
        xi_plus_best_exp = kernel(best_param_exp[0],best_param_exp[1])
        
        plt.scatter(distance,kk.xi)
        plt.plot(distance,xi_plus_best_exp)
        plt.xscale('log')
        
    sigma.on_changed(update)
    length.on_changed(update)
    plt.show()
    
if __name__=='__main__':

    gaussian_process_movie_gp(total=700,sub=100)
    
    #gaussian_process_interactif_gp()
    #gaussian_process_interactif_snia()

    #plot_interactif_2D_GP()
