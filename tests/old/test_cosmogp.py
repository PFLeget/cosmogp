""" Test between cosmogp and the version used in sugar."""

import numpy as np
import pylab as plt
import cosmogp
import sugar_gp
from test_gp_algorithm import generate_data_test


class compare_sugar_cosmogp:

    def __init__(self,number_data,number_point):

        """
        Comparaison between sugar gp and cosmogp.

        A lot of things changed since sugar gp, so 
        I want to check which is the best option. 

        number_data : int, number of data to generate
        
        number_point : int, number of point per data
        """

        assert type(number_data) is int, 'number_data should be integer'
        assert type(number_point) is int, 'number_point should be integer'
        
        self.x, self.y, kernel,det_kernel = generate_data_test(number_data,number_point,
                                                               kernel_amplitude=1.,correlation_length=1.,
                                                               white_noise=0,seed=1)

        del kernel
        del det_kernel

        self.hyperparameter_cosmogp = None
        self.hyperparameter_sugar = None

        self.gp_cosmogp = None
        self.gp_sugar = None

        self.bp_cosmogp = None
        self.bp_sugar = None
        

    def run_cosmogp(self):

        """
        Run cosmogp test.

        search of hyperparameters
        construction of the pull 
        """
        gp = cosmogp.gaussian_process_nobject(self.y,self.x)
        gp.find_hyperparameters(hyperparameter_guess=[1.,1.],svd_method=True)
        self.hyperparameter_cosmogp = gp.hyperparameters
        self.gp_cosmogp = gp

        bp = cosmogp.build_pull(self.y,self.x,gp.hyperparameters)
        bp.compute_pull(svd_method=True)
        self.bp_cosmogp = bp
        bp.plot_result()
        
        
    def run_sugar_gp(self):

        """
        Run sugar_gp test.

        search of hyperparameters
        construction of the pull 
        """

        gp = sugar_gp.find_global_hyperparameters(self.y,np.zeros_like(self.y),self.x,
                                                  self.x[0],np.zeros_like(self.x[0]))
        gp.find_hyperparameters(sigma_guess=1.,l_guess=1.)
        self.hyperparameter_sugar = gp.hyperparameters
        self.gp_sugar = gp
        
        bp = sugar_gp.build_pull(self.y,np.zeros_like(self.y),self.x,
                                 self.x[0],np.zeros(len(self.x[0])),gp.hyperparameters['sigma'],gp.hyperparameters['l'])
        bp.compute_pull(diFF=None)
        self.bp_sugar = bp
        bp.plot_result()
        


if __name__=='__main__':


    #csc = compare_sugar_cosmogp(20,10)
    #csc.run_cosmogp()
    #csc.run_sugar_gp()

    import cPickle
    #dico = cPickle.load(open('../../sugar/sugar/Prediction_Bin_42.pkl'))
    dico = cPickle.load(open('../../sugar/sugar/Gaussian_ProcessPrediction_Bin_42.pkl'))
    #dic['mean_time']
    #dic['y_time']
    #dic['y']
    #dic['diff']
    #dic['y_err']
    #dic'mean']

    import copy

    dic = copy.deepcopy(dico)
                                                                                            
    gpc = cosmogp.gaussian_process_nobject(dic['y'],dic['y_time'],y_err=dic['y_err'],diff=dic['diff'],
                                           Mean_Y=dic['mean'], Time_mean=dic['mean_time'],)
    gpc.nugget = 0.03
    gpc.find_hyperparameters(hyperparameter_guess=[0.5,8.],svd_method=True)

    
    gp = sugar_gp.find_global_hyperparameters(dic['y'],dic['y_err'],dic['y_time'],
                                              dic['mean_time'],dic['mean'])
    gp.substract_Mean(diff=dic['diff'])
    gp.find_hyperparameters(sigma_guess=0.5,l_guess=8.)

    dic = copy.deepcopy(dico)

    bp = sugar_gp.build_pull(dic['y'],dic['y_err'],dic['y_time'],dic['mean_time'],dic['mean'],gp.hyperparameters['sigma'],gp.hyperparameters['l'])
    bp.compute_pull(diFF=dic['diff'])
    #bp.compute_pull(diFF=None)

    dic = copy.deepcopy(dico)

    bpc = cosmogp.build_pull(dic['y'], dic['y_time'],[gp.hyperparameters['sigma'],gp.hyperparameters['l']],
                             nugget=0.03, y_err=dic['y_err'], y_mean=dic['mean'], x_axis_mean=dic['mean_time'], kernel='RBF1D')
    bpc.compute_pull(diff=dic['diff'])
