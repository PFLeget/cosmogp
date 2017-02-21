import numpy as N
from cosmogp import Gaussian_process
from scipy.stats import norm as normal



class build_pull:

    def __init__(self,y,y_err,Time,Time_mean,Mean_Y,sigma,L):

        self.y=y
        self.N_sn=len(y)
        self.y_err=y_err
        self.Time=Time
        self.Mean_Y=Mean_Y
        self.Time_mean=Time_mean
        self.sigma=sigma
        self.L=L

    def compute_pull(self,diFF=None):

        if diFF is None:
            diFF=N.zeros(self.N_sn)

        self.pull=[]
        self.PULL=[]

        for sn in range(self.N_sn):
            print '%i/%i'%((sn+1,self.N_sn))
            Pred=N.zeros(len(self.Time[sn]))
            Pred_var=N.zeros(len(self.Time[sn]))
            for t in range(len(self.Time[sn])):
                FILTRE=N.array([True]*len(self.Time[sn]))
                FILTRE[t]=False
                GPP=Gaussian_process([self.y[sn][FILTRE]],[self.y_err[sn][FILTRE]],[self.Time[sn][FILTRE]],self.Time_mean,self.Mean_Y)
                GPP.substract_Mean(diff=[diFF[sn]])
                GPP.hyperparameters.update({'sigma':self.sigma,
                                           'l':self.L})

                GPP.get_prediction(new_binning=self.Time[sn])
                Pred[t]=GPP.Prediction[0][t]
                Pred_var[t]=GPP.covariance_matrix[0][t,t]

            pull=(Pred-self.y[sn])/N.sqrt(self.y_err[sn]**2+Pred_var)
            self.pull.append(pull)
            for t in range(len(self.Time[sn])):
                self.PULL.append(pull[t])


        self.Moyenne_pull,self.ecart_type_pull=normal.fit(self.PULL)


    def plot_result(self,BIN=60,Lambda=None):

        import pylab as P
        P.hist(self.PULL,bins=BIN,normed=True)
        xmin, xmax = P.xlim()
        MAX=max([abs(xmin),abs(xmax)])
        P.xlim(-MAX,MAX)
        xmin, xmax = P.xlim()
        X = N.linspace(xmin, xmax, 100)
        PDF = normal.pdf(X, self.Moyenne_pull, self.ecart_type_pull)
        P.plot(X, PDF, 'r', linewidth=3)
        if Lambda is None:
            title = r"Fit results: $\mu$ = %.2f, $\sigma$ = %.2f" % (self.Moyenne_pull, self.ecart_type_pull)
        else:
            title = r"Fit results ($\lambda = %i \AA$): $\mu$ = %.2f, $\sigma$ = %.2f" %((Lambda,self.Moyenne_pull, self.ecart_type_pull))
        P.title(title)
        P.ylabel('Number of points (normed)')
        P.xlabel('Pull')
        P.show()
