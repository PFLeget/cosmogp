import numpy as N
import cosmogp
from scipy.stats import norm as normal



class build_pull:

    def __init__(self,y,y_err,Time,Time_mean,Mean_Y,hyperparameters,kernel='RBF1D'):

        self.y=y
        self.N_sn=len(y)
        self.y_err=y_err
        self.Time=Time
        self.Mean_Y=Mean_Y
        self.Time_mean=Time_mean
        self.hyperparameters=hyperparameters
        self.kernel=kernel
        
    def compute_pull(self,diFF=None):



        self.pull=[]
        self.PULL=[]
        self.RES=[]
        
        for sn in range(self.N_sn):
            #print '%i/%i'%((sn+1,self.N_sn))
            Pred=N.zeros(len(self.y[sn]))
            Pred_var=N.zeros(len(self.y[sn]))
            for t in range(len(self.y[sn])):
                FILTRE=N.array([True]*len(self.y[sn]))
                FILTRE[t]=False
                
                GPP=cosmogp.gaussian_process(self.y[sn][FILTRE],self.Time[sn][FILTRE],y_err=self.y_err[sn][FILTRE],Mean_Y=self.Mean_Y,Time_mean=self.Time_mean,kernel=self.kernel)

                if diFF is None:
                    GPP.substract_Mean(diff=None)
                else:
                    GPP.substract_Mean(diff=[diFF[sn]])
                
                GPP.hyperparameters=self.hyperparameters
                GPP.get_prediction(new_binning=self.Time[sn])
                Pred[t]=GPP.Prediction[0][t]
                Pred_var[t]=abs(GPP.covariance_matrix[0][t,t])
                
            self.Pred_var=Pred_var
            pull=(Pred-self.y[sn])/N.sqrt(self.y_err[sn]**2+Pred_var)
            res=Pred-self.y[sn]
            self.pull.append(pull)
            for t in range(len(self.y[sn])):
                self.PULL.append(pull[t])
                self.RES.append(res[t])

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
