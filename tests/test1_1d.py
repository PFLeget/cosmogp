




class generate_light_curves_SUGAR_MC:

    def __init__(self,SED):
        
        SED=N.loadtxt(SED)
        
        #SED=cPickle.load(open(SED))
        #SED_at_max=cPickle.load(open(SED_at_max))
        
        self.alpha=SED[:,3:6]
        self.M0=SED[:,2]
        self.grey=N.ones(len(self.alpha[:,0]))
        self.Red=SED[:,6]
        
        self.Time=N.linspace(-12,42,19)
        self.X=SED[:,1]
        self.Time_scale=19
        self.wavelength_scale=170
        
    def select_bin(self,Bin):

        self.alpha1_LC=self.alpha[:,0][Bin*self.Time_scale:(Bin+1)*self.Time_scale]
        self.alpha2_LC=self.alpha[:,1][Bin*self.Time_scale:(Bin+1)*self.Time_scale]
        self.alpha3_LC=self.alpha[:,2][Bin*self.Time_scale:(Bin+1)*self.Time_scale]
        
        self.M0_LC=self.M0[Bin*self.Time_scale:(Bin+1)*self.Time_scale]
        
        self.Red_LC=self.Red[Bin*self.Time_scale:(Bin+1)*self.Time_scale]
        self.grey_LC=N.ones(self.Time_scale)
        

    def generate_LC_MC(self,BIN,N_sn):
        
        self.TIME=[]
        self.select_bin(BIN)
        
        self.Supernovae_MC=N.zeros((N_sn,self.Time_scale))
        
        N.random.seed(1)
        
        for sn in range(N_sn):
            
            self.TIME.append(self.Time)
            self.Supernovae_MC[sn]+=self.M0_LC
            
            self.Supernovae_MC[sn]+=N.random.normal(scale=3.5)*self.alpha1_LC
            self.Supernovae_MC[sn]+=N.random.normal(scale=3.)*self.alpha2_LC
            self.Supernovae_MC[sn]+=N.random.normal(scale=1.)*self.alpha3_LC

            self.Supernovae_MC[sn]+=N.random.normal(scale=0.25)*self.Red_LC

            self.Supernovae_MC[sn]+=N.random.normal(scale=0.10)*self.grey_LC

        self.TIME=N.array(self.TIME)
        
        self.N_sn=N_sn
                                                                                                                                                                
    def mixe_time(self):

        Supernovae_MC=[]
        TIME=[]

        for i in range(self.N_sn):
            T=N.linspace(self.Time[0],self.Time[-1],10+i)
        
            Supernovae_MC.append(interpolate_mean(self.TIME[i],self.Supernovae_MC[i],T))
            TIME.append(T)
            
            
        self.Supernovae_MC=Supernovae_MC
        self.TIME=TIME

    def add_noise(self,sigma=0.15):
            
        N.random.seed(1)
        
        self.Y_err=[]
        
        for sn in range(self.N_sn) :
            self.Y_err.append(N.zeros(len(self.Supernovae_MC[sn])))
            for i in range(len(self.Supernovae_MC[sn])):
                noise = N.random.normal(scale=sigma)
                #noise=0
                self.Supernovae_MC[sn][i] += noise
                self.Y_err[sn][i]=N.sqrt(noise**2)
                                                                                                                                            


class generate_light_curves_hyper_parameter(generate_light_curves_SUGAR_MC):

    def __init__(self,SED,sigma=2,l=6):
        
        generate_light_curves_SUGAR_MC.__init__(self,SED)
        self.sigma=sigma
        self.l=l
        self.K=RBF_kernel_1D(self.Time ,self.sigma,self.l,0.)


    def generate_supernovae(self,Bin,N_sn):
        self.generate_LC_MC(Bin,1)
        self.Supernovae_MC=N.zeros((N_sn,len(self.Time)))
        TIME=[]
        for sn in range(N_sn):
            self.Supernovae_MC[sn]=N.random.multivariate_normal(self.M0_LC,self.K)
            TIME.append(self.Time)
            
        self.TIME=TIME
        self.Y_err=N.zeros(N.shape(self.Supernovae_MC))
        self.N_sn=N_sn



class Build_light_curves_from_SNF_data:

    def __init__(self,Repertoire_in,Bin,sn_list,Number_bin_wavelength=190):
        
        self.sn_name=N.loadtxt(sn_list,dtype='string')
        #self.sn_name=self.sn_name[:,8]
        
        self.Repertoire_in=Repertoire_in
        self.Bin=Bin
        self.Number_bin_wavelength=Number_bin_wavelength
        self.Y=[]
        self.Y_err=[]
        self.TIME=[]
        self.wavelength=[]


    def build_data(self):

        for i,sn in enumerate(self.sn_name):
            data=N.loadtxt(self.Repertoire_in+sn)
            Number_points=len(data[:,0])/self.Number_bin_wavelength
            Y=N.zeros(Number_points)
            Y_err=N.zeros(Number_points)
            Time=N.zeros(Number_points)
            wavelength=N.zeros(Number_points)
            
            for j in range(Number_points):
                Y[j]=data[(self.Number_bin_wavelength*j)+self.Bin,2]
                Y_err[j]=data[(self.Number_bin_wavelength*j)+self.Bin,3]
                Time[j]=data[(self.Number_bin_wavelength*j)+self.Bin,0]
                wavelength[j]=data[(self.Number_bin_wavelength*j)+self.Bin,1]
                
            self.Y.append(Y)
            self.Y_err.append(Y_err)
            self.TIME.append(Time)
            self.wavelength.append(wavelength)
        
                                                                                    
    def build_mean(self,Mean_file):

        data=N.loadtxt(Mean_file)
        Number_points=len(data[:,0])/self.Number_bin_wavelength
        Y=N.zeros(Number_points)
        Time=N.zeros(Number_points)
        wavelength=N.zeros(Number_points)
        self.Mean_file=Mean_file

        Y=data[Number_points*self.Bin:(Number_points*self.Bin)+Number_points,2]
        Time=data[Number_points*self.Bin:(Number_points*self.Bin)+Number_points,0]
        wavelength=data[Number_points*self.Bin:(Number_points*self.Bin)+Number_points,1]
        
        self.Time_Mean=Time
        self.Mean=Y
        self.W_mean=wavelength
                                                                       


    def build_difference_mean(self):

        self.difference=N.zeros(len(self.sn_name))
        data=N.loadtxt(self.Mean_file)

        delta_mean=0
        delta_lambda=0
    
        for i in range(len(data[:,0])):
            if data[:,0][0]==data[:,0][i]:
                delta_lambda+=1
            if data[:,1][0]==data[:,1][i]:
                delta_mean+=1
                                                                                    
        for i,sn in enumerate(self.sn_name):
            print sn
            Phase=self.TIME[i]
            Time=self.Time_Mean
            DELTA=len(Phase)
            self.Mean_new_binning=N.zeros(delta_lambda*len(Phase))
            
            for Bin in range(delta_lambda):
                self.Mean_new_binning[Bin*DELTA:(Bin+1)*DELTA]=interpolate_mean(self.Time_Mean,data[:,2][Bin*delta_mean:(Bin+1)*delta_mean],Phase)

            reorder = N.arange(delta_lambda*DELTA).reshape(delta_lambda, DELTA).T.reshape(-1)
            self.Mean_new_binning=self.Mean_new_binning[reorder]
            
            Data=N.loadtxt(self.Repertoire_in+sn)
            self.difference[i]=N.mean(Data[:,2]-self.Mean_new_binning)
            



    def control_plot(self):

        P.figure()
        for i in range(len(self.sn_name)):
            P.scatter(self.TIME[i],self.Y[i],c='b')
            P.errorbar(self.TIME[i],self.Y[i], linestyle='', yerr=self.Y_err[i],ecolor='red',alpha=0.9,marker='.',zorder=0)
        P.plot(self.Time_Mean,self.Mean)
        P.gca().invert_yaxis()
        P.show()
            
                                                                                                                                                                                                                     
