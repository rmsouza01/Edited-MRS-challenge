import numpy as np
import math


class TransientMaker:

    def __init__(self,fids=None,t=None, transients=40):

        if fids is None or t is None:
            raise Exception("both fids and t must be provided")
        
        self.ground_truth_fids=fids
        self.fids = np.expand_dims(self.ground_truth_fids,axis=3).copy()
        self.fids = np.repeat(self.fids,transients,axis=3)
        self.t=t
    

    # add amplitude normally generated amplitude noise to fids data
    def add_random_amplitude_noise(self,noise_level_base=10,noise_level_scan_var=3):
        """
            Add normal amplitude noise to time-domain data
            Parameters:
                - Noise level base: base Applied to all transients & scans
                - Noise level scan var: level of variation between different scans
            Returns: Updates Fids and Specs
        """
        base_noise = noise_level_base*np.ones(self.fids.shape[0])+np.random.uniform(low=-noise_level_scan_var,high=noise_level_scan_var,size=self.fids.shape[0])
        
        #adds real and imaginary noise
        noise_real = np.random.normal(0,base_noise.reshape(-1,1,1,1),size=self.fids.shape)
        noise_imag = 1j*np.random.normal(0,base_noise.reshape(-1,1,1,1),size=self.fids.shape)

        self.fids = self.fids + noise_real + noise_imag
    
    def add_random_frequency_noise(self,noise_level_base=7,noise_level_scan_var=3):
        """
            Adds frequency shift to scans according to normal distribution
            Parameters:
                - Noise level base: base applied to all transients and scans
                - Noise level scan var: level of variation between different scans
            Return: Updates Fids and Specs
        """
        base_noise = noise_level_base*np.ones(self.fids.shape[0])+np.random.uniform(low=-noise_level_scan_var,high=noise_level_scan_var,size=self.fids.shape[0])
        
        #noise = np.random.normal(0,base_noise.reshape(-1,1,1,1),size=(self.fids.shape[0],1,self.fids.shape[2],self.fids.shape[3]))
        noise = np.random.uniform(-base_noise.reshape(-1,1,1,1),base_noise.reshape(-1,1,1,1),size=(self.fids.shape[0],1,self.fids.shape[2],self.fids.shape[3]))

        self.fids = self.fids*np.exp(1j*self.t.reshape(self.t.shape[0],self.t.shape[1],1,1)*noise*2*math.pi)

    def add_random_phase_noise(self,noise_level_base=5,noise_level_scan_var=3):
        """
            Adds phase shift to scans according to normal distribution
            Parameters:
                - Noise level base: base applied to all transients and scans
                - Noise level scan var: level of variation between different scans
            Return: Updates Fids and Specs
        """
        base_noise = noise_level_base*np.ones(self.fids.shape[0])+np.random.uniform(low=-noise_level_scan_var,high=noise_level_scan_var,size=self.fids.shape[0])
        
        #noise = np.random.normal(0,base_noise.reshape(-1,1,1,1),size=(self.fids.shape[0],1,self.fids.shape[2],self.fids.shape[3]))
        noise = np.random.uniform(-base_noise.reshape(-1,1,1,1),base_noise.reshape(-1,1,1,1),size=(self.fids.shape[0],1,self.fids.shape[2],self.fids.shape[3]))

        self.fids = self.fids*np.exp(1j*noise*math.pi/180)

