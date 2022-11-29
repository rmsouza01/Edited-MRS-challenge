import numpy as np
import math
from scipy import optimize



def op_freqPhaseShiftComplexRangeNest(input,f,p):
    fid=input[0:int(input.shape[0]/3)]+input[int(input.shape[0]/3):int(input.shape[0]*2/3)]*1j
    t = input[int(input.shape[0]*2/3):]
    shifted=fid*np.exp(1j*t*f*2*math.pi)
    shifted=shifted*np.ones(shifted.shape)*np.exp(1j*p*math.pi/180)
    y=np.concatenate([np.real(shifted),np.imag(shifted)])
    return y


def basic_spectral_registration(x_in,t):
    x = x_in.copy()

    ## defining reference as first item
    #x_ref = x[:,:,:,0].copy()
    x_ref = x.mean(axis=3).copy()
    

    f_corrections = np.zeros((x.shape[0],1,x.shape[2],x.shape[3]))
    ph_corrections= np.zeros((x.shape[0],1,x.shape[2],x.shape[3]))

    for i in range(x.shape[0]):
        for sub in range(x.shape[2]):
            for j in range(x.shape[-1]):
                i_x = x[i,:,sub,j]
                i_x_flatted = np.concatenate([np.real(i_x).flatten(),np.imag(i_x).flatten(),t[i].flatten()])
                ref_flatted = np.concatenate([np.real(x_ref[i,:,sub]).flatten(),np.imag(x_ref[i,:,sub]).flatten()])

                try:
                    parfit,_ = optimize.curve_fit(op_freqPhaseShiftComplexRangeNest,i_x_flatted,ref_flatted,[0,0],maxfev=10000)
                except Exception as e:
                    print(f"Exception: {e}")
                    parfit=[0,0]
                
                #print(f"Finished {i}-{sub}-{j}")

                f_corrections[i,0,sub,j]=parfit[0]
                ph_corrections[i,0,sub,j]=parfit[1]
    
    # applying the corrections

    x = x*np.exp(1j*(2*math.pi*f_corrections*t.reshape(x.shape[0],-1,1,1) + ph_corrections*math.pi/180))

    return x