import h5py
import numpy as np


def save_submission(result_spectra,ppm,filename):
    '''
    Save the results in the submission format
    Parameters:
        - results_spectra (np.array): Resulting predictions from test in format scan x spectral points
        - ppm (np.array): ppm values associataed with results, in same format
        - filename (str): name of the file to save results in, should end in .h5
    
    '''

    with h5py.File(filename,"w") as hf:
        hf.create_dataset("result_spectra",result_spectra.shape,dtype=float,data=result_spectra)
        hf.create_dataset("ppm",ppm.shape,dtype=float,data=ppm)
