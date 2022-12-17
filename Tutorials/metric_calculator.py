## This script calculates the metrics for the challenge

import numpy as np



def calculate_metrics(x,y,ppm):
    # function to calculate all metrics and return dict with results
    # Args:
    #       - x: testing spectra
    #       - y: reference spectra
    #       - ppm: 1D ppm array
    # Output:
    #       - dictionary with metric name and value
    #

    mse = calculate_mse(x,y,ppm)
    snr = calculate_snr(x,ppm)
    linewidth = calculate_linewidth(x,ppm)
    shape_score = calculate_shape_score(x,y,ppm)

    output={
        "mse":mse,
        "snr":snr,
        "linewidth":linewidth,
        "shape_score":shape_score
    }

    return output


def calculate_mse(x,y,ppm):
    # Calculate the MSE of a region of interest based on reference
    # Args:
    #       - x: testing spectra
    #       - y: reference spectra
    #       - ppm: 1D ppm reference
    # Output:
    #       - mse
    #
    
    # selecting region of interest (1.5-4 ppm) 
    max_ind = np.argmin(ppm[ppm<=4])
    min_ind = np.argmax(ppm[ppm>=1.5])

    x_crop = x[:,min_ind:max_ind]
    y_crop = y[:,min_ind:max_ind]

    # calculating mse and returing
    return (np.square(y_crop-x_crop)).mean(axis=1).mean()

def calculate_snr(x,ppm):
    # Calculate the GABA SNR
    # Args:
    #       - x: testing spectra
    #       - ppm: 1D ppm reference
    # Output:
    #       - GABA SNR
    #

    # selecting indexes of regions of interest
    gaba_max_ind, gaba_min_ind = np.amax(np.where(ppm >= 2.8)), np.amin(np.where(ppm <= 3.2))
    dt_max_ind, dt_min_ind = np.amax(np.where(ppm >= 9.8)), np.amin(np.where(ppm <= 10.8))

    snrs=[]

    # looping over scans
    for i in range(x.shape[0]):

        # selecting scan and extracting region peak
        spec = x[i]
        max_peak = spec[gaba_min_ind:gaba_max_ind].max()

        # calculating fitted standard deviation of noise region
        dt = np.polyfit(ppm[dt_min_ind:dt_max_ind], spec[dt_min_ind:dt_max_ind], 2)
        sizeFreq = ppm[dt_min_ind:dt_max_ind].shape[0]
        stdev_Man = np.sqrt(np.sum(np.square(np.real(spec[dt_min_ind:dt_max_ind] - np.polyval(dt, ppm[dt_min_ind:dt_max_ind])))) / (sizeFreq - 1))
        
        # calculating snr as peak/(2*stds)
        snrs += [np.real(max_peak) / (2 * stdev_Man)]
    
    # return average of snrs
    return sum(snrs)/len(snrs)

def calculate_linewidth(x,ppm):
    # Calculate the GABA SNR
    # Args:
    #       - x: testing spectra
    #       - ppm: 1D ppm reference
    # Output:
    #       - GABA SNR
    #
    gaba_max_ind, gaba_min_ind = np.amax(np.where(ppm >= 2.8)), np.amin(np.where(ppm <= 3.2))
    linewidths=[]

    for i in range(x.shape[0]):
        spec = x[i,gaba_min_ind:gaba_max_ind]
        #print(spec.shape)
        ##normalizing spec
        spec = (spec-spec.min())/(spec.max()-spec.min())

        max_peak = spec.max()
        #print(max_peak)
        ind_max_peak = np.argmax(spec)
        left_side = spec[:ind_max_peak]
        #print(left_side)
        left_ind = np.amin(np.where(left_side>max_peak/2))+gaba_min_ind
        #print(left_ind)
        right_side = spec[ind_max_peak:]
        #print(right_side)
        right_ind = np.amax(np.where(right_side>max_peak/2))+gaba_min_ind+ind_max_peak
        #print([right_side>max_peak/2])
        #print(right_ind)
        left_ppm = ppm[left_ind]
        right_ppm = ppm[right_ind]

        linewidths.append(left_ppm-right_ppm)
    
    return sum(linewidths)/len(linewidths)

def calculate_shape_score(x,y,ppm):
    gaba_max_ind, gaba_min_ind = np.amax(np.where(ppm >= 2.8)), np.amin(np.where(ppm <= 3.2))
    glx_max_ind, glx_min_ind = np.amax(np.where(ppm >= 3.6)), np.amin(np.where(ppm <= 3.9))

    gaba_corrs=[]
    glx_corrs=[]


    for i in range(0,x.shape[0]):
        gaba_spec_x = x[i,gaba_min_ind:gaba_max_ind]
        gaba_spec_x = (gaba_spec_x-gaba_spec_x.min())/(gaba_spec_x.max()-gaba_spec_x.min())

        gaba_spec_y = y[i,gaba_min_ind:gaba_max_ind]
        gaba_spec_y = (gaba_spec_y-gaba_spec_y.min())/(gaba_spec_y.max()-gaba_spec_y.min())

        gaba_corrs.append(np.corrcoef(gaba_spec_x,gaba_spec_y)[0,1])

        glx_spec_x = x[i,glx_min_ind:glx_max_ind]
        glx_spec_x = (glx_spec_x-glx_spec_x.min())/(glx_spec_x.max()-glx_spec_x.min())

        glx_spec_y = y[i,glx_min_ind:glx_max_ind]
        glx_spec_y = (glx_spec_y-glx_spec_y.min())/(glx_spec_y.max()-glx_spec_y.min())

        glx_corrs.append(np.corrcoef(glx_spec_x,glx_spec_y)[0,1])
    
    gaba_score = sum(gaba_corrs)/len(gaba_corrs)
    glx_score = sum(glx_corrs)/len(glx_corrs)

    return gaba_score*0.6 + glx_score*0.4


