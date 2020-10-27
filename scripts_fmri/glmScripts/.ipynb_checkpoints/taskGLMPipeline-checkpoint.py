# Takuya Ito
# 10/21/2020

# Functions to run a GLM analysis

import numpy as np
import os
import glob
from nipy.modalities.fmri.hemodynamic_models import spm_hrf
import multiprocessing as mp
#import statsmodels.api as sm
import h5py
import scipy.stats as stats
from scipy import signal
import constructDesignMatrices as dmat
import nibabel as nib

## Define GLOBAL variables (variables accessible to all functions
# Define base data directory
projectdir = '/projects3/CPROCompositionality/'
datadir = projectdir + 'data/processedData/'
# Define number of frames to skip
framesToSkip = 5
# Define list of subjects
subjNums = ['013','014','016','017','018','021','023','024','026','027','028','030','031','032','033','034','035','037','038','039','040','041','042','043','045','046','047','048','049','050','053','055','056','057','058','062','063','066','067','068','069','070','072','074','075','076','077','081','085','086','087','088','090','092','093','094','095','097','098','099','101','102','103','104','105','106','108','109','110','111','112','114','115','117','119','120','121','122','123','124','125','126','127','128','129','130','131','132','134','135','136','137','138','139','140','141']
# Define all runs you want to preprocess
allRuns = ['Task1', 'Task2', 'Task3', 'Task4', 'Task5', 'Task6', 'Task7', 'Task8']
taskRuns = ['Task1', 'Task2', 'Task3', 'Task4', 'Task5', 'Task6', 'Task7', 'Task8']

nRunsPerTask = 8
taskNames = ['Task1', 'Task2', 'Task3', 'Task4', 'Task5', 'Task6', 'Task7', 'Task8']
taskLength = 581
# Define the *input* directory for nuisance regressors -- take from SRActflow project directory
nuis_reg_dir = '/projects3/SRActFlow/data/postProcessing/nuisanceRegressors/'
# Create directory if it doesn't exist
if not os.path.exists(nuis_reg_dir): os.makedirs(nuis_reg_dir)
# Define the directory containing the timing files
stimdir = projectdir + 'data/stimfiles/'
# Define the *output* directory for preprocessed data
outputdir = datadir 
# TRlength
trLength = .785

def runGroupTaskGLM(task, nuisModel='24pXaCompCorXVolterra', taskModel='canonical',nproc=1):
    if nproc==1:
        scount = 1
        for subj in subjNums:
            print('Running task GLM |', task, '| for subject', subj, '|', scount, '/', len(subjNums), '... model:', taskModel)
            taskGLM(subj, task, taskModel=taskModel, nuisModel=nuisModel)
            
            scount += 1
    else:
        print('Running task GLM on parallel CPUs...', nproc, 'parallel processes')
        inputs = []
        for subj in subjNums:
            inputs.append((subj,task,taskModel,nuisModel))

        pool = mp.Pool(processes=nproc)
        results = pool.starmap_async(taskGLM,inputs).get()
        pool.close()
        pool.join()

def taskGLM(subj, task='betaSeries', taskModel='canonical', nuisModel='24pXaCompCorXVolterra'):
    """
    This function runs a task-based GLM on a single subject
    Will only regress out task-timing, using either a canonical HRF model or FIR model

    Input parameters:
        subj        :   subject number as a string
        task        :   regression model (e.g. beta series) 
        nuisModel   :   nuisance regression model (to identify input data)
    """

    data_all = []
    for run in taskRuns:
        ## Load in data to be preprocessed - This needs to be a space x time 2d array
        inputfile = '/projects3/SRActFlow/data/' + subj + '/analysis/' + run + '_64kResampled.dtseries.nii'
        # Load data
        data = nib.load(inputfile).get_data()
        data = np.squeeze(data)

        tMask = np.ones((data.shape[0],))
        tMask[:framesToSkip] = 0

        # Skip frames
        data = data[framesToSkip:,:]
        # Demean each run
        data = signal.detrend(data,axis=0,type='constant')
        # Detrend each run
        data = signal.detrend(data,axis=0,type='linear')

        data_all.extend(data)

    data = np.asarray(data_all).T.copy()

    # Identify number of ROIs
    nROIs = data.shape[0]
    # Identify number of TRs
    nTRs = data.shape[1]

    # Load regressors for data
    X = loadTaskTiming(subj, task, taskModel=taskModel)

    taskRegs = X['taskRegressors']
    # Load nuisance regressors
    nuisanceRegressors = loadNuisanceRegressors(subj,model='24pXaCompCorXVolterra',spikeReg=False)

    allRegs = np.hstack((nuisanceRegressors,taskRegs))

    print('Running task GLM on subject', subj)
    print('\tNumber of spatial dimensions:', nROIs)
    print('\tNumber of task regressors:', taskRegs.shape[1])
    print('\tNumber of noise regressors:', nuisanceRegressors.shape[1])
    print('\tTask manipulation:', task)
    betas, resid = regression(data.T, allRegs, constant=True)
    
    nTaskRegressors = int(taskRegs.shape[1])
    
    betas = betas[-nTaskRegressors:,:].T # Exclude nuisance regressors
    residual_ts = resid.T
    
    h5f = h5py.File(outputdir + subj + '_glmOutput_data_' + task + '.h5','a')
    outname1 = 'taskRegression/' + task + '_' + nuisModel + '_taskReg_resid_' + taskModel
    outname2 = 'taskRegression/' + task + '_' + nuisModel + '_taskReg_betas_' + taskModel
    try:
        if task!='betaSeries': # don't output residuals if running the beta series regression
            h5f.create_dataset(outname1,data=residual_ts)
        h5f.create_dataset(outname2,data=betas)
    except:
        del h5f[outname1], h5f[outname2]
        if task!='betaSeries': # don't output residuals if running the beta series regression
            h5f.create_dataset(outname1,data=residual_ts)
        h5f.create_dataset(outname2,data=betas)
    h5f.close()

def loadTaskTiming(subj, task='betaSeries', taskModel='canonical'):
    if task=='betaSeries':
        # rules
        stimMat = dmat.loadBetaSeries(subj)

        #stimMat = np.hstack((logic,sensory,motor,colorStim,oriStim,pitchStim,constantStim,motorResp))
        stimIndex = []
        stimIndex.extend(np.repeat('Encoding',128))
        for i in range(128): # num miniblockss
            stimIndex.extend('Probe1')
            stimIndex.extend('Probe2')
            stimIndex.extend('Probe3')

    nTRsPerRun = int(stimMat.shape[0]/nRunsPerTask)

    ##### HRF CONVOLUTION STEP
    if taskModel=='canonical':
        # Convolve taskstim regressors based on SPM canonical HRF (likely period of task-induced activity)
        taskStims_HRF = np.zeros(stimMat.shape)
        spm_hrfTS = spm_hrf(trLength,oversampling=1)
       
        trcount = 0
        for run in range(nRunsPerTask):
            trstart = trcount
            trend = trstart + nTRsPerRun

            for stim in range(stimMat.shape[1]):

                # Perform convolution
                tmpconvolve = np.convolve(stimMat[trstart:trend,stim],spm_hrfTS)
                tmpconvolve_run = tmpconvolve[:nTRsPerRun] # Make sure to cut off at the end of the run
                taskStims_HRF[trstart:trend,stim] = tmpconvolve_run

            trcount += nTRsPerRun

        taskRegressors = taskStims_HRF.copy()
    
    # Create temporal mask (skipping which frames?)
    tMask = []
    for run in range(nRunsPerTask):
        tmp = np.ones((nTRsPerRun,), dtype=bool)
        tmp[:framesToSkip] = False
        tMask.extend(tmp)
    tMask = np.asarray(tMask,dtype=bool)

    output = {}
    # Commented out since we demean each run prior to loading data anyway
    output['taskRegressors'] = taskRegressors[tMask,:]
    output['taskDesignMat'] = stimMat[tMask,:]
    output['stimIndex'] = stimIndex

    return output

def loadNuisanceRegressors(subj,model='24pXaCompCorXVolterra',spikeReg=False):
    """
    LOAD and concatenate all nuisance regressors across all tasks
    """

    concatNuisRegressors = []
    # Load nuisance regressors for this data
    h5f = h5py.File(nuis_reg_dir + subj + '_nuisanceRegressors.h5','r') 
    for run in taskRuns:
        
        if model=='24pXaCompCorXVolterra':
            # Motion parameters + derivatives
            motion_parameters = h5f[run]['motionParams'][:].copy()
            motion_parameters_deriv = h5f[run]['motionParams_deriv'][:].copy()
            # WM aCompCor + derivatives
            aCompCor_WM = h5f[run]['aCompCor_WM'][:].copy()
            aCompCor_WM_deriv = h5f[run]['aCompCor_WM_deriv'][:].copy()
            # Ventricles aCompCor + derivatives
            aCompCor_ventricles = h5f[run]['aCompCor_ventricles'][:].copy()
            aCompCor_ventricles_deriv = h5f[run]['aCompCor_ventricles_deriv'][:].copy()
            # Create nuisance regressors design matrix
            nuisanceRegressors = np.hstack((motion_parameters, motion_parameters_deriv, aCompCor_WM, aCompCor_WM_deriv, aCompCor_ventricles, aCompCor_ventricles_deriv))
            quadraticRegressors = nuisanceRegressors**2
            nuisanceRegressors = np.hstack((nuisanceRegressors,quadraticRegressors))
        
        elif model=='18p':
            # Motion parameters + derivatives
            motion_parameters = h5f[run]['motionParams'][:].copy()
            motion_parameters_deriv = h5f[run]['motionParams_deriv'][:].copy()
            # Global signal + derivatives
            global_signal = h5f[run]['global_signal'][:].copy()
            global_signal_deriv = h5f[run]['global_signal_deriv'][:].copy()
            # white matter signal + derivatives
            wm_signal = h5f[run]['wm_signal'][:].copy()
            wm_signal_deriv = h5f[run]['wm_signal_deriv'][:].copy()
            # ventricle signal + derivatives
            ventricle_signal = h5f[run]['ventricle_signal'][:].copy()
            ventricle_signal_deriv = h5f[run]['ventricle_signal_deriv'][:].copy()
            # Create nuisance regressors design matrix
            tmp = np.vstack((global_signal,global_signal_deriv,wm_signal,wm_signal_deriv,ventricle_signal,ventricle_signal_deriv)).T # Need to vstack, since these are 1d arrays
            nuisanceRegressors = np.hstack((motion_parameters, motion_parameters_deriv, tmp))

        elif model=='16pNoGSR':
            # Motion parameters + derivatives
            motion_parameters = h5f[run]['motionParams'][:].copy()
            motion_parameters_deriv = h5f[run]['motionParams_deriv'][:].copy()
            # white matter signal + derivatives
            wm_signal = h5f[run]['wm_signal'][:].copy()
            wm_signal_deriv = h5f[run]['wm_signal_deriv'][:].copy()
            # ventricle signal + derivatives
            ventricle_signal = h5f[run]['ventricle_signal'][:].copy()
            ventricle_signal_deriv = h5f[run]['ventricle_signal_deriv'][:].copy()
            # Create nuisance regressors design matrix
            tmp = np.vstack((wm_signal,wm_signal_deriv,ventricle_signal,ventricle_signal_deriv)).T # Need to vstack, since these are 1d arrays
            nuisanceRegressors = np.hstack((motion_parameters, motion_parameters_deriv, tmp))
        
        elif model=='12pXaCompCor':
            # Motion parameters + derivatives
            motion_parameters = h5f[run]['motionParams'][:].copy()
            motion_parameters_deriv = h5f[run]['motionParams_deriv'][:].copy()
            # WM aCompCor + derivatives
            aCompCor_WM = h5f[run]['aCompCor_WM'][:].copy()
            aCompCor_WM_deriv = h5f[run]['aCompCor_WM_deriv'][:].copy()
            # Ventricles aCompCor + derivatives
            aCompCor_ventricles = h5f[run]['aCompCor_ventricles'][:].copy()
            aCompCor_ventricles_deriv = h5f[run]['aCompCor_ventricles_deriv'][:].copy()
            # Create nuisance regressors design matrix
            nuisanceRegressors = np.hstack((motion_parameters, motion_parameters_deriv, aCompCor_WM, aCompCor_WM_deriv, aCompCor_ventricles, aCompCor_ventricles_deriv))
        
        elif model=='36p':
            # Motion parameters + derivatives
            motion_parameters = h5f[run]['motionParams'][:].copy()
            motion_parameters_deriv = h5f[run]['motionParams_deriv'][:].copy()
            # Global signal + derivatives
            global_signal = h5f[run]['global_signal'][:].copy()
            global_signal_deriv = h5f[run]['global_signal_deriv'][:].copy()
            # white matter signal + derivatives
            wm_signal = h5f[run]['wm_signal'][:].copy()
            wm_signal_deriv = h5f[run]['wm_signal_deriv'][:].copy()
            # ventricle signal + derivatives
            ventricle_signal = h5f[run]['ventricle_signal'][:].copy()
            ventricle_signal_deriv = h5f[run]['ventricle_signal_deriv'][:].copy()
            # Create nuisance regressors design matrix
            tmp = np.vstack((global_signal,global_signal_deriv,wm_signal,wm_signal_deriv,ventricle_signal,ventricle_signal_deriv)).T # Need to vstack, since these are 1d arrays
            nuisanceRegressors = np.hstack((motion_parameters, motion_parameters_deriv, tmp))
            quadraticRegressors = nuisanceRegressors**2
            nuisanceRegressors = np.hstack((nuisanceRegressors,quadraticRegressors))


        if spikeReg:
            # Obtain motion spikes
            try:
                motion_spikes = h5f[run]['motionSpikes'][:].copy()
                nuisanceRegressors = np.hstack((nuisanceRegressors,motion_spikes))
            except:
                print('Spike regression option was chosen... but no motion spikes for subj', subj, '| run', run, '!')
            # Update the model name - to keep track of different model types for output naming
            model = model + '_spikeReg' 

        concatNuisRegressors.extend(nuisanceRegressors[framesToSkip:,:].copy())

    h5f.close()
    nuisanceRegressors = np.asarray(concatNuisRegressors)

    return nuisanceRegressors

#def runGroupRestGLM(nuisModel='24pXaCompCorXVolterra', taskModel='FIR'):
#    scount = 1
#    for subj in subjNums:
#        print('Running task regression matrix on resting-state data for subject', subj, '|', scount, '/', len(subjNums), '... model:', taskModel)
#        taskGLM_onRest(subj, taskModel=taskModel, nuisModel=nuisModel)
#        
#        scount += 1


def regression(data,regressors,alpha=0,constant=True):
    """
    Taku Ito
    2/21/2019

    Hand coded OLS regression using closed form equation: betas = (X'X + alpha*I)^(-1) X'y
    Set alpha = 0 for regular OLS.
    Set alpha > 0 for ridge penalty

    PARAMETERS:
        data = observation x feature matrix (e.g., time x regions)
        regressors = observation x feature matrix
        alpha = regularization term. 0 for regular multiple regression. >0 for ridge penalty
        constant = True/False - pad regressors with 1s?
    OUTPUT
        betas = coefficients X n target variables
        resid = observations X n target variables
    """
    # Add 'constant' regressor
    if constant:
        ones = np.ones((regressors.shape[0],1))
        regressors = np.hstack((ones,regressors))
    X = regressors.copy()

    # construct regularization term
    LAMBDA = np.identity(X.shape[1])*alpha

    # Least squares minimization
    try:
        C_ss_inv = np.linalg.pinv(np.dot(X.T,X) + LAMBDA)
    except np.linalg.LinAlgError as err:
        C_ss_inv = np.linalg.pinv(np.cov(X.T) + LAMBDA)

    betas = np.dot(C_ss_inv,np.dot(X.T,data))
    # Calculate residuals
    resid = data - (betas[0] + np.dot(X[:,1:],betas[1:]))

    return betas, resid

def _group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result

