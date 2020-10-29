import numpy as np
import scipy.stats as stats
import h5py
import nibabel as nib
from importlib import reload
import pandas as pd
import loadTaskBehavioralData as task
import tools
import multiprocessing as mp
import argparse

######################################
#### Basic parameters
projectdir = '/projects3/CPROCompositionality/'
datadir = projectdir + 'data/processedData/' 
resultdir = projectdir + 'data/results/'
subjNums = ['013','014','016','017','018','021','023','024','026','027','028',
            '030','031','032','033','034','035','037','038','039','040','041',
            '042','043','045','046','047','048','049','050','053','055','056',
            '057','058','062','063','066','067','068','069','070','072','074',
            '075','076','077','081','085','086','087','088','090','092','093',
            '094','095','097','098','099','101','102','103','104','105','106',
            '108','109','110','111','112','114','115','117','119','120','121',
            '122','123','124','125','126','127','128','129','130','131','132',
            '134','135','136','137','138','139','140','141']

glasser = projectdir + 'data/Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_RL.dlabel.nii'
glasser = nib.load(glasser).get_data()
glasser = np.squeeze(glasser)
rois = np.arange(1,361)

parser = argparse.ArgumentParser('./main.py', description='Run group decoding analysis on encoding miniblocks')
parser.add_argument('--logic', action='store_true', help="Run Logic rule decoding")
parser.add_argument('--sensory', action='store_true', help="Run sensory rule decoding")
parser.add_argument('--motor', action='store_true', help="Run motor rule decoding")
parser.add_argument('--nproc', type=int, default=10, help="num parallel processes to run (DEFAULT: 10)")
parser.add_argument('--kfold',type=int, default=10, help="number of CV folds (DEFAULT: 10)")
parser.add_argument('--normalize', action='store_true', help="Normalize features (DEFAULT: FALSE")
parser.add_argument('--classifier',type=str, default='distance', help="decoding method DEFAULT: 'distance' [distance, svm, logistic]")



def run(args):
    args 
    logic = args.logic
    sensory = args.sensory
    motor = args.motor
    nproc = args.nproc
    kfold = args.kfold
    normalize = args.normalize
    classifier = args.classifier

    permutation = False
    ##############################################################
    #### Set decoding parameters
    # function parameters
    confusion = True
    rois = np.arange(1,361)
    if not normalize:
        strlabel = '_unnormalized_'
    else:
        strlabel = ''
    
    ##############################################################
    #### Load fMRI data
    print('Loading data...')
    h5f = h5py.File(datadir + 'Data_EncodingMiniblock_AllSubjs.h5')
    fmri = h5f['data'][:]
    fmri = np.real(fmri)
    h5f.close()

    # Flatten data from 3d -> 2d matrix
    data_mat = np.zeros((64*len(subjNums),len(glasser)))
    logic_labels = []
    sensory_labels = []
    motor_labels = []
    subj_labels = []
    i = 0
    subjcount = 0
    for subj in subjNums:
        df_subj = task.loadExperimentalData(subj)
        df_subj = df_subj.loc[df_subj.TrialLabels=='Trial1'] # only want one row per miniblock -- all rules per miniblock are the same, anyway
        # Now find corresponding task IDs
        task64_id = df_subj.TaskID.values
        unique_tasks = np.sort(np.unique(task64_id))
        for taskid in unique_tasks: # iterate from 1-64
            ind = np.where(task64_id==taskid)[0]
            tmpmat = fmri[subjcount,:,ind]
            #print(tmpmat.shape)
            data_mat[i,:] = np.mean(tmpmat,axis=0) #tmp mat is cond x vertices
            # Create relevant labels for decoding
            tmpdf = df_subj.loc[df_subj.TaskID==taskid]
            logic_labels.append(tmpdf.LogicRules.values[0])
            sensory_labels.append(tmpdf.SensoryRules.values[0])
            motor_labels.append(tmpdf.MotorRules.values[0])
            subj_labels.append(subjcount)
            i += 1
        subjcount += 1

    
    subj_labels = np.asarray(subj_labels)
    logic_labels = np.asarray(logic_labels)
    sensory_labels = np.asarray(sensory_labels)
    motor_labels = np.asarray(motor_labels)


    ######################################
    #### Logic rule decoding
    if logic:
        print('Running Logic Rule decoding...')
        # pull out labels of interest
        task_labels = logic_labels
        inputs = []
        for roi in rois:
            roi_ind = np.where(glasser==roi)[0]
            roi_data = data_mat[:,roi_ind]
            inputs.append((roi_data,subj_labels,task_labels,kfold,normalize,classifier,confusion,permutation,roi))

        pool = mp.Pool(processes=nproc)
        results = pool.starmap_async(tools.decodeGroup,inputs).get()
        pool.close()
        pool.join()

        accuracies = []
        confusion_mats = []
        for result in results:
            acc, confusion_mat = result[0], result[1]
            accuracies.append(acc)
            confusion_mats.append(confusion_mat)

        #### Save accuracy data to pandas csv    
        mat = np.zeros((len(rois),))
        i = 0
        for roi in rois:
            mat[i] = np.mean(accuracies[i])
            i += 1

        np.savetxt(resultdir + 'CrossConditionGeneralization/LogicDecoding.csv',mat)

    ######################################
    #### Sensory rule decoding
    if sensory:
        print('Running Sensory Rule decoding...')
        # pull out labels of interest
        task_labels = sensory_labels
        inputs = []
        for roi in rois:
            roi_ind = np.where(glasser==roi)[0]
            roi_data = data_mat[:,roi_ind]
            inputs.append((roi_data,subj_labels,task_labels,kfold,normalize,classifier,confusion,permutation,roi))

        pool = mp.Pool(processes=nproc)
        results = pool.starmap_async(tools.decodeGroup,inputs).get()
        pool.close()
        pool.join()

        accuracies = []
        confusion_mats = []
        for result in results:
            acc, confusion_mat = result[0], result[1]
            accuracies.append(acc)
            confusion_mats.append(confusion_mat)

        #### Save accuracy data to pandas csv    
        mat = np.zeros((len(rois),))
        i = 0
        for roi in rois:
            mat[i] = np.mean(accuracies[i])
            i += 1

        np.savetxt(resultdir + 'CrossConditionGeneralization/SensoryDecoding.csv',mat)

    ######################################
    #### Motor rule decoding
    if motor:
        print('Running Motor Rule decoding...')
        # pull out labels of interest
        task_labels = motor_labels
        inputs = []
        for roi in rois:
            roi_ind = np.where(glasser==roi)[0]
            roi_data = data_mat[:,roi_ind]
            inputs.append((roi_data,subj_labels,task_labels,kfold,normalize,classifier,confusion,permutation,roi))

        pool = mp.Pool(processes=nproc)
        results = pool.starmap_async(tools.decodeGroup,inputs).get()
        pool.close()
        pool.join()

        accuracies = []
        confusion_mats = []
        for result in results:
            acc, confusion_mat = result[0], result[1]
            accuracies.append(acc)
            confusion_mats.append(confusion_mat)

        #### Save accuracy data to pandas csv    
        mat = np.zeros((len(rois),))
        i = 0
        for roi in rois:
            mat[i] = np.mean(accuracies[i])
            i += 1

        np.savetxt(resultdir + 'CrossConditionGeneralization/MotorDecoding.csv',mat)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
