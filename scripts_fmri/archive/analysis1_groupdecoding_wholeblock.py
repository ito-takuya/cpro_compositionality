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
parser.add_argument('--novelty', action='store_true', help="Run novelty decoding")
parser.add_argument('--task64', action='store_true', help="Run 64-way decoding")
parser.add_argument('--logic', action='store_true', help="Run Logic rule decoding")
parser.add_argument('--sensory', action='store_true', help="Run sensory rule decoding")
parser.add_argument('--motor', action='store_true', help="Run motor rule decoding")
parser.add_argument('--nproc', type=int, default=10, help="num parallel processes to run (DEFAULT: 10)")
parser.add_argument('--kfold',type=int, default=10, help="number of CV folds (DEFAULT: 10)")
parser.add_argument('--normalize', action='store_true', help="Normalize features (DEFAULT: FALSE")
parser.add_argument('--classifier',type=str, default='distance', help="decoding method DEFAULT: 'distance' [distance, svm, logistic]")
parser.add_argument('--permutation', action='store_true', help="Permuation -- shuffle training labels (DEFAULT: FALSE")



def run(args):
    args 
    novelty = args.novelty
    task64 = args.task64
    logic = args.logic
    sensory = args.sensory
    motor = args.motor
    nproc = args.nproc
    kfold = args.kfold
    normalize = args.normalize
    classifier = args.classifier
    permutation = args.permutation

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
    h5f = h5py.File(datadir + 'Data_WholeMiniblock_AllSubjs.h5','r')
    fmri = h5f['data'][:]
    fmri = np.real(fmri)
    h5f.close()

    # Flatten data from 3d -> 2d matrix
    data_mat = [] 
    for i in range(len(subjNums)):
        data_mat.extend(fmri[i,:,:].T)
    data_mat = np.asarray(data_mat)

    del fmri

    #### Load in behavioral data for all subjects
    df_all = tools.loadGroupBehavioralData(subjNums)
    # Make sure to only have one sample per task (rather than for each trial, since we're only decoding miniblocks)
    df = df_all.loc[df_all.TrialLabels=='Trial1']

    ######################################
    #### 64 context decoding
    if task64:
        print('Running 64-way task context decoding...')
        # pull out labels of interest
        task_labels = df.TaskID.values
        subj_labels = df.Subject.values
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
        i = 0
        for roi in rois:
            tmp = pd.DataFrame()
            tmp = tmp.append(df)
            # Delete irrelevant columns
            del tmp['TrialLabels'], tmp['Stim1_Ori'], tmp['Stim2_Ori'], tmp['Stim1_Color'], tmp['Stim2_Color'], tmp['Stim1_Pitch'], tmp['Stim2_Pitch'], tmp['Stim1_Constant']
            del tmp['Stim2_Constant'], tmp['MotorResponses']
            tmp.insert(0,'DecodingAccuracy',accuracies[i],True)
            tmp.insert(0,'ROI',np.repeat(roi,len(tmp)),True)
            tmp.to_csv(resultdir + 'CrossSubject64TaskDecoding/CrossSubjectWholeBlock' + strlabel + '64TaskDecoding_roi' + str(roi) + '.csv')
            
            i += 1

    ######################################
    #### Logic rule decoding
    if logic:
        print('Running Logic Rule decoding...')
        # pull out labels of interest
        task_labels = df.LogicRules.values
        subj_labels = df.Subject.values
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
        i = 0
        for roi in rois:
            tmp = pd.DataFrame()
            tmp = tmp.append(df)
            # Delete irrelevant columns
            del tmp['TrialLabels'], tmp['Stim1_Ori'], tmp['Stim2_Ori'], tmp['Stim1_Color'], tmp['Stim2_Color'], tmp['Stim1_Pitch'], tmp['Stim2_Pitch'], tmp['Stim1_Constant']
            del tmp['Stim2_Constant'], tmp['MotorResponses']
            tmp.insert(0,'DecodingAccuracy',accuracies[i],True)
            tmp.insert(0,'ROI',np.repeat(roi,len(tmp)),True)
            tmp.to_csv(resultdir + 'CrossSubjectLogicRuleDecoding/CrossSubjectWholeBlock' + strlabel + 'LogicRuleDecoding_roi' + str(roi) + '.csv')
            
            i += 1

    ######################################
    #### Sensory rule decoding
    if sensory:
        print('Running Sensory Rule decoding...')

        # pull out labels of interest
        task_labels = df.SensoryRules.values
        subj_labels = df.Subject.values
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
        i = 0
        for roi in rois:
            tmp = pd.DataFrame()
            tmp = tmp.append(df)
            # Delete irrelevant columns
            del tmp['TrialLabels'], tmp['Stim1_Ori'], tmp['Stim2_Ori'], tmp['Stim1_Color'], tmp['Stim2_Color'], tmp['Stim1_Pitch'], tmp['Stim2_Pitch'], tmp['Stim1_Constant']
            del tmp['Stim2_Constant'], tmp['MotorResponses']
            tmp.insert(0,'DecodingAccuracy',accuracies[i],True)
            tmp.insert(0,'ROI',np.repeat(roi,len(tmp)),True)
            tmp.to_csv(resultdir + 'CrossSubjectSensoryRuleDecoding/CrossSubjectWholeBlock' + strlabel + 'SensoryRuleDecoding_roi' + str(roi) + '.csv')
            
            i += 1

    ######################################
    #### Motor rule decoding
    if motor:
        print('Running Motor Rule decoding...')

        # pull out labels of interest
        task_labels = df.MotorRules.values
        subj_labels = df.Subject.values
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
        i = 0
        for roi in rois:
            tmp = pd.DataFrame()
            tmp = tmp.append(df)
            # Delete irrelevant columns
            del tmp['TrialLabels'], tmp['Stim1_Ori'], tmp['Stim2_Ori'], tmp['Stim1_Color'], tmp['Stim2_Color'], tmp['Stim1_Pitch'], tmp['Stim2_Pitch'], tmp['Stim1_Constant']
            del tmp['Stim2_Constant'], tmp['MotorResponses']
            tmp.insert(0,'DecodingAccuracy',accuracies[i],True)
            tmp.insert(0,'ROI',np.repeat(roi,len(tmp)),True)
            tmp.to_csv(resultdir + 'CrossSubjectMotorRuleDecoding/CrossSubjectWholeBlock' + strlabel + 'MotorRuleDecoding_roi' + str(roi) + '.csv')
            
            i += 1

    ######################################
    #### Motor rule decoding
    if novelty:
        print('Running Novelty decoding...')

        # pull out labels of interest
        task_labels = df.TaskNovelty.values
        subj_labels = df.Subject.values
        # Need to make only practiced v novel activations for each subject
        unique_subjs = np.unique(subj_labels)
        unique_task = np.unique(task_labels)
        data_mat2 = [] 
        subj_labels2 = []
        task_labels2 = []
        for subj in unique_subjs:
            subj_ind = np.where(subj_labels==subj)[0]
            for task in unique_task:
                task_ind = np.where(task_labels==task)[0]
                subjtask_ind = np.intersect1d(task_ind,subj_ind)
                # compute average activation of task condition for this subject
                data_mat2.append(np.mean(data_mat[subjtask_ind,:],axis=0))
                # Recreate new labels
                subj_labels2.append(subj)
                task_labels2.append(task)
        data_mat2 = np.asarray(data_mat2)
        subj_labels2 = np.asarray(subj_labels2)
        task_labels2 = np.asarray(task_labels2)

        inputs = []
        for roi in rois:
            roi_ind = np.where(glasser==roi)[0]
            roi_data = data_mat2[:,roi_ind]
            inputs.append((roi_data,subj_labels2,task_labels2,kfold,normalize,classifier,confusion,permutation,roi))

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
        tmp = {}
        tmp['ROI'] = []
        tmp['DecodingAccuracy'] = []
        i = 0
        for roi in rois:
            tmp['DecodingAccuracy'].extend(accuracies[i])
            tmp['ROI'].extend(np.repeat(roi,len(accuracies[i])))
            i += 1

        tmp = pd.DataFrame(data=tmp)
        tmp.to_csv(resultdir + 'CrossSubjectNoveltyDecoding/CrossSubjectWholeBlock' + strlabel + 'NoveltyDecoding_allROIs.csv')


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
