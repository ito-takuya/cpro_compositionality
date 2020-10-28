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

parser = argparse.ArgumentParser('./main.py', description='Calculate the parallelism score for each rule domain')
parser.add_argument('--logic', action='store_true', help="Run Logic rule decoding")
parser.add_argument('--sensory', action='store_true', help="Run sensory rule decoding")
parser.add_argument('--motor', action='store_true', help="Run motor rule decoding")
parser.add_argument('--nproc', type=int, default=10, help="num parallel processes to run (DEFAULT: 10)")
parser.add_argument('--permutation', action='store_true', help="Permuation -- shuffle training labels (DEFAULT: FALSE")



def run(args):
    args 
    logic = args.logic
    sensory = args.sensory
    motor = args.motor
    nproc = args.nproc
    permutation = args.permutation
    import loadTaskBehavioralData as task

    ##############################################################
    #### Set decoding parameters
    # function parameters
    confusion = True
    rois = np.arange(1,361)
    
    ##############################################################
    #### Load fMRI data
    print('Loading data...')
    h5f = h5py.File(datadir + 'Data_EncodingMiniblock_AllSubjs.h5')
    fmri = h5f['data'][:]
    fmri = np.real(fmri)
    h5f.close()

    # Flatten data from 3d -> 2d matrix
    data_mat = {} 
    logic_labels = {}
    sensory_labels = {}
    motor_labels = {}
    i = 0
    for subj in subjNums:
        df_subj = task.loadExperimentalData(subj)
        df_subj = df_subj.loc[df_subj.TrialLabels=='Trial1'] # only want one row per miniblock -- all rules per miniblock are the same, anyway
        # Now find corresponding task IDs
        task64_id = df_subj.TaskID.values
        unique_tasks = np.unique(task64_id)
        logic_labels[subj] = []
        sensory_labels[subj] = []
        motor_labels[subj] = []
        data_mat[subj] = np.zeros((len(glasser),len(unique_tasks)))
        for taskid in unique_tasks: # iterate from 1-64
            ind = np.where(task64_id==taskid)[0]
            tmpmat = fmri[i,:,ind]
            #print(tmpmat.shape)
            data_mat[subj][:,int(taskid-1)] = np.mean(tmpmat,axis=0) #tmp mat is cond x vertices
            tmpdf = df_subj.loc[df_subj.TaskID==taskid]
            logic_labels[subj].append(tmpdf.LogicRules.values[0])
            sensory_labels[subj].append(tmpdf.SensoryRules.values[0])
            motor_labels[subj].append(tmpdf.MotorRules.values[0])

        i += 1

    del fmri

    #### Load in behavioral data for all subjects
    df_all = tools.loadGroupBehavioralData(subjNums)
    # Make sure to only have one sample per task (rather than for each trial, since we're only decoding miniblocks)
    df = df_all.loc[df_all.TrialLabels=='Trial1']

    ######################################
    #### Logic parallelism calculation
    if logic:
        print('Running Logic Rule PS calculation...')
        ps_score = np.zeros((len(rois),len(subjNums)))
        roicount = 0
        for roi in rois:
            roi_ind = np.where(glasser==roi)[0]
            inputs = []
            for subj in subjNums:
                inputs.append((mat,logic_labels[subj],sensory_labels[subj],motor_labels[subj],permutation))
            subjcount = 0
            for subj in subjNums:
                mat = data_mat[subj][roi_ind,:].T
                ps, classes = tools.parallelismScore(mat,logic_labels[subj],sensory_labels[subj],motor_labels[subj],shuffle=permutation)
                ps_score[roicount,subjcount] = np.nanmean(ps)
                subjcount += 1

            print('Avg PS for ROI', roi, '=', np.mean(ps_score[roicount,:]))

            roicount += 1

        np.savetxt(resultdir + 'ParallelismScore_LogicRules.csv',ps_score)
            

    ######################################
    #### Sensory rule PS
    if sensory:
        print('Running Sensory Rule PS calculation...')
        ps_score = np.zeros((len(rois),len(subjNums)))
        roicount = 0
        for roi in rois:
            roi_ind = np.where(glasser==roi)[0]
            subjcount = 0
            for subj in subjNums:
                mat = data_mat[subj][roi_ind,:].T
                ps, classes = tools.parallelismScore(mat,sensory_labels[subj],logic_labels[subj],motor_labels[subj],shuffle=permutation)
                ps_score[roicount,subjcount] = np.nanmean(ps)
                subjcount += 1

            print('Avg PS for ROI', roi, '=', np.mean(ps_score[roicount,:]))

            roicount += 1

        np.savetxt(resultdir + 'ParallelismScore_SensoryRules.csv',ps_score)

    ######################################
    #### Motor rule PS
    if motor:
        print('Running Motor Rule PS calculation...')
        ps_score = np.zeros((len(rois),len(subjNums)))
        roicount = 0
        for roi in rois:
            roi_ind = np.where(glasser==roi)[0]
            subjcount = 0
            for subj in subjNums:
                mat = data_mat[subj][roi_ind,:].T
                ps, classes = tools.parallelismScore(mat,motor_labels[subj],sensory_labels[subj],logic_labels[subj],shuffle=permutation)
                ps_score[roicount,subjcount] = np.nanmean(ps)
                subjcount += 1

            print('Avg PS for ROI', roi, '=', np.mean(ps_score[roicount,:]))

            roicount += 1

        np.savetxt(resultdir + 'ParallelismScore_MotorRules.csv',ps_score)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
