import numpy as np
np.set_printoptions(suppress=True)
#import matplotlib.pyplot as plt
#import seaborn as sns
#import scipy.stats as stats
import os
from clearn.continual_learner import ContinualLearner

import model_cl as mod
import model.task as task
import time
import model.analysis as analysis
from importlib import reload
mod = reload(mod)
task = reload(task)
analysis = reload(analysis)
import torch
#from torch.autograd import Variable
#import torch.nn.functional as F


datadir = '../../../data/'

def runOnlineModelSI(datadir=datadir,practice=True,
             num_hidden=128,learning_rate=0.0001,thresh=0.5,
             create_new_batches=False,save_csv=False,save_hiddenrsm_pdf=False,save_model=None,verbose=True):
    """
    'online training model'
    num_hidden - # of hidden units
    learning_rate - learning rate 
    thresh - threshold for classifying output units
    create_new_batches - Create new training batches. Most likely not necessary; training batches already exist in data directory
    save_csv - Save out the RSM?
    save_hiddenrsm_pdf - save out a PDF of the RSM?
    """

    if practice:
        batchfilename = datadir + 'results/model/TrialBatches_4Prac60Nov'
    else:
        batchfilename = datadir + 'results/model/TrialBatches_AllTasks'

    if create_new_batches: 
        print('Creating new training batches')
        if practice:
            print('Creating separate practice and novel batches')
            TrialInfo = mod.TrialBatchesPracticeNovel(NUM_BATCHES=5000,
                                                      NUM_PRACTICE_TRIALS_PER_TASK=100,
                                                      NUM_NOVEL_TRIAlS_PER_TASK=100,
                                                      NUM_INPUT_ELEMENTS=28,
                                                      NUM_OUTPUT_ELEMENTS=4,
                                                      filename=batchfilename)
            TrialInfo.createBatches(condition='practice',nproc=4)
            TrialInfo.createBatches(condition='novel',nproc=4)
            TrialInfo.createBatches(condition='all',nproc=4)
        else:
            print('Creating batches with full task set')
            TrialInfo = mod.TrialBatchesTrainAll(NUM_BATCHES=5000,
                                                 NUM_TRIALS_PER_TASK=10,
                                                 NUM_INPUT_ELEMENTS=28,
                                                 NUM_OUTPUT_ELEMENTS=4,
                                                 filename=batchfilename)
            TrialInfo.createBatches(nproc=4)

    #### ANN construction
    if verbose: print('Instantiating new model, continual learner')
    Network = mod.ANN(num_rule_inputs=12,
                         num_sensory_inputs=16,
                         num_hidden=num_hidden,
                         num_motor_decision_outputs=4,
                         learning_rate=learning_rate,
                         thresh=thresh)
    #Network = ContinualLearner(Network)
    # Network.cuda = True
    Network = Network.cpu()

    return Network



    if practice:
        #### Load training batches
        if verbose: print('Loading practice and (all tasks) batches')
        TrialObj = mod.TrialBatchesPracticeNovel(filename=batchfilename)
        practice_input_batches, practice_output_batches = TrialObj.loadBatches(condition='practice',cuda=False)
        #novel_input_batches, novel_output_batches = TrialObj.loadBatches(condition='novel',cuda=False)
        novel_input_batches, novel_output_batches = TrialObj.loadBatches(condition='all',cuda=False)

        ###########################################################################################
        #### Online learning - practice tasks
        if verbose: print('Online training model on practiced tasks')
        timestart = time.time()
        
        batch_breaks = 10
        ntrials_viewed = 0
        trial_ids = np.arange(practice_input_batches.shape[1])
        np.random.shuffle(trial_ids)
        for batch in range(practice_input_batches.shape[0]):
            acc = []
            for trial in trial_ids:
                outputs, targets, loss = mod.train(Network,
                                                   practice_input_batches[batch,trial,:],
                                                   practice_output_batches[batch,trial,:])

                for out in range(len(targets)):
                    if targets[out] == 0: continue
                    response = outputs[out] # Identify response time points
                    target_resp = torch.ByteTensor([out]) # The correct target response
                    max_resp = outputs.argmax().byte()
                    if max_resp==target_resp and response>thresh: # Make sure network response is correct respnose, and that it exceeds some threshold
                        acc.append(1.0)
                    else:
                        acc.append(0)

                ntrials_viewed += 1

            if batch%batch_breaks==0: 
                if verbose: print('Batch', batch, 'achieved', np.mean(acc)*100.0, '% accuracy')
            if np.mean(acc)>0.9: 
                if verbose: print('Accuracy above 90%... exiting training') 
                break

        ###########################################################################################
        #### Online learning - all tasks
        if verbose: print('Training model on novel (and practiced) tasks')
        trial_ids = np.arange(novel_input_batches.shape[1])
        np.random.shuffle(trial_ids)
        for batch in range(novel_input_batches.shape[0]):
            acc = []
            for trial in trial_ids:
                outputs, targets, loss = mod.train(Network,
                                                   novel_input_batches[batch,trial,:],
                                                   novel_output_batches[batch,trial,:])

                for out in range(len(targets)):
                    if targets[out] == 0: continue
                    response = outputs[out] # Identify response time points
                    target_resp = torch.ByteTensor([out]) # The correct target response
                    max_resp = outputs.argmax().byte()
                    if max_resp==target_resp and response>thresh: # Make sure network response is correct respnose, and that it exceeds some threshold
                        acc.append(1.0)
                    else:
                        acc.append(0)

                ntrials_viewed += 1

            if batch%batch_breaks==0: 
                if verbose: print('Batch', batch, 'achieved', np.mean(acc)*100.0, '% accuracy')
            if np.mean(acc)>0.8: 
                if verbose: print('Accuracy above 80%... exiting training') 
                break

        timeend = time.time()
        if verbose: print('Time elapsed using CPU:', timeend-timestart)
        if verbose: print('Total number of trials viewed =', ntrials_viewed)
    #else:
    #    if verbose: print('Loading batches')
    #    TrialObj = mod.TrialBatchesTrainAll(filename=batchfilename)
    #    input_batches, output_batches = TrialObj.loadBatches(cuda=False)

    #    #### Train practice tasks
    #    if verbose: print('Training model on all tasks')
    #    timestart = time.time()
    #    nsamples_viewed, nbatches_trained = mod.batch_training(Network, input_batches,output_batches,cuda=False,verbose=False)  
    #    timeend = time.time()
    #    if verbose: print('Time elapsed using CPU:', timeend-timestart)
    #    print('Total number of batches =', nbatches_trained)
    #    print('Total number of samples viewed =', nsamples_viewed)

    if save_model is not None:
        torch.save(Network,datadir + 'results/model/' + save_model)

    #### Save out hidden layer RSM
    #hidden, rsm = analysis.rsa(Network,show=save_hiddenrsm_pdf,savepdf=save_hiddenrsm_pdf)
    ## hidden = hidden.detach().numpy()
    ## input_matrix = input_matrix.detach().numpy()

    ## Save out RSM 
    #if save_csv:
    #    np.savetxt('ANN1280_HiddenLayerRSM_NoDynamics.csv',rsm)

    return Network, TrialObj, ntrials_viewed
