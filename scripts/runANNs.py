import numpy as np
np.set_printoptions(suppress=True)
#import matplotlib.pyplot as plt
#import seaborn as sns
#import scipy.stats as stats
import os
#os.sys.path.append('model/')

import model.model as mod
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


def runModel(experiment, datadir=datadir,practice=True,learning='online',
             num_hidden=512,learning_rate=0.01,thresh=0.0,acc_cutoff=90.0,
             save_csv=False,save_hiddenrsm_pdf=False,save_model=None,verbose=True):
    """
    'online training model'
    num_hidden - # of hidden units
    learning_rate - learning rate 
    thresh - threshold for classifying output units
    save_csv - Save out the RSM?
    save_hiddenrsm_pdf - save out a PDF of the RSM?
    """

    #### ANN construction
    if verbose: print('Instantiating new model')
    Network = mod.ANN(num_rule_inputs=12,
                         num_sensory_inputs=16,
                         num_hidden=num_hidden,
                         num_motor_decision_outputs=4,
                         learning_rate=learning_rate,
                         thresh=thresh)
    # Network.cuda = True
    Network = Network.cpu()

    if practice:
        #### Load training batches
        if verbose: print('Loading practice and (all tasks) batches')
        practice_input_batches = experiment.practice_input_batches
        practice_output_batches = experiment.practice_output_batches

        #### train practiced tasks 
        ntrials_viewed, nbatches_trained1 = mod.task_training(Network,practice_input_batches,practice_output_batches,acc_cutoff=acc_cutoff,cuda=False,verbose=verbose)  

        online_accuracy = []
        ####
        if learning=='online':
            online_inputs = experiment.online_input_batches # task x stim x input
            online_outputs = experiment.online_output_batches # task x stim x output
            n_tasks = online_inputs.shape[0]
            n_stims = online_inputs.shape[1]
            count = 0
            accuracy = 0
            while accuracy*100.0 < acc_cutoff:
                acc = []
                for task in range(n_tasks):
                    stim = np.random.randint(n_stims) # pick a random stimulus set to pair with this task
                    outputs, targets, loss = mod.train(Network,
                                                       online_inputs[task,stim,:],
                                                       online_outputs[task,stim,:])
                    #### Now score output
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
                
                accuracy = np.mean(acc)
                online_accuracy.append(accuracy)
                if verbose and count%50==0:
                    print('Batch', count, 'achieved', accuracy*100.0,'%')

                count += 1

                if accuracy*100.0>acc_cutoff: 
                    if verbose: print('Achieved', accuracy*100.0, '%, greater than', acc_cutoff, '% threshold -  exiting training after', count, 'batches') 
                
                #if count >2: break
##TODO## not prepared
#        if learning=='batch':
#
#            input_trials = experiment.online_input_batches # task x stim x input
#            output_trials = experiment.online_output_batches # task x stim x output
#            n_tasks = input_trials.shape[0]
#            n_stims = input_trials.shape[1]
#            stim_ind = np.arange(n_stims)
#            np.random.shuffle(stim_ind)
#            i = 0
#            accuracy = 0
#            batch = 0 
#            while accuracy < acc_cutoff:
#                stim = stim_ind[i]
#                outputs, targets, loss = mod.train(Network,
#                                                   input_trials[:,stim,:],
#                                                   output_trials[:,stim,:])
#
#                accuracy = np.mean(mod.accuracyScore(Network,outputs,targets)) * 100.0
#
#                if verbose:
#                    print('Batch', batch, 'achieved', accuracy,'%')
#
#                i += 1
#                if i >= n_stims:
#                    i = 0 
#                    np.random.shuffle(stim_ind)
#
#                ntrials_viewed += n_tasks
#                batch += 1

    if save_model is not None:
        torch.save(Network,datadir + 'results/model/' + save_model)

    #### Save out hidden layer RSM
    #hidden, rsm = analysis.rsa(Network,show=save_hiddenrsm_pdf,savepdf=save_hiddenrsm_pdf)
    ## hidden = hidden.detach().numpy()
    ## input_matrix = input_matrix.detach().numpy()

    ## Save out RSM 
    #if save_csv:
    #    np.savetxt('ANN1280_HiddenLayerRSM_NoDynamics.csv',rsm)

    return Network, ntrials_viewed, online_accuracy

