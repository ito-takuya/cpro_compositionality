import numpy as np
np.set_printoptions(suppress=True)
#import matplotlib.pyplot as plt
#import seaborn as sns
#import scipy.stats as stats
import os
#os.sys.path.append('model/')

import model.model_cl as mod
import model.task as task
import time
import model.analysis as analysis
from importlib import reload
mod = reload(mod)
task = reload(task)
analysis = reload(analysis)
import torch
import pandas as pd
#from torch.autograd import Variable
#import torch.nn.functional as F

datadir = '../../data/'

def runModel(experiment,si_c=0,datadir=datadir,practice=True,learning='online',
             num_hidden=128,learning_rate=0.0001,thresh=0.5,acc_cutoff=95.0,
             save_rsm=False,save_hiddenrsm_pdf=False,save_model=None,verbose=True,
             lossfunc='MSE'):
    """
    'online training model'
    num_hidden - # of hidden units
    learning_rate - learning rate 
    thresh - threshold for classifying output units
    save_rsm - Save out the RSM?
    save_hiddenrsm_pdf - save out a PDF of the RSM?
    """

    #### ANN construction
    network = mod.ANN(num_rule_inputs=12,
                         si_c=si_c,
                         num_sensory_inputs=16,
                         num_hidden=num_hidden,
                         num_motor_decision_outputs=4,
                         learning_rate=learning_rate,
                         thresh=thresh,
                         lossfunc=lossfunc)
    # network.cuda = True
    network = network.cpu()

    if practice:
        #### Load training batches
        if verbose: print('Loading practice and (all tasks) batches')
        practice_input_batches = experiment.practice_input_batches
        practice_output_batches = experiment.practice_output_batches
        

        # Register starting param-values (needed for "intelligent synapses").
        if network.si_c>0:
            W = {}
            p_old = {}
            for n, p in network.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    # Set initial    
                    W[n] = p.data.clone().zero_()
                    p_old[n] = p.data.clone()
                    # Store initial tensors in state dict
                    network.register_buffer('{}_SI_prev_task'.format(n), p.data.clone())
                    #omega = p.detach().clone().zero_()
                    #network.register_buffer('{}_SI_omega'.format(n), omega)
            network.update_omega(W,network.epsilon)
        else:
            W = None

            


        #### train practiced tasks 
        tmpcutoff = 95.0
        ntrials_viewed, nbatches_trained = mod.task_training(network,
                                                              practice_input_batches,
                                                              practice_output_batches,
                                                              acc_cutoff=tmpcutoff,
                                                              si=W,
                                                              dropout=True,
                                                              cuda=False,
                                                              verbose=verbose)  

        if network.si_c>0:
            for n, p in network.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        W[n].add_(-p.grad*(p.detach()-p_old[n])) # parameter-specific contribution to changes in total loss of completed task

                    p_old[n] = p.data.clone()
                    
            network.update_omega(W, network.epsilon)

        online_accuracy = []
        ####
        if learning=='online':
            ntrials_per_task_online = 50
            online_inputs = experiment.online_input_batches # task x stim x input
            online_outputs = experiment.online_output_batches # task x stim x output
            n_tasks = online_inputs.shape[0]
            n_stims = online_inputs.shape[1]
            count = 0
            accuracy = 0
            while accuracy*100.0 < acc_cutoff:
                acc = []
                for task in range(n_tasks):
                    
                    #stim = np.random.randint(n_stims) # pick a random stimulus set to pair with this task
                    stim = np.random.choice(np.arange(n_stims),ntrials_per_task_online,replace=False) # pick a random stimulus set to pair with this task
                    if len(stim)==1: stim=stim[0]
                    if network.si_c>0:
                        outputs, targets, loss = mod.train(network,
                                                           online_inputs[task,stim,:],
                                                           online_outputs[task,stim,:],
                                                           si=True,dropout=False)
                    else:
                        outputs, targets, loss = mod.train(network,
                                                           online_inputs[task,stim,:],
                                                           online_outputs[task,stim,:],
                                                           si=False,dropout=False)

                    ### Update running parameter importance estimates in W
                    #if network.si_c>0:
                    #    for n, p in network.named_parameters():
                    #        if p.requires_grad:
                    #            n = n.replace('.', '__')
                    #            if p.grad is not None:
                    #                W[n].add_(-p.grad*(p.detach()-p_old[n]))
                    ##                print(p.detach()-p_old[n])
                    #            p_old[n] = p.detach().clone()

                    #    # SI: calculate and update the normalized path integral
                    #    network.update_omega(W, network.epsilon)
                        
                    if ntrials_per_task_online>1:
                        acc.extend(mod.accuracyScore(network,outputs,targets))
                        ntrials_viewed += len(stim)
                    else:
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
                #if count==2:
                #    raise Exception('still need to debug')

                count += 1

                if accuracy*100.0>acc_cutoff: 
                    if verbose: print('Achieved', accuracy*100.0, '%, greater than', acc_cutoff, '% threshold -  exiting training after', count, 'batches') 
                
                #if count >2: break

    ##TODO## if need to train trials in batches (rather than single trial)
        if learning=='batch':

            online_inputs = experiment.online_input_batches # task x stim x input
            online_outputs = experiment.online_output_batches # task x stim x output
            # Calculate the number of trials to show during this training phase

        #    print("Training all tasks with", nbatches_trained, "batches")
            n_stims = online_inputs.shape[1]
            accuracy = 0
            batch = 0
            #while batch < nbatches_trained:
            #while accuracy < acc_cutoff:
            task_ind = np.arange(64)
            np.random.shuffle(task_ind)
            for task in task_ind:
                #stim = np.random.choice(np.arange(n_stims),1,replace=False) # pick a random stimulus set to pair with this task
                #if len(stim)==1: stim=stim[0]
                if network.si_c>0:
                    #outputs, targets, loss = mod.train(network,
                    #                                   torch.flatten(online_inputs,start_dim=0,end_dim=1),
                    #                                   torch.flatten(online_outputs,start_dim=0,end_dim=1),
                    #                                   si=True,dropout=False)
                    outputs, targets, loss = mod.train(network,
                                                       online_inputs[task,:,:],
                                                       online_outputs[task,:,:],
                                                       si=True,dropout=False)
                
                else:
                    #outputs, targets, loss = mod.train(network,
                    #                                   torch.flatten(online_inputs,start_dim=0,end_dim=1),
                    #                                   torch.flatten(online_outputs,start_dim=0,end_dim=1),
                    #                                   si=False,dropout=False)
                    outputs, targets, loss = mod.train(network,
                                                       online_inputs[task,:,:],
                                                       online_outputs[task,:,:],
                                                       si=False,dropout=False)

                accuracy = np.mean(mod.accuracyScore(network,outputs,targets)) * 100.0

                if verbose:
                    if batch%50==0:
                        print('Achieved', accuracy, '%,  after', batch, 'batches') 

                batch += 1



            print('Exited training with accuracy', accuracy, '% and loss:', loss) 

    if save_model is not None:
        torch.save(network,datadir + 'results/model/' + save_model)

    #### Save out hidden layer RSM
    #hidden, rsm = analysis.rsa(network,show=save_hiddenrsm_pdf,savepdf=save_hiddenrsm_pdf)
    ## hidden = hidden.detach().numpy()
    ## input_matrix = input_matrix.detach().numpy()

    ## Save out RSM 
    #if save_rsm:
    #    np.savetxt('ANN1280_HiddenLayerRSM_NoDynamics.csv',rsm)

    return network, ntrials_viewed, online_accuracy

