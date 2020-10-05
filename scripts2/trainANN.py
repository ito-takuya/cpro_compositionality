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
from importlib import reload
mod = reload(mod)
task = reload(task)
import torch
import pandas as pd
#from torch.autograd import Variable
#import torch.nn.functional as F

datadir = '../../data/'

def train(experiment,si_c=0,datadir=datadir,practice=True,
             num_hidden=128,learning_rate=0.0001,acc_cutoff=95.0,
             save_model=None,verbose=True,save=True,
             lossfunc='MSE',pretraining=False,device='cpu'):
    """
    'online training model'
    num_hidden - # of hidden units
    learning_rate - learning rate 
    """

    #### ANN construction
    network = mod.ANN(num_rule_inputs=11,
                         si_c=si_c,
                         num_sensory_inputs=16,
                         num_hidden=num_hidden,
                         num_motor_decision_outputs=6,
                         learning_rate=learning_rate,
                         lossfunc=lossfunc,device=device)

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

    if pretraining:
        nbatches_pretraining = 200
        pretraining_input = experiment.pretraining_input
        pretraining_output= experiment.pretraining_output

        sensorimotor_pretraining_input = experiment.sensorimotor_pretraining_input
        sensorimotor_pretraining_output= experiment.sensorimotor_pretraining_output

        logicalsensory_pretraining_input = experiment.logicalsensory_pretraining_input
        logicalsensory_pretraining_output= experiment.logicalsensory_pretraining_output

        ##### First motor rule only pretraining
        loss = 1
        while loss>0.01: 

            #### Motor rule pretraining
            outputs, targets, loss = mod.train(network,
                                               pretraining_input,
                                               pretraining_output,
                                               si=W,dropout=True)
        #for i in range(nbatches_pretraining):
        #    outputs, targets, loss = mod.train(network,
        #                                       pretraining_input,
        #                                       pretraining_output,
        #                                       si=W,dropout=True)

        #if network.si_c>0:
        #    network.update_omega(W, network.epsilon)

    
        ##### Now train on simple logicalsensory rule pretraining
        loss1 = 1
        loss2 = 1
        count = 0
        while loss1>0.01 or loss2>0.01 or loss>0.01: 

            ##### Motor rule pretraining
            #outputs, targets, loss = mod.train(network,
            #                                   pretraining_input,
            #                                   pretraining_output,
            #                                   si=W,dropout=True)

            #### Logical sensory task pretraining
            outputs, targets, loss1 = mod.train(network,
                                               logicalsensory_pretraining_input,
                                               logicalsensory_pretraining_output,
                                               si=W,dropout=True)

            #accuracy = np.mean(mod.accuracyScore(network,outputs,targets))*100.0
            #accuracy1 = np.mean(mod.accuracyScore(network,outputs,targets))*100.0

            outputs, targets, loss2 = mod.train(network,
                                               sensorimotor_pretraining_input,
                                               sensorimotor_pretraining_output,
                                               si=W,dropout=True)

            #accuracy2 = np.mean(mod.accuracyScore(network,outputs,targets))*100.0

            #if verbose: 
            #    if count%200==0:
            #        print('**PRETRAINING**  iteration', count)
            #        print('\tloss on logicalsensory task:', loss1)
            #        print('\tloss on sensorimotor task:', loss2)
            #        #print('\tloss on motor rule pretraining:', loss)
            #      #  print('\taccuracy on practiced tasks:', accuracy)


            count += 1

        #### Sensorimotor task pretraining

        if network.si_c>0:
            network.update_omega(W, network.epsilon)

    ###### 
    accuracy = 0
    online_accuracy=[]
    if practice:
        #network.optimizer = torch.optim.SGD(network.parameters(), lr=0.01)
        #network.optimizer = torch.optim.SGD(network.parameters(), lr=0.025)
        network.optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)
        #### Load training batches
        if verbose: print('Loading practice and (all tasks) batches')
        prac_input2d = experiment.prac_input2d
        prac_target2d = experiment.prac_target2d

        ################################################################
        accuracy_prac = 0
        accuracy_simp = 0
        count = 0
        #while accuracy_prac < acc_cutoff or accuracy1 < acc_cutoff or accuracy2 < acc_cutoff: 
        ## old
        #while accuracy_prac < acc_cutoff:
        
        ## new
        # While alll tasks are under the accuracy cutoff
        while accuracy_prac < experiment.prac_inputs.shape[0]:
            
        #    outputs, targets, loss1 = mod.train(network,
        #                                       logicalsensory_pretraining_input,
        #                                       logicalsensory_pretraining_output,
        #                                       si=W,dropout=True)
        #    #if lossfunc=='CrossEntropy': network.lossfunc = torch.nn.CrossEntropyLoss()

        #    #accuracy = np.mean(mod.accuracyScore(network,outputs,targets))*100.0
        #    loss1 = loss1.detach().numpy()
        #    accuracy1 = np.mean(mod.accuracyScore(network,outputs,targets))*100.0

        #    outputs, targets, loss2 = mod.train(network,
        #                                       sensorimotor_pretraining_input,
        #                                       sensorimotor_pretraining_output,
        #                                       si=W,dropout=True)

        #    loss2 = loss2.detach().numpy()
        #    accuracy2 = np.mean(mod.accuracyScore(network,outputs,targets))*100.0
            #for i in range(len(experiment.practicedRuleSet)):
            #    outputs, targets, loss = mod.train(network,
            #                                       prac_input[i,:,:],
            #                                       prac_target2d[i,:,:],
            #                                       si=W,dropout=True)
            #    if loss<0.1:
            #        tmp_acc = np.mean(mod.accuracyScore(network,outputs,targets))*100.0

            acc = []
            order = np.arange(experiment.prac_inputs.shape[0])
            np.random.shuffle(order)
            for t in order:

                outputs, targets, loss = mod.train(network,
                                                   experiment.prac_inputs[t,:,:],
                                                   experiment.prac_targets[t,:],
                                                   si=W,dropout=True)

                acc.append(mod.accuracyScore(network,outputs,targets)*100.0)
            
            accuracy_prac = np.sum(np.asarray(acc)>acc_cutoff)

            if verbose: 
                if count%200==0:
                    print('**TRAINING**  iteration', count)
                    print('\tavg accuracy on practiced tasks:', np.mean(np.asarray(acc)), '|', np.sum(np.asarray(acc)>acc_cutoff))

            #### GOOD OLD VERSION
            #outputs, targets, loss = mod.train(network,
            #                                   prac_input2d,
            #                                   prac_target2d,
            #                                   si=W,dropout=True)
            #accuracy_prac = mod.accuracyScore(network,outputs,targets)*100.0



            #if loss<0.1:
            #accuracy_prac = np.mean(mod.accuracyScore(network,outputs,targets))*100.0


            count += 1

    if save:
        if save_model is not None:
            torch.save(network,datadir + 'results/model/' + save_model)

    return network, online_accuracy


