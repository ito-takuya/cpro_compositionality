import numpy as np
import model.task as task
np.set_printoptions(suppress=True)
import os

import model.model as mod
import time
from importlib import reload
mod = reload(mod)
import torch
import pandas as pd
#from torch.autograd import Variable
#import torch.nn.functional as F

datadir = '../../data/'

def train(experiment,si_c=0,datadir=datadir,practice=True,
          num_rule_inputs=11,num_hidden=128,num_hidden_layers=2,learning_rate=0.0001,
          acc_cutoff=95.0,n_epochs=None,optimizer='adam',
          save_model=None,verbose=True,save=True,
          lossfunc='MSE',pretraining=False,device='cpu'):
    """
    'online training model'
    num_hidden - # of hidden units
    learning_rate - learning rate 
    """

    #### ANN construction
    network = mod.ANN(num_rule_inputs=num_rule_inputs,
                         si_c=si_c,
                         num_sensory_inputs=16,
                         num_hidden_layers=num_hidden_layers,
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

        if hasattr(experiment,'posneg'):
            sensorimotor_pretraining_input_neg = experiment.sensorimotor_pretraining_input_neg
            sensorimotor_pretraining_output_neg = experiment.sensorimotor_pretraining_output_neg


        ##### First motor rule only pretraining
#        loss = 1
#        while loss>0.01: 
#
#            #### Motor rule pretraining
#            outputs, targets, loss = mod.train(network,
#                                               pretraining_input,
#                                               pretraining_output,
#                                               si=W,dropout=True)
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
        # If there's the negative equivalent of the sensorimotor task, then make sure it trains loss
        if hasattr(experiment,'posneg'):
            loss3 = 1
        else: 
            loss3 = 0
        count = 0
        lossmagnitude = 0.01
        while loss1>lossmagnitude or loss2>lossmagnitude or loss3>lossmagnitude: 

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

            if hasattr(experiment,'posneg'):
                outputs, targets, loss3 = mod.train(network,
                                                   sensorimotor_pretraining_input_neg,
                                                   sensorimotor_pretraining_output_neg,
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
        if optimizer =='sgd':
            network.optimizer = torch.optim.SGD(network.parameters(), lr=0.01,momentum=0.9)
            #network.optimizer = torch.optim.SGD(network.parameters(), lr=0.025)
        if optimizer == 'adam':
            network.optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)
        #### Load training batches
        if verbose: print('Loading practice and (all tasks) batches')
        prac_input2d = experiment.prac_input2d
        prac_target2d = experiment.prac_target2d

        ################################################################
        accuracy_prac = 0
        accuracy_simp = 0
        count = 0
       
        if n_epochs is not None:

            for epoch in range(n_epochs):

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

                #outputs, targets, loss = mod.train(network,
                #                                   prac_input2d,
                #                                   prac_target2d,
                #                                   si=W,dropout=True)
                #accuracy_prac = mod.accuracyScore(network,outputs,targets)*100.0

                accuracy_prac = np.sum(np.asarray(acc)>acc_cutoff)

            print('\tTraining on practiced tasks exits with:', np.mean(np.asarray(acc)),'% after', n_epochs, 'epochs')

        else:

            while accuracy_prac < experiment.prac_inputs.shape[0]:
                
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
            torch.save(network,save_model)

    return network, online_accuracy

