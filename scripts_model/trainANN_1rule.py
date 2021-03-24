import numpy as np
import model.task as task
np.set_printoptions(suppress=True)
import os

import model.model_v2 as mod
import time
from importlib import reload
mod = reload(mod)
import torch
import pandas as pd
#from torch.autograd import Variable
#import torch.nn.functional as F

datadir = '../../data/'

def train(experiment,si_c=0,datadir=datadir,practice=True,
          num_rule_inputs=11,num_sensory_inputs=16,num_hidden=128,num_hidden_layers=2,learning_rate=0.0001,
          acc_cutoff=95.0,n_epochs=None,optimizer='adam',
          save_model=None,verbose=True,save=True,
          lossfunc='MSE',pretraining=False,rule2pretraining=False,pretraining_acc=99.0,device='cpu'):
    """
    'online training model'
    num_hidden - # of hidden units
    learning_rate - learning rate 
    """

    #### ANN construction
    network = mod.ANN(num_rule_inputs=num_rule_inputs,
                         si_c=si_c,
                         num_sensory_inputs=num_sensory_inputs,
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


    pretraining_trials = 0
    if pretraining:
        logic_pretraining_input = experiment.logic_pretraining_input
        logic_pretraining_output= experiment.logic_pretraining_output

        sensory_pretraining_input = experiment.sensory_pretraining_input
        sensory_pretraining_output= experiment.sensory_pretraining_output

        motor_pretraining_input = experiment.motor_pretraining_input
        motor_pretraining_output= experiment.motor_pretraining_output

    
        loss1 = 1
        loss2 = 1
        loss3 = 1
        loss4 = 1
        loss5 = 1
        count = 0
        lossmagnitude = 0.0001
        #lossmagnitude = 0.001
        accuracy1 = 0
        accuracy2 = 0
        accuracy3 = 0
        accuracy_condition = False
        #while loss1>lossmagnitude or loss2>lossmagnitude or loss3>lossmagnitude:
        while accuracy1<pretraining_acc or accuracy2<pretraining_acc or accuracy3<pretraining_acc:

            #### Logic task pretraining
            outputs, targets, loss1 = mod.train(network,
                                               logic_pretraining_input,
                                               logic_pretraining_output,
                                               si=W,dropout=True)

            pretraining_trials += len(outputs)
            accuracy1 = np.mean(mod.accuracyScore(network,outputs,targets))*100.0

            outputs, targets, loss2 = mod.train(network,
                                               sensory_pretraining_input,
                                               sensory_pretraining_output,
                                               si=W,dropout=True)

            pretraining_trials += len(outputs)
            accuracy2 = np.mean(mod.accuracyScore(network,outputs,targets))*100.0

            outputs, targets, loss3 = mod.train(network,
                                               motor_pretraining_input,
                                               motor_pretraining_output,
                                               si=W,dropout=True)

            pretraining_trials += len(outputs)
            accuracy3 = np.mean(mod.accuracyScore(network,outputs,targets))*100.0



            count += 1



    if rule2pretraining:

    
        ##### Now train on simple logicalsensory rule pretraining
        loss4 = 1
        loss5 = 1
        count = 0
        lossmagnitude = 0.0001
        #lossmagnitude = 0.001
        # accuracy-based training
        accuracy4 = 0
        accuracy5 = 0
        while accuracy4<pretraining_acc or accuracy5>pretraining_acc:

            outputs, targets, loss4 = mod.train(network,
                                               experiment.logicalsensory_pretraining_input,
                                               experiment.logicalsensory_pretraining_output,
                                               si=W,dropout=True)

            pretraining_trials += len(outputs)
            accuracy4 = np.mean(mod.accuracyScore(network,outputs,targets))*100.0
#
            outputs, targets, loss5 = mod.train(network,
                                               experiment.sensorimotor_pretraining_input,
                                               experiment.sensorimotor_pretraining_output,
                                               si=W,dropout=True)
            
            pretraining_trials += len(outputs)
            accuracy5 = np.mean(mod.accuracyScore(network,outputs,targets))*100.0

            count += 1

        #### Sensorimotor task pretraining

        if network.si_c>0:
            network.update_omega(W, network.epsilon)

    ###### 
    accuracy = 0
    num_trials = 0
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
                    num_trials += len(outputs)


                    acc.append(mod.accuracyScore(network,outputs,targets)*100.0)
                
            #    accuracy_prac = np.sum(np.asarray(acc)>acc_cutoff)

                #outputs, targets, loss = mod.train(network,
                #                                   prac_input2d,
                #                                   prac_target2d,
                #                                   si=W,dropout=True)
                #accuracy_prac = mod.accuracyScore(network,outputs,targets)*100.0

            #    accuracy_prac = np.sum(np.asarray(acc)>acc_cutoff)

            print('\tTraining on practiced tasks exits with:', np.mean(np.asarray(acc)),'% after', n_epochs, 'epochs')

        else:

            i_epochs = 0
            while accuracy_prac < experiment.prac_inputs.shape[0]:
                
                acc = []
                order = np.arange(experiment.prac_inputs.shape[0])
                np.random.shuffle(order)
                for t in order:

                    outputs, targets, loss = mod.train(network,
                                                       experiment.prac_inputs[t,:,:],
                                                       experiment.prac_targets[t,:],
                                                       si=W,dropout=True)

                    num_trials += len(outputs)
                    acc.append(mod.accuracyScore(network,outputs,targets)*100.0)
                
                accuracy_prac = np.sum(np.asarray(acc)>acc_cutoff)
                i_epochs += 1

                if verbose: 
                    if count%200==0:
                        print('**TRAINING**  iteration', count)
                        print('\tavg accuracy on practiced tasks:', np.mean(np.asarray(acc)), '|', np.sum(np.asarray(acc)>acc_cutoff))
                count += 1
            
            print('\tTraining on practiced tasks exits with:', np.mean(np.asarray(acc)),'% after', i_epochs, 'epochs')

            #### GOOD OLD VERSION
            #outputs, targets, loss = mod.train(network,
            #                                   prac_input2d,
            #                                   prac_target2d,
            #                                   si=W,dropout=True)
            #accuracy_prac = mod.accuracyScore(network,outputs,targets)*100.0



            #if loss<0.1:
            #accuracy_prac = np.mean(mod.accuracyScore(network,outputs,targets))*100.0


    if save:
        if save_model is not None:
            torch.save(network,save_model)

    return network, pretraining_trials, num_trials 


