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
             num_hidden=128,learning_rate=0.0001,acc_cutoff=95.0,
             save_model=None,verbose=True,
             lossfunc='MSE',pretraining=False,device='cpu'):
    """
    'online training model'
    num_hidden - # of hidden units
    learning_rate - learning rate 
    """

    #### ANN construction
    network = mod.ANN(num_rule_inputs=12,
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

#        practice_input_batches = experiment.practice_input_batches
#        practice_output_batches = experiment.practice_output_batches        
        
        ##### First motor rule only pretraining
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
        loss = 0
        count = 0
        while loss1>0.01 or loss2>0.01 or loss>0.01: 

            ##### Motor rule pretraining
            #outputs, targets, loss = mod.train(network,
            #                                   pretraining_input,
            #                                   pretraining_output,
            #                                   si=W,dropout=True)
            #loss = loss.detach().numpy()

            #### Logical sensory task pretraining
            outputs, targets, loss1 = mod.train(network,
                                               logicalsensory_pretraining_input,
                                               logicalsensory_pretraining_output,
                                               si=W,dropout=True)

            #accuracy = np.mean(mod.accuracyScore(network,outputs,targets))*100.0
            loss1 = loss1
            #accuracy1 = np.mean(mod.accuracyScore(network,outputs,targets))*100.0

            outputs, targets, loss2 = mod.train(network,
                                               sensorimotor_pretraining_input,
                                               sensorimotor_pretraining_output,
                                               si=W,dropout=True)

            loss2 = loss2
            #accuracy2 = np.mean(mod.accuracyScore(network,outputs,targets))*100.0


#            outputs, targets, loss3 = mod.train(network,
#                                                practice_input_batches,
#                                                practice_output_batches,
#                                                si=W,dropout=True)

            #accuracy = np.mean(mod.accuracyScore(network,outputs,targets))*100.0

            if verbose: 
                if count%200==0:
                    print('**PRETRAINING**  iteration', count)
                    print('\tloss on logicalsensory task:', loss1)
                    print('\tloss on sensorimotor task:', loss2)
                    #print('\tloss on motor rule pretraining:', loss)
                  #  print('\taccuracy on practiced tasks:', accuracy)


            count += 1

        #### Sensorimotor task pretraining

        if network.si_c>0:
            network.update_omega(W, network.epsilon)

    ###### 
    accuracy = 0
    ntrials_viewed=0
    online_accuracy=[]
    if practice:
        network.optimizer = torch.optim.SGD(network.parameters(), lr=0.025)
        #### Load training batches
        if verbose: print('Loading practice and (all tasks) batches')
        practice_input_batches = experiment.practice_input_batches
        practice_output_batches = experiment.practice_output_batches        

        ################################################################
        accuracy_prac = 0
        accuracy_simp = 0
        count = 0
        #while accuracy_prac < acc_cutoff or accuracy1 < acc_cutoff or accuracy2 < acc_cutoff: 
        while accuracy_prac < acc_cutoff:
            
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

            network = network.to(device)
            outputs, targets, loss = mod.train(network,
                                               practice_input_batches,
                                               practice_output_batches,
                                               si=W,dropout=True)
            
            accuracy_prac = np.mean(mod.accuracyScore(network,outputs,targets))*100.0

            if verbose: 
                if count%10==0:
                    print('**PRACTICED training** iteration', count)
                    print('\tPracticed tasks:', accuracy_prac)
                 #   print('\tSensorimotor tasks:', accuracy2)
                 #   print('\tLogicalsensory tasks:', accuracy1)

            count += 1

        if network.si_c>0:
            network.update_omega(W, network.epsilon)


        ###############################################################

        if network.si_c>0:
            for n, p in network.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        W[n].add_(-p.grad*(p.detach()-p_old[n])) # parameter-specific contribution to changes in total loss of completed task

                    p_old[n] = p.data.clone()
                    
            network.update_omega(W, network.epsilon)

        #network.optimizer = torch.optim.SGD(network.parameters(), lr=0.05)
        online_accuracy = []
        ####
        if learning=='online':
            ntrials_per_task_online = 256
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
                            if max_resp==target_resp and response>network.thresh: # Make sure network response is correct respnose, and that it exceeds some threshold
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
            #np.random.shuffle(task_ind)
            for i in range(2):
                stim = np.random.choice(np.arange(n_stims),1,replace=False) # pick a random stimulus set to pair with this task
                if len(stim)==1: stim=stim[0]
                if network.si_c>0:
                    #outputs, targets, loss = mod.train(network,
                    #                                   torch.flatten(online_inputs,start_dim=0,end_dim=1),
                    #                                   torch.flatten(online_outputs,start_dim=0,end_dim=1),
                    #                                   si=True,dropout=False)
                    outputs, targets, loss = mod.train(network,
                                                       online_inputs[:,stim,:],
                                                       online_outputs[:,stim,:],
                                                       si=True,dropout=False)
                
                else:
                    #outputs, targets, loss = mod.train(network,
                    #                                   torch.flatten(online_inputs,start_dim=0,end_dim=1),
                    #                                   torch.flatten(online_outputs,start_dim=0,end_dim=1),
                    #                                   si=False,dropout=False)
                    outputs, targets, loss = mod.train(network,
                                                       online_inputs[:,stim,:],
                                                       online_outputs[:,stim,:],
                                                       si=False,dropout=False)

                accuracy = np.mean(mod.accuracyScore(network,outputs,targets)) * 100.0

                if verbose:
                    if batch%50==0:
                        print('Achieved', accuracy, '%,  after', batch, 'batches') 

                batch += 1



            print('Exited training with accuracy', accuracy, '% and loss:', loss) 

    if save_model is not None:
        torch.save(network,datadir + 'results/model/' + save_model)

    return network, ntrials_viewed, online_accuracy

