#### Experiment 2
# for each simulation, incrementally train additional tasks (i.e., include more practiced tasks) and assess performance etc.

import numpy as np
import argparse
np.set_printoptions(suppress=True)
import os
import model.model as mod
import model.task as task
import time
import model.analysis as analysis
from importlib import reload
import trainANN as trainANN
mod = reload(mod)
task = reload(task)
analysis = reload(analysis)
import torch
import pandas as pd
import tools

datadir = '../../data/'

parser = argparse.ArgumentParser('./main.py', description='Run a set of simulations/models')
parser.add_argument('--nsimulations', type=int, default=20, help='number of models/simulations to run')
parser.add_argument('--pretraining', action='store_true', help="pretrain network on simple tasks to improve compositionality")
parser.add_argument('--practice', action='store_true', help="Train on practiced tasks")
parser.add_argument('--acc_cutoff', type=float, default=95.0, help="condition for exiting ANN training")
parser.add_argument('--cuda', action='store_true', help="use gpu/cuda")
parser.add_argument('--save_model', type=str, default="ANN", help='string name to output models')
parser.add_argument('--verbose', action='store_true', help='verbose')
parser.add_argument('--num_hidden', type=int, default=256, help="number of units in hidden layers")
parser.add_argument('--learning_rate', type=float, default=0.001, help="learning rate for pretraining sessions (ADAM default)")
parser.add_argument('--save', action='store_true', help="save or don't save model")
parser.add_argument('--batchname', type=str, default='Experiment_FullTaskSet_11LogicInputs', help='string name for the experiment filename')
parser.add_argument('--si_c', type=float, default=0.0, help='synaptic intelligence parameter (Zenke et al. 2017); default=0, meaning no synaptic intelligence implemented')

def run(args):
    args 
    nsimulations = args.nsimulations
    si_c = args.si_c
    practice = args.practice
    num_hidden = args.num_hidden
    learning_rate = args.learning_rate
    acc_cutoff = args.acc_cutoff
    save_model = args.save_model
    save = args.save
    batchname = args.batchname
    pretraining = args.pretraining
    cuda = args.cuda
    verbose = args.verbose


    #save_model = save_model + '_' + batchname
    save_model = save_model + '_' + str(int(acc_cutoff)) + 'acc'
    if pretraining:
        save_model = save_model + '_pretraining'

    if practice:
        save_model = save_model + '_practice'

    if cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'

    # batchfilename = datadir + 'results/model/TrialBatches_4Prac60Nov_FullStimSets'
    batchfilename = datadir + 'results/model/' + batchname
    experiment = task.Experiment(filename=batchfilename)


    #########################################
    print("create task sets -- all simulations start with the same base 'practiced set'")
    #### Construct test set for 'practiced' trials based on task similarity
    prac_inputs, prac_targets = task.create_all_trials(experiment.practicedRuleSet)
    prac_inputs = torch.from_numpy(prac_inputs.T).float()
    prac_targets = torch.from_numpy(prac_targets.T).long()
    if cuda:
        prac_inputs = prac_inputs.cuda()
        prac_targets = prac_targets.cuda()
    
    prac_input2d = prac_inputs.reshape(prac_inputs.shape[0]*prac_inputs.shape[1],prac_inputs.shape[2])
    prac_target2d = torch.flatten(prac_targets)

    experiment.prac_input2d = prac_input2d
    experiment.prac_target2d = prac_target2d
    # Also store the unflattened version 
    experiment.prac_inputs = prac_inputs
    experiment.prac_targets = prac_targets

    #### Load novel tasks
    novel_inputs, novel_targets = task.create_all_trials(experiment.novelRuleSet)
    novel_inputs = torch.from_numpy(novel_inputs.T).float() # task x trials/stimuli x input units
    novel_targets = torch.from_numpy(novel_targets.T).long() # task x stimuli
    if cuda:
        novel_inputs = novel_inputs.cuda()
        novel_targets = novel_targets.cuda()
    experiment.novel_inputs = novel_inputs 
    experiment.novel_targets = novel_targets

    #### Load pretraining task data
    if pretraining:
        pretraining_input, pretraining_output = task.create_motorrule_pretraining()  
        pretraining_input = torch.from_numpy(pretraining_input).float()
        pretraining_output = torch.from_numpy(pretraining_output).long()
        if cuda:
            pretraining_input = pretraining_input.cuda()
            pretraining_output = pretraining_output.cuda()
        experiment.pretraining_input = pretraining_input
        experiment.pretraining_output = pretraining_output

        sensorimotor_pretraining_input, sensorimotor_pretraining_output = task.create_sensorimotor_pretraining()
        sensorimotor_pretraining_input = torch.from_numpy(sensorimotor_pretraining_input).float()
        sensorimotor_pretraining_output = torch.from_numpy(sensorimotor_pretraining_output).long()
        if cuda:
            sensorimotor_pretraining_input = sensorimotor_pretraining_input.cuda()
            sensorimotor_pretraining_output = sensorimotor_pretraining_output.cuda()
        experiment.sensorimotor_pretraining_input = sensorimotor_pretraining_input
        experiment.sensorimotor_pretraining_output = sensorimotor_pretraining_output

        logicalsensory_pretraining_input, logicalsensory_pretraining_output = task.create_logicalsensory_pretraining()
        logicalsensory_pretraining_input = torch.from_numpy(logicalsensory_pretraining_input).float()
        logicalsensory_pretraining_output = torch.from_numpy(logicalsensory_pretraining_output).long()
        if cuda:
            logicalsensory_pretraining_input = logicalsensory_pretraining_input.cuda()
            logicalsensory_pretraining_output = logicalsensory_pretraining_output.cuda()
        experiment.logicalsensory_pretraining_input = logicalsensory_pretraining_input
        experiment.logicalsensory_pretraining_output = logicalsensory_pretraining_output


    full_inputs = torch.cat((prac_inputs,novel_inputs),0)
    full_targets = torch.cat((prac_targets,novel_targets),0)


    ###########################################
    #### run simulations
    df_group = {}
    df_group['Accuracy'] = []
    df_group['ContextDimensionality'] = []
    df_group['ResponseDimensionality'] = []
    df_group['Simulation'] = []
    df_group['NumPracticedTasks'] = []

    df_pertask = {}
    df_pertask['Accuracy'] = []
    df_pertask['Condition'] = []
    df_pertask['Logic'] = []
    df_pertask['Sensory'] = []
    df_pertask['Motor'] = []
    df_pertask['Simulation'] = []
    df_pertask['NumPracticedTasks'] = []
    n_practiced_tasks = len(experiment.practicedRuleSet)
    while n_practiced_tasks < len(experiment.taskRuleSet):

        #### Run simulation
        avg_nov_acc = []
        for i in range(nsimulations):
            modelname = save_model + str(i) + '.pt'
            if verbose: print('** TRAINING ON', n_practiced_tasks, 'PRACTICED TASKS ** ... simulation', i, ' |', modelname, '| cuda:', cuda)
            network, acc = trainANN.train(experiment,si_c=si_c,acc_cutoff=acc_cutoff,datadir=datadir,practice=practice,
                                          num_hidden=num_hidden,learning_rate=learning_rate,save=save,
                                          save_model=modelname,verbose=False,lossfunc='CrossEntropy',pretraining=pretraining,device=device)
        

            network.eval()
            
            #### Save accuracies by task
            for i in range(len(experiment.practicedRuleSet)):
                outputs, hidden = network.forward(experiment.prac_inputs[i,:,:],noise=False)
                outputs[:,4:] = 0
                acc = mod.accuracyScore(network,outputs,experiment.prac_targets[i,:])
                df_pertask['Accuracy'].append(acc)
                df_pertask['Condition'].append('Practiced')
                df_pertask['Logic'].append(experiment.practicedRuleSet.Logic[i])
                df_pertask['Sensory'].append(experiment.practicedRuleSet.Sensory[i])
                df_pertask['Motor'].append(experiment.practicedRuleSet.Motor[i])
                df_pertask['Simulation'].append(i)
                df_pertask['NumPracticedTasks'].append(n_practiced_tasks)

            novel_acc = []
            for i in range(len(experiment.novelRuleSet)):
                outputs, hidden = network.forward(experiment.novel_inputs[i,:,:],noise=False)
                outputs[:,4:] = 0
                acc = mod.accuracyScore(network,outputs,experiment.novel_targets[i,:])
                novel_acc.append(acc)
                df_pertask['Accuracy'].append(acc)
                df_pertask['Condition'].append('Novel')
                df_pertask['Logic'].append(experiment.novelRuleSet.Logic[i])
                df_pertask['Sensory'].append(experiment.novelRuleSet.Sensory[i])
                df_pertask['Motor'].append(experiment.novelRuleSet.Motor[i])
                df_pertask['Simulation'].append(i)
                df_pertask['NumPracticedTasks'].append(n_practiced_tasks)

                
            if verbose: print('\tAccuracy on novel tasks:', np.mean(novel_acc)) 

            # novel trial accuracy
            df_group['Accuracy'].append(np.mean(novel_acc))
            df_group['Simulation'].append(i)
            df_group['NumPracticedTasks'].append(n_practiced_tasks)
            hidden, rsm_corr = analysis.rsa_context(network,batchfilename=batchfilename,measure='corr')
            df_group['ContextDimensionality'].append(tools.dimensionality(rsm_corr))
            #### response dimensionality - requires the task input/output set
            hidden, rsm_corr = analysis.rsa_behavior(network,full_inputs,full_targets,measure='corr')
            df_group['ResponseDimensionality'].append(tools.dimensionality(rsm_corr))

            avg_nov_acc.append(np.mean(novel_acc))

            
        print('**Averages across simulations** |', n_practiced_tasks, 'number of training tasks', ' |', modelname, '| cuda:', cuda)
        print('\t Novel task acc =',np.mean(avg_nov_acc))


        
        #### Update and transfer novel task to practiced tasks
        practicedRuleSet, novelRuleSet, nov2prac_ind = experiment.addPracticedTasks(n=1) # Add a random novel task to the practiced set
        nov2prac_ind = nov2prac_ind[0]
        new_novel_ind = np.where(experiment.novelRuleSet.index!=nov2prac_ind)[0] 
        experiment.practicedRuleSet = practicedRuleSet
        experiment.novelRuleSet = novelRuleSet
        #
        experiment.prac_input2d = torch.cat((experiment.prac_input2d, experiment.novel_inputs[nov2prac_ind,:]),0)
        experiment.prac_target2d = torch.cat((experiment.prac_target2d, experiment.novel_targets[nov2prac_ind,:]),0)
        #
        new_novel_input = experiment.novel_inputs[nov2prac_ind,:].unsqueeze(0) # add empty dimension to stack
        new_novel_target = experiment.novel_targets[nov2prac_ind,:].unsqueeze(0) # add empty dimension to stack
        experiment.prac_inputs = torch.cat((experiment.prac_inputs, new_novel_input),0)
        experiment.prac_targets = torch.cat((experiment.prac_targets, new_novel_target),0)
        #
        experiment.novel_inputs = experiment.novel_inputs[new_novel_ind,:,:]
        experiment.novel_targets = experiment.novel_targets[new_novel_ind,:]

        n_practiced_tasks += 1

    outputdir = datadir + '/results/experiment2/'
    
    df_group = pd.DataFrame(df_group) 
    df_group.to_csv(outputdir + save_model + '_GroupData' + '.csv')

    df_pertask = pd.DataFrame(df_pertask)
    df_pertask.to_csv(outputdir + save_model + '_PerTaskData' + '.csv')


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
