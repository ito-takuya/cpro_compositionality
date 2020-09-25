import numpy as np
import argparse
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
import runANNs_CL as runModel
mod = reload(mod)
task = reload(task)
analysis = reload(analysis)
import torch
import pandas as pd
#from torch.autograd import Variable
#import torch.nn.functional as F

datadir = '../../data/'

parser = argparse.ArgumentParser('./main.py', description='Run a set of simulations/models')
parser.add_argument('--nsimulations', type=int, default=20, help='number of models/simulations to run')
parser.add_argument('--si_c', type=float, default=0.0, help='synaptic intelligence parameter (Zenke et al. 2017); default=0, meaning no synaptic intelligence implemented')
parser.add_argument('--create_new_tasks', action='store_true', help="don't create new task sets")
parser.add_argument('--practice', action='store_true', help="Train on 4 practiced tasks")
parser.add_argument('--num_hidden', type=int, default=256, help="number of units in hidden layers")
parser.add_argument('--learning', type=str, default=None, help="type of learning performed *after* practiced training")
parser.add_argument('--learning_rate', type=float, default=0.001, help="learning rate for pretraining sessions (ADAM default)")
parser.add_argument('--acc_cutoff', type=float, default=95.0, help="condition for exiting ANN training")
parser.add_argument('--save_model', type=str, default="ANN", help='string name to output models')
parser.add_argument('--batchname', type=str, default='Experiment_FullTaskSet', help='string name for the experiment filename')
parser.add_argument('--lossfunc', type=str, default='CrossEntropy', help='default: CrossEntropy, options: MSE or CrossEntropy, which determines the loss function')
parser.add_argument('--pretraining', action='store_true', help="pretrain network on simple tasks to improve compositionality")
parser.add_argument('--cuda', action='store_true', help="use gpu/cuda")
parser.add_argument('--verbose', action='store_false', help='verbose')

def run(args):
    args 
    nsimulations = args.nsimulations
    si_c = args.si_c
    create_new_tasks = args.create_new_tasks
    practice = args.practice
    learning = args.learning
    num_hidden = args.num_hidden
    learning_rate = args.learning_rate
    acc_cutoff = args.acc_cutoff
    save_model = args.save_model
    batchname = args.batchname
    lossfunc = args.lossfunc
    pretraining = args.pretraining
    cuda = args.cuda
    verbose = args.verbose

    # batchfilename = datadir + 'results/model/TrialBatches_4Prac60Nov_FullStimSets'
    batchfilename = datadir + 'results/model/' + batchname
    experiment = task.Experiment(NUM_INPUT_ELEMENTS=28,
				 NUM_OUTPUT_ELEMENTS=4,
				 filename=batchfilename)


    if create_new_tasks:
         print("creating practiced batches")
         experiment.createFullTaskSet(condition='practice')
         print("creating novel batches")
         experiment.createFullTaskSet(condition='novel')
         print("creating all batches")
         experiment.createFullTaskSet(condition='all')

    #########################################
    print("loading experimental tasks for training")
    #### Construct training set for 'practiced' and 'full' task set
    prac_inputs, prac_targets = experiment.loadFullTask(condition='practice')
    prac_inputs = prac_inputs.reshape(prac_inputs.shape[0]*prac_inputs.shape[1],prac_inputs.shape[2])
    prac_targets = prac_targets.reshape(prac_targets.shape[0]*prac_targets.shape[1],prac_targets.shape[2])
    full_inputs, full_targets = experiment.loadFullTask(condition='all')

    experiment.practice_input_batches = prac_inputs
    experiment.practice_output_batches = prac_targets

    experiment.online_input_batches = full_inputs
    experiment.online_output_batches = full_targets

    #########################################
    print("create test set experimental tasks")
    #### Construct test set
    ntrials_per_task = 100
    #### Construct test set for 'practiced' trials based on task similarity
    test_prac_inputs, test_prac_targets = task.create_all_trials(experiment.practicedRuleSet)
    test_prac_inputs = torch.from_numpy(test_prac_inputs.T).float()
    test_prac_targets = torch.from_numpy(test_prac_targets.T).float()
    test_prac_inputs = test_prac_inputs.reshape(test_prac_inputs.shape[0]*test_prac_inputs.shape[1],test_prac_inputs.shape[2])
    test_prac_targets = test_prac_targets.reshape(test_prac_targets.shape[0]*test_prac_targets.shape[1],test_prac_targets.shape[2])
    #test_prac_inputs, test_prac_targets = task.create_random_trials(experiment.practicedRuleSet,ntrials_per_task,np.random.randint(1000000))
    #test_prac_inputs = torch.from_numpy(test_prac_inputs.T).float()
    #test_prac_targets = torch.from_numpy(test_prac_targets.T).float()

    #### Now identify overlap with practiced tasks in the novel task set
    taskSim1Set, taskSim2Set = experiment.taskSimilarity(experiment.practicedRuleSet,experiment.novelRuleSet)

    # Make even the number of trials for each task set
    # ntrials1 = int((ntrials*len(trialobj.practicedRuleSet))/len(taskSim1Set))
    # ntrials2 = int((ntrials*len(trialobj.practicedRuleSet))/len(taskSim2Set))
    # ntrials1 = ntrials
    # ntrials2 = ntrials

    #### Simulate task sets with 2-rule similarities
    sim2_inputs, sim2_targets = task.create_all_trials(taskSim2Set)
    sim2_inputs = torch.from_numpy(sim2_inputs.T).float()
    sim2_targets = torch.from_numpy(sim2_targets.T).float()
    sim2_inputs = sim2_inputs.reshape(sim2_inputs.shape[0]*sim2_inputs.shape[1],sim2_inputs.shape[2])
    sim2_targets = sim2_targets.reshape(sim2_targets.shape[0]*sim2_targets.shape[1],sim2_targets.shape[2])

    #### Simulate task sets with 1-rule similarities
    sim1_inputs, sim1_targets = task.create_all_trials(taskSim1Set)
    sim1_inputs = torch.from_numpy(sim1_inputs.T).float()
    sim1_targets = torch.from_numpy(sim1_targets.T).float()
    sim1_inputs = sim1_inputs.reshape(sim1_inputs.shape[0]*sim1_inputs.shape[1],sim1_inputs.shape[2])
    sim1_targets = sim1_targets.reshape(sim1_targets.shape[0]*sim1_targets.shape[1],sim1_targets.shape[2])

    #### Load pretraining data
    if pretraining:
        pretraining_input, pretraining_output = task.create_motorrule_pretraining()  
        pretraining_input = torch.from_numpy(pretraining_input).float()
        pretraining_output = torch.from_numpy(pretraining_output).float()
        experiment.pretraining_input = pretraining_input
        experiment.pretraining_output = pretraining_output

        sensorimotor_pretraining_input, sensorimotor_pretraining_output = task.create_sensorimotor_pretraining()
        sensorimotor_pretraining_input = torch.from_numpy(sensorimotor_pretraining_input).float()
        sensorimotor_pretraining_output = torch.from_numpy(sensorimotor_pretraining_output).float()
        experiment.sensorimotor_pretraining_input = sensorimotor_pretraining_input
        experiment.sensorimotor_pretraining_output = sensorimotor_pretraining_output

        logicalsensory_pretraining_input, logicalsensory_pretraining_output = task.create_logicalsensory_pretraining()
        logicalsensory_pretraining_input = torch.from_numpy(logicalsensory_pretraining_input).float()
        logicalsensory_pretraining_output = torch.from_numpy(logicalsensory_pretraining_output).float()
        experiment.logicalsensory_pretraining_input = logicalsensory_pretraining_input
        experiment.logicalsensory_pretraining_output = logicalsensory_pretraining_output


    ###########################################
    #### run simulations
    df = {}
    df['Accuracy'] = []
    df['Condition'] = []
    df['Simulation'] = []
    df['Trials viewed'] = []
    #### Run simulation
    online_accuracies = []
    for i in range(nsimulations):
        modelname = save_model + str(i) + '.pt'
        print('**SIMULATION**', i, 'saving to file:', modelname, '| cuda:', cuda)
        network_prac2nov, ntrials_viewed, acc = runModel.runModel(experiment,si_c=si_c,acc_cutoff=acc_cutoff,learning=learning,datadir=datadir,practice=practice,
                                                                  num_hidden=num_hidden,learning_rate=learning_rate,
                                                                  save_model=modelname,verbose=True,lossfunc=lossfunc,pretraining=pretraining,cuda=cuda)

        network_prac2nov.eval()
        online_accuracies.append(acc)
            
        # practice trials
        outputs, hidden = network_prac2nov.forward(test_prac_inputs[:,:],noise=False)
        #### Set to 0 the pretraining practice outputs
        outputs[:,4:] = 0
        acc = np.mean(mod.accuracyScore(network_prac2nov,outputs,test_prac_targets[:,:]))
        df['Accuracy'].append(acc)
        df['Condition'].append('Practiced')
        df['Simulation'].append(i)
        df['Trials viewed'].append(ntrials_viewed)
        print('\t Practiced acc =',acc)
        
        # 2-rule overlap
        outputs, hidden = network_prac2nov.forward(sim2_inputs,noise=False)
        #### Set to 0 the pretraining practice outputs
        outputs[:,4:] = 0
        acc = np.mean(mod.accuracyScore(network_prac2nov,outputs,sim2_targets))
        df['Accuracy'].append(acc)
        df['Condition'].append('2-rule overlap')
        df['Simulation'].append(i)
        df['Trials viewed'].append(ntrials_viewed)
        print('\t 2-rule overlap acc =',acc)    

        # 1-rule overlap
        outputs, hidden = network_prac2nov.forward(sim1_inputs,noise=False)
        #### Set to 0 the pretraining practice outputs
        outputs[:,4:] = 0
        acc = np.mean(mod.accuracyScore(network_prac2nov,outputs,sim1_targets))
        df['Accuracy'].append(acc)
        df['Condition'].append('1-rule overlap')
        df['Simulation'].append(i)
        df['Trials viewed'].append(ntrials_viewed)
        print('\t 1-rule overlap acc =',acc)   
        print('\toutputs device:', outputs.device)

    df = pd.DataFrame(df) 
    df.to_csv(save_model + '.csv')

if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
