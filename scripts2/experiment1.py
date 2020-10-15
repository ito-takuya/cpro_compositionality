import numpy as np
import argparse
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
import trainANN as trainANN
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
parser.add_argument('--practice', action='store_true', help="Train on 4 practiced tasks")
parser.add_argument('--num_hidden', type=int, default=256, help="number of units in hidden layers")
parser.add_argument('--learning_rate', type=float, default=0.001, help="learning rate for pretraining sessions (ADAM default)")
parser.add_argument('--acc_cutoff', type=float, default=95.0, help="condition for exiting ANN training")
parser.add_argument('--save_model', type=str, default="ANN", help='string name to output models')
parser.add_argument('--save', action='store_true', help="save or don't save model")
parser.add_argument('--batchname', type=str, default='Experiment_FullTaskSet_11LogicInputs', help='string name for the experiment filename')
parser.add_argument('--lossfunc', type=str, default='CrossEntropy', help='default: CrossEntropy, options: MSE or CrossEntropy, which determines the loss function')
parser.add_argument('--pretraining', action='store_true', help="pretrain network on simple tasks to improve compositionality")
parser.add_argument('--cuda', action='store_true', help="use gpu/cuda")
parser.add_argument('--verbose', action='store_false', help='verbose')

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
    lossfunc = args.lossfunc
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
    print("create task sets")
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



    #### Now identify overlap with practiced tasks in the novel task set
    taskSim1Set, taskSim2Set = experiment.taskSimilarity(experiment.practicedRuleSet,experiment.novelRuleSet)

    #### Simulate task sets with 2-rule similarities
    sim2_inputs, sim2_targets = task.create_all_trials(taskSim2Set)
    sim2_inputs = torch.from_numpy(sim2_inputs.T).float()
    sim2_targets = torch.from_numpy(sim2_targets.T).long()
    sim2_inputs = sim2_inputs.reshape(sim2_inputs.shape[0]*sim2_inputs.shape[1],sim2_inputs.shape[2])
    sim2_targets = torch.flatten(sim2_targets)

    #### Simulate task sets with 1-rule similarities
    sim1_inputs, sim1_targets = task.create_all_trials(taskSim1Set)
    sim1_inputs = torch.from_numpy(sim1_inputs.T).float()
    sim1_targets = torch.from_numpy(sim1_targets.T).long()
    sim1_inputs = sim1_inputs.reshape(sim1_inputs.shape[0]*sim1_inputs.shape[1],sim1_inputs.shape[2])
    sim1_targets = torch.flatten(sim1_targets)

    if cuda:
        sim1_inputs = sim1_inputs.cuda()
        sim2_inputs = sim2_inputs.cuda()
        sim1_targets = sim1_targets.cuda()
        sim2_targets = sim2_targets.cuda()

    #### Load pretraining data
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


    ###########################################
    #### run simulations
    df = {}
    df['Accuracy'] = []
    df['Condition'] = []
    df['Simulation'] = []
    #### Run simulation
    for i in range(nsimulations):
        modelname = save_model + str(i) + '.pt'
        print('**SIMULATION**', i, 'saving to file:', modelname, '| cuda:', cuda)
        network, acc = trainANN.train(experiment,si_c=si_c,acc_cutoff=acc_cutoff,datadir=datadir,practice=practice,
                                      num_hidden=num_hidden,learning_rate=learning_rate,save=save,
                                      save_model=modelname,verbose=True,lossfunc=lossfunc,pretraining=pretraining,device=device)


        network.eval()
            
        # practice trials
        outputs, hidden = network.forward(prac_input2d,noise=False)
        #### Set to 0 the pretraining practice outputs
        outputs[:,4:] = 0
        acc = mod.accuracyScore(network,outputs,prac_target2d)
        df['Accuracy'].append(acc)
        df['Condition'].append('Practiced')
        df['Simulation'].append(i)
        print('\t Practiced acc =',acc)
        
        # 2-rule overlap
        outputs, hidden = network.forward(sim2_inputs,noise=False)
        #### Set to 0 the pretraining practice outputs
        outputs[:,4:] = 0
        acc = mod.accuracyScore(network,outputs,sim2_targets)
        df['Accuracy'].append(acc)
        df['Condition'].append('2-rule overlap')
        df['Simulation'].append(i)
        print('\t 2-rule overlap acc =',acc)    

        # 1-rule overlap
        outputs, hidden = network.forward(sim1_inputs,noise=False)
        #### Set to 0 the pretraining practice outputs
        outputs[:,4:] = 0
        acc = mod.accuracyScore(network,outputs,sim1_targets)
        df['Accuracy'].append(acc)
        df['Condition'].append('1-rule overlap')
        df['Simulation'].append(i)
        print('\t 1-rule overlap acc =',acc)   

    df = pd.DataFrame(df) 
    df.to_csv(save_model + '.csv')

    prac_acc = np.mean(df.loc[df['Condition']=='Practiced'].Accuracy.values) 
    rule2_acc = np.mean(df.loc[df['Condition']=='2-rule overlap'].Accuracy.values) 
    rule1_acc = np.mean(df.loc[df['Condition']=='1-rule overlap'].Accuracy.values) 
    print('**Averages across simulations**')
    print('\tPracticed accuracy:', prac_acc)
    print('\t2-rule overlap accuracy:', rule2_acc)
    print('\t1-rule overlap accuracy:', rule1_acc)

if __name__ == '__main__':
    args = parser.parse_args()
    run(args)