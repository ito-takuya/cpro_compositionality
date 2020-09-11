# Taku Ito
# 05/10/2019
# RSA analysis for RNN model training with no trial dynamics
import pandas as pd
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import model.task as task
import multiprocessing as mp
import h5py
from importlib import reload
task = reload(task)
import time
import matplotlib.pyplot as plt
import seaborn as sns
import model.model as mod


basedir = '../../../data/'



def rsa_pracnov(network,batchfilename='../../data/results/model/TrialBatches_4Prac60Nov',measure='corr'):
    """
    For each input element, inject a single element representing each rule/stimulus
    Observe the representational space
    """

    #batchfilename = datadir + 'results/model/TrialBatches_4Prac60Nov'

    print('Loading practice and novel batches')
    TrialObj = mod.TrialBatchesPracticeNovel(filename=batchfilename)
    #practice_input_batches, practice_output_batches = TrialObj.loadBatches(condition='practice',cuda=False) # batches x trials x input elements
    #novel_input_batches, novel_output_batches = TrialObj.loadBatches(condition='novel',cuda=False)
    prac_ruleset = pd.read_csv(batchfilename + '_practice.csv')
    nov_ruleset = pd.read_csv(batchfilename + '_novel.csv')

    prac_ruleset_arr = prac_ruleset.Code.apply(lambda x: x[1:-1].split(',')).values
    nov_ruleset_arr = nov_ruleset.Code.apply(lambda x: x[1:-1].split(',')).values

    #### Set input matrix parameters
    rule_ind = np.arange(network.num_rule_inputs) # rules are the first 12 indices of input vector
    stim_ind = np.arange(network.num_rule_inputs, network.num_rule_inputs+network.num_sensory_inputs)
    input_size = len(rule_ind) + len(stim_ind)
    input_matrix = np.zeros((len(prac_ruleset)+len(nov_ruleset),input_size)) # Activation for each input element separately

    
    #### Now fill input matrix
    # For loop is inefficient, but allows for consistent structuring of rule permutations
    taskRuleSets = task.createRulePermutations()
    taskcount = 0
    for logic in np.unique(taskRuleSets.Logic):
        logic_ind = prac_ruleset.Logic==logic
        for sensory in np.unique(taskRuleSets.Sensory):
            sensory_ind = prac_ruleset.Sensory==sensory
            for motor in np.unique(taskRuleSets.Motor):
                motor_ind = prac_ruleset.Motor==motor
                # View tasks in order; if task exists in practice set, include the input code
                ind = np.where(np.multiply(np.multiply(logic_ind.values,sensory_ind.values),motor_ind.values))[0]
                if len(ind)>0:
                    input_matrix[taskcount,rule_ind] = np.asarray(prac_ruleset_arr[ind][0][0].split(' '),dtype=float) 
                    taskcount += 1

    # Now fill in task rules for 'novel' tasks 
    for logic in np.unique(taskRuleSets.Logic):
        logic_ind = nov_ruleset.Logic==logic
        for sensory in np.unique(taskRuleSets.Sensory):
            sensory_ind = nov_ruleset.Sensory==sensory
            for motor in np.unique(taskRuleSets.Motor):
                motor_ind = nov_ruleset.Motor==motor
                # View tasks in order; if task exists in novel set, include the input code
                ind = np.where(np.multiply(np.multiply(logic_ind.values,sensory_ind.values),motor_ind.values))[0]
                if len(ind)>0:
                    input_matrix[taskcount,rule_ind] = np.asarray(nov_ruleset_arr[ind][0][0].split(' '),dtype=float) 
                    taskcount += 1

    input_matrix = torch.from_numpy(input_matrix).float()
    # Now run a forward pass for all activations
    outputs, hidden = network.forward(input_matrix,noise=False)

    ## Now plot RSM
    hidden = hidden.detach().numpy()
    
    if measure=='corr':
        rsm = np.corrcoef(hidden)
        #np.fill_diagonal(rsm,0)
    if measure=='cov':
        rsm = np.cov(hidden)

    return hidden, rsm

    

