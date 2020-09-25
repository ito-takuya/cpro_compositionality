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
import runANNs_CL as runModel
mod = reload(mod)
task = reload(task)
analysis = reload(analysis)
import torch
import pandas as pd
#from torch.autograd import Variable
#import torch.nn.functional as F

datadir = '../../data/'

def run(nsimulations,si_c=0,create_new_tasks=False, practice=True,learning='online',
        num_hidden=512,learning_rate=0.01,thresh=0.0,acc_cutoff=90.0,
        save_rsm=False,save_hiddenrsm_pdf=False,
        save_model='ANN_OnlineLearning',batchname='Experiment_FullTaskSet',
        lossfunc='MSE',pretraining=True,
        verbose=True):

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
        print('Training simulation', i, 'saving to file:', modelname, '| synaptic intelligence:', si_c)
        network_prac2nov, ntrials_viewed, acc = runModel.runModel(experiment,si_c=si_c,acc_cutoff=acc_cutoff,learning=learning,datadir=datadir,practice=practice,
                                                                  num_hidden=num_hidden,thresh=thresh,learning_rate=learning_rate,
                                                                  save_model=modelname,verbose=True,lossfunc=lossfunc,pretraining=pretraining)
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

    df = pd.DataFrame(df) 
    df.to_csv(save_model + '.csv')

if __name__ == '__main__':
    nsimulations = 20
    si_c = 0.0
    lrate = 0.001
    cutoff = 80.0 # 85
    lossfunc = 'CrossEntropy'
    num_hidden = 256
    #args = parser.parse_args()
    #args = set_default_values(args)
    run(nsimulations,
        si_c=si_c,
        create_new_tasks=False,
        practice=True,
        learning=None,
        num_hidden=num_hidden,
        learning_rate=lrate,
        thresh=0.0,
        acc_cutoff=cutoff,
        save_rsm=False,
        save_hiddenrsm_pdf=False,
        save_model='ANN_OnlineLearning_256_DO_noPretraining',
        batchname='Experiment_FullTaskSet',
        verbose=True,
        lossfunc=lossfunc,
        pretraining=False)
