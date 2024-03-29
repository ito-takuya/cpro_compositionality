#### Experiment 6
# for each simulation, include a regularizer that penalizes the cost function according to the parallelism score (maximize PS)
# fix number of epochs of data trained on, rather than using an 'accuracy cut-off'
# Calculate PS for each simulation

import numpy as np
import argparse
np.set_printoptions(suppress=True)
import os
import model.model as mod
import model.task_1ruletasks as task
import time
import model.analysis as analysis
from importlib import reload
import trainANN_costPS as trainANN
mod = reload(mod)
task = reload(task)
analysis = reload(analysis)
import torch
import pandas as pd
import tools

datadir = '../../data/'

parser = argparse.ArgumentParser('./main.py', description='Run a set of simulations/models')
parser.add_argument('--nsimulations', type=int, default=20, help='number of models/simulations to run')
parser.add_argument('--ps', type=float, default=0.0, help="include regularization term to maximize PS in cost function")
parser.add_argument('--pretraining', action='store_true', help="pretrain network on simple tasks to improve compositionality")
parser.add_argument('--negation', action='store_true', help="use the negative/negation version of the sensorimotor pretraining task")
parser.add_argument('--posneg', action='store_true', help="use BOTH positive & negative/negation version of the sensorimotor pretraining task")
parser.add_argument('--nepochs', type=int, default=100, help='number of epochs to run on practiced data')
parser.add_argument('--optimizer', type=str, default='adam', help='default optimizer to train on practiced tasks (DEFAULT: adam')
parser.add_argument('--practice', action='store_true', help="Train on practiced tasks")
parser.add_argument('--cuda', action='store_true', help="use gpu/cuda")
parser.add_argument('--save_model', type=str, default="expt6", help='string name to output models (DEFAULT: ANN)')
parser.add_argument('--verbose', action='store_true', help='verbose')
parser.add_argument('--num_layers', type=int, default=2, help="number of hidden layers (DEFAULT: 2")
parser.add_argument('--num_hidden', type=int, default=256, help="number of units in hidden layers (DEFAULT: 256")
parser.add_argument('--learning_rate', type=float, default=0.001, help="learning rate for pretraining sessions (DEFAULT: 0.001)")
parser.add_argument('--save', action='store_true', help="save or don't save model")
parser.add_argument('--batchname', type=str, default='Experiment_FullTaskSet_11LogicInputs', help='string name for the experiment filename')
parser.add_argument('--si_c', type=float, default=0.0, help='synaptic intelligence parameter (Zenke et al. 2017); default=0, meaning no synaptic intelligence implemented')


def run(args):
    args 
    nsimulations = args.nsimulations
    #if args.ps==0.0:
    #    ps_optim = None
    #else:
    #    ps_optim = args.ps
    ps_optim = args.ps
    si_c = args.si_c
    practice = args.practice
    negation = args.negation
    posneg = args.posneg
    n_epochs = args.nepochs
    optimizer = args.optimizer
    num_layers = args.num_layers
    num_hidden = args.num_hidden
    learning_rate = args.learning_rate
    save_model = args.save_model
    save = args.save
    batchname = args.batchname
    pretraining = args.pretraining
    cuda = args.cuda
    verbose = args.verbose

    outputdir = datadir + '/results/experiment6/'

    #save_model = save_model + '_' + batchname
    save_model = save_model + '_' + optimizer
    if practice:
        save_model = save_model + '_' + str(int(n_epochs)) + 'epochs' # only include epochs if there's actually a practiced training session
    else:
        save_model = save_model + '_zeroshot' # only include epochs if there's actually a practiced training session

    save_model = save_model + '_' + str(int(num_layers)) + 'layers'
    
    if ps_optim is not None:
        save_model = save_model + '_optimizePS' + str(ps_optim)

    if pretraining:
        save_model = save_model + '_pretraining'

    if negation:
        save_model = save_model + '_negation'

    if posneg and not negation:
        save_model = save_model + '_posneg'
    if posneg and negation:
        raise Exception("Can't have both positive and balanced (pos & neg) pretraining")

    if practice:
        save_model = save_model + '_practice'


    if cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'

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
        motor_pretraining_input, motor_pretraining_output = task.create_motor_pretraining(negation=negation)  
        motor_pretraining_input = torch.from_numpy(motor_pretraining_input).float()
        motor_pretraining_output = torch.from_numpy(motor_pretraining_output).long()
        if cuda:
            motor_pretraining_input = motor_pretraining_input.cuda()
            motor_pretraining_output = motor_pretraining_output.cuda()
        experiment.motor_pretraining_input = motor_pretraining_input
        experiment.motor_pretraining_output = motor_pretraining_output

        sensory_pretraining_input, sensory_pretraining_output = task.create_sensory_pretraining(negation=negation)
        sensory_pretraining_input = torch.from_numpy(sensory_pretraining_input).float()
        sensory_pretraining_output = torch.from_numpy(sensory_pretraining_output).long()
        if cuda:
            sensory_pretraining_input = sensory_pretraining_input.cuda()
            sensory_pretraining_output = sensory_pretraining_output.cuda()
        experiment.sensory_pretraining_input = sensory_pretraining_input
        experiment.sensory_pretraining_output = sensory_pretraining_output


        logic_pretraining_input, logic_pretraining_output = task.create_logic_pretraining(negation=negation)
        logic_pretraining_input = torch.from_numpy(logic_pretraining_input).float()
        logic_pretraining_output = torch.from_numpy(logic_pretraining_output).long()
        if cuda:
            logic_pretraining_input = logic_pretraining_input.cuda()
            logic_pretraining_output = logic_pretraining_output.cuda()
        experiment.logic_pretraining_input = logic_pretraining_input
        experiment.logic_pretraining_output = logic_pretraining_output


        logicalsensory_pretraining_input, logicalsensory_pretraining_output = task.create_logicalsensory_pretraining()
        logicalsensory_pretraining_input = torch.from_numpy(logicalsensory_pretraining_input).float()
        logicalsensory_pretraining_output = torch.from_numpy(logicalsensory_pretraining_output).long()
        if cuda:
            logicalsensory_pretraining_input = logicalsensory_pretraining_input.cuda()
            logicalsensory_pretraining_output = logicalsensory_pretraining_output.cuda()
        experiment.logicalsensory_pretraining_input = logicalsensory_pretraining_input
        experiment.logicalsensory_pretraining_output = logicalsensory_pretraining_output


        sensorimotor_pretraining_input, sensorimotor_pretraining_output = task.create_sensorimotor_pretraining()
        sensorimotor_pretraining_input = torch.from_numpy(sensorimotor_pretraining_input).float()
        sensorimotor_pretraining_output = torch.from_numpy(sensorimotor_pretraining_output).long()
        if cuda:
            sensorimotor_pretraining_input = sensorimotor_pretraining_input.cuda()
            sensorimotor_pretraining_output = sensorimotor_pretraining_output.cuda()
        experiment.sensorimotor_pretraining_input = sensorimotor_pretraining_input
        experiment.sensorimotor_pretraining_output = sensorimotor_pretraining_output



    ###########################################
    #### run simulations
    for sim in range(nsimulations):

        #########################################
        # Load or reload practiced and novel tasks (each simulation needs to refresh task data
        batchfilename = datadir + 'results/model/' + batchname
        # Reset practiced and novel task sets
        orig = task.Experiment(filename=batchfilename)
        experiment.practicedRuleSet = orig.practicedRuleSet
        experiment.novelRuleSet = orig.novelRuleSet
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

        full_inputs = torch.cat((prac_inputs,novel_inputs),0)
        full_targets = torch.cat((prac_targets,novel_targets),0)

        df_sim = {}
        df_sim['Accuracy'] = []
        df_sim['ContextDimensionality'] = []
        df_sim['ResponseDimensionality'] = []
        df_sim['NumPracticedTasks'] = []
        df_sim['LogicPS'] = []
        df_sim['SensoryPS'] = []
        df_sim['MotorPS'] = []

        df_pertask = {}
        df_pertask['Accuracy'] = []
        df_pertask['Condition'] = []
        df_pertask['Logic'] = []
        df_pertask['Sensory'] = []
        df_pertask['Motor'] = []
        df_pertask['NumPracticedTasks'] = []

        n_practiced_tasks = len(experiment.practicedRuleSet)
        while n_practiced_tasks < len(experiment.taskRuleSet):
            if n_practiced_tasks not in [4,60]:
                n_practiced_tasks += 1
                continue
            else:
                modelname = save_model + str(sim)

                #### Create conditional such that if practice==False, then don't train on any practiced tasks an exit immediately
                if not practice:
                    n_practiced_tasks = 0

                #if verbose: print('** TRAINING ON', n_practiced_tasks, 'PRACTICED TASKS ** ... simulation', sim, ' |', modelname, '| cuda:', cuda)
                network, acc = trainANN.train(experiment,si_c=si_c,n_epochs=n_epochs,datadir=datadir,practice=practice,optimizer=optimizer,
                                              num_hidden=num_hidden,num_hidden_layers=num_layers,learning_rate=learning_rate,save=save,
                                              save_model=outputdir+modelname+'.pt',verbose=False,lossfunc='CrossEntropy',ps_optim=ps_optim, pretraining=pretraining,device=device)
            

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
                    df_pertask['NumPracticedTasks'].append(n_practiced_tasks)

                    
                if verbose: 
                    print('Model:', modelname, '| Simulation', sim, ' | # of practiced tasks:', n_practiced_tasks, '| Acc on novel tasks:', np.mean(novel_acc))

                # novel trial accuracy
                df_sim['Accuracy'].append(np.mean(novel_acc))
                df_sim['NumPracticedTasks'].append(n_practiced_tasks)
                hidden, rsm_corr = analysis.rsa_context(network,batchfilename=batchfilename,measure='corr')
                df_sim['ContextDimensionality'].append(tools.dimensionality(rsm_corr))
                #### response dimensionality - requires the task input/output set
                hidden, rsm_corr = analysis.rsa_behavior(network,full_inputs,full_targets,measure='corr')
                df_sim['ResponseDimensionality'].append(tools.dimensionality(rsm_corr))


                #### Compute PS scores for each rule dimension
                # Logic PS
                taskcontext_inputs = task.create_taskcontext_inputsOnly(experiment.taskRuleSet)
                ps, classes = tools.parallelismScore(taskcontext_inputs,
                                                     experiment.taskRuleSet.Logic.values,
                                                     experiment.taskRuleSet.Sensory.values,
                                                     experiment.taskRuleSet.Motor.values)
                df_sim['LogicPS'].append(np.nanmean(ps))
                # Sensory PS
                ps, classes = tools.parallelismScore(taskcontext_inputs,
                                                     experiment.taskRuleSet.Sensory.values,
                                                     experiment.taskRuleSet.Logic.values,
                                                     experiment.taskRuleSet.Motor.values)
                df_sim['SensoryPS'].append(np.nanmean(ps))
                # Motor PS
                ps, classes = tools.parallelismScore(taskcontext_inputs,
                                                     experiment.taskRuleSet.Motor.values,
                                                     experiment.taskRuleSet.Logic.values,
                                                     experiment.taskRuleSet.Sensory.values)
                df_sim['MotorPS'].append(np.nanmean(ps))

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

            # exit if practice==False
            if not practice:
                break





        df_sim = pd.DataFrame(df_sim) 
        df_sim.to_csv(outputdir + save_model + '_simData' + str(sim) + '.csv')

        df_pertask = pd.DataFrame(df_pertask)
        df_pertask.to_csv(outputdir + save_model + '_PerTaskData' + str(sim) + '.csv')
        
        


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
