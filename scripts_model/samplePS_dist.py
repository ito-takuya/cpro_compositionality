import numpy as np
import model.task as task
np.set_printoptions(suppress=True)
import os

import model.model as mod
import time
from importlib import reload
mod = reload(mod)
import torch


def run():


    device ='cpu'
    experiment = task.Experiment(filename='tmp')
    taskcontext_inputs = task.create_taskcontext_inputsOnly(experiment.taskRuleSet)
    nsamples = taskcontext_inputs.shape[0]
    targets_ps = torch.zeros(nsamples,4)

    taskcontext_inputs = torch.from_numpy(taskcontext_inputs).float()
    taskcontext_inputs = taskcontext_inputs.to(device)
    targets_ps = targets_ps.float().to(device)

    # Create object to pass
    ps_object = type('', (), {})()
    ps_object.match_logic_ind = mod.sortConditionsPS(experiment.taskRuleSet.Logic.values,
                                                     experiment.taskRuleSet.Sensory.values,
                                                     experiment.taskRuleSet.Motor.values) 
    ps_object.match_sensory_ind = mod.sortConditionsPS(experiment.taskRuleSet.Sensory.values,
                                                       experiment.taskRuleSet.Logic.values,
                                                       experiment.taskRuleSet.Motor.values) 
    ps_object.match_motor_ind = mod.sortConditionsPS(experiment.taskRuleSet.Motor.values,
                                                     experiment.taskRuleSet.Logic.values,
                                                     experiment.taskRuleSet.Sensory.values) 
    ps_object.inputs_ps = taskcontext_inputs



    networks = []
    logicscore = []
    sensoryscore = []
    motorscore = []
    for i in range(10000):
        network = mod.ANN(num_rule_inputs=11,
                             si_c=0,
                             num_sensory_inputs=16,
                             num_hidden_layers=2,
                             num_hidden=256,
                             num_motor_decision_outputs=6,
                             learning_rate=0.001,
                             lossfunc='CrossEntropy',device='cpu')
        with torch.no_grad():
            network.w_in.weight.copy_(-10+torch.randn(network.w_in.out_features,network.w_in.in_features)/.01)
            network.w_rec.weight.copy_(-10+torch.randn(network.w_rec.out_features,network.w_rec.out_features)/.01)

        logicps, sensoryps, motorps = mod.trainps(network,taskcontext_inputs,targets_ps,ps_object,dropout=False)
        networks.append(network)
        logicscore.append(logicps)
        sensoryscore.append(sensoryps)
        motorscore.append(motorps)

    return networks,logicscore, sensoryscore, motorscore
