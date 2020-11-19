import numpy as np
import model.task as task
np.set_printoptions(suppress=True)
import os

import model.model as mod
import time
from importlib import reload
mod = reload(mod)
import torch


def run()
    networks = []
    logicscore = []
    sensoryscore = []
    motorscore = []
    for i in range(1000):
        network = mod.ANN(num_rule_inputs=11,
                             si_c=0,
                             num_sensory_inputs=16,
                             num_hidden_layers=2,
                             num_hidden=256,
                             num_motor_decision_outputs=6,
                             learning_rate=0.001,
                             lossfunc='CrossEntropy',device='cpu')

        logicps, sensoryps, motorps = mod.trainps(network,taskcontext_inputs,targets_ps,ps_object,dropout=False)
        networks.append(network)
        logicscore.append(logicps)
        sensoryscore.append(sensoryps)
        motorscore.append(motorps)

    return logicscore, sensoryscore, motorscore
