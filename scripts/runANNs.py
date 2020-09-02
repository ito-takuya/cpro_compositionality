import numpy as np
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
#import torch
#from torch.autograd import Variable
#import torch.nn.functional as F


datadir = '../../../data/'

def runModel(datadir=datadir,num_hidden=512,learning_rate=0.0001,thresh=0.9,create_new_batches=False,save_csv=False,save_hiddenrsm_pdf=False,save_model=None):
    """
    num_hidden - # of hidden units
    learning_rate - learning rate 
    thresh - threshold for classifying output units
    create_new_batches - Create new training batches. Most likely not necessary; training batches already exist in data directory
    save_csv - Save out the RSM?
    save_hiddenrsm_pdf - save out a PDF of the RSM?
    """


    batchfilename = datadir + 'results/model/TrialBatches_4Prac60Nov'
    if create_new_batches: 
        print('Creating new training batches')
        TrialInfo = mod.TrialBatchesPracticeNovel(NUM_BATCHES=30000,
                                                  NUM_PRACTICE_TRIALS_PER_TASK=10,
                                                  NUM_NOVEL_TRIAlS_PER_TASK=10,
                                                  NUM_INPUT_ELEMENTS=28,
                                                  NUM_OUTPUT_ELEMENTS=4,
                                                  filename=batchfilename)
        TrialInfo.createBatches(condition='practice',nproc=4)
        TrialInfo.createBatches(condition='novel',nproc=4)

    #### ANN construction
    print('Instantiating new model')
    Network = mod.ANN(num_rule_inputs=12,
                         num_sensory_inputs=16,
                         num_hidden=num_hidden,
                         num_motor_decision_outputs=4,
                         learning_rate=learning_rate,
                         thresh=thresh)
    # Network.cuda = True
    Network = Network.cpu()

    #### Load training batches
    print('Loading practice batches')
    TrialObj = mod.TrialBatchesPracticeNovel(filename=batchfilename)
    practice_input_batches, practice_output_batches = TrialObj.loadBatches(condition='practice',cuda=False)
    novel_input_batches, novel_output_batches = TrialObj.loadBatches(condition='novel',cuda=False)

    #### Train practice tasks
    print('Training model on practiced tasks')
    timestart = time.time()
    mod.batch_training(Network, practice_input_batches,practice_output_batches,cuda=False)  
    timeend = time.time()
    print('Time elapsed using CPU:', timeend-timestart)

    print('Training model on novel tasks')
    timestart = time.time()
    mod.batch_training(Network, novel_input_batches,novel_output_batches,cuda=False)  
    timeend = time.time()
    print('Time elapsed using CPU:', timeend-timestart)

    if save_model is not None:
        torch.save(Network,save_model)

    #### Save out hidden layer RSM
    hidden, rsm = analysis.rsa(Network,show=save_hiddenrsm_pdf,savepdf=save_hiddenrsm_pdf)
    # hidden = hidden.detach().numpy()
    # input_matrix = input_matrix.detach().numpy()

    # Save out RSM 
    if save_csv:
        np.savetxt('ANN1280_HiddenLayerRSM_NoDynamics.csv',rsm)

    return Network
