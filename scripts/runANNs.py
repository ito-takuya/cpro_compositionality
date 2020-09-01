import numpy as np
np.set_printoptions(suppress=True)
#import matplotlib.pyplot as plt
#import seaborn as sns
#import scipy.stats as stats
import os
os.sys.path.append('model/')
import model.model_nodynamics as model
import model.task as task
import time
import model.analysis as analysis
#import torch
#from torch.autograd import Variable
#import torch.nn.functional as F


datadir = '../../../data/'

def runModel(num_hidden=1280,learning_rate=0.0001,thresh=0.9,create_new_batches=False,save_csv=False,save_hiddenrsm_pdf=False):
    """
    num_hidden - # of hidden units
    learning_rate - learning rate 
    thresh - threshold for classifying output units
    create_new_batches - Create new training batches. Most likely not necessary; training batches already exist in data directory
    save_csv - Save out the RSM?
    save_hiddenrsm_pdf - save out a PDF of the RSM?
    """


    if create_new_batches: 
        TrialInfo = model.TrialBatchesPracticeNovel(NUM_BATCHES=30000,
                                                    NUM_TRAINING_TRIAlS_PER_TASK=10,
                                                    NUM_TESTING_TRIAlS_PER_TASK=100,
                                                    NUM_INPUT_ELEMENTS=28,
                                                    NUM_OUTPUT_ELEMENTS=4,
                                                    filename=datadir + 'results/MODEL/TrialBatches_Default_NoDynamics'):
        TrialInfo.createAllBatches(nproc=30)

    #### Load training batches
    print('Loading training batches')
    input_batches, output_batches = model.load_training_batches(cuda=False,filename=projectdir + 'data/results/MODEL/TrialBatches_Default_NoDynamics')

    Network = model.RNN(num_rule_inputs=12,
                         num_sensory_inputs=16,
                         num_hidden=num_hidden,
                         num_motor_decision_outputs=4,
                         learning_rate=learning_rate,
                         thresh=thresh)
    # Network.cuda = True
    Network = Network.cpu()

    #### Train model
    print('Training model')
    timestart = time.time()
    model.batch_training(Network, input_batches,output_batches,cuda=False)  
    timeend = time.time()
    print('Time elapsed using CPU:', timeend-timestart)

    #### Save out hidden layer RSM
    hidden, rsm = analysis.rsa(Network,show=save_hiddenrsm_pdf,savepdf=save_hiddenrsm_pdf)
    # hidden = hidden.detach().numpy()
    # input_matrix = input_matrix.detach().numpy()

    # Save out RSM 
    if save_csv:
        np.savetxt('ANN1280_HiddenLayerRSM_NoDynamics.csv',rsm)

