# Taku Ito
# 05/10/2019
# RNN model training with no trial dynamics
import pandas as pd
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import task
import multiprocessing as mp
import h5py
from importlib import reload
task = reload(task)
np.set_printoptions(suppress=True)
import time


datadir = '../../../data/'

class ANN(torch.nn.Module):
    """
    Neural network object
    """

    def __init__(self,
                 num_rule_inputs=12,
                 num_sensory_inputs=16,
                 num_hidden=128,
                 num_motor_decision_outputs=4,
                 learning_rate=0.0001,
                 thresh=0.8,
                 cuda=False):

        # Define general parameters
        self.num_rule_inputs = num_rule_inputs
        self.num_sensory_inputs =  num_sensory_inputs
        self.num_hidden = num_hidden
        self.num_motor_decision_outputs = num_motor_decision_outputs
        self.cuda = cuda

        # Define entwork architectural parameters
        super(RNN,self).__init__()

        self.w_in = torch.nn.Linear(num_sensory_inputs+num_rule_inputs,num_hidden)
        self.w_rec = torch.nn.Linear(num_hidden,num_hidden)
        self.w_out = torch.nn.Linear(num_hidden,num_motor_decision_outputs)
        self.sigmoid = torch.nn.Sigmoid()
        self.func = torch.nn.ReLU()

        # Initialize RNN units
        self.units = torch.nn.Parameter(self.initHidden())

        # Define loss function
        self.lossfunc = torch.nn.MSELoss(reduce=False)

        # Decision threshhold for behavior
        self.thresh = thresh

        # Construct optimizer
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        ##optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
    
    def initHidden(self):
        return torch.randn(1, self.num_hidden)

    def forward(self,inputs,noise=False):
        """
        Run a forward pass of a trial by input_elements matrix
        """
        # Map inputs into RNN space
        rnn_input = self.w_in(inputs)
        rnn_input = self.func(rnn_input)
        # Define rnn private noise/spont_act
        if noise:
            spont_act = torch.randn(rnn_input.shape, dtype=torch.float)/self.num_hidden
            # Add private noise to each unit, and add to the input
            rnn_input = rnn_input + spont_act

        # Run RNN
        hidden = self.w_rec(rnn_input)
        hidden = self.func(hidden)
        
        # Compute outputs
        h2o = self.w_out(hidden) # Generate linear outupts
        outputs = self.sigmoid(h2o) # Pass through nonlinearity

        return outputs, hidden

def train(network, inputs, targets):
    """Train network"""
    network.train()
    network.zero_grad()
    network.optimizer.zero_grad()

    outputs, hidden = network.forward(inputs,noise=True)

    # Calculate loss
    loss = network.lossfunc(outputs,targets)
    loss = torch.mean(loss)
    
    # Backprop and update weights
    loss.backward()
    network.optimizer.step() # Update parameters using optimizer

    return outputs, targets, loss

def batch_training(network,train_inputs,train_outputs,
                     cuda=False):

    accuracy_per_batch = []

    batch_ordering = np.arange(train_inputs.shape[0])

    np.random.shuffle(batch_ordering)

    timestart = time.time()
    for batch in np.arange(train_inputs.shape[0]):
        batch_id = batch_ordering[batch]

        if cuda:
            train_input_batch = train_inputs[batch_id,:,:].cuda()
            train_output_batch = train_outputs[batch_id,:,:].cuda()
        else:
            train_input_batch = train_inputs[batch_id,:,:]
            train_output_batch = train_outputs[batch_id,:,:]

        outputs, targets, loss = train(network,
                                       train_inputs[batch_id,:,:],
                                       train_outputs[batch_id,:,:])

        if batch % 1000 == 0:
            targets = targets.cpu()
            outputs = outputs.cpu()
    
            acc = [] # accuracy array
            for mb in range(targets.shape[0]):
                for out in range(targets.shape[1]):
                    if targets[mb,out] == 0: continue
                    response = outputs[mb,out] # Identify response time points
                    thresh = network.thresh # decision thresh
                    target_resp = torch.ByteTensor([out]) # The correct target response
                    max_resp = outputs[mb,:].argmax().byte()
                    if max_resp==target_resp and response>thresh: # Make sure network response is correct respnose, and that it exceeds some threshold
                        acc.append(1.0)
                    else:
                        acc.append(0)

            timeend = time.time()
            print('Iteration:', batch)
            print('\tloss:', loss.item())
            print('Time elapsed...', timeend-timestart)
            timestart = timeend
            print('\tAccuracy: ', str(round(np.mean(acc)*100.0,4)),'%')
        
            nbatches_break = 1000
            if batch>nbatches_break:
                if np.sum(np.asarray(accuracy_per_batch[-nbatches_break:])>99.0)==nbatches_break:
                    print('Last', nbatches_break, 'batches had above 99.5% accuracy... stopping training')
                    break

        accuracy_per_batch.append(np.mean(acc)*100.0)

def eval(network,test_inputs,targets,cuda=False):
    network.eval()
    network.zero_grad()
    network.optimizer.zero_grad()

    outputs, hidden = network.forward(test_inputs,noise=True)

    # Calculate loss
    loss = network.lossfunc(outputs,targets)
    loss = torch.mean(loss)


    acc = [] # accuracy array
    for mb in range(targets.shape[0]):
        for out in range(targets.shape[1]):
            if targets[mb,out] == 0: continue
            response = outputs[mb,out] # Identify response time points
            thresh = network.thresh # decision thresh
            target_resp = torch.ByteTensor([out]) # The correct target response
            max_resp = outputs[mb,:].argmax().byte()
            if max_resp==target_resp and response>thresh: # Make sure network response is correct respnose, and that it exceeds some threshold
                acc.append(1.0)
            else:
                acc.append(0)

    print('\tloss:', loss.item())
    print('\tAccuracy: ',str(round(np.mean(acc)*100.0,4)),'%')
    return outputs, hidden

def load_practice_batches(cuda=False,filename=datadir + 'results/MODEL/TrialBatches_Default_NoDynamics'):
    TrialObj = TrialBatches(filename=filename)
    inputs, outputs = TrialObj.loadTrainingBatches()

    inputs = inputs.float()
    outputs = outputs.float()
    if cuda==True:
        inputs = inputs.cuda()
        outputs = outputs.cuda()

    return inputs, outputs

def load_novel_batches(cuda=False,filename=datadir + 'results/MODEL/TrialBatches_Default_NoDynamics'):
    TrialObj = TrialBatches(filename=filename)
    test_inputs, test_outputs = TrialObj.loadTestset()

    test_inputs = test_inputs.float()
    test_outputs = test_outputs.float()
    if cuda==True:
        test_inputs = test_inputs.cuda()
        test_outputs = test_outputs.cuda()
    return test_inputs, test_outputs

class TrialBatchesPracticeNovel(object):
    """
    Batch trials, but specifically separate practiced versus novel task sets (4 practiced, 60 novel)
    """
    def __init__(self,
                 NUM_BATCHES=10000,
                 NUM_PRACTICE_TRIAlS_PER_TASK=10,
                 NUM_NOVEL_TRIAlS_PER_TASK=10,
                 NUM_INPUT_ELEMENTS=28,
                 NUM_OUTPUT_ELEMENTS=4,
                 filename=datadir + 'results/MODEL/TrialBatches_Default_NoDynamics'):


        self.NUM_BATCHES = NUM_BATCHES
        self.NUM_OUTPUT_ELEMENTS = NUM_OUTPUT_ELEMENTS
        self.NUM_NOVEL_TRIAlS_PER_TASK = NUM_NOVEL_TRIAlS_PER_TASK
        self.NUM_PRACTICE_TRIAlS_PER_TASK = NUM_PRACTICE_TRIAlS_PER_TASK
        self.NUM_INPUT_ELEMENTS = NUM_INPUT_ELEMENTS
        self.splitPracticedNovelTaskSets()
        self.filename = filename

    def createBatches(self,condition='practice',nproc=10):
        if condition=='practice':
            ntrials = self.NUM_PRACTICE_TRIALS_PER_TASK
            ruleset = self.practicedRuleSet
        elif condition=='novel':
            ntrials = self.NUM_NOVEL_TRIALS_PER_TASK
            ruleset = self.novelRuleSet
        # Initialize empty tensor for batches
        batch_inputtensor = np.zeros((self.NUM_INPUT_ELEMENTS, len(ruleset)*ntrials, self.NUM_BATCHES))
        batch_outputtensor = np.zeros((self.NUM_OUTPUT_ELEMENTS, len(ruleset)*ntrials, self.NUM_BATCHES))

        inputs = []
        for batch in range(self.NUM_BATCHES):
            shuffle = True
            seed = np.random.randint(1000000)
            inputs.append((ruleset,ntrials,shuffle,batch,seed))

        pool = mp.Pool(processes=nproc)
        results = pool.starmap_async(create_trial_batches,inputs).get()
        pool.close()
        pool.join()

        batch = 0
        for result in results:
            batch_inputtensor[:,:,batch] = result[0]
            batch_outputtensor[:,:,batch] = result[1]
            batch += 1

        h5f = h5py.File(self.filename + '.h5','a')
        try:
            h5f.create_dataset(condition + '/inputs',data=batch_inputtensor)
            h5f.create_dataset(condition + '/outputs',data=batch_outputtensor)
        except:
            del h5f[condition + '/inputs'], h5f[condition + '/outputs']
            h5f.create_dataset(condition + '/inputs',data=batch_inputtensor)
            h5f.create_dataset(condition + '/outputs',data=batch_outputtensor)
        h5f.close()

    def loadBatches(self,condition='practice',cuda=False):
        h5f = h5py.File(self.filename + '.h5','r')
        inputs = h5f[condition + '/inputs'][:].copy()
        outputs = h5f[condition + '/outputs'][:].copy()
        h5f.close()

        # Input dimensions: input features, nMiniblocks, nBatches
        inputs = np.transpose(inputs, (2, 1, 0)) # convert to: nBatches, nMiniblocks, input dimensions
        outputs = np.transpose(outputs, (2, 1, 0)) # convert to: nBatches, nMiniblocks, input dimensions

        inputs = torch.from_numpy(inputs)
        outputs = torch.from_numpy(outputs)

        if cuda:
            inputs = inputs.cuda()
            outputs = outputs.cuda()

        return inputs, outputs
    
    def splitPracticedNovelTaskSets(self):
        taskRuleSet = task.createRulePermutations()
        practicedRuleSet, novelRuleSet = task.create4Practiced60NovelTaskContexts(taskRuleSet)

        self.taskRuleSet = taskRuleSet
        self.practicedRuleSet = practicedRuleSet
        self.novelRuleSet = novelRuleSet

def create_trial_batches(taskRuleSet,ntrials_per_task,shuffle,batchNum,seed):
    """
    Randomly generates a set of stimuli (nStimuli) for each task rule
    Will end up with 64 (task rules) * nStimuli total number of input stimuli
    
    If shuffle keyword is True, will randomly shuffle the training set
    Otherwise will start with taskrule1 (nStimuli), taskrule2 (nStimuli), etc.
    """
    # instantiate random seed (since this is function called in parallel)
    np.random.seed(seed)

    if batchNum%100==0:
        print('Running batch', batchNum)
    
    stimuliSet = task.createSensoryInputs()

    # Create 1d array to randomly sample indices from
    stimIndices = np.arange(len(stimuliSet))
    taskIndices = np.arange(len(taskRuleSet))

    shuffle=True
    

    #randomTaskIndices = np.random.choice(taskIndices,len(taskIndices),replace=False)
    #randomTaskIndices = np.random.choice(taskIndices,nTasks,replace=False)
    #taskRuleSet2 = taskRuleSet.iloc[randomTaskIndices].copy(deep=True)
    #taskRuleSet = taskRuleSet.reset_index(drop=True)
    taskRuleSet = taskRuleSet.reset_index(drop=False)
    #taskRuleSet = taskRuleSet2.copy(deep=True)

    ntrials_total = ntrials_per_task * len(taskRuleSet)
    ####
    # Construct trial dynamics
    rule_ind = np.arange(12) # rules are the first 12 indices of input vector
    stim_ind = np.arange(12,28) # stimuli are the last 16 indices of input vector
    input_size = len(rule_ind) + len(stim_ind)
    input_matrix = np.zeros((input_size,ntrials_total))
    output_matrix = np.zeros((4,ntrials_total))
    trialcount = 0
    for tasknum in range(len(taskRuleSet)):
        

        for i in range(ntrials_per_task):
            rand_stim_ind = np.random.choice(stimIndices,1,replace=False)
            stimuliSet2 = stimuliSet.iloc[rand_stim_ind].copy(deep=True)
            stimuliSet2 = stimuliSet2.reset_index(drop=True)
        
            ## Create trial array
            # Find input code for this task set
            input_matrix[rule_ind,trialcount] = taskRuleSet.Code[tasknum] 
            # Solve task to get the output code
            tmpresp, out_code = task.solveInputs(taskRuleSet.iloc[tasknum], stimuliSet2.iloc[0])

            input_matrix[stim_ind,trialcount] = stimuliSet2.Code[0]
            output_matrix[:,trialcount] = out_code

            trialcount += 1
            
    if shuffle:
        ind = np.arange(input_matrix.shape[1],dtype=int)
        np.random.shuffle(ind)
        input_matrix = input_matrix[:,ind]
        output_matrix = output_matrix[:,ind]
        
    return input_matrix, output_matrix 

