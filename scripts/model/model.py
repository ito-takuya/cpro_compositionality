# Taku Ito
# 05/10/2019
# RNN model training with no trial dynamics
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
                 thresh=0.9,
                 cuda=False):

        # Define general parameters
        self.num_rule_inputs = num_rule_inputs
        self.num_sensory_inputs =  num_sensory_inputs
        self.num_hidden = num_hidden
        self.num_motor_decision_outputs = num_motor_decision_outputs
        self.cuda = cuda

        # Define entwork architectural parameters
        super(ANN,self).__init__()

        self.w_in = torch.nn.Linear(num_sensory_inputs+num_rule_inputs,num_hidden)
        self.w_rec = torch.nn.Linear(num_hidden,num_hidden)
        self.w_out = torch.nn.Linear(num_hidden,num_motor_decision_outputs)
        self.func_out = torch.nn.Softmax(dim=1)
        self.func_out1d = torch.nn.Softmax(dim=0) # used for online learning (since only one trial is trained at a time)
        self.func = torch.nn.ReLU()

        self.dropout_in = torch.nn.Dropout(p=0.2)
        self.dropout_rec = torch.nn.Dropout(p=0.5)

        # Initialize RNN units
        self.units = torch.nn.Parameter(self.initHidden())

        # Define loss function
        self.lossfunc = torch.nn.MSELoss(reduction='none')

        # Decision threshhold for behavior
        self.thresh = thresh

        # Construct optimizer
        self.learning_rate = learning_rate
        #self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.optimizer = torch.optim.Adagrad(self.parameters(), lr=learning_rate)
        #self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        ##optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
    def initHidden(self):
        return torch.randn(1, self.num_hidden)

    def forward(self,inputs,noise=False,dropout=True):
        """
        Run a forward pass of a trial by input_elements matrix
        """
        # Map inputs into RNN space
        rnn_input = self.w_in(inputs) 
        if dropout: rnn_input = self.dropout_in(rnn_input)
        rnn_input = self.func(rnn_input)
        # Define rnn private noise/spont_act
        if noise:
            spont_act = torch.randn(rnn_input.shape, dtype=torch.float)/self.num_hidden
            # Add private noise to each unit, and add to the input
            rnn_input = rnn_input + spont_act

        # Run RNN
        hidden = self.w_rec(rnn_input)
        if dropout: hidden = self.dropout_rec(hidden)
        hidden = self.func(hidden)
        
        # Compute outputs
        h2o = self.w_out(hidden) # Generate linear outupts
        if h2o.dim()==1: # for online learning
            outputs = self.func_out1d(h2o)
        else:
            outputs = self.func_out(h2o) # Pass through nonlinearity

        return outputs, hidden

def train(network, inputs, targets):
    """Train network"""
    network.train()
    network.zero_grad()
    network.optimizer.zero_grad()

    outputs, hidden = network.forward(inputs,noise=False)

    # Calculate loss
    loss = network.lossfunc(outputs,targets)
    loss = torch.mean(loss)
    
    # Backprop and update weights
    loss.backward()
    network.optimizer.step() # Update parameters using optimizer

    return outputs, targets, loss


def batch_training(network,train_inputs,train_outputs,acc_cutoff=99.5,
                     cuda=False,verbose=True):

    accuracy_per_batch = []

    batch_ordering = np.arange(train_inputs.shape[0])

    np.random.shuffle(batch_ordering)

    nsamples_viewed = 0
    nbatches_trained = 0
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


        nbatches_trained += 1
        nsamples_viewed += train_inputs.shape[0]

        targets = targets.cpu()
        outputs = outputs.cpu()

        acc = np.mean(accuracyScore(network,outputs,targets))

        if verbose and batch%50==0:
            timeend = time.time()
            print('Iteration:', batch)
            print('\tloss:', loss.item())
            print('Time elapsed...', timeend-timestart)
            timestart = timeend
            print('\tAccuracy: ', str(round(np.mean(acc)*100.0,4)),'%')
    
        if batch>nbatches_break:
            if acc*100.0>acc_cutoff:
                if verbose: print('Last batch had', np.mean(acc)*100.0, '> above', acc_cutoff, 'accuracy... stopping training')
                break

    return nsamples_viewed, nbatches_trained

def task_training(network,train_inputs,train_outputs,acc_cutoff=99.5,
                     cuda=False,verbose=True):
    """
    training for tasks (using all stimulus combinations)
    """

    nsamples_viewed = 0
    timestart = time.time()

    nbatches_break = 10
    batch = 0
    accuracy = 0
    while accuracy < acc_cutoff:

        if cuda:
            train_input = train_inputs[:,:].cuda()
            train_output = train_outputs[:,:].cuda()

        outputs, targets, loss = train(network,
                                       train_inputs,
                                       train_outputs)


        targets = targets.cpu()
        outputs = outputs.cpu()

        acc = accuracyScore(network,outputs,targets)

        accuracy = np.mean(np.asarray(acc))*100.0

        if verbose and batch%50==0:
            timeend = time.time()
            print('Iteration:', batch)
            print('\tloss:', loss.item())
            print('\tAccuracy: ', str(round(accuracy,4)),'%')
    
        if accuracy>acc_cutoff:
            if verbose: print('Achieved', accuracy, '% accuracy, greater than cutoff (', acc_cutoff, '%) accuracy... stopping training after', batch, 'batches')
        
        batch += 1
        nsamples_viewed += train_inputs.shape[0]

    return nsamples_viewed, batch

def accuracyScore(network,outputs,targets):
    """
    return accuracy given a set of outputs and targets
    """
    thresh = network.thresh # decision thresh
    acc = [] # accuracy array
    for mb in range(targets.shape[0]):
        for out in range(targets.shape[1]):
            if targets[mb,out] == 0: continue
            response = outputs[mb,out] # Identify response time points
            target_resp = torch.ByteTensor([out]) # The correct target response
            max_resp = outputs[mb,:].argmax().byte()
            if max_resp==target_resp and response>thresh: # Make sure network response is correct respnose, and that it exceeds some threshold
                acc.append(1.0)
            else:
                acc.append(0)

    return acc


