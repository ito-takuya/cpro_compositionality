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
                 num_rule_inputs=11,
                 num_sensory_inputs=16,
                 num_hidden=128,
                 num_motor_decision_outputs=6,
                 num_hidden_layers=2,
                 learning_rate=0.0001,
                 thresh=0.0,
                 si_c=0,
                 device='cpu',
                 lossfunc='MSE'):

        # Define general parameters
        self.num_rule_inputs = num_rule_inputs
        self.num_sensory_inputs =  num_sensory_inputs
        self.num_hidden = num_hidden
        self.num_motor_decision_outputs = num_motor_decision_outputs
        self.num_hidden_layers = num_hidden_layers
        
        # Define entwork architectural parameters
        super(ANN,self).__init__()

        self.w_in = torch.nn.Linear(num_sensory_inputs+num_rule_inputs,num_hidden)
        self.w_rec = torch.nn.Linear(num_hidden,num_hidden)
        self.w_out = torch.nn.Linear(num_hidden,num_motor_decision_outputs)
        #self.func_out = torch.nn.Softmax(dim=1)
        #self.func_out1d = torch.nn.Softmax(dim=0) # used for online learning (since only one trial is trained at a time)
        self.func = torch.nn.ReLU()
        # Create weights for PS training
        self.w_ps = torch.nn.Linear(num_hidden, 1)
        with torch.no_grad():
            self.w_ps.weight.copy_(torch.ones(1,num_hidden))

        self.dropout_in = torch.nn.Dropout(p=0.2)
        self.dropout_rec = torch.nn.Dropout(p=0.5)

        # Initialize RNN units
        self.units = torch.nn.Parameter(self.initHidden())

        # Define loss function
        if lossfunc=='MSE':
            self.lossfunc = torch.nn.MSELoss(reduction='none')
        if lossfunc=='CrossEntropy':
            self.lossfunc = torch.nn.CrossEntropyLoss()

        # Decision threshhold for behavior
        self.thresh = thresh

        # Construct optimizer
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        #self.optimizer = torch.optim.Adagrad(self.parameters(), lr=learning_rate)
        #self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        #self.optimizer = torch.optim.RMSprop(self.parameters())
        
        #### Zenke et al. SI parameters
        self.si_c = si_c           #-> hyperparam: how strong to weigh SI-loss ("regularisation strength")
        self.epsilon = 0.1      #-> dampening parameter: bounds 'omega' when squared parameter-change goes to 0

        self.device = device
        self.to(device)
    
    def initHidden(self):
        return torch.randn(1, self.num_hidden)

    def forward(self,inputs,noise=False,dropout=True,return1st=False):
        """
        Run a forward pass of a trial by input_elements matrix
        """
        #Add noise to inputs
        if noise:
            inputs = inputs + torch.randn(inputs.shape, device=self.device, dtype=torch.float)/5 #(self.num_sensory_inputs+self.num_rule_inputs)
            #inputs = inputs + torch.randn(inputs.shape, device=self.device, dtype=torch.float)/(self.num_sensory_inputs+self.num_rule_inputs)

        # Map inputs into RNN space
        hidden = self.w_in(inputs) 
        if dropout: hidden = self.dropout_in(hidden)
        hidden = self.func(hidden)
        hidden1st = hidden
        # Define rnn private noise/spont_act
        #if noise:
        #    spont_act = torch.randn(hidden.shape, device=self.device,dtype=torch.float)/self.num_hidden
        #    #spont_act = torch.randn(hidden.shape, device=self.device,dtype=torch.float)/5 #self.num_hidden
#       #     spont_act = torch.randn(hidden.shape, device=self.device,dtype=torch.float)
        #    # Add private noise to each unit, and add to the input
        #    hidden = hidden + spont_act

        # Run RNN
        if self.num_hidden_layers>1:
            for i in range(self.num_hidden_layers-1):
                hidden = self.w_rec(hidden)
                if dropout: hidden = self.dropout_rec(hidden)
                hidden = self.func(hidden)
        
        # Compute outputs
        h2o = self.w_out(hidden) # Generate linear outupts
        outputs = self.w_out(hidden) # Generate linear outupts
        #if h2o.dim()==1: # for online learning
        #    outputs = self.func_out1d(h2o)
        #else:
        #    outputs = self.func_out(h2o) # Pass through nonlinearity

        if return1st:
            return outputs, hidden, hidden1st
        else:
            return outputs, hidden


    def forward_ps(self,inputs,noise=False,dropout=True,return1st=False):
        """
        Run a forward pass of a trial by input_elements matrix
        """
        #Add noise to inputs
        #if noise:
        #    inputs = inputs + torch.randn(inputs.shape, device=self.device, dtype=torch.float)/5 #(self.num_sensory_inputs+self.num_rule_inputs)

        # Map inputs into RNN space
        hidden = self.w_in(inputs) 
        if dropout: hidden = self.dropout_in(hidden)
        hidden = self.func(hidden)
        hidden1st = hidden
        # Define rnn private noise/spont_act
        if noise:
            spont_act = torch.randn(hidden.shape, device=self.device,dtype=torch.float)/self.num_hidden
        #    #spont_act = torch.randn(hidden.shape, device=self.device,dtype=torch.float)/5 #self.num_hidden
#       #     spont_act = torch.randn(hidden.shape, device=self.device,dtype=torch.float)
        #    # Add private noise to each unit, and add to the input
            hidden = hidden + spont_act

        # Run RNN
        if self.num_hidden_layers>1:
            for i in range(self.num_hidden_layers-1):
                hidden = self.w_rec(hidden)
                if dropout: hidden = self.dropout_rec(hidden)
                hidden = self.func(hidden)
        
        # Compute outputs
        h2o = self.w_out(hidden) # Generate linear outupts
        outputs = self.w_ps(hidden) # Generate linear outupts
        #if h2o.dim()==1: # for online learning
        #    outputs = self.func_out1d(h2o)
        #else:
        #    outputs = self.func_out(h2o) # Pass through nonlinearity

        if return1st:
            return outputs, hidden, hidden1st
        else:
            return outputs, hidden

    def update_omega(self, W, epsilon):
        '''After completing training on a task, update the per-parameter regularization strength.

        [W]         <dict> estimated parameter-specific contribution to changes in total loss of completed task
        [epsilon]   <float> dampening parameter (to bound [omega] when [p_change] goes to 0)'''

        # Loop over all parameters
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')

                # Find/calculate new values for quadratic penalty on parameters
                p_prev = getattr(self, '{}_SI_prev_task'.format(n))
                p_current = p.detach().clone()
                p_change = p_current - p_prev
                omega_add = W[n]/(p_change**2 + epsilon)
                try:
                    omega = getattr(self, '{}_SI_omega'.format(n))
                except AttributeError:
                    omega = p.detach().clone().zero_()
                omega_new = omega + omega_add

                # Store initial tensors in state dict
                #self.register_buffer('{}_SI_prev_task'.format(n), p_current) ### Originally this was not commented out
                self.register_buffer('{}_SI_omega'.format(n), omega_new)


    def surrogate_loss(self):
        '''Calculate SI's surrogate loss.'''
        try:
            losses = []
            for n, p in self.named_parameters():
                if p.requires_grad:
                    # Retrieve previous parameter values and their normalized path integral (i.e., omega)
                    n = n.replace('.', '__')
                    prev_values = getattr(self, '{}_SI_prev_task'.format(n))
                    omega = getattr(self, '{}_SI_omega'.format(n))
                    # Calculate SI's surrogate loss, sum over all parameters
                    losses.append((omega * (p-prev_values)**2).sum())

                    ##### New -- update previous p AFTER surrogate loss is computed
                    self.register_buffer('{}_SI_prev_task'.format(n), p.detach().clone())
                    ##### EDIT finished
            return sum(losses)
        except AttributeError:
            # SI-loss is 0 if there is no stored omega yet
            return torch.tensor(0., device=self._device())

def train(network, inputs, targets, si=True, ps_optim=None, dropout=False):
    """Train network"""

    network.train()
    network.zero_grad()
    network.optimizer.zero_grad()


    outputs, hidden = network.forward(inputs,noise=False,dropout=dropout)

    # Calculate loss
    if isinstance(network.lossfunc,torch.nn.CrossEntropyLoss):
       # tmp_target = []
       # if targets.dim()==1:
       #     tmp_target.append(torch.where(targets)[0])
       # else:
       #     for i in range(targets.shape[0]):
       #         tmp_target.append(torch.where(targets[i,:])[0][0])
        #tmp_target = np.asarray(tmp_target)
        #tmp_target = torch.from_numpy(tmp_target)
        #loss = network.lossfunc(outputs,tmp_target)
        loss = network.lossfunc(outputs,targets)
        loss = torch.mean(loss)
    else:
        loss = network.lossfunc(outputs,targets)
        loss = torch.mean(loss)
    #if si is not None:
    #    if network.si_c>0:
    #        loss += (network.si_c * network.surrogate_loss()).detach()

    ### PS regularization
    if ps_optim is not None:
        ps_outputs, hidden = network.forward(ps_optim.inputs_ps,noise=False,dropout=False)
        ps = calculatePS(hidden,ps_optim.match_logic_ind)
        logicps = ps # Want to maximize
        # Sensory PS
        ps = calculatePS(hidden,ps_optim.match_sensory_ind)
        sensoryps =ps
        # Motor PS
        ps = calculatePS(hidden,ps_optim.match_motor_ind)
        motorps = ps
        #ps_reg = (logicps + sensoryps + motorps) * ps_optim.ps
        msefunc = torch.nn.MSELoss(reduction='mean')

        ps_reg = torch.tensor(0., requires_grad=True).to(network.device)
        ps_reg += (3.0-logicps+sensoryps+motorps) * ps_optim.ps
        #ps_reg += (sensoryps) * ps_optim.ps
        #loss = -msefunc(torch.mean(ps_outputs),ps_reg)
        loss += ps_reg


        #print('PS reg:', ps_reg)
        #print(loss)
        #print(ps_loss)

    # Backprop and update weights
    loss.backward()
    network.optimizer.step() # Update parameters using optimizer
    
    if ps_optim is not None:
        return outputs, targets, loss.item(), logicps, sensoryps, motorps
    else:
        return outputs, targets, loss.item()
    #return outputs, targets, ps_reg 

#def trainps(network,inputs_ps,targets_ps,ps_optim,dropout=False):
#    """Train network"""
#
#
#    ps_outputs, hidden = network.forward_ps(ps_optim.inputs_ps,noise=True,dropout=dropout)
#    ps = calculatePS(hidden,ps_optim.match_logic_ind)
#    logicps = ps # Want to maximize
#    # Sensory PS
#    ps = calculatePS(hidden,ps_optim.match_sensory_ind)
#    sensoryps = ps
#    # Motor PS
#    ps = calculatePS(hidden,ps_optim.match_motor_ind)
#    motorps = ps
#
#    return logicps, sensoryps, motorps


def trainps(network,inputs_ps,targets_ps,ps_optim,dropout=False):
    """Train network"""

    network.train()
    network.zero_grad()
    network.optimizer.zero_grad()


    ps_outputs, hidden = network.forward_ps(ps_optim.inputs_ps,noise=True,dropout=dropout)
    ps = calculatePS(hidden,ps_optim.match_logic_ind)
    logicps = ps # Want to maximize
    # Sensory PS
    ps = calculatePS(hidden,ps_optim.match_sensory_ind)
    sensoryps = ps
    # Motor PS
    ps = calculatePS(hidden,ps_optim.match_motor_ind)
    motorps = ps
    #ps_reg = (logicps + sensoryps + motorps) * ps_optim.ps
    msefunc = torch.nn.MSELoss(reduction='mean')

    ps_reg = torch.tensor(0., requires_grad=True).to(network.device)
    ps_reg += (3.0-logicps+sensoryps+motorps) * ps_optim.ps
    #ps_reg += (sensoryps) * ps_optim.ps
    loss = msefunc(torch.mean(ps_outputs),ps_reg)

    #l2loss = torch.tensor(0., requires_grad=True).to(network.device)
    #for name, param in network.named_parameters():
    #    if 'weight' in name:
    #        l2loss += param.norm(2)*0.1 + .5
    ##loss += l2loss 
    #loss = l2loss
    
    # Backprop and update weights
    loss.backward()
    network.optimizer.step() # Update parameters using optimizer

    return ps_outputs, targets_ps, loss.item(), logicps, sensoryps, motorps

def batch_training(network,train_inputs,train_outputs,acc_cutoff=99.5,si=True,
                     verbose=True):

    accuracy_per_batch = []

    batch_ordering = np.arange(train_inputs.shape[0])

    np.random.shuffle(batch_ordering)

    nsamples_viewed = 0
    nbatches_trained = 0
    timestart = time.time()
    for batch in np.arange(train_inputs.shape[0]):
        batch_id = batch_ordering[batch]

        train_input_batch = train_inputs[batch_id,:,:]
        train_output_batch = train_outputs[batch_id,:,:]

        outputs, targets, loss = train(network,
                                       train_inputs[batch_id,:,:],
                                       train_outputs[batch_id,:,:],
                                       si=si)


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

def task_training(network,train_inputs,train_outputs,acc_cutoff=99.5,si=True,dropout=False,
                     verbose=True):
    """
    training for tasks (using all stimulus combinations)
    """

    nsamples_viewed = 0
    timestart = time.time()

    nbatches_break = 10
    batch = 0
    accuracy = 0
    while accuracy < acc_cutoff:

        outputs, targets, loss = train(network,
                                       train_inputs,
                                       train_outputs,
                                       si=si,
                                       dropout=dropout)

        #if si is not None:
        #    W = si
        #    if network.si_c>0:
        #        for n, p in network.named_parameters():
        #            if p.requires_grad:
        #                n = n.replace('.', '__')

        #                # Find/calculate new values for quadratic penalty on parameters
        #                p_prev = getattr(network, '{}_SI_prev_task'.format(n))
        #                if p.grad is not None:
        #                    W[n].add_(-p.grad*(p.detach()-p_prev)) # parameter-specific contribution to changes in total loss of completed task

        #        network.update_omega(W, network.epsilon)


        targets = targets.cpu()
        outputs = outputs.cpu()

        acc = accuracyScore(network,outputs,targets)

        accuracy = np.mean(np.asarray(acc))*100.0

        if verbose and batch%50==0:
            timeend = time.time()
            print('Iteration:', batch)
            print('\tloss:', loss.item())
            print('\tAccuracy: ', str(round(accuracy,4)),'%')
    
        
        batch += 1

        if accuracy>acc_cutoff:
            if verbose: print('Achieved', accuracy, '% accuracy, greater than cutoff (', acc_cutoff, '%; loss:', loss,') accuracy... stopping training after', batch, 'batches')

        nsamples_viewed += train_inputs.shape[0]

    return nsamples_viewed, batch

def accuracyScore(network,outputs,targets):
    """
    return accuracy given a set of outputs and targets
    """
    #thresh = network.thresh # decision thresh
    #acc = [] # accuracy array
    #for mb in range(targets.shape[0]):
    #    response = outputs[mb,targets[mb]] # Identify response time points
    #    #target_resp = torch.ByteTensor([targets[mb]]) # The correct target response
    #    target_resp = targets[mb] # The correct target response
    #    max_resp = outputs[mb,:].argmax().byte()
    #    if max_resp==target_resp and response>thresh: # Make sure network response is correct respnose, and that it exceeds some threshold
    #        acc.append(1.0)
    #    else:
    #        acc.append(0)

    thresh = network.thresh # decision thresh
    acc = [] # accuracy array
    if outputs.dim()==2:
        max_resp = torch.argmax(outputs,1)
    else:
        max_resp = torch.argmax(outputs)
    acc = max_resp==targets
    acc = torch.mean(acc.float()).detach().item()

    return acc


def sortConditionsPS(labels,labels2,labels3,shuffle=False):
    """
    Returns the indices required to perform PS
    labels - 1d array/list of labels from which to build decoders (can be binary or multi-class)
    labels2 - 1d array/list of secondary labels from which to maximize similarity/matches to build cosine similarities
    labels3 - 1d array/list of tertiary labels from which to maximize similarity/matches to build cosine similarities
    
    returns match indices: 3d matrix (condition pairs X 16 matches X contexts (to subtract)

    """
    if shuffle:
        indices = np.arange(len(labels))
        np.random.shuffle(indices)
        labels = labels[indices]
        labels2 = labels2[indices]
        labels3 = labels3[indices]

    classes = np.unique(labels) # return the unique class labels
    match_ind = []
    ps_score = np.zeros((len(classes),len(classes))) # create matrix to indicate the ps scores for each pair of conditions
    i = 0
    for cond1 in classes: # first condition
        ind_cond1 = np.where(labels==cond1)[0] # return the indices for the first class
        
        j = 0
        #match_ind.append([])
        for cond2 in classes: # second condition
            if i == j: 
                j+=1
                # skip if same conditions or if doing a symmetrical calculation - this is not informative (dot product is symmetrical)
                continue 
            ind_cond2 = np.where(labels==cond2)[0] # return the indices for the second class

            #### Now for each condition, find a pair that maximizes rule similarity
            match_ind.append([])
            for ind1 in ind_cond1: # for each index in condition 1
                label2_instance = labels2[ind1]
                label3_instance = labels3[ind1]

                # Now find these two label indices in the second condition for matching
                cond2_label2_ind = np.where(labels2==label2_instance)[0]
                cond2_label3_ind = np.where(labels3==label3_instance)[0]

                matchinglabels_ind = np.intersect1d(cond2_label2_ind,cond2_label3_ind)
                ind2 = np.intersect1d(matchinglabels_ind,ind_cond2)
                if len(ind2)>1:
                    raise Exception("Something's wrong... this should be a unique index")
                
                ind2 = ind2[0]
                indices = np.hstack((ind1,ind2))
                match_ind[-1].append(indices)

            j += 1

        i += 1

#    ps_score = ps_score + ps_score.T # make the matrix symmetric and fill out other half of the matrix
    match_ind = np.asarray(match_ind)
    match_ind = torch.from_numpy(match_ind)

    return match_ind

def calculatePS(data,match_indices):
    """
    data - observations X features 2d matrix (features correspond to unit activations)
    """
    n_contrasts = match_indices.size(0) #contrast is e.g., BOTH V EITHER
    n_matches = match_indices.size(1) # number of contexts with matches e.g., RED + LMID in 2nd and 3rd matches

    triu_ind = torch.triu_indices(n_matches,n_matches,offset=1)
    ps_avg = torch.empty(n_contrasts)
    for i in range(n_contrasts):
        data1 = data[match_indices[i,:,0],:]
        data2 = data[match_indices[i,:,1],:]

        diff_vec = data1-data2

        diff_vec = diff_vec.transpose(1,0) / torch.norm(diff_vec,dim=1)
        ps_mat = torch.matmul(diff_vec.transpose(1,0),diff_vec)
        ps_avg[i] = torch.mean(ps_mat[triu_ind]).item()

    return torch.mean(ps_avg).item()

