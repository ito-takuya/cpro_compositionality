# Module to construct task features (e.g., inputs/outputs of tasks) that are independent of the model
# Is used as a 'helper' module for model.py
# Inputs and outputs of CPRO taks
import numpy as np
import torch
import pandas as pd
import os
from ast import literal_eval
import h5py

datadir = '../../../data/'

motorCode = {0:'l_mid',
             1:'l_ind',
             2:'r_ind',
             3:'r_mid'}

class Experiment(object):
    """
    Batch trials, but specifically separate practiced versus novel task sets (4 practiced, 60 novel)
    """
    def __init__(self,
                 NUM_INPUT_ELEMENTS=28,
                 NUM_OUTPUT_ELEMENTS=4,
                 filename=datadir + 'results/MODEL/Trials_PracticeVsNovel_Default'):


        self.NUM_OUTPUT_ELEMENTS = NUM_OUTPUT_ELEMENTS
        self.NUM_INPUT_ELEMENTS = NUM_INPUT_ELEMENTS
        self.filename = filename

        if os.path.exists(filename + '_practice.csv'):
            print('Loading previously constructed practiced and novel rule sets...')
            self.practicedRuleSet = pd.read_csv(filename + '_practice.csv')
            self.novelRuleSet = pd.read_csv(filename + '_novel.csv')
            self.taskRuleSet = pd.read_csv(filename + '_all.csv')

            # Convert codes to arrays
            for i in self.practicedRuleSet.index:
                self.practicedRuleSet.Code[i] = literal_eval(self.practicedRuleSet.Code[i])
            for i in self.novelRuleSet.index:
                self.novelRuleSet.Code[i] = literal_eval(self.novelRuleSet.Code[i])
            for i in self.taskRuleSet.index:
                self.taskRuleSet.Code[i] = literal_eval(self.taskRuleSet.Code[i])
        else:
            print('Creating new practiced and novel rule sets...')
            self.splitPracticedNovelTaskSets()


    def createFullTaskSet(self,condition='practice',shuffle=False):
        if os.path.exists(self.filename + '.h5'):
            print('WARNING... overwriting previous task data set @', self.filename+'.h5')

        if condition=='practice':
            ntrials = len(createSensoryInputs())
            ruleset = self.practicedRuleSet
        elif condition=='novel':
            ntrials = len(createSensoryInputs())
            ruleset = self.novelRuleSet
        elif condition=='all':
            ntrials = len(createSensoryInputs())
            ruleset = self.taskRuleSet

        batchnum = 1 # only one batch if we sample the entire stimulus set equally
        batch_inputtensor, batch_outputtensor = create_all_trials(ruleset)

        h5f = h5py.File(self.filename + '.h5','a')
        try:
            h5f.create_dataset(condition + '/inputs',data=batch_inputtensor)
            h5f.create_dataset(condition + '/outputs',data=batch_outputtensor)
        except:
            del h5f[condition + '/inputs'], h5f[condition + '/outputs']
            h5f.create_dataset(condition + '/inputs',data=batch_inputtensor)
            h5f.create_dataset(condition + '/outputs',data=batch_outputtensor)
        h5f.close()

    def createRandomTrialBatches(self,nbatches,ntrials_per_task,condition='practice',nproc=10):
        if os.path.exists(self.filename + '.h5'):
            print('WARNING... overwriting previous task data set @', self.filename+'.h5')

        ntrials = ntrials_per_task

        if condition=='practice':
            ruleset = self.practicedRuleSet
        elif condition=='novel':
            ruleset = self.novelRuleSet
        elif condition=='all':
            ruleset = self.taskRuleSet
        # Initialize empty tensor for batches
        batch_inputtensor = np.zeros((self.NUM_INPUT_ELEMENTS, len(ruleset)*ntrials, nbatches))
        batch_outputtensor = np.zeros((self.NUM_OUTPUT_ELEMENTS, len(ruleset)*ntrials, nbatches))

        inputs = []
        for batch in range(nbatches):
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

    def loadFullTask(self,condition='practice',cuda=False):
        h5f = h5py.File(self.filename + '.h5','r')
        inputs = h5f[condition + '/inputs'][:].copy()
        outputs = h5f[condition + '/outputs'][:].copy()
        h5f.close()

        inputs = inputs.T
        outputs = outputs.T

        inputs = torch.from_numpy(inputs)
        outputs = torch.from_numpy(outputs)

        inputs = inputs.float()
        outputs = outputs.float()

        if cuda:
            inputs = inputs.cuda()
            outputs = outputs.cuda()

        return inputs, outputs

    def loadBatches(self,condition='practice',cuda=False):
        h5f = h5py.File(self.filename + '.h5','r')
        inputs = h5f[condition + '/inputs'][:].copy()
        outputs = h5f[condition + '/outputs'][:].copy()
        h5f.close()

        # Input dimensions: input features, nMiniblocks, nBatches
        inputs = np.transpose(inputs, (0, 2, 1)) # convert to: nBatches, nMiniblocks, input dimensions
        outputs = np.transpose(outputs, (0, 2, 1)) # convert to: nBatches, nMiniblocks, input dimensions

        inputs = torch.from_numpy(inputs)
        outputs = torch.from_numpy(outputs)

        inputs = inputs.float()
        outputs = outputs.float()

        if cuda:
            inputs = inputs.cuda()
            outputs = outputs.cuda()

        return inputs, outputs
    
    def splitPracticedNovelTaskSets(self):
        taskRuleSet = createRulePermutations()
        practicedRuleSet, novelRuleSet = create4Practiced60NovelTaskContexts(taskRuleSet)

        self.taskRuleSet = taskRuleSet
        self.practicedRuleSet = practicedRuleSet
        self.novelRuleSet = novelRuleSet
        # Save to file 
        taskRuleSet.to_csv(self.filename + '_all.csv')
        practicedRuleSet.to_csv(self.filename + '_practice.csv')
        novelRuleSet.to_csv(self.filename + '_novel.csv')

    def taskSimilarity(self,practicedSet, novelSet):
        practicedSet = practicedSet.reset_index()
        novelSet = novelSet.reset_index()
        task_similarity_arr = np.ones((len(novelSet),))
        for i in range(len(novelSet)):
            log = novelSet.Logic[i]
            sen = novelSet.Sensory[i]
            mot = novelSet.Motor[i]
            for j in range(len(practicedSet)):
                if log == practicedSet.Logic[j] and sen == practicedSet.Sensory[j]:
                    task_similarity_arr[i] = 2
                if log == practicedSet.Logic[j] and mot == practicedSet.Motor[j]:
                    task_similarity_arr[i] = 2
                if sen == practicedSet.Sensory[j] and mot == practicedSet.Motor[j]:
                    task_similarity_arr[i] = 2
                    
        sim1_ind = np.where(task_similarity_arr==1)[0]
        sim2_ind = np.where(task_similarity_arr==2)[0]

        taskSim1Set = novelSet.iloc[sim1_ind]
        taskSim2Set = novelSet.iloc[sim2_ind]

        return taskSim1Set, taskSim2Set

def create_random_trials(taskRuleSet,ntrials_per_task,seed):
    """
    Randomly generates a set of stimuli (nStimuli) for each task rule
    Will end up with 64 (task rules) * nStimuli total number of input stimuli
    
    If shuffle keyword is True, will randomly shuffle the training set
    Otherwise will start with taskrule1 (nStimuli), taskrule2 (nStimuli), etc.
    """
    np.random.seed(seed)

    stimuliSet = createSensoryInputs()

    # Create 1d array to randomly sample indices from
    stimIndices = np.arange(len(stimuliSet))
    taskIndices = np.arange(len(taskRuleSet))

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
            tmpresp, out_code = solveInputs(taskRuleSet.iloc[tasknum], stimuliSet2.iloc[0])

            input_matrix[stim_ind,trialcount] = stimuliSet2.Code[0]
            output_matrix[:,trialcount] = out_code

            trialcount += 1
            

    # Pad output with 2 additional units for pretraining tasks
    tmp_zeros = np.zeros((2,output_matrix.shape[1]))
    output_matrix = np.vstack((output_matrix,tmp_zeros))
    return input_matrix, output_matrix 

def create_all_trials(taskRuleSet):
    """
    Creates all possible trials given a task rule set (iterates through all possible stimulus combinations)
    Will end up with 64 (task rules) * nStimuli total number of input stimuli
    """
    stimuliSet = createSensoryInputs()

    # Create 1d array to randomly sample indices from
    stimIndices = np.arange(len(stimuliSet))
    taskIndices = np.arange(len(taskRuleSet))

    #randomTaskIndices = np.random.choice(taskIndices,len(taskIndices),replace=False)
    #randomTaskIndices = np.random.choice(taskIndices,nTasks,replace=False)
    #taskRuleSet2 = taskRuleSet.iloc[randomTaskIndices].copy(deep=True)
    #taskRuleSet = taskRuleSet.reset_index(drop=True)
    taskRuleSet = taskRuleSet.reset_index(drop=False)
    #taskRuleSet = taskRuleSet2.copy(deep=True)

    ntrials_total = len(stimuliSet) * len(taskRuleSet)
    ####
    # Construct trial dynamics
    rule_ind = np.arange(12) # rules are the first 12 indices of input vector
    stim_ind = np.arange(12,28) # stimuli are the last 16 indices of input vector
    input_size = len(rule_ind) + len(stim_ind)
    input_matrix = np.zeros((input_size,len(stimuliSet),len(taskRuleSet)))
    output_matrix = np.zeros((4,len(stimuliSet),len(taskRuleSet)))
    for tasknum in range(len(taskRuleSet)):
        
        trialcount = 0
        for i in stimuliSet.index:
    
            ## Create trial array
            # Find input code for this task set
            input_matrix[rule_ind,trialcount,tasknum] = taskRuleSet.Code[tasknum] 
            # Solve task to get the output code
            tmpresp, out_code = solveInputs(taskRuleSet.iloc[tasknum], stimuliSet.iloc[i])

            input_matrix[stim_ind,trialcount,tasknum] = stimuliSet.Code[i]
            output_matrix[:,trialcount,tasknum] = out_code

            trialcount += 1
            
    # Pad output with 2 additional units for pretraining tasks
    tmp_zeros = np.zeros((2,output_matrix.shape[1],output_matrix.shape[2]))
    output_matrix = np.vstack((output_matrix,tmp_zeros))
    return input_matrix, output_matrix 

def createSensoryInputs(nStims=2):
    stimdata = {}
    # Stim 1 empty columns
    stimdata['Color1'] = []
    stimdata['Orientation1'] = []
    stimdata['Pitch1'] = []
    stimdata['Constant1'] = []
    # Stim 2 empty columns
    stimdata['Color2'] = []
    stimdata['Orientation2'] = []
    stimdata['Pitch2'] = []
    stimdata['Constant2'] = []
    # Code for RNN training
    stimdata['Code'] = []

    # Property index tells us which columns ID the property in question
    color = {0:'red',
             1:'blue'}
    orientation = {2:'vertical',
                   3:'horizontal'}
    pitch = {4:'high',
             5:'low'}
    constant = {6:'constant',
                7:'beeping'}
    
    for col1 in color:
        for col2 in color:
            for ori1 in orientation:
                for ori2 in orientation:
                    for pit1 in pitch:
                        for pit2 in pitch:
                            for con1 in constant:
                                for con2 in constant:
                                    code = np.zeros((8*nStims,))
                                    # Stim 1
                                    code[col1] = 1
                                    stimdata['Color1'].append(color[col1])
                                    code[ori1] = 1
                                    stimdata['Orientation1'].append(orientation[ori1])
                                    code[pit1] = 1
                                    stimdata['Pitch1'].append(pitch[pit1])
                                    code[con1] = 1
                                    stimdata['Constant1'].append(constant[con1])
                                    # Stim 2 -- need to add 8, since this is the second stimuli
                                    code[col2+8] = 1
                                    stimdata['Color2'].append(color[col2])
                                    code[ori2+8] = 1
                                    stimdata['Orientation2'].append(orientation[ori2])
                                    code[pit2+8] = 1
                                    stimdata['Pitch2'].append(pitch[pit2])
                                    code[con2+8] = 1
                                    stimdata['Constant2'].append(constant[con2])
                                    
                                    # Code
                                    stimdata['Code'].append(list(code))
                    
    return pd.DataFrame(stimdata) 

def createRulePermutations():
    """
    May need to change this - both and not both are negations of each other, as are either and neither
    """    
    logicRules = {0: 'both',
                  1: 'notboth',
                  2: 'either',
                  3: 'neither'}
    sensoryRules = {4: 'red',
                    5: 'vertical',
                    6: 'high',
                    7: 'constant'}
    motorRules = {8: 'l_mid',
                  9: 'l_ind',
                  10: 'r_ind',
                  11: 'r_mid'}
    
    
    taskrules = {}
    taskrules['Logic'] = []
    taskrules['Sensory'] = []
    taskrules['Motor'] = []
    # Create another field for the sensory category (to select stimuli from)
    taskrules['SensoryCategory'] = []
    # For RNN training
    taskrules['Code'] = []
    
    for lo in logicRules:
        for se in sensoryRules:
            for mo in motorRules:
                code = np.zeros((12,))
                # Logic rule
                taskrules['Logic'].append(logicRules[lo])
                code[lo] = 1
                
                # Sensory rule
                taskrules['Sensory'].append(sensoryRules[se])
                code[se] = 1
                # Define sensory category
                if sensoryRules[se]=='red': category = 'Color'
                if sensoryRules[se]=='vertical': category = 'Orientation'
                if sensoryRules[se]=='high': category = 'Pitch'
                if sensoryRules[se]=='constant': category = 'Constant'
                taskrules['SensoryCategory'].append(category)
                
                # Motor rule
                taskrules['Motor'].append(motorRules[mo])
                code[mo] = 1
                
                taskrules['Code'].append(list(code))
                
    return pd.DataFrame(taskrules)

def solveInputs(task_rules, stimuli, printTask=False):
    """
    Solves CPRO task given a set of inputs and a task rule
    """
    logicRule = task_rules.Logic
    sensoryRule = task_rules.Sensory
    motorRule = task_rules.Motor

    sensoryCategory = task_rules.SensoryCategory

    # Isolate the property for each stimulus relevant to the sensory rule
    stim1 = stimuli[sensoryCategory + '1']
    stim2 = stimuli[sensoryCategory + '2']

    # Run through logic rule gates
    if logicRule == 'both':
        if stim1==sensoryRule and stim2==sensoryRule:
            gate = True
        else:
            gate = False

    if logicRule == 'notboth':
        if stim1!=sensoryRule or stim2!=sensoryRule:
            gate = True
        else:
            gate = False

    if logicRule == 'either':
        if stim1==sensoryRule or stim2==sensoryRule:
            gate = True
        else:
            gate = False

    if logicRule == 'neither':
        if stim1!=sensoryRule and stim2!=sensoryRule:
            gate = True
        else:
            gate = False


    ## Print task first
    if printTask:
        print('Logic rule:', logicRule)
        print('Sensory rule:', sensoryRule)
        print('Motor rule:', motorRule)
        print('**Stimuli**')
        print(stim1, stim2)

    # Apply logic gating to motor rules
    if motorRule=='l_mid':
        if gate==True:
            motorOutput = 'l_mid'
        else:
            motorOutput = 'l_ind'

    if motorRule=='l_ind':
        if gate==True:
            motorOutput = 'l_ind'
        else:
            motorOutput = 'l_mid'

    if motorRule=='r_mid':
        if gate==True:
            motorOutput = 'r_mid'
        else:
            motorOutput = 'r_ind'

    if motorRule=='r_ind':
        if gate==True:
            motorOutput = 'r_ind'
        else:
            motorOutput = 'r_mid'

    outputcode = np.zeros((4,))
    if motorOutput=='l_mid': outputcode[0] = 1
    if motorOutput=='l_ind': outputcode[1] = 1
    if motorOutput=='r_mid': outputcode[2] = 1
    if motorOutput=='r_ind': outputcode[3] = 1

    return motorOutput, outputcode

def createTrainTestTaskRules(taskRuleSet,nTrainSet=32,nTestSet=32):
    """
    Construct partitions of training and test set task contexts
    """
    nRulesPerTrainSet = nTrainSet/4.0
    nRulesPerTestSet = nTestSet/4.0
    if nRulesPerTrainSet%4.0!=0.0:
        print('WARNING: Number of rules per train/test set is not divisible by 4')
        #raise Exception('ERROR: Number of rules per train/test set needs to be divisible by 4!')

    df_test = pd.DataFrame()
    df_train = pd.DataFrame()
    df_test = []
    df_train = []
    # Make sure all columns exist 
    #df_train = df_train.append(taskRuleSet.iloc[0])

    # Iterate through tasks in a random manner
    ind = np.arange(len(taskRuleSet))
    np.random.shuffle(ind)
    taskcount = 0
    for i in ind:
        # Identify the rules in this task set
        #logic = taskRuleSet.Logic[i]
        #sensory = taskRuleSet.Sensory[i]
        #motor = taskRuleSet.Motor[i]

        ## Count number of logic rules for this task set
        #nLogic = np.sum(df_train.Logic==logic)
        #nSensory = np.sum(df_train.Sensory==sensory)
        #nMotor = np.sum(df_train.Motor==motor)
        #if nLogic<nRulesPerTrainSet and nSensory<nRulesPerTrainSet and nMotor<nRulesPerTrainSet:
        #    df_train = df_train.append(taskRuleSet.iloc[i])
        #else:
        #    df_test = df_test.append(taskRuleSet.iloc[i])
        if taskcount<nTrainSet:
            df_train.append(taskRuleSet.iloc[i])
        else:
            df_test.append(taskRuleSet.iloc[i])

        taskcount += 1

    df_test = pd.DataFrame(df_test)
    df_train = pd.DataFrame(df_train)

    return df_train, df_test

def create4Practiced60NovelTaskContexts(taskRuleSet):
    """
    Construct a set of 4 practiced task contexts, where each rule (within each rule domain) is presented exactly once
    CPRO constructs a task context from 3 unique task rules (a logic, sensory, and motor rule). Thus, across 4 practiced task contexts, exactly 12 unique task rules will be employed.  
    N.B.: This is the task design in the experimental data
    """
    df_prac = pd.DataFrame()
    df_nov = pd.DataFrame()
    df_prac = []
    df_nov = []

    # Identify the unique rule values
    logic_rules = np.unique(taskRuleSet.Logic.values)
    sensory_rules = np.unique(taskRuleSet.Sensory.values)
    motor_rules = np.unique(taskRuleSet.Motor.values)

    prac_ind = []
    available_tasksets = np.arange(0,64)
    while len(available_tasksets)>0:
        i = np.random.choice(available_tasksets,1)[0] # Pick random task context
        prac_ind.append(i)
        df_prac.append(taskRuleSet.iloc[i])
        #available_tasksets = np.delete(available_tasksets,np.where(available_tasksets==i)[0][0]) # delete this task context from available set of tasks
        # Remove any tasks with those specific task rules
        logic_ind = np.where(taskRuleSet.Logic.values==str(taskRuleSet.iloc[i].Logic))[0]
        for ind in logic_ind: 
            if ind in available_tasksets:
                available_tasksets = np.delete(available_tasksets,np.where(available_tasksets==ind)[0][0])
        # sensory
        sensory_ind = np.where(taskRuleSet.Sensory.values==taskRuleSet.iloc[i].Sensory)[0]
        for ind in sensory_ind: 
            if ind in available_tasksets:
                available_tasksets = np.delete(available_tasksets,np.where(available_tasksets==ind)[0][0])
        # motor
        motor_ind = np.where(taskRuleSet.Motor.values==taskRuleSet.iloc[i].Motor)[0]
        for ind in motor_ind: 
            if ind in available_tasksets:
                available_tasksets = np.delete(available_tasksets,np.where(available_tasksets==ind)[0][0])

    ####  Now identify the novel tasks
    for i in np.arange(64):
        if i not in prac_ind:
            df_nov.append(taskRuleSet.iloc[i])

    df_nov = pd.DataFrame(df_nov)
    df_prac = pd.DataFrame(df_prac)

    return df_prac, df_nov


def _create_sensorimotor_pretraining_rules():
    """
    create rule combinations with one sensory rule and one motor rule
    """    
#    logicRules = {0: 'both',
#                  1: 'notboth',
#                  2: 'either',
#                  3: 'neither'}
    sensoryRules = {4: 'red',
                    5: 'vertical',
                    6: 'high',
                    7: 'constant'}
    motorRules = {8: 'l_mid',
                  9: 'l_ind',
                  10: 'r_ind',
                  11: 'r_mid'}
    
    
    taskrules = {}
    taskrules['Sensory'] = []
    taskrules['Motor'] = []
    # Create another field for the sensory category (to select stimuli from)
    taskrules['SensoryCategory'] = []
    # For RNN training
    taskrules['Code'] = []
    
    for se in sensoryRules:
        for mo in motorRules:
            code = np.zeros((12,))
            # Sensory rule
            taskrules['Sensory'].append(sensoryRules[se])
            code[se] = 1
            # Define sensory category
            if sensoryRules[se]=='red': category = 'Color'
            if sensoryRules[se]=='vertical': category = 'Orientation'
            if sensoryRules[se]=='high': category = 'Pitch'
            if sensoryRules[se]=='constant': category = 'Constant'
            taskrules['SensoryCategory'].append(category)
            
            # Motor rule
            taskrules['Motor'].append(motorRules[mo])
            code[mo] = 1
            
            taskrules['Code'].append(list(code))
                
    return pd.DataFrame(taskrules)

def _create_logicalsensory_pretraining_rules():
    """
    create rule combinations with one sensory rule and one motor rule
    """    
    logicRules = {0: 'both',
                  1: 'notboth',
                  2: 'either',
                  3: 'neither'}
    sensoryRules = {4: 'red',
                    5: 'vertical',
                    6: 'high',
                    7: 'constant'}
#    motorRules = {8: 'l_mid',
#                  9: 'l_ind',
#                  10: 'r_ind',
#                  11: 'r_mid'}
    
    
    taskrules = {}
    taskrules['Sensory'] = []
    taskrules['Logic'] = []
    # Create another field for the sensory category (to select stimuli from)
    taskrules['SensoryCategory'] = []
    # For RNN training
    taskrules['Code'] = []
    
    for se in sensoryRules:
        for lo in logicRules:
            code = np.zeros((12,))
            # Sensory rule
            taskrules['Sensory'].append(sensoryRules[se])
            code[se] = 1
            # Define sensory category
            if sensoryRules[se]=='red': category = 'Color'
            if sensoryRules[se]=='vertical': category = 'Orientation'
            if sensoryRules[se]=='high': category = 'Pitch'
            if sensoryRules[se]=='constant': category = 'Constant'
            taskrules['SensoryCategory'].append(category)
            
            # Motor rule
            taskrules['Logic'].append(logicRules[lo])
            code[lo] = 1
            
            taskrules['Code'].append(list(code))
                
    return pd.DataFrame(taskrules)

def _solve_sensorimotor_pretraining_task(task_rules,stimuli,printTask=False):
    """
    Solves simple stimulus-response associations given a stimulus, sensory rule, and motor rule 
    'stim' parameter indicates whether one should focus on the first or second stim
    """
    sensoryRule = task_rules.Sensory
    motorRule = task_rules.Motor

    sensoryCategory = task_rules.SensoryCategory

    # Isolate the property for each stimulus relevant to the sensory rule
    stim = stimuli[sensoryCategory]

    # Run through logic rule gates
    if stim==sensoryRule:
        gate = True
    else:
        gate = False

    ## Print task first
    if printTask:
        print('Sensory rule:', sensoryRule)
        print('Motor rule:', motorRule)
        print('**Stimuli**')
        print(stim)

    # Apply logic gating to motor rules
    if motorRule=='l_mid':
        if gate==True:
            motorOutput = 'l_mid'
        else:
            motorOutput = 'l_ind'

    if motorRule=='l_ind':
        if gate==True:
            motorOutput = 'l_ind'
        else:
            motorOutput = 'l_mid'

    if motorRule=='r_mid':
        if gate==True:
            motorOutput = 'r_mid'
        else:
            motorOutput = 'r_ind'

    if motorRule=='r_ind':
        if gate==True:
            motorOutput = 'r_ind'
        else:
            motorOutput = 'r_mid'

    outputcode = np.zeros((4,))
    if motorOutput=='l_mid': outputcode[0] = 1
    if motorOutput=='l_ind': outputcode[1] = 1
    if motorOutput=='r_mid': outputcode[2] = 1
    if motorOutput=='r_ind': outputcode[3] = 1

    return motorOutput, outputcode

def _solve_logicalsensory_pretraining_task(task_rules,stimuli,printTask=False):
    """
    Solves simple stimulus-response associations given a stimulus, sensory rule, and motor rule 
    'stim' parameter indicates whether one should focus on the first or second stim
    """
    logicRule = task_rules.Logic
    sensoryRule = task_rules.Sensory

    sensoryCategory = task_rules.SensoryCategory

    # Isolate the property for each stimulus relevant to the sensory rule
    stim1 = stimuli[sensoryCategory + '1']
    stim2 = stimuli[sensoryCategory + '2']

    # Run through logic rule gates
    if logicRule == 'both':
        if stim1==sensoryRule and stim2==sensoryRule:
            gate = True
        else:
            gate = False

    if logicRule == 'notboth':
        if stim1!=sensoryRule or stim2!=sensoryRule:
            gate = True
        else:
            gate = False

    if logicRule == 'either':
        if stim1==sensoryRule or stim2==sensoryRule:
            gate = True
        else:
            gate = False

    if logicRule == 'neither':
        if stim1!=sensoryRule and stim2!=sensoryRule:
            gate = True
        else:
            gate = False


    ## Print task first
    if printTask:
        print('Logic rule:', logicRule)
        print('Sensory rule:', sensoryRule)
        print('**Stimuli**')
        print(stim1, stim2)

    # If gate is 'true', then indicate as all 1s
    outputcode = np.zeros((6,))
    if gate:
        outputcode[4] = 1
    else:
        outputcode[5] = 1

    return gate, outputcode

def _create_pretraining_stimuli():
    stimdata = {}
    # Stim 1 empty columns
    stimdata['Color'] = []
    stimdata['Orientation'] = []
    stimdata['Pitch'] = []
    stimdata['Constant'] = []
    # Code for RNN training
    stimdata['Code'] = []

    stim_ind = np.arange(16) # 16 total stimulus combinations

    # Property index tells us which columns ID the property in question
    color = {0:'red',
             1:'blue'}
    orientation = {2:'vertical',
                   3:'horizontal'}
    pitch = {4:'high',
             5:'low'}
    constant = {6:'constant',
                7:'beeping'}
    
    # Code in the sensory stimulus for the 1st stimulus presentation
    for col1 in color:
        for ori1 in orientation:
            for pit1 in pitch:
                for con1 in constant:
                    code = np.zeros((len(stim_ind),))
                    # Stim 1
                    code[col1] = 1
                    stimdata['Color'].append(color[col1])
                    code[ori1] = 1
                    stimdata['Orientation'].append(orientation[ori1])
                    code[pit1] = 1
                    stimdata['Pitch'].append(pitch[pit1])
                    code[con1] = 1
                    stimdata['Constant'].append(constant[con1])

                    # Code
                    stimdata['Code'].append(list(code))

    # Code in the sensory stimulus for the 2nd stimulus presentation
    for col2 in color:
        for ori2 in orientation:
            for pit2 in pitch:
                for con2 in constant:
                    code = np.zeros((len(stim_ind),))
                    # Stim 2 -- need to add 8, since this is the second stimuli
                    code[col2+8] = 1
                    stimdata['Color'].append(color[col2])
                    code[ori2+8] = 1
                    stimdata['Orientation'].append(orientation[ori2])
                    code[pit2+8] = 1
                    stimdata['Pitch'].append(pitch[pit2])
                    code[con2+8] = 1
                    stimdata['Constant'].append(constant[con2])
                    
                    # Code
                    stimdata['Code'].append(list(code))
                    
    return pd.DataFrame(stimdata) 

def create_motorrule_pretraining():
    """
    Creates all possible trials given a task rule set (iterates through all possible stimulus combinations)
    Will end up with 64 (task rules) * nStimuli total number of input stimuli
    """

    ####
    # Construct trial dynamics
    rule_ind = np.arange(12) # rules are the first 12 indices of input vector
    stim_ind = np.arange(12,28) # stimuli are the last 16 indices of input vector
    motor_ind = np.arange(8,12)
    input_size = len(rule_ind) + len(stim_ind)
    input_matrix = []
    output_matrix = []

    # Now create motor rule primitives
    for mot in motorCode:
        input_arr = np.zeros((input_size,))
        output_arr = np.zeros((4,))
        input_arr[motor_ind[mot]] = 1
        output_arr[mot] = 1

        input_matrix.append(input_arr)
        output_matrix.append(output_arr)
    
    input_matrix = np.asarray(input_matrix)
    output_matrix = np.asarray(output_matrix)
    
    tmp_zeros = np.zeros((output_matrix.shape[0],2))
    output_matrix = np.hstack((output_matrix,tmp_zeros))
            
    return input_matrix, output_matrix 

def create_sensorimotor_pretraining():
    """
    Creates simple pretraining tasks 
    motor rule only (i.e., motor rule -> motor response)
    sensory + motor rules (stimulus-motor associations)
    """
    stimuliSet = _create_pretraining_stimuli()
    taskRuleSet = _create_sensorimotor_pretraining_rules()

    # Create 1d array to randomly sample indices from
    stimIndices = np.arange(len(stimuliSet))
    taskIndices = np.arange(len(taskRuleSet))

    ####
    # Construct trial dynamics
    rule_ind = np.arange(12) # rules are the first 12 indices of input vector
    stim_ind = np.arange(12,28) # stimuli are the last 16 indices of input vector
    input_size = len(rule_ind) + len(stim_ind)
    input_matrix = []
    output_matrix = []
    for tasknum in range(len(taskRuleSet)):
        
        for i in stimuliSet.index:
            input_arr = np.zeros((input_size,))

            ## Create trial array -- 1st stim
            # Find input code for this task set
            input_arr[rule_ind] = taskRuleSet.Code[tasknum] 
            input_arr[stim_ind] = stimuliSet.Code[i]
            input_matrix.append(input_arr)

            # Solve task to get the output code
            tmpresp, out_code = _solve_sensorimotor_pretraining_task(taskRuleSet.iloc[tasknum],stimuliSet.iloc[i])
            output_matrix.append(out_code)

    input_matrix = np.asarray(input_matrix)
    output_matrix = np.asarray(output_matrix)

    tmp_zeros = np.zeros((output_matrix.shape[0],2))
    output_matrix = np.hstack((output_matrix,tmp_zeros))
            
    return input_matrix, output_matrix 

def create_logicalsensory_pretraining():
    """
    Creates simple pretraining tasks 
    logic + sensory rules (logical sensory associations)
    outputs will be all 1s or 0s, depending on a TRUE/FALSE assessment
    """
    stimuliSet = createSensoryInputs()
    taskRuleSet = _create_logicalsensory_pretraining_rules()

    # Create 1d array to randomly sample indices from
    stimIndices = np.arange(len(stimuliSet))
    taskIndices = np.arange(len(taskRuleSet))

    ####
    # Construct trial dynamics
    rule_ind = np.arange(12) # rules are the first 12 indices of input vector
    stim_ind = np.arange(12,28) # stimuli are the last 16 indices of input vector
    input_size = len(rule_ind) + len(stim_ind)
    input_matrix = []
    output_matrix = []
    for tasknum in range(len(taskRuleSet)):
        
        for i in stimuliSet.index:
            input_arr = np.zeros((input_size,))

            ## Create trial array -- 1st stim
            # Find input code for this task set
            input_arr[rule_ind] = taskRuleSet.Code[tasknum] 
            input_arr[stim_ind] = stimuliSet.Code[i]
            input_matrix.append(input_arr)

            # Solve task to get the output code
            tmpresp, out_code = _solve_logicalsensory_pretraining_task(taskRuleSet.iloc[tasknum],stimuliSet.iloc[i])
            output_matrix.append(out_code)

    input_matrix = np.asarray(input_matrix)
    output_matrix = np.asarray(output_matrix)
            
    return input_matrix, output_matrix 
