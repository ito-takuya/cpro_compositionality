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
                 filename=datadir + 'results/MODEL/Trials_PracticeVsNovel_Default'):

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

    def addPracticedTasks(self,n=1,ordered=False):
        """
        adds additional novel tasks to the 'practicedRuleSet' variable
        """
        if ordered:
            novel2prac_ind = np.arange(n) 
        else:
            novel2prac_ind = np.random.choice(self.novelRuleSet.index,n,replace=False)

        # Add to random tasks to the practiced set
        practicedRuleSet = self.practicedRuleSet.append(self.novelRuleSet.iloc[novel2prac_ind],ignore_index=True)

        novelRuleSet = self.novelRuleSet.drop(novel2prac_ind,axis='index')
        novelRuleSet = novelRuleSet.reset_index(drop=True) # reset index, don't create new index column

        return practicedRuleSet, novelRuleSet, novel2prac_ind




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
    rule_ind = np.arange(11) # rules are the first 11 indices of input vector
    stim_ind = np.arange(11,27) # stimuli are the last 16 indices of input vector
    input_size = len(rule_ind) + len(stim_ind)
    input_matrix = np.zeros((input_size,ntrials_total))
    #output_matrix = np.zeros((4,ntrials_total))
    output_matrix = np.zeros((ntrials_total,))
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
            #output_matrix[:,trialcount] = out_code
            output_matrix[trialcount] = np.where(out_code==1)[0][0]

            trialcount += 1
            

    # Pad output with 2 additional units for pretraining tasks
    #tmp_zeros = np.zeros((2,output_matrix.shape[1]))
    #output_matrix = np.vstack((output_matrix,tmp_zeros))
    return input_matrix, output_matrix 

def create_all_trials(taskRuleSet,output_taskinfo=False):
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

    # Create taskinfo data frame to keep track of specific trial types
    taskinfo = {}
    taskinfo['Logic'] = []
    taskinfo['Sensory'] = []
    taskinfo['Motor'] = []

    ntrials_total = len(stimuliSet) * len(taskRuleSet)
    ####
    # Construct trial dynamics
    rule_ind = np.arange(11) # rules are the first 11 indices of input vector
    stim_ind = np.arange(11,27) # stimuli are the last 16 indices of input vector
    input_size = len(rule_ind) + len(stim_ind)
    input_matrix = np.zeros((input_size,len(stimuliSet),len(taskRuleSet)))
    #output_matrix = np.zeros((4,len(stimuliSet),len(taskRuleSet)))
    output_matrix = np.zeros((len(stimuliSet),len(taskRuleSet)))
    for tasknum in range(len(taskRuleSet)):
        
        trialcount = 0
        for i in stimuliSet.index:
    
            # Create full task design dataframe
            taskinfo['Logic'].append(taskRuleSet.Logic[tasknum])
            taskinfo['Sensory'].append(taskRuleSet.Sensory[tasknum])
            taskinfo['Motor'].append(taskRuleSet.Motor[tasknum])

            ## Create trial array
            # Find input code for this task set
            input_matrix[rule_ind,trialcount,tasknum] = taskRuleSet.Code[tasknum] 
            # Solve task to get the output code
            tmpresp, out_code = solveInputs(taskRuleSet.iloc[tasknum], stimuliSet.iloc[i])

            input_matrix[stim_ind,trialcount,tasknum] = stimuliSet.Code[i]
            #output_matrix[:,trialcount,tasknum] = out_code
            output_matrix[trialcount,tasknum] = np.where(out_code==1)[0][0]

            trialcount += 1
            
    # Pad output with 2 additional units for pretraining tasks
    #tmp_zeros = np.zeros((2,output_matrix.shape[1],output_matrix.shape[2]))
    #output_matrix = np.vstack((output_matrix,tmp_zeros))
    if output_taskinfo:
        return input_matrix, output_matrix, pd.DataFrame(taskinfo)
    else:
        return input_matrix, output_matrix 

def create_taskcontext_inputsOnly(taskRuleSet,output_taskinfo=False):
    """
    Creates input vectors for only the set of tasks included
    returns only the input matrix
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

    # Create taskinfo data frame to keep track of specific trial types
    taskinfo = {}
    taskinfo['Logic'] = []
    taskinfo['Sensory'] = []
    taskinfo['Motor'] = []

    ntrials_total = len(stimuliSet) * len(taskRuleSet)
    ####
    # Construct trial dynamics
    rule_ind = np.arange(11) # rules are the first 11 indices of input vector
    stim_ind = np.arange(11,27) # stimuli are the last 16 indices of input vector
    input_size = len(rule_ind) + len(stim_ind)
    input_matrix = np.zeros((len(taskRuleSet),input_size))
    for tasknum in range(len(taskRuleSet)):
        

        # Create full task design dataframe
        taskinfo['Logic'].append(taskRuleSet.Logic[tasknum])
        taskinfo['Sensory'].append(taskRuleSet.Sensory[tasknum])
        taskinfo['Motor'].append(taskRuleSet.Motor[tasknum])

        ## Create trial array
        # Find input code for this task set
        input_matrix[tasknum,rule_ind] = taskRuleSet.Code[tasknum] 
            
    # Pad output with 2 additional units for pretraining tasks
    #tmp_zeros = np.zeros((2,output_matrix.shape[1],output_matrix.shape[2]))
    #output_matrix = np.vstack((output_matrix,tmp_zeros))
    if output_taskinfo:
        return input_matrix, pd.DataFrame(taskinfo)
    else:
        return input_matrix 

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
    logicRules = {'both':1,
                  'either':2,
                  'notboth':[1,0],
                  'neither':[2,0]} # 0 - 'not', 1 - 'both', 2 - 'either'
    sensoryRules = {'red':3,
                    'vertical':4,
                    'high':5,
                    'constant':6}
    motorRules = {'l_mid':7,
                  'l_ind':8,
                  'r_ind':9,
                  'r_mid':10}
    
    
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
                code = np.zeros((11,))
                # Logic rule
                taskrules['Logic'].append(lo)
                code[logicRules[lo]] = 1
                
                # Sensory rule
                taskrules['Sensory'].append(se)
                code[sensoryRules[se]] = 1
                # Define sensory category
                if se=='red': category = 'Color'
                if se=='vertical': category = 'Orientation'
                if se=='high': category = 'Pitch'
                if se=='constant': category = 'Constant'
                taskrules['SensoryCategory'].append(category)
                
                # Motor rule
                taskrules['Motor'].append(mo)
                code[motorRules[mo]] = 1
                
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


def _create_sensory_pretraining_rules(negation=True):
    """
    create rule combinations with one sensory rule and one motor rule
    """    
    #logicRules = {'both':1,
    #              'either':2,
    #              'notboth':[1,0],
    #              'neither':[2,0]} # 0 - 'not', 1 - 'both', 2 - 'either'
    sensoryRules = {'red':3,
                    'vertical':4,
                    'high':5,
                    'constant':6}
    #motorRules = {'l_mid':7,
    #              'l_ind':8,
    #              'r_ind':9,
    #              'r_mid':10}
    
    
    taskrules = {}
    taskrules['Sensory'] = []
    # Create another field for the sensory category (to select stimuli from)
    taskrules['SensoryCategory'] = []
    # For RNN training
    taskrules['Code'] = []
    
    for se in sensoryRules:
        code = np.zeros((11,))
        # Sensory rule
        taskrules['Sensory'].append(se)
        code[sensoryRules[se]] = 1
        # Define sensory category
        if se=='red': category = 'Color'
        if se=='vertical': category = 'Orientation'
        if se=='high': category = 'Pitch'
        if se=='constant': category = 'Constant'
        taskrules['SensoryCategory'].append(category)
        
        taskrules['Code'].append(list(code))

    # If we have the negation of this task (e.g., If *NOT* RED ...)
    if negation:
        for se in sensoryRules:
            code = np.zeros((11,))
            #### Negation:
            code[0] = 1

            # Sensory rule
            taskrules['Sensory'].append(se)
            code[sensoryRules[se]] = 1
            # Define sensory category
            if se=='red': category = 'Color'
            if se=='vertical': category = 'Orientation'
            if se=='high': category = 'Pitch'
            if se=='constant': category = 'Constant'
            taskrules['SensoryCategory'].append(category)
            
            taskrules['Code'].append(list(code))
                
    return pd.DataFrame(taskrules)

def _solve_sensory_pretraining_task(task_rules,stimuli,printTask=False):
    """
    Solves simple stimulus-response associations given a stimulus, sensory rule, and motor rule 
    'stim' parameter indicates whether one should focus on the first or second stim
    """
    negation = True if task_rules.Code[0]==1 else False
    sensoryRule = task_rules.Sensory

    sensoryCategory = task_rules.SensoryCategory

    # Isolate the property for each stimulus relevant to the sensory rule
    stim = stimuli[sensoryCategory]

    # Run through logic rule gates
    if stim==sensoryRule:
        if negation:
            gate = False
        else:
            gate = True
    else:
        if negation:
            gate = True
        else:
            gate = False

    ## Print task first
    if printTask:
        print('Sensory rule:', sensoryRule)
        print('**Stimuli**')
        print(stim)


    # If gate is 'true', then indicate as all 1s
    outputcode = np.zeros((6,))
    if gate:
        outputcode[4] = 1
    else:
        outputcode[5] = 1

    return gate, outputcode

def _create_sensorytask_stimuli():
    """
    Create stimuli input for the sensory primitive task only
    """
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

def create_sensory_pretraining(negation=True):
    """
    Creates simple pretraining tasks 
    motor rule only (i.e., motor rule -> motor response)
    sensory + motor rules (stimulus-motor associations)
    """
    stimuliSet = _create_sensorytask_stimuli()
    taskRuleSet = _create_sensory_pretraining_rules(negation=negation)

    # Create 1d array to randomly sample indices from
    stimIndices = np.arange(len(stimuliSet))
    taskIndices = np.arange(len(taskRuleSet))

    ####
    # Construct trial dynamics
    rule_ind = np.arange(11) # rules are the first 12 indices of input vector
    stim_ind = np.arange(11,27) # stimuli are the last 16 indices of input vector
    input_size = len(rule_ind) + len(stim_ind)
    input_matrix = []
    output_matrix = []
    for tasknum in range(len(taskRuleSet)):
        
        for i in stimuliSet.index:
            input_arr = np.zeros((input_size,))

            ## Create trial array -- 1st stim
            # Find input code for this task set
            input_arr[rule_ind] = taskRuleSet.Code[tasknum] 
            #
            input_arr[stim_ind] = stimuliSet.Code[i]
            input_matrix.append(input_arr)

            # Solve task to get the output code
            tmpresp, out_code = _solve_sensory_pretraining_task(taskRuleSet.iloc[tasknum],stimuliSet.iloc[i])
            output_matrix.append(np.where(out_code)[0][0])

    input_matrix = np.asarray(input_matrix)
    output_matrix = np.asarray(output_matrix)

    return input_matrix, output_matrix 



def _create_motor_pretraining_rules(negation=True):
    """
    create rule combinations with one sensory rule and one motor rule
    """    
    #logicRules = {'both':1,
    #              'either':2,
    #              'notboth':[1,0],
    #              'neither':[2,0]} # 0 - 'not', 1 - 'both', 2 - 'either'
    #sensoryRules = {'red':3,
    #                'vertical':4,
    #                'high':5,
    #                'constant':6}
    motorRules = {'l_mid':7,
                  'l_ind':8,
                  'r_ind':9,
                  'r_mid':10}
    
    
    taskrules = {}
    taskrules['Motor'] = []
    taskrules['Code'] = []
    
    for mo in motorRules:
        code = np.zeros((11,))
        # Motor rule
        taskrules['Motor'].append(mo)
        code[motorRules[mo]] = 1
        
        taskrules['Code'].append(list(code))

    # If we have the negation of this task (e.g., If *NOT* RED ...)
    if negation:
        for mo in motorRules:
            code = np.zeros((11,))
            #### Negation:
            code[0] = 1

            # Motor rule
            taskrules['Motor'].append(mo)
            code[motorRules[mo]] = 1
            
            taskrules['Code'].append(list(code))
                
    return pd.DataFrame(taskrules)

def _solve_motor_pretraining_task(task_rules,stimuli,printTask=False):
    """
    Solves a motor rule task given a stimulus + motor rule 
    'stim' parameter indicates whether one should focus on the first or second stim
    """
    negation = True if task_rules.Code[0]==1 else False
    motorRule = task_rules.Motor

    
    # Apply logic gating to motor rules
    if motorRule=='l_mid':
        if negation:
            motorOutput = 'l_ind'
        else:
            motorOutput = 'l_mid'

    if motorRule=='l_ind':
        if negation:
            motorOutput = 'l_mid'
        else:
            motorOutput = 'l_ind'

    if motorRule=='r_mid':
        if negation:
            motorOutput = 'r_ind'
        else:
            motorOutput = 'r_mid'

    if motorRule=='r_ind':
        if negation:
            motorOutput = 'r_mid'
        else:
            motorOutput = 'r_ind'

    outputcode = np.zeros((4,))
    if motorOutput=='l_mid': outputcode[0] = 1
    if motorOutput=='l_ind': outputcode[1] = 1
    if motorOutput=='r_mid': outputcode[2] = 1
    if motorOutput=='r_ind': outputcode[3] = 1

    return motorOutput, outputcode

def create_motor_pretraining(negation=True):
    """
    Creates simple pretraining tasks 
    motor rule only (i.e., motor rule -> motor response)
    sensory + motor rules (stimulus-motor associations)
    """
    #stimuliSet = createSensoryInputs()
    #taskRuleSet = _create_motor_pretraining_rules(negation=negation)

    ## Create 1d array to randomly sample indices from
    #stimIndices = np.arange(len(stimuliSet))
    #taskIndices = np.arange(len(taskRuleSet))

    #####
    ## Construct trial dynamics
    #rule_ind = np.arange(11) # rules are the first 12 indices of input vector
    #stim_ind = np.arange(11,27) # stimuli are the last 16 indices of input vector
    #input_size = len(rule_ind) + len(stim_ind)
    #input_matrix = []
    #output_matrix = []
    #for tasknum in range(len(taskRuleSet)):
    #    
    #    for i in stimuliSet.index:
    #        input_arr = np.zeros((input_size,))

    #        ## Create trial array -- 1st stim
    #        # Find input code for this task set
    #        input_arr[rule_ind] = taskRuleSet.Code[tasknum] 
    #        #
    #        input_arr[stim_ind] = stimuliSet.Code[i]
    #        input_matrix.append(input_arr)

    #        # Solve task to get the output code
    #        tmpresp, out_code = _solve_motor_pretraining_task(taskRuleSet.iloc[tasknum],stimuliSet.iloc[i])
    #        output_matrix.append(np.where(out_code)[0][0])

    #input_matrix = np.asarray(input_matrix)
    #output_matrix = np.asarray(output_matrix)

    stimuliSet = createSensoryInputs()
    taskRuleSet = _create_motor_pretraining_rules(negation=negation)

    # Create 1d array to randomly sample indices from
    stimIndices = np.arange(len(stimuliSet))
    taskIndices = np.arange(len(taskRuleSet))

    ####
    # Construct trial dynamics
    rule_ind = np.arange(11) # rules are the first 12 indices of input vector
    stim_ind = np.arange(11,27) # stimuli are the last 16 indices of input vector
    input_size = len(rule_ind) + len(stim_ind)
    input_matrix = []
    output_matrix = []
    for tasknum in range(len(taskRuleSet)):
        
        input_arr = np.zeros((input_size,))

        ## Create trial array -- 1st stim
        # Find input code for this task set
        input_arr[rule_ind] = taskRuleSet.Code[tasknum] 
        #
        #input_arr[stim_ind] = stimuliSet.Code[i]
        input_matrix.append(input_arr)

        # Solve task to get the output code
        tmpresp, out_code = _solve_motor_pretraining_task(taskRuleSet.iloc[tasknum],None)
        output_matrix.append(np.where(out_code)[0][0])

    input_matrix = np.asarray(input_matrix)
    output_matrix = np.asarray(output_matrix)

    return input_matrix, output_matrix 




def _create_logic_pretraining_rules(negation=True):
    """
    create rule combinations with one sensory rule and one motor rule
    """    
#    logicRules = {'both':1,
#                  'either':2}
    logicRules = {'both':1,
                  'either':2,
                  'notboth':[0,1],
                  'neither':[0,2]} # 0 - 'not', 1 - 'both', 2 - 'either'
    #sensoryRules = {'red':3,
    #                'vertical':4,
    #                'high':5,
    #                'constant':6}
    #motorRules = {'l_mid':7,
    #              'l_ind':8,
    #              'r_ind':9,
    #              'r_mid':10}
    
    
    taskrules = {}
    taskrules['Logic'] = []
    # For RNN training
    taskrules['Code'] = []
    
    for lo in logicRules:
        code = np.zeros((11,))
        
        # Logic rule
        taskrules['Logic'].append(lo)
        code[logicRules[lo]] = 1
        
        taskrules['Code'].append(list(code))

    return pd.DataFrame(taskrules)

def _solve_logic_pretraining_task(task_rules,stimuli,printTask=False):
    """
    Solves simple stimulus-response associations given a stimulus, sensory rule, and motor rule 
    'stim' parameter indicates whether one should focus on the first or second stim
    """
    logicRule = task_rules.Logic

    BOOL = stimuli.BOOL

    # Run through logic rule gates
    # AND/'both' are the same (logically speaking)
    if logicRule == 'both':
        if BOOL=='AND':
            gate = True
        else:
            gate = False

    # AND/'notboth' are not logically the same (notboth is an 'or' operation)
    if logicRule == 'notboth':
        if BOOL=='AND':
            gate = False
        else:
            gate = True

    # AND/'either' are not logically the same (either is an 'or' operation)
    if logicRule == 'either':
        gate = True

    # AND/'neither' are logically the same (neither is an 'and' operation)
    if logicRule == 'neither':
        if BOOL=='AND':
            gate = False
        else:
            gate = False


    ## Print task first
    if printTask:
        print('Logic rule:', logicRule)
        print('**stim bool**')
        print(BOOL)

    # If gate is 'true', then indicate as all 1s
    outputcode = np.zeros((6,))
    if gate:
        outputcode[4] = 1
    else:
        outputcode[5] = 1

    return gate, outputcode

def _create_logictask_stimuli():
    """
    Create stimuli input for the logic rule primitive task only
    Actually stimulus labels don't matter -- just keep track of whether or not they are 'AND' or 'OR'
    Also, only present one stimulus dimension at a time
    EX: Color only: RED x RED == AND | RED X BLUE == OR
    EX: Color only: VERTICAL x VERTICAL == AND | VERTICAL X HORIZONTAL == OR
    """
    stimdata = {}

    stimdata['BOOL'] = []
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
    
    for col1 in color:
        for col2 in color:
            code = np.zeros((len(stim_ind),))
            # Stim 1
            code[col1] = 1
            code[col2+8] = 1

            # Code
            stimdata['Code'].append(list(code))

            ## Logical evaluation
            if col1==col2:
                stimdata['BOOL'].append('AND')
            else:
                stimdata['BOOL'].append('OR')

    for ori1 in orientation:
        for ori2 in orientation:
            code = np.zeros((len(stim_ind),))
            # Stim 1
            code[ori1] = 1
            code[ori2+8] = 1

            # Code
            stimdata['Code'].append(list(code))

            ## Logical evaluation
            if ori1==ori2:
                stimdata['BOOL'].append('AND')
            else:
                stimdata['BOOL'].append('OR')

    for pit1 in pitch:
        for pit2 in pitch:
            code = np.zeros((len(stim_ind),))
            # Stim 1
            code[pit1] = 1
            code[pit2+8] = 1

            # Code
            stimdata['Code'].append(list(code))

            ## Logical evaluation
            if pit1==pit2:
                stimdata['BOOL'].append('AND')
            else:
                stimdata['BOOL'].append('OR')

    for con1 in constant:
        for con2 in constant:
            code = np.zeros((len(stim_ind),))
            # Stim 1
            code[con1] = 1
            code[con2+8] = 1

            # Code
            stimdata['Code'].append(list(code))

            ## Logical evaluation
            if con1==con2:
                stimdata['BOOL'].append('AND')
            else:
                stimdata['BOOL'].append('OR')

    return pd.DataFrame(stimdata) 

def create_logic_pretraining(negation=True):
    """
    Creates simple pretraining tasks 
    logic rule only (i.e., logic rule -> true/false gating output)
    """
    stimuliSet = _create_logictask_stimuli()
    taskRuleSet = _create_logic_pretraining_rules(negation=negation)

    # Create 1d array to randomly sample indices from
    stimIndices = np.arange(len(stimuliSet))
    taskIndices = np.arange(len(taskRuleSet))

    ####
    # Construct trial dynamics
    rule_ind = np.arange(11) # rules are the first 12 indices of input vector
    stim_ind = np.arange(11,27) # stimuli are the last 16 indices of input vector
    input_size = len(rule_ind) + len(stim_ind)
    input_matrix = []
    output_matrix = []
    for tasknum in range(len(taskRuleSet)):
        
        for i in stimuliSet.index:
            input_arr = np.zeros((input_size,))

            ## Create trial array -- 1st stim
            # Find input code for this task set
            input_arr[rule_ind] = taskRuleSet.Code[tasknum] 
            #
            input_arr[stim_ind] = stimuliSet.Code[i]
            input_matrix.append(input_arr)

            # Solve task to get the output code
            tmpresp, out_code = _solve_logic_pretraining_task(taskRuleSet.iloc[tasknum],stimuliSet.iloc[i])
            output_matrix.append(np.where(out_code)[0][0])

    input_matrix = np.asarray(input_matrix)
    output_matrix = np.asarray(output_matrix)

    return input_matrix, output_matrix 






def _create_logicalsensory_pretraining_rules():
    """
    create rule combinations with one sensory rule and one motor rule
    """    
    logicRules = {'both':1,
                  'either':2,
                  'notboth':[0,1],
                  'neither':[0,2]} # 0 - 'not', 1 - 'both', 2 - 'either'
    sensoryRules = {'red':3,
                    'vertical':4,
                    'high':5,
                    'constant':6}
    #motorRules = {'l_mid':7,
    #              'l_ind':8,
    #              'r_ind':9,
    #              'r_mid':10}

    logickeys = ['both','either','notboth','neither']
    sensorykeys = ['red','vertical','high','constant']
    
    
    taskrules = {}
    taskrules['Sensory'] = []
    taskrules['Logic'] = []
    # Create another field for the sensory category (to select stimuli from)
    taskrules['SensoryCategory'] = []
    # For RNN training
    taskrules['Code'] = []
    
    #for i in range(4):
    for lo in logicRules:
        for se in sensoryRules:
            #se = sensorykeys[i]
            #lo = logickeys[i]

            code = np.zeros((11,))
            # Sensory rule
            taskrules['Sensory'].append(se)
            code[sensoryRules[se]] = 1
            # Define sensory category
            if se=='red': category = 'Color'
            if se=='vertical': category = 'Orientation'
            if se=='high': category = 'Pitch'
            if se=='constant': category = 'Constant'
            taskrules['SensoryCategory'].append(category)
            
            # Motor rule
            taskrules['Logic'].append(lo)
            code[logicRules[lo]] = 1
            
            taskrules['Code'].append(list(code))
                
    return pd.DataFrame(taskrules)

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
    rule_ind = np.arange(11) # rules are the first 12 indices of input vector
    stim_ind = np.arange(11,27) # stimuli are the last 16 indices of input vector
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
            output_matrix.append(np.where(out_code)[0][0])

    input_matrix = np.asarray(input_matrix)
    output_matrix = np.asarray(output_matrix)
            
    return input_matrix, output_matrix 


def _create_sensorimotor_pretraining_rules(negation=True):
    """
    create rule combinations with one sensory rule and one motor rule
    """    
    #logicRules = {'both':1,
    #              'either':2,
    #              'notboth':[1,0],
    #              'neither':[2,0]} # 0 - 'not', 1 - 'both', 2 - 'either'
    sensoryRules = {'red':3,
                    'vertical':4,
                    'high':5,
                    'constant':6}
    motorRules = {'l_mid':7,
                  'l_ind':8,
                  'r_ind':9,
                  'r_mid':10}

    sensorykeys = ['red','vertical','high','constant']
    motorkeys = ['l_mid', 'l_ind', 'r_ind', 'r_mid']
    
    
    taskrules = {}
    taskrules['Sensory'] = []
    taskrules['Motor'] = []
    # Create another field for the sensory category (to select stimuli from)
    taskrules['SensoryCategory'] = []
    # For RNN training
    taskrules['Code'] = []
    
    #for i in range(4):
    for se in sensoryRules:
        for mo in motorRules:
        
            code = np.zeros((11,))
            # Sensory rule
            taskrules['Sensory'].append(se)
            code[sensoryRules[se]] = 1
            # Define sensory category
            if se=='red': category = 'Color'
            if se=='vertical': category = 'Orientation'
            if se=='high': category = 'Pitch'
            if se=='constant': category = 'Constant'
            taskrules['SensoryCategory'].append(category)
            
            # Motor rule
            taskrules['Motor'].append(mo)
            code[motorRules[mo]] = 1
            
            taskrules['Code'].append(list(code))

    #### Add negation
    if negation:
        for se in sensoryRules:
            for mo in motorRules:
                
                code = np.zeros((11,))

                # Negation
                code[0] = 1
                # Sensory rule
                taskrules['Sensory'].append(se)
                code[sensoryRules[se]] = 1
                # Define sensory category
                if se=='red': category = 'Color'
                if se=='vertical': category = 'Orientation'
                if se=='high': category = 'Pitch'
                if se=='constant': category = 'Constant'
                taskrules['SensoryCategory'].append(category)
                
                # Motor rule
                taskrules['Motor'].append(mo)
                code[motorRules[mo]] = 1
                
                taskrules['Code'].append(list(code))
                    
    return pd.DataFrame(taskrules)

def _solve_sensorimotor_pretraining_task(task_rules,stimuli,printTask=False):
    """
    Solves simple stimulus-response associations given a stimulus, sensory rule, and motor rule 
    'stim' parameter indicates whether one should focus on the first or second stim
    """
    negation = True if task_rules.Code[0]==1 else False

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
    if not negation: # normal task 
        if motorOutput=='l_mid': outputcode[0] = 1
        if motorOutput=='l_ind': outputcode[1] = 1
        if motorOutput=='r_mid': outputcode[2] = 1
        if motorOutput=='r_ind': outputcode[3] = 1
    else: # remember if negation, need to include the 'not' rule 
        if motorOutput=='l_mid': outputcode[1] = 1
        if motorOutput=='l_ind': outputcode[0] = 1
        if motorOutput=='r_mid': outputcode[3] = 1
        if motorOutput=='r_ind': outputcode[2] = 1

    return motorOutput, outputcode

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

def create_sensorimotor_pretraining(negation=False):
    """
    Creates simple pretraining tasks 
    motor rule only (i.e., motor rule -> motor response)
    sensory + motor rules (stimulus-motor associations)
    """
    stimuliSet = _create_pretraining_stimuli()
    taskRuleSet = _create_sensorimotor_pretraining_rules(negation=negation)

    # Create 1d array to randomly sample indices from
    stimIndices = np.arange(len(stimuliSet))
    taskIndices = np.arange(len(taskRuleSet))

    ####
    # Construct trial dynamics
    rule_ind = np.arange(11) # rules are the first 12 indices of input vector
    stim_ind = np.arange(11,27) # stimuli are the last 16 indices of input vector
    input_size = len(rule_ind) + len(stim_ind)
    input_matrix = []
    output_matrix = []
    for tasknum in range(len(taskRuleSet)):
        
        for i in stimuliSet.index:
            input_arr = np.zeros((input_size,))

            ## Create trial array -- 1st stim
            # Find input code for this task set
            input_arr[rule_ind] = taskRuleSet.Code[tasknum] 
            # If this is the 'negation' version of the task, make sure to include all 1s for the 'not' rule
            #
            input_arr[stim_ind] = stimuliSet.Code[i]
            input_matrix.append(input_arr)

            # Solve task to get the output code
            tmpresp, out_code = _solve_sensorimotor_pretraining_task(taskRuleSet.iloc[tasknum],stimuliSet.iloc[i])
            output_matrix.append(np.where(out_code)[0][0])

    input_matrix = np.asarray(input_matrix)
    output_matrix = np.asarray(output_matrix)

    #tmp_zeros = np.zeros((output_matrix.shape[0],2))
    #output_matrix = np.hstack((output_matrix,tmp_zeros))
            
    return input_matrix, output_matrix 



#def _create_logicalsensory_pretraining_rules():
#    """
#    create rule combinations with one sensory rule and one motor rule
#    """    
#    logicRules = {'both':1,
#                  'either':2,
#                  'notboth':[0,1],
#                  'neither':[0,2]} # 0 - 'not', 1 - 'both', 2 - 'either'
#    sensoryRules = {'red':3,
#                    'vertical':4,
#                    'high':5,
#                    'constant':6}
#    #motorRules = {'l_mid':7,
#    #              'l_ind':8,
#    #              'r_ind':9,
#    #              'r_mid':10}
#    
#    
#    taskrules = {}
#    taskrules['Sensory'] = []
#    taskrules['Logic'] = []
#    # Create another field for the sensory category (to select stimuli from)
#    taskrules['SensoryCategory'] = []
#    # For RNN training
#    taskrules['Code'] = []
#    
#    for se in sensoryRules:
#        for lo in logicRules:
#            code = np.zeros((11,))
#            # Sensory rule
#            taskrules['Sensory'].append(se)
#            code[sensoryRules[se]] = 1
#            # Define sensory category
#            if se=='red': category = 'Color'
#            if se=='vertical': category = 'Orientation'
#            if se=='high': category = 'Pitch'
#            if se=='constant': category = 'Constant'
#            taskrules['SensoryCategory'].append(category)
#            
#            # Motor rule
#            taskrules['Logic'].append(lo)
#            code[logicRules[lo]] = 1
#            
#            taskrules['Code'].append(list(code))
#                
#    return pd.DataFrame(taskrules)
#
#def _solve_logicalsensory_pretraining_task(task_rules,stimuli,printTask=False):
#    """
#    Solves simple stimulus-response associations given a stimulus, sensory rule, and motor rule 
#    'stim' parameter indicates whether one should focus on the first or second stim
#    """
#    logicRule = task_rules.Logic
#    sensoryRule = task_rules.Sensory
#
#    sensoryCategory = task_rules.SensoryCategory
#
#    # Isolate the property for each stimulus relevant to the sensory rule
#    stim1 = stimuli[sensoryCategory + '1']
#    stim2 = stimuli[sensoryCategory + '2']
#
#    # Run through logic rule gates
#    if logicRule == 'both':
#        if stim1==sensoryRule and stim2==sensoryRule:
#            gate = True
#        else:
#            gate = False
#
#    if logicRule == 'notboth':
#        if stim1!=sensoryRule or stim2!=sensoryRule:
#            gate = True
#        else:
#            gate = False
#
#    if logicRule == 'either':
#        if stim1==sensoryRule or stim2==sensoryRule:
#            gate = True
#        else:
#            gate = False
#
#    if logicRule == 'neither':
#        if stim1!=sensoryRule and stim2!=sensoryRule:
#            gate = True
#        else:
#            gate = False
#
#
#    ## Print task first
#    if printTask:
#        print('Logic rule:', logicRule)
#        print('Sensory rule:', sensoryRule)
#        print('**Stimuli**')
#        print(stim1, stim2)
#
#    # If gate is 'true', then indicate as all 1s
#    outputcode = np.zeros((6,))
#    if gate:
#        outputcode[4] = 1
#    else:
#        outputcode[5] = 1
#
#    return gate, outputcode
#
#def create_logicalsensory_pretraining():
#    """
#    Creates simple pretraining tasks 
#    logic + sensory rules (logical sensory associations)
#    outputs will be all 1s or 0s, depending on a TRUE/FALSE assessment
#    """
#    stimuliSet = createSensoryInputs()
#    taskRuleSet = _create_logicalsensory_pretraining_rules()
#
#    # Create 1d array to randomly sample indices from
#    stimIndices = np.arange(len(stimuliSet))
#    taskIndices = np.arange(len(taskRuleSet))
#
#    ####
#    # Construct trial dynamics
#    rule_ind = np.arange(11) # rules are the first 12 indices of input vector
#    stim_ind = np.arange(11,27) # stimuli are the last 16 indices of input vector
#    input_size = len(rule_ind) + len(stim_ind)
#    input_matrix = []
#    output_matrix = []
#    for tasknum in range(len(taskRuleSet)):
#        
#        for i in stimuliSet.index:
#            input_arr = np.zeros((input_size,))
#
#            ## Create trial array -- 1st stim
#            # Find input code for this task set
#            input_arr[rule_ind] = taskRuleSet.Code[tasknum] 
#            input_arr[stim_ind] = stimuliSet.Code[i]
#            input_matrix.append(input_arr)
#
#            # Solve task to get the output code
#            tmpresp, out_code = _solve_logicalsensory_pretraining_task(taskRuleSet.iloc[tasknum],stimuliSet.iloc[i])
#            output_matrix.append(np.where(out_code)[0][0])
#
#    input_matrix = np.asarray(input_matrix)
#    output_matrix = np.asarray(output_matrix)
#            
#    return input_matrix, output_matrix 
