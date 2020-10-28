import numpy as np
import pandas as pd
import scipy.stats as stats

datadir = '../../../data/fMRI_BehavData/'

keyValues = {'Logic':'LogicCue[LogLevel5]', 'Sensory':'SemanticCue[LogLevel5]', 'Motor':'ResponseCue[LogLevel5]','Accuracy':'Feedback[LogLevel6]', 'Novelty':'TaskType_rec', 'TaskNum':'TaskName[LogLevel5]',
             'PracTask1':'PracTaskA','PracTask2':'PracTaskB','PracTask3':'PracTaskC', 'PracTask4':'PracTaskD','PracTaskIntro':'PracIntroExampleList',
             'LogicExample':'LogicCue[SubTrial]','SensoryExample':'SemanticCue[SubTrial]','MotorExample':'ResponseCue[SubTrial]',
             'PracIntroAccuracy':'Feedback[LogLevel5]'}

def behaviorOfNovelty(subjNums,behavior='Accuracy'):
    prac_acc = []
    nov_acc = []
    df_acc = {}
    df_acc[behavior] = []
    df_acc['Condition'] = []
    df_acc['Subject'] = []
    for subj in subjNums:

        df = pd.read_csv(datadir + subj + '_behavdata_reformatted.csv')
        zeroRT_ind = np.where(df.RT.values==0)[0]
        df.RT.values[zeroRT_ind] = np.nan

        intro_tasks = np.unique(df['TaskNum'].values[1:9])
    #     print('Subject', subj, intro_tasks)

        #### Identify practiced tasks
        prac_ind = np.where(df['Novelty'].values=='Prac')[0]
        prac_tasks = np.unique(df['TaskNum'].values[prac_ind])
        #### Identify 1st and 2nd presentations of practiced task blocks
        prac_ind1 = []
        prac_ind2 = []
        for i in prac_tasks:
            ind1 = np.where(df['TaskNum'].values==i)[0][2:5] # first miniblock (3 trials)
            ind2 = np.where(df['TaskNum'].values==i)[0][5:] # second miniblock (3 trials)
            prac_ind1.extend(ind1)
            prac_ind2.extend(ind2)
        prac_ind1 = np.asarray(prac_ind1)
        prac_ind2 = np.asarray(prac_ind2)
        # calculate accuracy for practiced tasks (all)
        if behavior=='Accuracy':
            acc = np.nanmean(df[behavior].values[prac_ind]=='Correct')
        if behavior=='RT':
            acc = np.nanmean(df[behavior].values[prac_ind])
        df_acc[behavior].append(acc)
        df_acc['Subject'].append(subj)
        df_acc['Condition'].append('Practiced')
        # calculate accuracy for 1st set of practiced tasks
        if behavior=='Accuracy':
            acc = np.nanmean(df[behavior].values[prac_ind1]=='Correct')
        if behavior=='RT':
            acc = np.nanmean(df[behavior].values[prac_ind1])
        df_acc[behavior].append(acc)
        df_acc['Subject'].append(subj)
        df_acc['Condition'].append('1st Practiced')
        # calculate accuracy for 2nd set of practiced tasks
        if behavior=='Accuracy':
            acc = np.nanmean(df[behavior].values[prac_ind2]=='Correct')
        if behavior=='RT':
            acc = np.nanmean(df[behavior].values[prac_ind2])
        df_acc[behavior].append(acc)
        df_acc['Subject'].append(subj)
        df_acc['Condition'].append('2nd Practiced')
        
        
        #### Identify novel tasks
        nov_ind = np.where(df['Novelty'].values=='Novel')[0]
        nov_tasks = np.unique(df['TaskNum'].values[nov_ind])
        #### Identify 1st and 2nd presentations of novel task blocks
        nov_ind1 = []
        nov_ind2 = []
        for i in nov_tasks:
            ind1 = np.where(df['TaskNum'].values==i)[0][:3] # first miniblock
            ind2 = np.where(df['TaskNum'].values==i)[0][3:] # second miniblock
            nov_ind1.extend(ind1)
            nov_ind2.extend(ind2)
        nov_ind1 = np.asarray(nov_ind1)
        nov_ind2 = np.asarray(nov_ind2)
        # Calculate accuracy for all novel tasks
        if behavior=='Accuracy':
            acc = np.nanmean(df[behavior].values[nov_ind]=='Correct')
        if behavior=='RT':
            acc = np.nanmean(df[behavior].values[nov_ind])
        df_acc[behavior].append(acc)
        df_acc['Subject'].append(subj)
        df_acc['Condition'].append('Novel')
        # Calculate accuracy for 1st set of novel tasks
        if behavior=='Accuracy':
            acc = np.nanmean(df[behavior].values[nov_ind1]=='Correct')
        if behavior=='RT':
            acc = np.nanmean(df[behavior].values[nov_ind1])
        df_acc[behavior].append(acc)
        df_acc['Subject'].append(subj)
        df_acc['Condition'].append('1st Novel')
        # Calculate accuracy for 2nd set of novel tasks
        if behavior=='Accuracy':
            acc = np.nanmean(df[behavior].values[nov_ind2]=='Correct')
        if behavior=='RT':
            acc = np.nanmean(df[behavior].values[nov_ind2])
        df_acc[behavior].append(acc)
        df_acc['Subject'].append(subj)
        df_acc['Condition'].append('2nd Novel')

    #     practice_tasks = np.unique(df['TaskNum'].values[prac_ind])

    df_acc = pd.DataFrame(df_acc)
    return df_acc

def behaviorOfTaskSimilarity(subjNums, firstOnly=False, behavior='Accuracy'):
    """
    Returns a dataframe with the behavioral performance of trials based on task similarity
    Conditions: Practiced tasks, tasks with 2-rules that are similar, tasks with 3-rules that are similar

    firstOnly : If True, only analyze the behavior of the first presentation of task rule sets
    """
    prac_acc = []
    nov_acc = []
    df_acc = {}
    df_acc[behavior] = []
    df_acc['Condition'] = []
    df_acc['Subject'] = []
    for subj in subjNums:

        df = pd.read_csv(datadir + subj + '_behavdata_reformatted.csv')
        zeroRT_ind = np.where(df.RT.values==0)[0]
        df.RT.values[zeroRT_ind] = np.nan
        # Exclude the pre-run practice trials
        df.TaskNum.values[:9] = np.nan
        #### Identify practiced tasks
        prac_ind = np.where(df['Novelty'].values=='Prac')[0]
        prac_tasks = np.unique(df['TaskNum'].values[prac_ind])
        # Use only first miniblock of all practiced tasks
        if firstOnly:
            ind1 = []
            for i in prac_tasks:
                ind = np.where(df['TaskNum'].values==i)[0][:3] # first miniblock (3 trials)
                ind1.extend(ind)
            prac_ind = np.asarray(ind1)
        logic_rule = []
        sensory_rule = []
        motor_rule = []
        for task in prac_tasks:
            ind = np.where(df.TaskNum.values==task)[0]
            # Note all tasks have the same rule set so only need to identify the first rule
            logic_rule.append(df.Logic.values[ind][0]) 
            sensory_rule.append(df.Sensory.values[ind][0])
            motor_rule.append(df.Motor.values[ind][0])
        
        #### Now identify novel tasks 
        novel_ind = np.where(df.Novelty.values=='Novel')[0]
        novel_tasks = np.unique(df.TaskNum.values[novel_ind])
        task_similarity = np.ones((len(novel_tasks),))
        noveltaskcount = 0
        for task in novel_tasks:
            ind = np.where(df.TaskNum.values==task)[0]
            # Note all tasks have the same rule set so only need to identify the first rule
            log = df.Logic.values[ind][0]
            sen = df.Sensory.values[ind][0]
            mot = df.Motor.values[ind][0]
            # Iterate through all practiced tasks and identify task similarity score
            similarityscore = 0
            for i in range(len(prac_tasks)):
                if log==logic_rule[i] and sen==sensory_rule[i]: 
                    task_similarity[noveltaskcount] = 2
                if log==logic_rule[i] and mot==motor_rule[i]: 
                    task_similarity[noveltaskcount] = 2
                if mot==motor_rule[i] and sen==sensory_rule[i]: 
                    task_similarity[noveltaskcount] = 2
    #         if task_similarity[noveltaskcount]==1:
    #             for i in range(len(prac_tasks)):
    #                 print(log, sen, mot, '|', logic_rule[i], sensory_rule[i], motor_rule[i])
            noveltaskcount += 1
            
        #### Identify practiced tasks
        if behavior=='Accuracy':
            acc = np.nanmean(df[behavior].values[prac_ind]=='Correct')
        if behavior=='RT':
            acc = np.nanmean(df[behavior].values[prac_ind])
        df_acc[behavior].append(acc)
        df_acc['Subject'].append(subj)
        df_acc['Condition'].append('Practiced')
        
        #### Identify tasks with 1 and 2 rule similarities
        sim2_ind = np.where(task_similarity==2)[0]
        accs = []
        for task in sim2_ind:
            if firstOnly:
                ind = np.where(df.TaskNum.values==novel_tasks[task])[0][:3] # important bc similarity indices are obtained on novel tasks only
            else:
                ind = np.where(df.TaskNum.values==novel_tasks[task])[0][:] # use all trials
            accs.extend(df[behavior].values[ind])
        accs = np.asarray(accs)
        if behavior=='Accuracy':
            acc = np.nanmean(accs=='Correct') 
        if behavior=='RT':
            acc = np.nanmean(accs) 
        df_acc[behavior].append(acc)
        df_acc['Subject'].append(subj)
        df_acc['Condition'].append('2-rule similarity')

        sim1_ind = np.where(task_similarity==1)[0]
        accs = []
        for task in sim1_ind:
            if firstOnly:
                ind = np.where(df.TaskNum.values==novel_tasks[task])[0][:3] # important bc similarity indices are obtained on novel tasks only
            else:
                ind = np.where(df.TaskNum.values==novel_tasks[task])[0][:] # use all trials
            accs.extend(df[behavior].values[ind])
        accs = np.asarray(accs)
        if behavior=='Accuracy':
            acc = np.nanmean(accs=='Correct') 
        if behavior=='RT':
            acc = np.nanmean(accs) 
        df_acc[behavior].append(acc)
        df_acc['Subject'].append(subj)
        df_acc['Condition'].append('1-rule similarity')
        
    df_acc = pd.DataFrame(df_acc)

    return  df_acc

def behaviorAcrossRuleInstances(subjNums, behavior='Accuracy',novelOnly=False):
    """
    Returns a dataframe with the behavioral performance of trials based on task similarity
    Conditions: Practiced tasks, tasks with 2-rules that are similar, tasks with 3-rules that are similar

    firstOnly : If True, only analyze the behavior of the first presentation of task rule sets
    """
    nov_acc = []
    df_acc = {}
    df_acc[behavior] = []
    df_acc['Condition'] = []
    df_acc['Subject'] = []
    df_acc['Rule'] = []
    df_acc['Block'] = []
    df_acc['RuleInstance'] = []
    for subj in subjNums:

        #### Load data and remove 0 RT scores
        df = pd.read_csv(datadir + subj + '_behavdata_reformatted.csv')
        zeroRT_ind = np.where(df.RT.values==0)[0]
        df.RT.values[zeroRT_ind] = np.nan

        # Exclude the pre-run practice trials
        df.TaskNum.values[:9] = np.nan
        
        log_rules = np.unique(df.Logic.values[9:]) # exclude the nans
        sen_rules = np.unique(df.Sensory.values[9:])
        mot_rules = np.unique(df.Motor.values[9:])
        df.Logic.values[:9] = np.nan
        df.Sensory.values[:9] = np.nan
        df.Motor.values[:9] = np.nan

        prac_ind = np.where(df.Novelty.values=='Prac')[0]
        prac_tasks = np.unique(df.TaskNum.values[prac_ind])

        firstblock_ind = []
        secondblock_ind = []
        for i in np.unique(df.TaskNum.values[9:]):
            ind = np.where(df.TaskNum.values==i)[0]
            if novelOnly:
                if i not in prac_tasks: # Ensure this is a novel task only
                    firstblock_ind.extend(ind[:3]) # novel tasks have 2 blocks 3 trials each
            else:
                firstblock_ind.extend(ind[:3]) # novel tasks have 2 blocks 3 trials each
            secondblock_ind.extend(ind[3:]) 


        for rule in log_rules:
            inds = np.where(df.Logic.values==rule)[0]
            # There are 6 trials of each rule (1st and 2nd blocks, 3 trials each) -- indicate whether it's the first block
            instance = 1
            i = 0
            while i < len(inds):
                ind = inds[np.arange(i,i+3)] # group trials associated with a miniblock together (3 trials/miniblock)
                if ind[0] in firstblock_ind:
                    if behavior=='Accuracy':
                        df_acc[behavior].append(np.mean(df[behavior].values[ind]=='Correct'))
                    if behavior=='RT':
                        df_acc[behavior].append(np.nanmean(df[behavior].values[ind]))
                    df_acc['Rule'].append(rule)
                    df_acc['Subject'].append(subj)
                    df_acc['Condition'].append(df.Novelty.values[ind[0]])
                    df_acc['Block'].append(1)
                    df_acc['RuleInstance'].append(instance)
                    instance+=1
                #if ind[0] in secondblock_ind:
                #    df_acc['Block'].append(2)
                #    df_acc['RuleInstance'].append(instance)
                i += 3

        for rule in sen_rules:
            inds = np.where(df.Sensory.values==rule)[0]
            # There are 6 trials of each rule (1st and 2nd blocks, 3 trials each) -- indicate whether it's the first block
            instance = 1
            i = 0
            while i < len(inds):
                ind = inds[np.arange(i,i+3)] # group trials associated with a miniblock together (3 trials/miniblock)
                if ind[0] in firstblock_ind:
                    if behavior=='Accuracy':
                        df_acc[behavior].append(np.mean(df[behavior].values[ind]=='Correct'))
                    if behavior=='RT':
                        df_acc[behavior].append(np.nanmean(df[behavior].values[ind]))
                    df_acc['Rule'].append(rule)
                    df_acc['Subject'].append(subj)
                    df_acc['Condition'].append(df.Novelty.values[ind[0]])
                    df_acc['Block'].append(1)
                    df_acc['RuleInstance'].append(instance)
                    instance+=1
                #if ind[0] in secondblock_ind:
                #    df_acc['Block'].append(2)
                #    df_acc['RuleInstance'].append(instance)
                i += 3

        for rule in mot_rules:
            inds = np.where(df.Motor.values==rule)[0]
            # There are 6 trials of each rule (1st and 2nd blocks, 3 trials each) -- indicate whether it's the first block
            instance = 1
            i = 0
            while i < len(inds):
                ind = inds[np.arange(i,i+3)] # group trials associated with a miniblock together (3 trials/miniblock)
                if ind[0] in firstblock_ind:
                    if behavior=='Accuracy':
                        df_acc[behavior].append(np.mean(df[behavior].values[ind]=='Correct'))
                    if behavior=='RT':
                        df_acc[behavior].append(np.nanmean(df[behavior].values[ind]))
                    df_acc['Rule'].append(rule)
                    df_acc['Subject'].append(subj)
                    df_acc['Condition'].append(df.Novelty.values[ind[0]])
                    df_acc['Block'].append(1)
                    df_acc['RuleInstance'].append(instance)
                    instance+=1
                #if ind[0] in secondblock_ind:
                #    df_acc['Block'].append(2)
                #    df_acc['RuleInstance'].append(instance)
                i += 3

    df_acc = pd.DataFrame(df_acc)

    return  df_acc
