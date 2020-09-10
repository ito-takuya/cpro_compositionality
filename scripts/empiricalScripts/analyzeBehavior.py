import numpy as np
import pandas as pd
import scipy.stats as stats

datadir = '../../../data/fMRI_BehavData/'

subjNums = ['013','014','016','017','018','021','023','024','025','026','027','028','030','031','032','033','034','035','037','038','039','040','041','042','043','045','046','047','048','049','050','053','055','056','057','058','062','063','064','066','067','068','069','070','072','074','075','076','077','081','082','085','086','087','088','090','092','093','094','095','097','098','099','101','102','103','104','105','106','108','109','110','111','112','114','115','117','118','119','120','121','122','123','124','125','126','127','128','129','130','131','132','134','135','136','137','138','139','140','141']

keyValues = {'Logic':'LogicCue[LogLevel5]', 'Sensory':'SemanticCue[LogLevel5]', 'Motor':'ResponseCue[LogLevel5]','Accuracy':'Feedback[LogLevel6]', 'Novelty':'TaskType_rec', 'TaskNum':'TaskName[LogLevel5]',
             'PracTask1':'PracTaskA','PracTask2':'PracTaskB','PracTask3':'PracTaskC', 'PracTask4':'PracTaskD','PracTaskIntro':'PracIntroExampleList',
             'LogicExample':'LogicCue[SubTrial]','SensoryExample':'SemanticCue[SubTrial]','MotorExample':'ResponseCue[SubTrial]',
             'PracIntroAccuracy':'Feedback[LogLevel5]'}

def behaviorOfNovelty(subjlist):
    prac_acc = []
    nov_acc = []
    df_acc = {}
    df_acc['Accuracy'] = []
    df_acc['Condition'] = []
    df_acc['Subject'] = []
    for subj in subjNums:

        df = pd.read_csv(datadir + subj + '_behavdata_reformatted.csv')


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
        acc = np.mean(df['Accuracy'].values[prac_ind]=='Correct')
        df_acc['Accuracy'].append(acc)
        df_acc['Subject'].append(subj)
        df_acc['Condition'].append('Practiced')
        # calculate accuracy for 1st set of practiced tasks
        acc = np.mean(df['Accuracy'].values[prac_ind1]=='Correct')
        df_acc['Accuracy'].append(acc)
        df_acc['Subject'].append(subj)
        df_acc['Condition'].append('1st Practiced')
        # calculate accuracy for 2nd set of practiced tasks
        acc = np.mean(df['Accuracy'].values[prac_ind2]=='Correct')
        df_acc['Accuracy'].append(acc)
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
        acc = np.mean(df['Accuracy'].values[nov_ind]=='Correct')
        df_acc['Accuracy'].append(acc)
        df_acc['Subject'].append(subj)
        df_acc['Condition'].append('Novel')
        # Calculate accuracy for 1st set of novel tasks
        acc = np.mean(df['Accuracy'].values[nov_ind1]=='Correct')
        df_acc['Accuracy'].append(acc)
        df_acc['Subject'].append(subj)
        df_acc['Condition'].append('1st Novel')
        # Calculate accuracy for 2nd set of novel tasks
        acc = np.mean(df['Accuracy'].values[nov_ind2]=='Correct')
        df_acc['Accuracy'].append(acc)
        df_acc['Subject'].append(subj)
        df_acc['Condition'].append('2nd Novel')

    #     practice_tasks = np.unique(df['TaskNum'].values[prac_ind])

    df_acc = pd.DataFrame(df_acc)
    return df_acc

def behaviorOfTaskSimilarity(subjlist, firstOnly=False):
    """
    Returns a dataframe with the behavioral performance of trials based on task similarity
    Conditions: Practiced tasks, tasks with 2-rules that are similar, tasks with 3-rules that are similar

    firstOnly : If True, only analyze the behavior of the first presentation of task rule sets
    """
    prac_acc = []
    nov_acc = []
    df_acc = {}
    df_acc['Accuracy'] = []
    df_acc['Condition'] = []
    df_acc['Subject'] = []
    for subj in subjNums:

        df = pd.read_csv(datadir + subj + '_behavdata_reformatted.csv')
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
        acc = np.mean(df['Accuracy'].values[prac_ind]=='Correct')
        df_acc['Accuracy'].append(acc)
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
            accs.extend(df.Accuracy.values[ind])
        accs = np.asarray(accs)
        acc = np.mean(accs=='Correct') 
        df_acc['Accuracy'].append(acc)
        df_acc['Subject'].append(subj)
        df_acc['Condition'].append('2-rule similarity')

        sim1_ind = np.where(task_similarity==1)[0]
        accs = []
        for task in sim1_ind:
            if firstOnly:
                ind = np.where(df.TaskNum.values==novel_tasks[task])[0][:3] # important bc similarity indices are obtained on novel tasks only
            else:
                ind = np.where(df.TaskNum.values==novel_tasks[task])[0][:] # use all trials
            accs.extend(df.Accuracy.values[ind])
        accs = np.asarray(accs)
        acc = np.mean(accs=='Correct') 
        df_acc['Accuracy'].append(acc)
        df_acc['Subject'].append(subj)
        df_acc['Condition'].append('1-rule similarity')
        
    df_acc = pd.DataFrame(df_acc)

    return  df_acc
