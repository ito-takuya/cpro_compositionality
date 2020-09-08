import numpy as np
import pandas as pd
import scipy.stats as stats

datadir = '../../../data/fMRI_BehavData/'

subjNums = ['013','014','016','017','018','021','023','024','025','026','027','028','030','031','032','033','034','035','037','038','039','040','041','042','043','045','046','047','048','049','050','053','055','056','057','058','062','063','064','066','067','068','069','070','072','074','075','076','077','081','082','085','086','087','088','090','092','093','094','095','097','098','099','101','102','103','104','105','106','108','109','110','111','112','114','115','117','118','119','120','121','122','123','124','125','126','127','128','129','130','131','132','134','135','136','137','138','139','140','141']

keyValues = {'Logic':'LogicCue[LogLevel5]', 'Sensory':'SemanticCue[LogLevel5]', 'Motor':'ResponseCue[LogLevel5]','Accuracy':'Feedback[LogLevel6]', 'Novelty':'TaskType_rec', 'TaskNum':'TaskName[LogLevel5]',
             'PracTask1':'PracTaskA','PracTask2':'PracTaskB','PracTask3':'PracTaskC', 'PracTask4':'PracTaskD','PracTaskIntro':'PracIntroExampleList',
             'LogicExample':'LogicCue[SubTrial]','SensoryExample':'SemanticCue[SubTrial]','MotorExample':'ResponseCue[SubTrial]',
             'PracIntroAccuracy':'Feedback[LogLevel5]'}

intro_acc = []
prac_acc = []
nov_acc = []
nov_acc1 = []
nov_acc2 = []
prac_acc1 = []
prac_acc2 = []
for subj in subjNums:
    df = pd.read_csv(datadir + subj + '_behavdata.csv') 

    intro_tasks = np.unique(df['TaskName[LogLevel5]'].values[1:9])
    print('Subject', subj, intro_tasks)
#    print('\t', df['PracTaskA'].values[5],df['PracTaskB'].values[5],df['PracTaskC'].values[5],df['PracTaskD'].values[5])

    prac_ind = np.where(df['TaskType_rec'].values=='Prac')[0]
    prac_tasks = np.unique(df['TaskName[LogLevel5]'].values[prac_ind])
    prac_ind1 = []
    prac_ind2 = []
    for i in prac_tasks:
        ind1 = np.where(df['TaskName[LogLevel5]'].values==i)[0][2:5] # first miniblock
        ind2 = np.where(df['TaskName[LogLevel5]'].values==i)[0][5:] # second miniblock
        print(np.where(df['TaskName[LogLevel5]'].values==i)[0].shape) # second miniblock
        prac_ind1.extend(ind1)
        prac_ind2.extend(ind2)
    prac_ind1 = np.asarray(prac_ind1)
    prac_ind2 = np.asarray(prac_ind2)

    # all practiced tasks
    acc = np.mean(df['Feedback[LogLevel6]'].values[prac_ind]=='Correct')
    prac_acc.append(acc)
    # nov acc 1
    acc = np.mean(df['Feedback[LogLevel6]'].values[prac_ind1]=='Correct')
    prac_acc1.append(acc)
    # nov acc 2
    acc = np.mean(df['Feedback[LogLevel6]'].values[prac_ind2]=='Correct')
    prac_acc2.append(acc)

    nov_ind = np.where(df['TaskType_rec'].values=='Novel')[0]
    nov_tasks = np.unique(df['TaskName[LogLevel5]'].values[nov_ind])
    nov_ind1 = []
    nov_ind2 = []
    for i in nov_tasks:
        ind1 = np.where(df['TaskName[LogLevel5]'].values==i)[0][:3] # first miniblock
        ind2 = np.where(df['TaskName[LogLevel5]'].values==i)[0][3:] # second miniblock
        nov_ind1.extend(ind1)
        nov_ind2.extend(ind2)
    nov_ind1 = np.asarray(nov_ind1)
    nov_ind2 = np.asarray(nov_ind2)

    # all novel tasks
    acc = np.mean(df['Feedback[LogLevel6]'].values[nov_ind]=='Correct')
    nov_acc.append(acc)
    # nov acc 1
    acc = np.mean(df['Feedback[LogLevel6]'].values[nov_ind1]=='Correct')
    nov_acc1.append(acc)
    # nov acc 2
    acc = np.mean(df['Feedback[LogLevel6]'].values[nov_ind2]=='Correct')
    nov_acc2.append(acc)

    practice_tasks = np.unique(df['TaskName[LogLevel5]'].values[prac_ind])
    #print('\t', practice_tasks)

print('Practice 1 accuracy:', np.mean(prac_acc1))
print('Practice 2 accuracy:', np.mean(prac_acc2))
print('Practice accuracy:', np.mean(prac_acc))
print('Novel 1 accuracy:', np.mean(nov_acc1))
print('Novel 2 accuracy:', np.mean(nov_acc2))
print('Novel accuracy:', np.mean(nov_acc))
print('Practiced versus Novel:', stats.ttest_rel(prac_acc,nov_acc))

