import numpy as np
import itertools
import scipy.stats as stats
import multiprocessing as mp
import statsmodels.api as sm
import sklearn
import sklearn.svm as svm
import loadTaskBehavioralData as task
import pandas as pd
import nibabel as nib
import os

#glasserfile = '/projects3/CPROCompositionality/data/Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_RL.dlabel.nii'
glasserfile = '/projectsn/f_mc1689_1/CPROCompositionality/data/Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_RL.dlabel.nii'
glasser = nib.load(glasserfile).get_data()
glasser = np.squeeze(glasser)


def loadGroupBehavioralData(subjNums):
    """
    Load and concatenate all subjects' task labels
    """
    df_all = pd.DataFrame()
    for subj in subjNums:
        tmpdf = task.loadExperimentalData(subj)
        df_all = df_all.append(tmpdf)

    return df_all

def decodeGroup(data, subj_labels, task_labels,
                kfold=10, normalize=True, classifier='distance',
                confusion=False, permutation=False, roi=None):
    """
    Group decoding for a given set of task_labels
    Ensures labels are separated by subject
    data: sample x feature 2d matrix
    subj_labels: subject labels
    task_labels: task labels to be decoded
    kfold: number of cross-validation folds -- default=10
    normalize: Perform feature-wise normalization within each CV fold -- default=True
    classifier: classification method -- default='distance' (correlation-based) alternatives: ['svm','logistic']
    permutation: Randomly permute task_labels (default=False)
    roi: Print out the ROI being decoded (default=None, which doesn't print out anything)
    """
   
    # Create accuracies array to remember exactly the predictions for each sample
    accuracies = np.zeros((len(task_labels),))
    confusion_mats = [] 
    # Begin CV
    groupkfold = sklearn.model_selection.GroupKFold(n_splits=kfold)
    for train_index, test_index in groupkfold.split(data,y=task_labels,groups=subj_labels):
        X_train, X_test = data[train_index,:], data[test_index,:]
        y_train, y_test = task_labels[train_index], task_labels[test_index]

        if permutation:
            np.random.seed(roi)
            np.random.shuffle(y_train)

        if normalize:
            mean = np.mean(X_train,axis=0)
            mean.shape = (1,len(mean))
            std = np.std(X_train,axis=0)
            std.shape = (1,len(std))

            X_train = np.divide((X_train - mean),std)
            X_test = np.divide((X_test - mean),std)

        if confusion:
            acc, confusion_mat = _decoding(X_train,X_test,y_train,y_test,classifier=classifier,confusion=confusion)
            confusion_mats.append(confusion_mat)
        else:
            acc = _decoding(X_train,X_test,y_train,y_test,classifier=classifier,confusion=confusion)

        accuracies[test_index] = acc

    if roi is not None and not permutation:
        print('Decoding ROI', roi, '| Average accuracy:', np.mean(accuracies))

    if confusion:
        confusion_mat = np.sum(np.asarray(confusion_mats),axis=0) # sum across all confusion matrices (for each CV fold)
        return accuracies, confusion_mat
    else:
        return accuracies

def decodeGroup2(data, subj_labels, task_labels, secondary_labels, tertiary_labels, taskid_labels,
                normalize=True, classifier='logistic',
                confusion=False, permutation=False, roi=None):
    """
    Group decoding for a given set of task_labels
    Ensures labels are separated by subject
    data: sample x feature 2d matrix
    subj_labels: subject labels
    task_labels: task labels to be decoded
    normalize: Perform feature-wise normalization within each CV fold -- default=True
    classifier: classification method -- default='logistic' (linear-based) alternatives: ['svm','logistic','distance']
    permutation: Randomly permute task_labels (default=False)
    roi: Print out the ROI being decoded (default=None, which doesn't print out anything)


    To maintain consistency with the CCGP decoding analysis, this will be a leave 16-out cross validation (but where contexts are not held-out for the purpose of testing generalization
    """
    kfold = 16
   
    # Create accuracies array to remember exactly the predictions for each sample
    accuracies = np.zeros((len(task_labels),))
    confusion_mats = [] 
    # Begin CV
    groupkfold = sklearn.model_selection.GroupKFold(n_splits=kfold)
    for train_index, test_index in groupkfold.split(data,y=task_labels,groups=subj_labels):
        X_train, X_test = data[train_index,:], data[test_index,:]
        y_train, y_test = task_labels[train_index], task_labels[test_index]

        if permutation:
            np.random.seed(roi)
            np.random.shuffle(y_train)

        if normalize:
            mean = np.mean(X_train,axis=0)
            mean.shape = (1,len(mean))
            std = np.std(X_train,axis=0)
            std.shape = (1,len(std))

            X_train = np.divide((X_train - mean),std)
            X_test = np.divide((X_test - mean),std)

        if confusion:
            acc, confusion_mat = _decoding(X_train,X_test,y_train,y_test,classifier=classifier,confusion=confusion)
            confusion_mats.append(confusion_mat)
        else:
            acc = _decoding(X_train,X_test,y_train,y_test,classifier=classifier,confusion=confusion)

        accuracies[test_index] = acc

    if roi is not None and not permutation:
        print('Decoding ROI', roi, '| Average accuracy:', np.mean(accuracies))

    if confusion:
        confusion_mat = np.sum(np.asarray(confusion_mats),axis=0) # sum across all confusion matrices (for each CV fold)
        return accuracies, confusion_mat
    else:
        return accuracies

def ccgpGroup(data, subj_labels, task_labels, secondary_labels, tertiary_labels, taskid_labels,
                normalize=True, classifier='logistic',
                confusion=False, permutation=False, roi=None):
    """
    Group decoding for a given set of task_labels
    Ensures labels are separated by subject
    data: sample x feature 2d matrix
    subj_labels: subject labels
    task_labels: task labels to be decoded
    normalize: Perform feature-wise normalization within each CV fold -- default=True
    classifier: classification method -- default='logistic' (linear-based) alternatives: ['svm','logistic','distance']
    permutation: Randomly permute task_labels (default=False)
    roi: Print out the ROI being decoded (default=None, which doesn't print out anything)
    """
   
    # Create accuracies array to remember exactly the predictions for each sample
    accuracies = np.zeros((len(task_labels),))
    confusion_mats = [] 

    # Create cross-validation folds (specifically, identify the to-be-predicted incdices for each fold)
    train_folds = []
    test_folds = []
    for label3 in np.unique(tertiary_labels):
        label3_ind = np.where(tertiary_labels==label3)[0]
        for label2 in np.unique(seconday_labels):
            label2_ind = np.where(secondary_labels==label2)[0]
            # matching context samples
            context_matching_ind = []
            for label in np.unique(task_labels):
                label1_ind = np.where(task_labels==label)[0]
                # find the intersection (i.e., unique task id)
                task_ind = np.intersect1d(label3_ind,label2_ind)
                task_ind = np.intersect1d(task_ind, label1_ind)
                context_matching_ind.append(task_ind)
            # Now add to the test_folds
            test_folds.append(np.asarray(context_matching_ind))
    
    # Run cross-validation
    all_indices = np.arange(len(task_labels))
    for test_fold in test_folds:
        # Define the training set as all samples not in the test set
        train_fold = np.where(all_indices!=test_fold)[0]

        X_train, X_test = data[train_fold,:], data[test_fold,:]
        y_train, y_test = task_labels[train_fold], task_labels[test_index]

        if permutation:
            np.random.seed(roi)
            np.random.shuffle(y_train)

        if normalize:
            mean = np.mean(X_train,axis=0)
            mean.shape = (1,len(mean))
            std = np.std(X_train,axis=0)
            std.shape = (1,len(std))

            X_train = np.divide((X_train - mean),std)
            X_test = np.divide((X_test - mean),std)

        if confusion:
            acc, confusion_mat = _decoding(X_train,X_test,y_train,y_test,classifier=classifier,confusion=confusion)
            confusion_mats.append(confusion_mat)
        else:
            acc = _decoding(X_train,X_test,y_train,y_test,classifier=classifier,confusion=confusion)

        accuracies[test_index] = acc

    if roi is not None and not permutation:
        print('Decoding ROI', roi, '| Average accuracy:', np.mean(accuracies))

    if confusion:
        confusion_mat = np.sum(np.asarray(confusion_mats),axis=0) # sum across all confusion matrices (for each CV fold)
        return accuracies, confusion_mat
    else:
        return accuracies
      


    
def _decoding(trainset,testset,trainlabels,testlabels,classifier='distance',confusion=False):
    unique_labels = np.unique(trainlabels)
    
    if classifier == 'distance':
        #### Create prototypes from trainset
        prototypes = {}
        for label in unique_labels:
            ind = np.where(trainlabels==label)[0]
            prototypes[label] = np.mean(trainset[ind,:],axis=0)

        #### Now classifiy each sample n the testset
        predictions = []
        for i in range(testset.shape[0]):
            # Correlate sampple with each prototype
            rs = []
            for label in prototypes:
                rs.append(stats.pearsonr(prototypes[label],testset[i,:])[0])
            
            # Find the closest prototype for sample
            max_ind = np.argmax(np.asarray(rs))
            predictions.append(unique_labels[max_ind])

        predictions = np.asarray(predictions)

    if classifier == 'logistic':

        clf = sklearn.linear_model.LogisticRegression()
        clf.fit(trainset,trainlabels)
        predictions = clf.predict(testset)

    if classifier == 'svm':
        clf = svm.SVC(kernel='linear')
        clf.fit(trainset,trainlabels)
        predictions = clf.predict(testset)

    accuracy = predictions == np.asarray(testlabels)
    confusion_mat = sklearn.metrics.confusion_matrix(testlabels, predictions, labels=unique_labels)

    if confusion:
        return accuracy, confusion_mat
    else:
        return accuracy

def dimensionality(data):
    """
    data needs to be a square, symmetric matrix
    """
    corrmat = data
    eigenvalues, eigenvectors = np.linalg.eig(corrmat)
    dimensionality_nom = 0
    dimensionality_denom = 0
    for eig in eigenvalues:
        dimensionality_nom += np.real(eig)
        dimensionality_denom += np.real(eig)**2
    
    dimensionality = dimensionality_nom**2/dimensionality_denom

    return dimensionality



def parallelismScore(data,labels,labels2,labels3,shuffle=False):
    """
    Computes parallelism score (PS) on either binary or multi-class problems 
    Parallelism on multi-class problems is computed as the average cosine(angle) of all pairs of activation vectors
    Partitions each class into two equal partitions, and then computes the cosine between all pairs of vectors 
    Partitions are determined by maximizing the task rule conditions. 
    Ex: If BOTH and NEITHER are the rules of interest, then we would pair BOTH - X [RED] - Y [LMID] with EITHER - X [RED] - Y [LMID], where X and Y are the same across pairs

    data - observations X features 2d matrix (features correspond to unit activations)
    labels - 1d array/list of labels from which to build decoders (can be binary or multi-class)
    labels2 - 1d array/list of secondary labels from which to maximize similarity/matches to build cosine similarities
    labels3 - 1d array/list of tertiary labels from which to maximize similarity/matches to build cosine similarities
    """
    labels = np.asarray(labels)
    labels2 = np.asarray(labels2)
    labels3 = np.asarray(labels3)
    classes = np.unique(labels) # return the unique class labels

    if shuffle!=False:
        np.random.seed(shuffle)
        indices = np.arange(len(labels))
        np.random.shuffle(indices)
        labels = labels[indices]
        labels2 = labels2[indices]
        labels3 = labels3[indices]

    ps_score = np.zeros((len(classes),len(classes))) # create matrix to indicate the ps scores for each pair of conditions
    i = 0
    for cond1 in classes: # first condition
        ind_cond1 = np.where(labels==cond1)[0] # return the indices for the first class
        
        j = 0
        for cond2 in classes: # second condition
            if i == j: 
                # skip if same conditions or if doing a symmetrical calculation - this is not informative (dot product is symmetrical)
                j += 1
                continue 
            ind_cond2 = np.where(labels==cond2)[0] # return the indices for the second class

            #### Now for each condition, find a pair that maximizes rule similarity
            diff_vec = []
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
                vec1 = data[ind1,:]
                vec2 = data[ind2,:]
                diff_vec.append(vec1-vec2)
                
#                print("Matching: (", labels[ind1], labels2[ind1], labels3[ind1], ') \tX\t (', labels[ind2], labels2[ind2], labels3[ind2],')')


            diff_vec = np.asarray(diff_vec) 
            diff_vec = diff_vec.T / np.linalg.norm(diff_vec,axis=1)
            ps = np.matmul(diff_vec.T,diff_vec)
            np.fill_diagonal(ps,np.nan) # dot product with self will be 0

            ps_score[i,j] = np.nanmean(ps) # compute average ps
            j += 1

        i += 1

    ps_score = ps_score + ps_score.T # make the matrix symmetric and fill out other half of the matrix

    return ps_score, classes

def parallelismScoreTrial(data,labels,labels2,labels3,shuffle=False):
    """
    Computes parallelism score (PS) on either binary or multi-class problems 
    Parallelism on multi-class problems is computed as the average cosine(angle) of all pairs of activation vectors
    Partitions each class into two equal partitions, and then computes the cosine between all pairs of vectors 
    Partitions are determined by maximizing the task rule conditions. 
    Ex: If BOTH and NEITHER are the rules of interest, then we would pair BOTH - X [RED] - Y [LMID] with EITHER - X [RED] - Y [LMID], where X and Y are the same across pairs

    data - observations X features 2d matrix (features correspond to unit activations)
    labels - 1d array/list of labels from which to build decoders (can be binary or multi-class)
    labels2 - 1d array/list of secondary labels from which to maximize similarity/matches to build cosine similarities
    labels3 - 1d array/list of tertiary labels from which to maximize similarity/matches to build cosine similarities
    """
    labels = np.asarray(labels)
    labels2 = np.asarray(labels2)
    labels3 = np.asarray(labels3)
    classes = np.unique(labels) # return the unique class labels

    if shuffle!=False:
        np.random.seed(shuffle)
        indices = np.arange(len(labels))
        np.random.shuffle(indices)
        labels = labels[indices]
        labels2 = labels2[indices]
        labels3 = labels3[indices]

    ps_score = np.zeros((len(labels),len(classes))) # create matrix to indicate the ps scores for each pair of conditions
    i = 0
    for cond1 in classes: # first condition
        ind_cond1 = np.where(labels==cond1)[0] # return the indices for the first class
        
        j = 0
        for cond2 in classes: # second condition
            if i == j: 
                # skip if same conditions or if doing a symmetrical calculation - this is not informative (dot product is symmetrical)
                j += 1
                continue 
            ind_cond2 = np.where(labels==cond2)[0] # return the indices for the second class

            #### Now for each condition, find a pair that maximizes rule similarity
            diff_vec = []
            index_labels  = [] # keep track of which tasks/trials these belong to
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
                vec1 = data[ind1,:]
                vec2 = data[ind2,:]
                diff_vec.append(vec1-vec2)
                index_labels.append(ind1)
                
#                print("Matching: (", labels[ind1], labels2[ind1], labels3[ind1], ') \tX\t (', labels[ind2], labels2[ind2], labels3[ind2],')')


            diff_vec = np.asarray(diff_vec) 
            diff_vec = diff_vec.T / np.linalg.norm(diff_vec,axis=1)
            ps = np.matmul(diff_vec.T,diff_vec)
            np.fill_diagonal(ps,np.nan) # dot product with self will be 0

            index_labels = np.asarray(index_labels)
            #test - make sure don't overwrite a previous obtained score
            if np.sum(ps_score[index_labels,j])>0: raise Exception('error')
            ps_score[index_labels,j] = np.nanmean(ps,axis=1) # compute average ps for each trial
            j += 1

        i += 1

    ps_score = np.mean(ps_score,axis=1) # get average ps score per trial/label

    return ps_score




def mapBackToSurface(array,filename):
    """
    array can either be 360 array or ~59k array. If 360, will automatically map back to ~59k
    """
    #### Map back to surface
    if array.shape[0]==360:
        out_array = np.zeros((glasser.shape[0],3))

        roicount = 0
        for roi in range(360):
            for col in range(array.shape[1]):
                vertex_ind = np.where(glasser==roi+1)[0]
                out_array[vertex_ind,col] = array[roicount,col]

            roicount += 1

    else:
        out_array = array

    #### 
    # Write file to csv and run wb_command
    np.savetxt(filename + '.csv', out_array,fmt='%s')
    wb_file = filename + '.dscalar.nii'
    wb_command = 'wb_command -cifti-convert -from-text ' + filename + '.csv ' + glasserfile + ' ' + wb_file + ' -reset-scalars'
    os.system(wb_command)
    os.remove(filename + '.csv')
