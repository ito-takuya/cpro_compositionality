import numpy as np
import itertools

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

    if shuffle:
        indices = np.arange(len(labels))
        np.random.shuffle(indices)
        labels = labels[indices]
        labels2 = labels2[indices]
        labels3 = labels3[indices]

    classes = np.unique(labels) # return the unique class labels
    ps_score = np.zeros((len(classes),len(classes))) # create matrix to indicate the ps scores for each pair of conditions
    i = 0
    for cond1 in classes: # first condition
        ind_cond1 = np.where(labels==cond1)[0] # return the indices for the first class
        
        j = 0
        for cond2 in classes: # second condition
            if i == j: 
                j+=1
                # skip if same conditions or if doing a symmetrical calculation - this is not informative (dot product is symmetrical)
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

#    ps_score = ps_score + ps_score.T # make the matrix symmetric and fill out other half of the matrix

    return ps_score, classes

def parallelismScoreRandomPartitions(data,labels,labels2,labels3,shuffle=False):
    """
    Computes parallelism score (PS) on either binary or multi-class problems 
    Parallelism on multi-class problems is computed as the average cosine(angle) of all pairs of activation vectors
    Partitions each class into two equal partitions, and then computes the cosine between all pairs of vectors 
    Partitions are randomly sampled 1000 times (since the full permutation of combination pairs would be far too expensive

    data - observations X features 2d matrix (features correspond to unit activations)
    labels - 1d array/list of labels from which to build decoders (can be binary or multi-class)
    labels2 - 1d array/list of secondary labels from which to maximize similarity/matches to build cosine similarities
    labels3 - 1d array/list of tertiary labels from which to maximize similarity/matches to build cosine similarities
    """

    if shuffle:
        indices = np.arange(len(labels))
        np.random.shuffle(indices)
        labels = labels[indices]
        labels2 = labels2[indices]
        labels3 = labels3[indices]

    classes = np.unique(labels) # return the unique class labels
    ps_score = np.zeros((len(classes),len(classes))) # create matrix to indicate the ps scores for each pair of conditions
    i = 0
    for cond1 in classes: # first condition
        ind_cond1 = np.where(labels==cond1)[0] # return the indices for the first class
        
        j = 0
        for cond2 in classes: # second condition
            if i == j: 
                j+=1
                # skip if same conditions or if doing a symmetrical calculation - this is not informative (dot product is symmetrical)
                continue 
            ind_cond2 = np.where(labels==cond2)[0] # return the indices for the second class

            #### Now for each condition, iterate through random possible combinations
            cosine_sim = []
            for n in range(1000): # for each index in condition 1
                np.random.shuffle(ind_cond1)
                np.random.shuffle(ind_cond2)

                vec1 = data[ind_cond1,:]
                vec2 = data[ind_cond2,:]

                diff_vec = vec1 - vec2
                diff_vec = diff_vec.T / np.linalg.norm(diff_vec,axis=1)
                ps = np.matmul(diff_vec.T,diff_vec)
                np.fill_diagonal(ps,np.nan) # dot product with self will be 0
                cosine_sim.append(np.nanmean(ps))


            ps_score[i,j] = np.max(cosine_sim) # compute average ps
            j += 1

        i += 1

    ps_score = ps_score + ps_score.T # make the matrix symmetric and fill out other half of the matrix

    return ps_score, classes




