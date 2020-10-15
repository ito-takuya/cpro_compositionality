import numpy as np

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


def parallelismScore(data,labels):
    """
    Computes parallelism score (PS) on either binary or multi-class problems 
    Parallelism on multi-class problems is computed as the average cosine(angle) of all pairs of activation vectors
    Partitions each class into two equal partitions, and then computes the cosine between all pairs of vectors 

    data - observations X features 2d matrix (features correspond to unit activations)
    labels - 1d array/list of labels from which to build decoders (can be binary or multi-class)
    """

    classes = np.unique(labels) # return the unique class labels
    ps_score = np.zeros((len(classes),len(classes))) # create matrix to indicate the ps scores for each pair of conditions
    i = 0
    for cond1 in classes: # first condition
        ind_cond1 = np.where(labels==cond1)[0] # return the indices for the first class
        np.random.shuffle(ind_cond1) # randomize order
        ind_split1_cond1, ind_split2_cond1 = ind_cond1[:int(len(ind_cond1)/2)], ind_cond1[int(len(ind_cond1)/2):] # randomly split into pairs
        split1_cond1, split2_cond1 = data[ind_split1_cond1,:], data[ind_split2_cond1,:]
        j = 0
        for cond2 in classes: # second condition
            if i == j or i<j: 
                # skip if same conditions or if doing a symmetrical calculation - this is not informative (dot product is symmetrical)
                continue 
            ind_cond2 = np.where(labels==cond2)[0] # return the indices for the second class
            np.random.shuffle(ind_cond2) # randomize order
            ind_split1_cond2, ind_split2_cond2 = ind_cond2[:int(len(ind_cond2)/2)], ind_cond2[int(len(ind_cond2)/2):] # randomly split into pairs
            split1_cond2, split2_cond2 = data[ind_split1_cond2,:], data[ind_split2_cond2,:]

            #### Calculate the vectors
            # To calculate the vector between cond1 and cond2, need to subtract them
            split1_vec = split1_cond1 - split1_cond2 # first split
            split2_vec = split2_cond1 - split2_cond2 # second split

            # Normalize vectors
            split1_norm = np.linalg.norm(split1_vec,axis=1)
            split1_norm.shape = (len(split1_norm),1)
            split2_norm = np.linalg.norm(split2_vec,axis=1)
            split2_norm.shape = (len(split2_norm),1)
            split1_vec = np.divide(split1_vec,split1_norm)
            split2_vec = np.divide(split2_vec,split2_norm)

            # If parallel, the dot product of the difference vectors should be close to 1 for all random pairs
            ps = np.dot(split1_vec,split2_vec.T) # each mat is samples x features
            ps_score[i,j] = np.mean(ps) # compute average ps
            j += 1

        i += 1

    ps_score = ps_score + ps_score.T # make the matrix symmetric and fill out other half of the matrix

    return ps_score, classes




