import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from time import time
from tqdm import tqdm 

######## for styling ########
from matplotlib import font_manager, rcParams
font_manager.findSystemFonts(fontpaths="/Users/charlie/Library/Fonts/", fontext="ttf")
myfont = {'fontname':'Iosevka', 'fontsize':'15', 'fontweight': 'bold'}
#############################

########### Setup ###########
dataset = scipy.io.loadmat('face.mat')
data    = np.array(dataset['X'])     # (2576, 520) 
labels  = np.array(dataset['l'][0])  # (1, 520)


test_indices  = np.concatenate((np.arange(data.shape[1]//10)*10+7, 
                               (np.arange(data.shape[1]//10)*10+8)))
train_indices = [i for i in range(data.shape[1]) if i not in test_indices]

train_data  = data[:, train_indices]
train_label = labels[train_indices]

test_data  = data[:, test_indices]
test_label = labels[test_indices]

def show_face(x, title=None):
    '''
    Input: 
        x: (2576,) vector representing face
        title (Optional): the title of the plot
    Output:
        None. Plot the given face vector
    '''
    plt.title(title, **myfont, pad=13)
    face = x.reshape((46, 56)).T
    plt.imshow (face,cmap='gray')

    filename = "../figures/result.png"
    if title:
        filename = f"../figures/{title}.png"
    plt.savefig(filename, bbox_inches='tight')

    plt.show ()

def plot_acc(acc, title=None):
    '''
    Input: 
        x: (2576,) vector representing face
        title (Optional): the title of the plot
    Output:
        None. Plot the given face vector
    '''
    plt.title(title, **myfont, pad=13)
    x = range(len(acc))
    plt.plot(x, acc)

    filename = "../figures/acc_result.png"
    if title:
        filename = f"../figures/{title}.png"
    plt.savefig(filename, bbox_inches='tight')

    plt.show ()
#############################

############ Q1 #############
def time_func(name, func, arg):
    start = time()
    result = func(arg)
    print(f"{name} took {time()-start}")
    return result

def face_recon(i, eival, eivec):
    '''
    Input: 
        X: (2576, N) matrix with face vector along 
        title (Optional): the title of the plot
    Output:
        None. Plot the given face vector
    '''
    M = [10, 50, 100, 200, 300, 400]
    pass

def eig(M):
    D, V = np.linalg.eigh(M)
    '''
    ascen_idx = D.argsort()
    V = np.real_if_close(V[:, ascen_idx], tol=1)
    D = D[ascen_idx]
    '''
    return D, V

def train():
    mean = train_data.mean(axis=1)
    A = (train_data.T - mean.T).T
    N = A.shape[1]

    eival_slow, eivec_slow = time_func("slow method", eig, (A @ A.T) / N)
    eival_fast, eivec_fast = time_func("fast method", eig, (A.T @ A) / N)
    eivec_fast_ = A @ eivec_fast
    eivec_fast_ = eivec_fast_ / np.expand_dims(np.linalg.norm(eivec_fast_,axis=0), axis=0)

    # fix the signs
    eivec_slow  = eivec_slow *  (eivec_slow[0,:] / np.absolute(eivec_slow[0,:]))
    eivec_fast_ = eivec_fast_ * (eivec_fast_[0,:] / np.absolute(eivec_fast_[0,:]))

    print("number of non-zero eigen values slow: ", np.sum(eival_slow > 1e-8))
    print("number of non-zero eigen values fast: ", np.sum(eival_fast > 1e-8))
    print(f"eigen value are same? {(eival_slow[-416:]-eival_fast < 1e-8).all()}")

    return mean, A, eival_slow, eivec_slow, eival_fast, eivec_fast_

mean, A, eival_slow, eivec_slow, eival_fast, eivec_fast = train()


show_face(mean, "The Mean Face")
for i in range(1, 7):
    print(eivec_slow[:, -i])
    print(eivec_fast[:, -i])
    show_face(eivec_slow[:, -i], f"{i} Eigenface (slow)")
    show_face(eivec_fast[:, -i], f"{i} Eigenface (fast)")


def reconstruct_face (mean, phi_w, V):
    face = mean.copy()
    for i, eig in enumerate(V.T):
        face += eig * phi_w[i]
    return face

# N_nonzero = np.sum(eival_slow > 1e-8)
def test():
    acc_list = []
    sample = 42
    for M in tqdm(range(1, N_nonzero+1), total=N_nonzero):
        V = eivec_slow[:, -M:] # eigen space consist of top M eigen vectors
        W = V.T @ A            # column vectors are w_n 

        # testing 
        test_data  = data[:, test_indices]
        test_label = labels[test_indices]

        acc = 0
        test_N = test_data.shape[1]
        for i, test_face in enumerate(test_data.T):
            phi   = test_face - mean 
            phi_w = (V.T @ phi) # project into eigenspace
            nn    = np.argmin([np.linalg.norm(p - phi_w) for p in W.T]) # index of nearest neighbor

            pred_label = train_label[nn]
            recon_face = reconstruct_face(mean, phi_w, V)
            if (i == sample and M % 50 == 0):
                show_face(test_face, f"original face, M={M}")
                show_face(phi, f"normalized face, M={M}")
                show_face(recon_face, f"reconstucted face, M={M}")
                show_face(train_data[:, nn], f"closest face in eigenspace, M={M}")
            acc += (pred_label == test_label[i])

        acc_list.append(acc/test_N)

    plot_acc(acc_list, "Accuracy vs. M")

# test()
#############################

############ Q2 #############

# split training subsets
subset1_indices = np.concatenate((np.arange(train_data.shape[1]//8)*8+0, (np.arange(train_data.shape[1]//8)*8+1)))
train_subdata_1  = train_data[:, subset1_indices]
train_sublabel_1 = train_label[subset1_indices]

subset2_indices = np.concatenate((np.arange(train_data.shape[1]//8)*8+2, (np.arange(train_data.shape[1]//8)*8+3)))
train_subdata_2  = train_data[:, subset2_indices]
train_sublabel_2 = train_label[subset2_indices]

subset3_indices = np.concatenate((np.arange(train_data.shape[1]//8)*8+4, (np.arange(train_data.shape[1]//8)*8+5)))
train_subdata_3  = train_data[:, subset3_indices]
train_sublabel_3 = train_label[subset3_indices]

subset4_indices = np.concatenate((np.arange(train_data.shape[1]//8)*8+6, (np.arange(train_data.shape[1]//8)*8+7)))
train_subdata_4  = train_data[:, subset4_indices]
train_sublabel_4 = train_label[subset4_indices]

def recon_error (mean, V, X):
    N   = X.shape[1] # x_n is column of X
    err = zeros_like(mean)
    for i, face in enumerate(X.T):
        phi_w = V.T @ (face - mean) 
        recon_face = reconstruct_face(mean, phi_w, V)
        err += np.linalg.norm(face - recon_face)
    return err / N

def face_reg_acc (A, V, test_data, test_label, title=None):
    W = V.T @ A            # column vectors are w_n 

    acc = 0
    test_N = test_data.shape[1]
    for i, test_face in enumerate(test_data.T):
        phi   = test_face - mean 
        phi_w = (V.T @ phi) # project into eigenspace
        nn    = np.argmin([np.linalg.norm(p - phi_w) for p in W.T]) # index of nearest neighbor

        pred_label = train_label[nn]
        recon_face = reconstruct_face(mean, phi_w, V)
        acc += (pred_label == test_label[i])
    acc /= tet_N
    print(f"{title} face recognition accuracy: {acc}")
    return acc

def minimal_pca(train_data):
    mean = train_data.mean(axis=1)
    A = (train_data.T - mean.T).T
    N = A.shape[1]
    S = A @ A.T / N

    D, V = np.linalg.eigh(S)
    return mean, N, V, D, S

def batch_pca():
    mean, N, V, D, S = minimal_pca (train_data)

def first_set_pca():
    mean, N, V, D, S = minimal_pca (train_subdata_1)

def increment_pca(steps=3):
    assert(step >= 2)
    # in order of mean, N, eigenvectors, eigenvalues
    def combine_pca(mean1, N1, V1, D1, S1, mean2, N2, V2, D2, S2):
        N3 = N1 + N2
        mean3 = (N1*mean1 + N2*mean2) / N3
        mean_diff = np.expand_dims(mean1 - mean2, axis=1) # mean difference
        S3 = (N1/N3)*S1 + (N2/N3)*S2 + (N1*N2/N3)*(mean_diff @ mean_diff.T)
        combined_eig = np.hstack((V1, V2, mean_diff))
        Phi, _ = np.linalg.qr(combined_eig)
        D3, R = np.linalg.eigh(Phi @ S3 @ Phi)
        return mean3, N3, Phi@R, D3, S3
    
    train_subdata = [train_subdata_1, train_subdata_2, 
                     train_subdata_3, train_subdata_4]

    mean0, N0, V0, D0, S0 = minimal_pca(train_subdata[0])
    mean1, N1, V1, D1, S1 = minimal_pca(train_subdata[1])
    mean, N, V, D, S      = combine_pca(mean0, N0, V0, D0, S0, mean1, N1, V1, D1, S1)

    for step in range(2, steps):
        mean_, N_, V_, D_, S_ = minimal_pca(train_subdata[i])
        mean, N, V, D, S      = combine_pca(mean, N, V, D, S, mean_, N_, V_, D_, S_)

    return mean, N, V, D, S

# increment_pca()

#############################

############ Q3 #############
# TODO
#############################

############ Q4 #############
# TODO
#############################
