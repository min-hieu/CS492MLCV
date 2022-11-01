import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from time import time
from tqdm import tqdm, trange
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from multiprocessing import Pool

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

classes = np.unique(labels)

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
def time_func(func, arg, name=None):
    start = time()
    if arg is None or arg is (None):
        result = func()
    else:
        result = func(arg)
    elapsed_time = time() - start
    if name is not None:
        print(f"{name} took {elapsed_time}")

    return result, elapsed_time

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
    print(f"total number of training faces: {N}")

    (eival_slow, eivec_slow),_ = time_func(eig, (A @ A.T) / N, "slow method")
    (eival_fast, eivec_fast),_ = time_func(eig, (A.T @ A) / N, "fast method")
    eivec_fast_ = A @ eivec_fast
    eivec_fast_ = eivec_fast_ / np.expand_dims(np.linalg.norm(eivec_fast_,axis=0), axis=0)

    # fix the signs
    eivec_slow  = eivec_slow *  (eivec_slow[0,:] / np.absolute(eivec_slow[0,:]))
    eivec_fast_ = eivec_fast_ * (eivec_fast_[0,:] / np.absolute(eivec_fast_[0,:]))

    print("total number of eigen value (slow): ", len(eival_slow))
    print("total number of eigen value (fast): ", len(eival_fast))
    print("number of non-zero eigen values slow: ", np.sum(eival_slow > 1e-8))
    print("number of non-zero eigen values fast: ", np.sum(eival_fast > 1e-8))
    print(f"eigen value are same? {(eival_slow[-416:]-eival_fast < 1e-8).all()}")

    return mean, A, eival_slow, eivec_slow, eival_fast, eivec_fast_

# mean, A, eival_slow, eivec_slow, eival_fast, eivec_fast = train()


# show_face(mean, "The Mean Face")
# for i in range(1, 7):
#     print(eivec_slow[:, -i])
#     print(eivec_fast[:, -i])
#     show_face(eivec_slow[:, -i], f"{i} Eigenface (slow)")
#     show_face(eivec_fast[:, -i], f"{i} Eigenface (fast)")


def reconstruct_face (mean, phi_w, V):
    face = mean.copy()
    for i, eig in enumerate(V.T):
        face += eig * phi_w[i]
    return face

# N_nonzero = np.sum(eival_slow > 1e-8)
def test():
    acc_list = []
    err_list = []
    sample = 69
    show_face(data.T[sample], f"sample-face")
    show_face(data.T[sample] - mean, f"normalized-face")
    for M in tqdm(range(1, N_nonzero+1), total=N_nonzero):
        V = eivec_slow[:, -M:] # eigen space consist of top M eigen vectors
        W = V.T @ A            # column vectors are w_n

        # testing
        test_data  = data[:, test_indices]
        test_label = labels[test_indices]

        acc = 0
        err = 0
        test_N = test_data.shape[1]

        for i, test_face in enumerate(test_data.T):
            phi   = test_face - mean
            phi_w = (V.T @ phi) # project into eigenspace
            nn    = np.argmin([np.linalg.norm(p - phi_w) for p in W.T]) # index of nearest neighbor

            pred_label = train_label[nn]
            recon_face = reconstruct_face(mean, phi_w, V)
            if (i == sample and M % 50 == 0):
                show_face(recon_face, f"recon-face-M={M}")
                show_face(train_data[:, nn], f"NN-face-M={M}")
            acc += (pred_label == test_label[i])
            err += np.linalg.norm(recon_face - test_face)

        acc_list.append(acc/test_N)
        err_list.append(err/test_N)

    plot_acc(acc_list, "Accuracy vs. M")
    plot_acc(err_list, "Reconstruction Error vs. M")

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

def face_reg_acc (mean, V, A, train_label, test_data, test_label, time, title=None):
    W = V.T @ A            # column vectors are w_n

    acc = 0
    err = 0
    test_N = test_data.shape[1]
    for i, test_face in enumerate(test_data.T):
        phi   = test_face - mean
        phi_w = (V.T @ phi) # project into eigenspace
        nn    = np.argmin([np.linalg.norm(p - phi_w) for p in W.T]) # index of nearest neighbor

        pred_label = train_label[nn]
        recon_face = reconstruct_face(mean, phi_w, V)
        acc += (pred_label == test_label[i])
        err += np.linalg.norm(recon_face - test_face)
    acc /= test_N
    err /= test_N
    print(f"{title} & {100*acc:5.5}\\% & {time:5.5} & {err:5.5} \\\\")
    return acc

def minimal_pca(data):
    mean = data.mean(axis=1)
    A = (data.T - mean.T).T
    N = A.shape[1]
    S = (A @ A.T) / N

    D, V = np.linalg.eigh(S)
    return mean, N, V, D, S, A

def batch_pca():
    print("running batch PCA")
    mean, N, V, D, S, A = minimal_pca (train_data)
    return (mean, V, A, train_label)

def first_set_pca():
    print("running first set PCA")
    mean, N, V, D, S, A = minimal_pca (train_subdata_1)
    return (mean, V, A, train_sublabel_1)

def increment_pca(steps=3):
    assert(steps >= 2)
    print(f"running increment PCA with step {steps}")
    # in order of mean, N, eigenvectors, eigenvalues
    def combine_pca(mean1, N1, V1, D1, S1, A1, mean2, N2, V2, D2, S2, A2):
        N3        = N1 + N2
        A3        = np.hstack((A1, A2))
        mean3     = (N1*mean1 + N2*mean2) / N3
        mean_diff = np.expand_dims(mean1 - mean2, axis=1) # mean difference
        S3        = (N1/N3)*S1 + (N2/N3)*S2 + (N1*N2/N3)*(mean_diff @ mean_diff.T)

        combined_eig = np.hstack((V1, V2, mean_diff))
        Phi, _       = np.linalg.qr(combined_eig)
        D3, R        = np.linalg.eigh(Phi @ S3 @ Phi)

        return mean3, N3, Phi@R, D3, S3, A3

    train_subdata = [train_subdata_1, train_subdata_2,
                     train_subdata_3, train_subdata_4]
    train_sublabel = [train_sublabel_1, train_sublabel_2,
                      train_sublabel_3, train_sublabel_4]

    args0 = minimal_pca(train_subdata[0])
    args1 = minimal_pca(train_subdata[1])
    mean, N, V, D, S, A = combine_pca(*args0, *args1)
    train_lab = np.hstack((train_sublabel[0], train_sublabel[1]))

    for step in tqdm(range(2, steps)):
        tmp_args            = minimal_pca(train_subdata[step])
        mean, N, V, D, S, A = combine_pca(mean, N, V, D, S, A, *tmp_args)
        train_lab           = np.hstack((train_lab, train_sublabel[step]))

    return (mean, V, A, train_lab)

def test2():
    args_first, time_first = time_func(first_set_pca, None)
    args_batch, time_batch = time_func(batch_pca, None)
    args_incr2, time_incr2 = time_func(increment_pca, 2)
    args_incr3, time_incr3 = time_func(increment_pca, 3)
    args_incr4, time_incr4 = time_func(increment_pca, 4)

    out = sys.stdout
    
    with open('../tables/q2.txt', 'w') as f:
        sys.stdout = f
        print("\\begin{table}[ht]")
        print("\\centering")
        print("\\begin{tabular}[t]{lrrr}")
        print("\\hline")
        print("& avg. accuracy($\\uparrow$) & training time($\\downarrow$) & reconstruction error\\\\")
        print("\\hline")
        face_reg_acc(*args_first, test_data, test_label, time_first, title="first batch")
        face_reg_acc(*args_batch, test_data, test_label, time_batch, title="whole batch")
        face_reg_acc(*args_incr2, test_data, test_label, time_incr2, title="increment[2]")
        face_reg_acc(*args_incr3, test_data, test_label, time_incr3, title="increment[3]")
        face_reg_acc(*args_incr4, test_data, test_label, time_incr4, title="increment[4]")
        print("\\hline\\\\")
        print("\\end{tabular}")
        print("\\caption{Accuracy of }")
        print("\\label{tab:eigspeed}")
        print("\\end{table}")
        sys.stdout = out

    print("test 2 finished!")


# test2()

#############################

############ Q3 #############

def get_bags (data, n_, m):
    D, N    = data.shape
    bags    = np.zeros((m, D, n_))
    idx     = np.zeros((m, n_)) 
    for i in trange(m, desc="getting bags"):
        idx[i]  = np.random.choice(N, size=n_) 
        bags[i] = data[:, idx[i]]

    return bags, idx

def ensemble (M0=5, M1=10):
    _, _, V, D, _, A = minimal_pca (train_data)
    nz      = D > 1e-8 # filter non-zero
    D       = D[nz] 
    W       = V[:, nz] # (D, N-1) eigenfaces
    N       = W.shape[1]
    W0      = W[:, -M0:]
    W1      = W[:, np.random.choice(len(D)-M1, size=M1, replace=False)]
    data    = np.hstack((W1, W0)) 
    bags    = get_bags(data)

test3()

#############################

