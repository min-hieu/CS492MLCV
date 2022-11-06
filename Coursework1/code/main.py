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
from multiprocessing import Pool, freeze_support
from glob import glob
import argparse

######## for styling ########
from matplotlib import font_manager, rcParams
font_manager.findSystemFonts(fontpaths="/Users/charlie/Library/Fonts/", fontext="ttc")
myfont      = {'fontname':'Iosevka', 'fontsize':'15', 'fontweight': 'bold'}
mysmolfont  = {'fontname':'Iosevka', 'fontsize':'11', 'fontweight': 'regular'}
#############################


########### Argparse ###########
parser  = argparse.ArgumentParser()

parser.add_argument('--test', choices=["1","2","3"])

args    = parser.parse_args()
################################

########### Setup ###########
dataset = scipy.io.loadmat('face.mat')
data    = np.array(dataset['X'])     # (2576, 520)
labels  = np.array(dataset['l'][0])  # (1, 520)


test_indices  = np.concatenate((np.arange(data.shape[1]//10)*10+7,
                               (np.arange(data.shape[1]//10)*10+8)))
train_indices = [i for i in range(data.shape[1]) if i not in test_indices]

train_data  = data[:, train_indices]
train_label = labels[train_indices]

test_data   = data[:, test_indices]
test_label  = labels[test_indices]

classes     = np.unique(labels)

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
  
if args.test == "1":
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

if args.test == "1":
    N_nonzero = np.sum(eival_slow > 1e-8)
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

if args.test == "1":
    test()
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

if args.test == "2":
    test2()

#############################

############ Q3 #############

def self_scatter(c_data, c_mean):
    c_data_center = c_data - c_mean.reshape((*c_mean.shape, 1))
    return c_data_center @ c_data_center.T

def get_scatter_matrices(data, label, classes):
    m = data.mean(axis=1)
    S_w = np.zeros((data.shape[0], data.shape[0]))
    S_b = np.zeros((data.shape[0], data.shape[0]))

    for c in tqdm(classes, desc="Getting scatter matrices", leave=False):
        c_idx   = label == c
        c_data  = data[:, c_idx]
        c_mean  = c_data.mean(axis=1)

        c_centered  = c_data - c_mean.reshape((*c_mean.shape, 1))
        S_i         = c_centered  @ c_centered.T
        S_w         += S_i

        N_i         = c_data.shape[1]
        c_diff      = c_mean - m
        S_b         += N_i * np.outer(c_diff, c_diff)

    return S_w, S_b


def get_W_pca (M_pca, data):
    _, _, V, D, _, A = minimal_pca (data)
    nz      = D > 1e-8 # filter non-zero
    return V[:, nz][:, -M_pca:] 

def get_W_lda (M_lda, W_pca, S_w, S_b):
    SW      = W_pca.T @ S_w @ W_pca
    SB      = W_pca.T @ S_b @ W_pca
    S       = np.linalg.inv(SW) @ SB
    D, V    = np.linalg.eigh(S)
    nz      = D > 1e-8 # filter non-zero

    return V[:, nz][:, -M_lda:] 

def mini_reg_acc (V, train, train_lab, test, test_lab, classes):
    mean    = train.mean(axis=1)
    A       = train - mean.reshape((train.shape[0], 1))
    W       = V.T @ A  # column vectors are w_n

    acc     = 0
    err     = 0
    test_N  = test.shape[1]

    c       = len(classes)
    con_mat = np.zeros((c,c))

    for i, test_face in enumerate(test.T):
        phi   = test_face - mean
        phi_w = (V.T @ phi) # project into eigenspace
        nn    = np.argmin([np.linalg.norm(p - phi_w) for p in W.T]) # index of nearest neighbor

        pred_lab    = train_lab[nn]
        recon_face  = reconstruct_face(mean, phi_w, V)

        acc += (pred_lab == test_lab[i])
        err += np.linalg.norm(recon_face - test_face)
        con_mat[pred_lab-1, test_lab[i]-1] += 1
        
    return acc/test_N, err/test_N, con_mat


train_N     = train_data.shape[1]
train_c     = len(np.unique(train_label))
M_pca_r     = np.linspace(1, train_N-train_c, num=30, dtype=int) 
M_lda_r     = np.linspace(1, train_c-1, num=30, dtype=int)
M_pca_r_smol    = np.linspace(1, train_N-train_c, num=5, dtype=int) 
M_lda_r_smol    = np.linspace(1, train_c-1, num=5, dtype=int) 


def pca_lda_minimal (data, label, m_pca, m_lda):
    classes     = np.unique(label)
    N           = data.shape[1]
    c           = len(classes)

    S_w, S_b    = get_scatter_matrices(data, label, classes)
    W_pca       = get_W_pca(m_pca, train_data)
    W_lda       = get_W_lda(m_lda, W_pca, S_w, S_b)
    W           = (W_lda.T @ W_pca.T).T

    return W


def test3 ():
    classes     = np.unique(train_label)
    N           = train_data.shape[1]
    c           = len(classes)

    start_s     = time()
    S_w, S_b    = get_scatter_matrices(train_data, train_label, classes)
    total_t_s   = start_s - time()

    print("collecting statistics...")
    size_s_w    = sys.getsizeof(S_w)
    size_s_b    = sys.getsizeof(S_b)
    rank_s_w    = np.linalg.matrix_rank(S_w)
    rank_s_b    = np.linalg.matrix_rank(S_b)

    total_conf  = np.zeros((c,c))
    acc_matrix  = np.zeros((len(M_pca_r), len(M_lda_r)))
    err_matrix  = np.zeros((len(M_pca_r), len(M_lda_r)))
    mem_matrix  = np.zeros((len(M_pca_r), len(M_lda_r)))
    tim_matrix  = np.zeros((len(M_pca_r), len(M_lda_r)))
    print("DONE")

    for p, M_pca in enumerate(tqdm(M_pca_r, desc="running test")):
        for l, M_lda in enumerate(tqdm(M_lda_r, leave=False)):
            start   = time()

            W_pca   = get_W_pca(M_pca, train_data)
            W_lda   = get_W_lda(M_lda, W_pca, S_w, S_b)
            W       = (W_lda.T @ W_pca.T).T

            total_t     = time() - start
            mem_pca     = sys.getsizeof(W_pca)
            mem_lda     = sys.getsizeof(W_lda)
            total_m     = mem_pca + mem_lda

            acc, err, con_mat = mini_reg_acc(W, train_data, train_label, 
                                             test_data, test_label, classes)
            if p % 10 == 0 and l % 10 == 0:
                np.save(f"./conf_pca{M_pca_r[p]}_lda{M_lda_r[l]}", con_mat)

            acc_matrix[p][l] = acc
            err_matrix[p][l] = err
            mem_matrix[p][l] = total_m
            tim_matrix[p][l] = total_t
            total_conf += con_mat

    np.save("./acc_matrix", acc_matrix)
    np.save("./err_matrix", err_matrix)
    np.save("./mem_matrix", mem_matrix)
    np.save("./tim_matrix", tim_matrix)
    np.save("./total_conf", total_conf)
    
def visualize_matrix(M, fname, title, show=False, conf=False):
    classes     = np.unique(train_label)
    N           = train_data.shape[1]
    c           = len(classes)

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    fig, ax = plt.subplots()

    cax = ax.matshow(M, cmap='viridis')
    if conf:
        bounds = np.sort(np.unique(M)).astype(int)
        print(bounds.shape)
        bounds = np.vstack((np.array(0).reshape(1,), bounds))
        cbar = fig.colorbar(cax, boundaries=bounds)
        cbar.ax.set_ylabel("counts", **mysmolfont)
        ax.set_title(title, **myfont)
        ax.set_xlabel('real class', **mysmolfont)
        ax.set_ylabel('predicted class', **mysmolfont)
    else:
        cbar = fig.colorbar(cax)
        cbar.ax.set_ylabel(title, **mysmolfont)
        ax.set_xlabel('M_lda', **mysmolfont)
        ax.set_ylabel('M_pca', **mysmolfont)
        ax.set_xticklabels(M_lda_r, **mysmolfont)
        ax.set_yticklabels(M_pca_r, **mysmolfont)

    if show:
        plt.show()
    plt.savefig(fname)

def save_figs_test3():
    matrix_list = ["acc", "err", "mem", "tim"]
    title = {"acc": "accuracy (%)", 
             "err": "reconstruction error", 
             "mem": "total memory (bytes)", 
             "tim": "total time (s)"}

    for M in matrix_list:
        try:
            matrix = np.load(f"./{M}_matrix.npy")
            fname  = f"../figures/q3/{M}_plot.png"
            visualize_matrix(matrix, fname, title[M])
        except Exception as e:
            print(e)

    matrix  = np.load(f"./total_conf.npy")
    fname   = f"../figures/q3/conf_plot.png"
    title   = "total confusion matrix"
    visualize_matrix(matrix, fname, title=title, conf=True)

    for con_mat in glob("./conf_pca*.npy"):
        pca     = con_mat.split("_")[1][3:]
        lda     = con_mat.split("_")[2].split(".")[0][3:]
        matrix  = np.load(con_mat)
        fname   = f"../figures/q3/{con_mat[2:][:-4]}_plot.png"
        title   = f"confusion matrix pca = {pca} lda = {lda}"
        visualize_matrix(matrix, fname, title=title, conf=True)



def get_best_hyperparam(show=True):
    acc_matrix = np.load(f"./acc_matrix.npy")
    err_matrix = np.load(f"./err_matrix.npy")
    
    score   = acc_matrix * err_matrix
    argmax  = np.unravel_index(score.argmax(), score.shape)

    fname   = f"../figures/q3/score_plot.png"
    title   = "score matrix"
    visualize_matrix(score, fname, title)
    if show:
        print(f"score at {argmax} is {score[argmax]}")
        print(f"M_pca is {M_pca_r[argmax[0]]}, M_lda is {M_lda_r[argmax[1]]}")

    return argmax
    

# visualize_test3()
# test3()
# save_figs_test3()

def get_bags (data, label, classes, n, T):
    D, N        = data.shape
    c           = len(classes)
    bags        = np.zeros((T, D, c*n))
    bags_label  = np.zeros((T, c*n)) 
    for t in trange(T, desc="getting bags", leave=False):
        for i, cl in enumerate(classes):
            train_class = data[:,label == cl]
            idx_class   = np.random.choice(train_class.shape[1], size=n,
                                            replace=False)
            # TODO: replace = True makes huge difference in error

            bags[t,:,i*n:(i+1)*n]       = train_class[:, idx_class] 
            bags_label[t,i*n:(i+1)*n]   = cl


    return bags, bags_label

def eval_pred(pred, label, classes):
    N       = len(pred)
    acc     = np.sum(pred == label) / N

    c       = len(classes)
    con_mat = np.zeros((c,c))

    for i in range(N):
        con_mat[pred[i]-1, label[i]-1] += 1

    return acc, con_mat

def get_pred(V, train, label, test):
    mean = train.mean(axis=1)
    A    = train - mean.reshape((train.shape[0], 1))
    W    = V.T @ A
    pred = np.zeros(test.shape[1])

    for i, test_face in enumerate(test.T):
        phi     = test_face - mean
        phi_w   = (V.T @ phi) # project into eigenspace
        nn      = np.argmin([np.linalg.norm(p - phi_w) for p in W.T]) # index of nearest neighbor
        pred[i] = label[nn]

    return pred

def fusion_vote(preds, classes):
    T, N    = preds.shape
    c       = len(classes)
    preds_p = np.zeros((T, c, N)) # probability distribution

    for t, p_t in enumerate(preds): # p_t as shape (N,)
        p_t_ = (p_t-1).astype(int)
        preds_p[t, p_t_, range(len(p_t))] += 1

    preds_dist = preds_p.mean(axis=0) # (c, N) 
    # column of preds_dist is p(y|x)

    return preds_dist.argmax(axis=0) + 1

# c1=8 T=4 pca=100 lda=40 give 66%
def ensemble_random_data(c1=8, T=4, m_pca=100, m_lda=40, fusion='vote', pos=0):
    bags, bags_lab  = get_bags(train_data, train_label, classes, c1, T)
    bags_pred       = np.zeros((T, len(test_label)))

    for i in trange(T, desc="running ensemble", leave=False, position=pos):
        W               = pca_lda_minimal(bags[i], bags_lab[i], m_pca, m_lda)
        bags_pred[i]    = get_pred(W, bags[i], bags_lab[i], test_data)

    if fusion == 'sum':
        pred = fusion_sum(bags_pred)
    elif fusion == 'vote':
        pred = fusion_vote(bags_pred, classes)
    else:
        raise Exception

    return eval_pred(pred, test_label, classes)

acc_matrix_rd = np.zeros((6,6,5,5))

def tune_ensemble_rdata(i):
    # change m_pca, m_lda, c1, T
    T_range     = [4,8,10,20,40,60][i:i+2]
    c1_range    = [4,6,8,10,15,20][i:i+2]
    

    for t_i, T in enumerate(T_range):
        for c_i, c1 in enumerate(c1_range):
            for p, m_pca in enumerate(M_pca_r_smol):
                for l, m_lda in enumerate(M_lda_r_smol):
                    acc, conf = ensemble_random_data(c1,T,m_pca,m_lda,pos=i)
                    acc_matrix_rd[t_i+2*i, c_i+2*i, p, l] = acc
                    np.save(f"./q3/conf_T{T}_c1{c1}_pca{m_pca}_lda{m_lda}_rd", conf)
                    np.save("./q3/acc_matrix_rd", acc_matrix_rd)


'''
if __name__ == '__main__':
    freeze_support()
    with Pool(3) as p:
        p.map(tune_ensemble_rdata, range(3))
'''
    

def get_random_feature(eigspace, M1, T, fixed_V):
    D, N    = eigspace.shape
    M0      = fixed_V.shape[1]
    rand_Vs = np.zeros((T, D, M1+M0))

    for t in range(T):
        rand_idx    = np.random.choice(N, size=M1, replace=False)
        rand_V      = eigspace[:, rand_idx]
        rand_Vs[t]  = np.hstack((rand_V, fixed_V))

    return rand_Vs

def ensemble_random_feature(M0=14, M1=30, T=6, m_lda=40, fusion='vote', pos=0):
    classes     = np.unique(train_label)
    N           = data.shape[1]
    c           = len(classes)

    _, _, V, D, _, A = minimal_pca (data)
    nz          = D > 1e-8 # filter non-zero
    eigenspace  = V[:, nz]
    fixed_V     = eigenspace[:, -M0:] 
    rand_Vs     = get_random_feature(eigenspace[:, :-M0], M1, T, fixed_V)
    preds       = np.zeros((T, *test_label.shape))

    S_w, S_b    = get_scatter_matrices(train_data, train_label, classes)
    for t in trange(T, desc="random feature ensemble", position=pos):
        W_lda       = get_W_lda(m_lda, rand_Vs[t], S_w, S_b)
        W           = (W_lda.T @ rand_Vs[t].T).T
        preds[t]    = get_pred(W, train_data, train_label, test_data)
        
    preds = fusion_vote(preds, classes)

    return eval_pred(preds, test_label, classes)

acc_matrix_rf = np.zeros((6,6,5,5))

def tune_ensemble_rfeat(i):
    # change m_pca, m_lda, c1, T
    T_range     = [4,8,10,20,40,60][i:i+2]
    M0_range    = [20,50,100,200,300][i:i+2]
    M1_range    = [5,20,40,60,80,100]
    

    for t_i, T in enumerate(T_range):
        for m0_i, m0 in enumerate(M0_range):
            for m1_i, m1 in enumerate(M1_range):
                for l, m_lda in enumerate(M_lda_r_smol):
                    acc, conf = ensemble_random_feature(m0,m1,T,m_lda,pos=i)
                    acc_matrix_rf[t_i+2*i, m0_i+2*i, m1_i, l] = acc
                    np.save(f"./q3/conf_T{T}_m0{m1}_m1{m1}_lda{m_lda}_rf", conf)
                    np.save("./q3/acc_matrix_rf", acc_matrix_rf)


if __name__ == '__main__':
    freeze_support()
    with Pool(3) as p:
        p.map(tune_ensemble_rfeat, range(3))


def test3_ensemble():
    pass


#############################

