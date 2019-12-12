import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

def add_noise(x_, contrast_reduce=0.5, std=3, add_pause=0):
    '''
    reduce the contrast of the MNIST data and apply an
    additive gaussian white noise filter:
    following: http://csc.lsu.edu/~saikat/n-mnist/
    '''
    x_ = x_*0.5
    noise = np.random.normal(0,np.mean(x_)*std,(x_.shape))
    return(x_ + noise)

def gen_order_idx(y_, order):
    if ((order =='ascending') | (order=='a')):
        s_idx = np.argsort(y_)
    elif ((order=='descending') | (order=='d')):
        s_idx = np.argsort(y_)[::-1]
    _, counts = np.unique(y_[s_idx], return_counts=True)
    n_idx = (np.zeros(11)).astype(int)
    n_idx[1:] = np.cumsum(counts)
    return(s_idx, n_idx)

def gen_ordered_data(n, t, x_, y_, order):
    s_idx, n_idx = gen_order_idx(y_, order)
    yt = np.ones((n,2)).astype(int)
    if ((order =='ascending') | (order=='a')):
        yt[:,0] = np.zeros(n)
    elif ((order=='descending') | (order=='d')):
        yt[:,1] = np.zeros(n)
    xt = x_[s_idx]
    xt_i = np.zeros((n,t)).astype(int)
    for i in range(n):
        n_set = np.sort(np.random.choice(np.arange(10), size=t, replace=False))
        for j in range(t):
            xt_i[i,j] = np.random.choice(np.arange(n_idx[n_set[j]],\
                    n_idx[n_set[j]+1]))
    return(xt[xt_i], yt)

def time_series_func(x, y, deg):
    pass

def gen_continuous_data(n, t, x_, y_, add_pause=0, transform_func=np.polyfit):
    ''' 
    Generate based on random numbers, calculated the OLS regression slope 
    and intercept, but we want to used numbers as pictures. Also the 
    sign will be determined by the ascending and descending classification 
    that we hope to pull out of the regression.
    '''
    x_t = np.zeros((n, t+add_pause, x_.shape[1], x_.shape[2])).astype(int)
    yt_1 = np.zeros((n,3)).astype(int)
    yt_2 = np.zeros((n,2))
    y_out = []
    # yt is described in 3 classes, {0:'stationary', 1:'ascending', 2:'descending'}
    # and two regression coefficients 
    thr = 0.05 #threshold for the regression to be basically flat
    for i in range(n):
        n_set = np.random.choice(len(y_), size=t, replace=True)
        x_values = x_[n_set]
        pause_set = np.random.choice(t+add_pause, size=t+add_pause, replace=False)
        pause_set = (pause_set>=add_pause)
        x_t[i][pause_set] = x_[n_set]
        # now fit a regression to this
        m, b = transform_func(np.arange(t), y_[n_set], 1)
        if np.abs(m) <= thr:
            yt_1[i][0] = 1
        elif(m>0):
            yt_1[i][1] = 1
        else:
            yt_1[i][2] = 1
        yt_2[i] = [np.abs(m),b]
    y_out = [yt_1, yt_2]
    x_out = pre_process_x(x_t)
    return(x_out, y_out)  

def pre_process_x(x_):
    x_ = np.reshape(x_, (x_.shape[0], x_.shape[1], x_.shape[2],\
            x_.shape[3], 1))  # add a channels dimension
    return(x_/255)

def gen_data(n, t, x_, y_):
    xt_des, yt_des = gen_ordered_data(int(n/2), t, x_, y_, 'a')
    xt_asc, yt_asc = gen_ordered_data(int(n/2), t, x_, y_, 'd')
    x_out = np.concatenate((xt_des, xt_asc))
    y_out = np.concatenate((yt_des, yt_asc))
    shuffle_idx = np.random.choice(np.arange(n),size=n,replace=False)
    x_out = pre_process_x(x_out[shuffle_idx])
    y_out = y_out[shuffle_idx]
    return(x_out, y_out)

def plot_confusion_matrix(y_, y_hat, classes, normalize=False, \
        title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm =cnf_matrix = confusion_matrix(y_, y_hat) 

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",\
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    
