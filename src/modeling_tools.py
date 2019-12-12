import numpy as np
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
import keras
from keras import optimizers
from keras import backend as K
from src.tsNet_model import chimera, rmse, det_coeff
from src.analysis_tools import save_accuracies
import datetime

def train_kfold(X_train, y_train, X_val, y_val, k, batch_size=100, \
                epochs=10, learning_rates=[1e-3,1e-4,1e-5]):
    '''
    Trains the kth-fold of the chimera model using an Adam optimizer
    and step-down learning rate scheduling.

    Parameters:
    X_train : <list>
    y_train : <list>
    k       : <int>
    batch_size: <int>
    epochs:   <int>
    learning_rates: <list>
    '''
    input_shape = [X_train[i].shape[1:] for i in range(len(X_train))]
    output_shape = [y_train[i].shape[1:][0] for i in range(len(y_train))]
    tsnet_model = chimera(input_shape, output_shape)
    model_history = []
    nb_epochs = 1 
    sample_size = len(X_train[0])
    steps_per_epoch = (sample_size // batch_size)
    learning_rates = [1e-3, 1e-4, 1e-5]
    class_weights = gen_class_weights(y_train)
    #set up optimizer
    opt = optimizers.Adam(lr=learning_rates[0], beta_1=0.9, beta_2=0.999, \
                          epsilon=1e-08, decay=0.0)
    #compile model
    tsnet_model.compile(loss=['categorical_crossentropy',rmse],\
                              optimizer=opt, metrics=['accuracy',det_coeff])
    #set up decay learning rate
    counter = 0
    current_lr = scheduler(0, epochs, learning_rates)
    #start up model training
    print('Fitting Model K-fold:%s! Learning Rate: %s'%(k,current_lr))
    for epoch in range(epochs):
        lr = scheduler(epoch, epochs, learning_rates)  
        if current_lr != lr:
            current_lr = lr
            print('Changing Learning Rate to: %s' %current_lr)
            K.set_value(tsnet_model.optimizer.lr, current_lr)
        gen = data_generator(X_train,y_train,batch_size=batch_size,augment_img=True,landsat=False)
        model_history.append(tsnet_model.fit_generator(gen, steps_per_epoch=steps_per_epoch,\
                                                  nb_epoch=1, validation_data=(X_val, y_val),\
                                                  class_weight=[class_weights, [1]]))
    return(tsnet_model, model_history)

def write_model(model, model_history, model_num, nepochs, prefix='chimera'):
    today = datetime.datetime.now().strftime("%Y%m%d")
    model_json = model.to_json()
    version = '%s_%s'%(today, model_num)
    with open('./models/%s_%s.json'%(prefix,version), 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('./models/%s_%s.h5'%(prefix,version))
    model_out = save_accuracies(model_history, nepochs-1, 1, \
                                './models/%s_model_history_%s.npy'%(prefix,version)) 
    return()

def data_split(X,y,proportion=0.8,idx_return=False, seed=1234):
    '''
    Splits data into training and validation sets
    
    Parameters:
    X : <list>
    y : <list>
    proportion : <float>
    '''
    np.random.seed(seed=seed) 
    n_samples = y[0].shape[0]
    idx = np.random.permutation(n_samples)
    t_end = np.floor(proportion*n_samples).astype(int)
    train_idx = idx[:t_end] 
    val_idx = idx[t_end:]
    X_train = [X[i][train_idx] for i in range(len(X))] 
    X_val = [X[i][val_idx] for i in range(len(X))] 
    y_train = [y[i][train_idx] for i in range(len(y))] 
    y_val = [y[i][val_idx] for i in range(len(y))] 
    if idx_return:
        return(X_train, y_train, X_val, y_val, [train_idx, val_idx])
    else:
        return(X_train, y_train, X_val, y_val)

def k_fold_sampler(n,k,seed=1234):
    '''
    Splits data into training and validation sets for k fold 
    cross validation
    
    Parameters:
    n : <int> number of samples
    k : <int> k folds desired

    Returns:
    train_indices: <list> k-element list of indices for n samples
    val_indices: <list> k-element list of indices for n samples
   '''
    kf = KFold(k, shuffle=True, random_state=seed)
    kf.get_n_splits(np.arange(n))
    train_idx, val_idx = [], []
    for t, v in kf.split(np.arange(n)):
        train_idx.append(t)
        val_idx.append(v)
    return(train_idx, val_idx)

def get_samples(X, y, train_idx, val_idx):
    '''
    Takes in indices and samples and sub-samples into training 
    and validation batches
    '''     
    X_train = [X[i][train_idx] for i in range(len(X))]
    y_train = [y[i][train_idx] for i in range(len(y))]
    X_val = [X[i][val_idx] for i in range(len(X))]
    y_val = [y[i][val_idx] for i in range(len(y))]
    return(X_train, y_train, X_val, y_val)

def data_augment(x, seed=None, timeseries=False):
    '''
    Takes in a pre-process set of image data,
    and randomly transforms it with a 
    rotation, flip, or mirror. 
    Concatenates new data to end of inputs

    Parameters:
    x : numpy array of raw n x training data
    '''
    if seed is None:
        seed = np.random.choice([0,1,2,3])
    # create a seed for rotation scale -1, 0, or 1
    # flip/mirror/neither : 0-2,
    # apply these values to the new data generated
    if seed == 0:
        return(x) #no change
    if timeseries:
        if seed == 1: #flip
            im_x = np.array([np.fliplr(x[i]) for i in range(len(x))])
        elif seed == 2: #mirror
            im_x = np.array([np.flipud(x[i]) for i in range(len(x))])
        elif seed == 3: #flip and mirror
            im_x = np.array([np.fliplr(np.flipud(x[i])) for i in range(len(x))])
        return(im_x)
    else:
        if seed == 1: #flip
            return(np.fliplr(x))
        elif seed == 2: #mirror
            return(np.flipud(x))
        else: #flip & mirror
            return(np.fliplr(np.flipud(x)))

def data_generator(X, y, batch_size=None, augment_img=False, landsat=False):
    total_data_size = len(y[0])
    data_indices = np.arange(total_data_size)
    if batch_size is None:
        batch_size = total_data_size
    n_batches = total_data_size // batch_size
    batch_list = np.random.permutation(total_data_size)
    for b in range(n_batches):
        permut = batch_list[b*batch_size:((b+1)*batch_size)]
        x_batch = [X[i][permut] for i in range(len(X))]
        y_batch = [y[i][permut] for i in range(len(y))]
        if augment_img:
            for i in range(len(permut)):
                seed = np.random.choice([0,1,2,3])
                x_batch[0][i] = data_augment(x_batch[0][i], seed=seed)
                if landsat:
                    x_batch[-1][i] = data_augment(x_batch[-1][i], seed=seed, timeseries=True)
        yield(x_batch, y_batch)

def gen_class_weights(y_train):
    unhot = np.argmax(y_train[0], axis=1)
    cw = compute_class_weight('balanced', np.unique(unhot), unhot)
    n_classes = np.max(unhot)+1
    class_weights = np.zeros(n_classes)
    c = np.arange(n_classes)
    for i in range(n_classes):
        class_weights[c[i]] = cw[i]
    print(class_weights)
    return(class_weights)

def scheduler(epoch, total_epochs, learning_rates):
    learning_rate_step = total_epochs/len(learning_rates)
    return(learning_rates[int(epoch // learning_rate_step)])

