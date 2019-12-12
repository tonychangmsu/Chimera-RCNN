import os
import numpy as np
import gdal
from osgeo import osr
import itertools
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
from keras.models import model_from_json
import matplotlib.pyplot as plt

def get_data_paths(partition, scenario, data_dir='data'):
    '''
    opens glued data list and labels

    Parameters:
    -----------
    partition: <str> glued data partition
    scenario: <str> glued scenario
    data_dir: <str> path to glued data
    '''
    filename_suffix = '%s-set-%s-ts.csv' % (partition, scenario)
    xy_filename = 'glued-labels-%s' % (filename_suffix)
    y_lookup_filename = 'glued-label-lookup-%s' % (filename_suffix)
    return os.path.join(data_dir, xy_filename), \
            os.path.join(data_dir, y_lookup_filename)


def get_class_weight(datalist):
    y_train = datalist['label_int'].values
    class_weight = compute_class_weight('balanced', np.unique(y_train), y_train)
    return(class_weight)

def get_x_data(xfile):
    return(np.array(Image.open(xfile)))

def pre_process_x(x_):
    return(np.array(x_).astype('float32')/255.)

def pre_process_y(y_, n_classes):
    y_one_hot = np.zeros(n_classes)
    y_one_hot[y_] = 1
    return(y_one_hot)

def random_channel_shift(x, intensity):
    '''
    Applies channel shift to image by set intensity
    percent. 

    Parameters:
    ----------
    x : <numpy array> of raw 4D x training data assumes
        channels are the last dimension 
    '''
    min_x, max_x = np.min(x), np.max(x)
    out = np.zeros(x.shape)
    for channel in range(x.shape[-1]):
        out[:,:,channel] = np.clip(x[:,:,channel] +\
                (max_x*np.random.uniform(-intensity, intensity)),\
                min_x, max_x)
    return(out)

def data_augment(x):
    '''
    Takes in a pre-process set of image data,
    and randomly transforms it with a 
    rotation, flip, or mirror. 
    Concatenates new data to end of inputs

    Parameters:
    x : numpy array of raw 4D x training data
    '''
    seed = [np.random.choice([-1,0,1]), \
    np.random.choice([0,1,2])]
    # create a seed for rotation scale -1, 0, or 1
    # flip/mirror/neither : 0-2,
    # apply these values to the new data generated
    im_x = Image.fromarray(x).rotate(90*seed[0])
    if seed[1] == 0: #flip
        im_x = im_x.transpose(Image.FLIP_LEFT_RIGHT)
    if seed[1] == 1: #mirror
        im_x = im_x.transpose(Image.FLIP_TOP_BOTTOM)
    x_ = np.array(im_x)
    return(x_)

def data_generator(datalist, data_dirc='./data',\
        n_classes=10, batch_size=1, augment_img=False,
        channel_shift=None):
    total_data_size = len(datalist)
    data_indices = np.arange(total_data_size)
    if batch_size is None:
        batch_size = total_data_size
    n_batches = total_data_size // batch_size
    batch_list = np.random.permutation(total_data_size)
    for b in range(n_batches):
        permutation = batch_list[b*batch_size:((b+1)*batch_size)]
        batch = data_indices[permutation]
        x_batch, y_batch = [], []
        for i in batch:
            x_ = get_x_data('%s/%s'%(data_dirc, datalist['file_name'].iloc[i]))
            y_ = int(datalist.iloc[i]['label_int'])
            if augment_img:
                x_ = data_augment(x_)
            if channel_shift is not None:
                x_ = random_channel_shift(x_, channel_shift)
            x_im = pre_process_x(x_)
            x_batch.append(x_im)
            y_batch.append(pre_process_y(y_, n_classes))
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        yield (x_batch, y_batch)

###################################################################
def get_ts_data(datalist, get_y=False):
    bandlist = [1,2,3,4,5,7]
    ts_data = []
    for q in range(1,5):
        ts_data.append([datalist['B%s_med_q%s'%(b,q)] for b in bandlist])
    if get_y:
        y_data = datalist['label_int']
        n_classes = len(np.unique(y_data))
        y_out = []
        for y in y_data:
            y_out.append(pre_process_y(y, n_classes))
        return(np.array(ts_data), np.array(y_out))
    return(np.array(ts_data))

def data_generator_two(datalist, data_dirc='./data',\
        n_classes=10, batch_size=1, augment_img=False,
        channel_shift=None):
    total_data_size = len(datalist)
    data_indices = np.arange(total_data_size)
    if batch_size is None:
        batch_size = total_data_size
    n_batches = total_data_size // batch_size
    batch_list = np.random.permutation(total_data_size)
    for b in range(n_batches):
        permutation = batch_list[b*batch_size:((b+1)*batch_size)]
        batch = data_indices[permutation]
        x1_batch, x2_batch, y_batch = [], [], []
        for i in batch:
            x_ = get_x_data('%s/%s'%(data_dirc, datalist['file_name'].iloc[i]))
            y_ = int(datalist.iloc[i]['label_int'])
            if augment_img:
                x_ = data_augment(x_)
            if channel_shift is not None:
                x_ = random_channel_shift(x_, channel_shift)
            # 2 inputs
            x_ts = get_ts_data(datalist.iloc[i])
            x1_batch.append(pre_process_x(x_))
            x2_batch.append(x_ts)
            y_batch.append(pre_process_y(y_, n_classes))
        x_batch = [np.array(x1_batch),np.array(x2_batch)]
        y_batch = np.array(y_batch)
        yield (x_batch, y_batch)

def import_model(model_json, model_weights):
    '''
    Imports a keras model architecture and 
    associated weights.

    Parameters:
    -----------
    model_json : <str> of keras model in json
    format

    model_weights : <str> of keras model parameters weights
    '''

    json_file = open(model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_weights)
    return(loaded_model)

def write_raster(test_img, y_hat, stride, continuous=False):
    if continuous:
        #os.makedirs('./predictions/continuous')
        outfilename = './predictions/continuous/%s'%(os.path.basename(test_img))
    else:
        #os.makedirs('./predictions/classified')
        outfilename = './predictions/classified/%s'%(os.path.basename(test_img))
    y_hat = np.expand_dims(y_hat, axis=-1)
    n_classes = np.shape(y_hat)[-1]
    ref = gdal.Open(test_img)
    ref_gt = ref.GetGeoTransform()
    out_proj = osr.SpatialReference()
    out_proj.ImportFromWkt(ref.GetProjectionRef())
    out_gt = (ref_gt[0], stride[0], 0, ref_gt[1], 0, -stride[1])
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(outfilename, n_cols, n_rows, n_classes, gdal.GDT_Float32)
    ds.SetGeoTransform(out_gt)
    ds.SetProjection(out_proj.ExportToWkt())
    for i in range(n_classes):
        ds.GetRasterBand(i+1).WriteArray(y_hat[:,:,i])
    ds = None
    print('Wrote %s'%outfilename)

def plot_confusion_matrix(cm, classes, normalize=False,\
        title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(im,fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=14)
    plt.yticks(tick_marks, classes, fontsize=14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",\
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
