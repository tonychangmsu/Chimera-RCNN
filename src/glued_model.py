from keras import models, layers
from keras import utils
from keras.regularizers import l2
import keras.backend as K

def bn_block(input_tensor, filters, kernel_size, strides, dropout=None):
    ''' Standard convolutional block with batch normalization,
    activation using a Rectified Linear Unit and dropout.

    Parameters:
    -----------
    input_tensor : 4d tensor of shape (samples, nrows, ncols, channels)
    filters : int scaler of number of filters 
    kernel_size : int tuple of convolution kernel size
    strides : int tuple of stride for convolution
    dropout : float of dropout rate
    '''
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, \
    padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    if dropout is not None:
        x = layers.Dropout(dropout)(x)
    return(x)

#########################################################
#########################################################
#########################################################
def residual_block(x, filters, kernel_size, dropout, conv_layers):
    '''
    Residual block constructor 

    Parameters:
    -----------
    x : 4d tensor of shape (samples, nrows, ncols, channels)
    filters : int scaler of number of filters 
    kernel_size : int tuple of convolution kernel size
    strides : int tuple of stride for convolution
    dropout : float of dropout rate
    conv_layers : int scalar number of bn_block layers to construct
    '''
    shortcut = layers.Conv2D(filters, kernel_size=(1,1), strides=(1,1), \
            padding='same')(x)
    for c in range(conv_layers-1):
        x = bn_block(x, filters, kernel_size, (1,1), dropout)
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=(1,1),\
            padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout)(x)
    return(x)

def res_net(input_shape, output_shape, initial_filter):
    input_layer = layers.Input(shape=input_shape)
    # first convolution block 
    x = bn_block(input_layer, initial_filter, (7,7), (2,2), dropout=0.15)
    # second convolution block
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = residual_block(x, initial_filter*2, (3,3), dropout=0.4, conv_layers=2)
    # third convolution block
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = residual_block(x, initial_filter*4, (3,3), dropout=0.4, conv_layers=4)
    # fourth convolution block
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = residual_block(x, initial_filter*8, (3,3), dropout=0.4, conv_layers=8)
    # fifth convolution block
    #x = layers.MaxPooling2D(pool_size=(2,2))(x)
    #x = residual_block(x, initial_filter*16, (3,3), dropout=0.4, conv_layers=3)

    pool = layers.AveragePooling2D(pool_size=(2,2), strides=(1,1))(x)
    flatten = layers.Flatten()(pool)
    output_layer = layers.Dense(units=output_shape, activation="softmax")(flatten)
    model = models.Model(inputs=input_layer, outputs=output_layer)
    return(model)

#########################################################
#########################################################
#########################################################
def identity_block(input_tensor, filters, kernel_size):
    '''
    Identity block is a residual block that
    does not convolve input and merges to final layer
    
    Parameters:
    -----------
    input_tensor : <keras tensor>
    filters : list(<int>, <int>, <int>) 
    kernel_size : tuple(<int>, <int>)
    '''
    x = bn_block(input_tensor, filters[0], kernel_size=(1,1), \
            strides=(1,1))
    # convolution
    x = bn_block(x, filters[1], kernel_size=kernel_size,\
            strides=(1,1))
    # 1x1 conv for filter match
    x = layers.Conv2D(filters[2], kernel_size=(1,1), strides=(1,1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return(x)


def conv_block(input_tensor, filters, kernel_size, stride, \
        dropout=None):
    '''
    Residual block creates a short cut layer that connects
    input tensor to output of 3 convolutions. 
    
    Parameters:
    -----------
    input_tensor : <keras tensor>
    filters : list(<int>, <int>, <int>) 
    kernel_size : tuple(<int>, <int>)
    stride : tuple(<int>, <int>)
    dropout : <float>
    ''' 
    shortcut = layers.Conv2D(filters[2], kernel_size=kernel_size, \
        strides=stride, padding='same')(input_tensor)
    # 1x1 conv
    x = bn_block(input_tensor, filters[0], kernel_size=(1,1), \
            strides=stride, dropout=dropout)
    # convolution
    x = bn_block(x, filters[1], kernel_size=kernel_size,\
            strides=(1,1), dropout=dropout)
    # 1x1 conv for filter match
    x = layers.Conv2D(filters[2], kernel_size=(1,1), strides=(1,1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return(x)

def split_branch(input_tensor, cropped_size, filters, kernel_size, \
        stride, dropout):
    c = (((input_tensor._keras_shape[1]-cropped_size[0])//2),\
            ((input_tensor._keras_shape[2]-cropped_size[1])//2))
    x_s = layers.Cropping2D(c)(input_tensor)
    x_s = bn_block(x_s, filters[0], kernel_size, \
           stride, dropout)
    x_s = conv_block(x_s, filters, kernel_size, stride,\
            dropout)
    x_s = identity_block(x_s, filters, kernel_size)
    x_s = conv_block(x_s, filters*2, kernel_size, stride,\
            dropout)
    x_s = identity_block(x_s, filters*2, kernel_size)
    x_s = identity_block(x_s, filters*2, kernel_size)
    x_s = layers.AveragePooling2D((3,3))(x_s)
    split_flat = layers.Flatten()(x_s)
    return(split_flat)

def main_branch(input_tensor, filters, kernel_size, \
        stride, dropout):
    x = bn_block(input_tensor, filters[0], (7,7), stride, dropout)
    x = bn_block(x, filters[0], (3,3), stride, dropout)
    x = conv_block(x, filters, kernel_size, stride,\
            dropout)
    x = identity_block(x, filters, kernel_size)
    x = identity_block(x, filters, kernel_size)
    x = conv_block(x, filters*2, kernel_size, stride,\
            dropout)
    x = identity_block(x, filters*2, kernel_size)
    x = identity_block(x, filters*2, kernel_size)
    x = identity_block(x, filters*2, kernel_size)
    x = layers.AveragePooling2D((7,7))(x)
    main_flat = layers.Flatten()(x)
    return(main_flat)

def merged_net(input_shape, output_shape, n_classes, \
        filters, cropped_size=(31,31)):
    input_layer = layers.Input(shape=input_shape)
    # split branch
    split_flat = split_branch(input_layer, cropped_size=cropped_size, \
            filters=filters, kernel_size=(3,3), \
            stride=(2,2), dropout=0.15)
    # main branch
    main_flat = main_branch(input_layer, filters, kernel_size=(3,3),\
            stride=(2,2), dropout=0.15)
    ## flat merged
    merged_layers = layers.Concatenate()([main_flat, split_flat])
    output_layer = layers.Dense(units=n_classes, \
            activation="softmax")(merged_layers)
    model = models.Model(inputs=input_layer, outputs=output_layer)
    return(model)

#########################################################
#########################################################
#########################################################

def dense_block(input_tensor, filters, dropout, weight_decay):
    x = layers.BatchNormalization(gamma_regularizer=l2(weight_decay), \
            beta_regularizer=l2(weight_decay))(input_tensor)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=filters, kernel_size=(3,3), \
            padding='same', use_bias=False,
            kernel_regularizer=l2(weight_decay))(x)
    conv_1 = layers.Dropout(dropout)(x)
    x = layers.Concatenate()([input_tensor,conv_1])
    x = layers.BatchNormalization(gamma_regularizer=l2(weight_decay), \
            beta_regularizer=l2(weight_decay))(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=filters, kernel_size=(3,3), \
            padding='same', use_bias=False,
            kernel_regularizer=l2(weight_decay))(x)
    conv_2 = layers.Dropout(dropout)(x)
    output_tensor = layers.Concatenate()([input_tensor, conv_1, conv_2])
    return(output_tensor)

def transistion_block(input_tensor, filters, dropout, weight_decay):
    x = layers.BatchNormalization(gamma_regularizer=l2(weight_decay), \
            beta_regularizer=l2(weight_decay))(input_tensor)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=filters, kernel_size=(1,1), \
            padding='same', use_bias=False,
            kernel_regularizer=l2(weight_decay))(x)
    x = layers.Dropout(dropout)(x)
    output_tensor = layers.AveragePooling2D(pool_size=(2,2), \
            strides=(2,2))(x)
    return(output_tensor)

def dense_net(input_shape, output_shape, \
        dropout, cropped_size=(31,31), weight_decay=1E-4, \
        merged_model=False):
    input_layer = layers.Input(shape=input_shape)
    c = (((input_layer._keras_shape[1]-cropped_size[0])//2),\
            ((input_layer._keras_shape[2]-cropped_size[1])//2))
    crop_layer = layers.Cropping2D(c)(input_layer)
    init_conv = layers.Conv2D(filters=16, kernel_size=(1,1), \
            padding='same', use_bias=False,
            kernel_regularizer=l2(weight_decay))(crop_layer)
    #dense/transistion block 1
    dense = dense_block(init_conv, 12, dropout, weight_decay)
    trans = transistion_block(dense, 40, dropout, weight_decay)
    #dense/transistion block 2 
    dense = dense_block(trans, 12, dropout, weight_decay)
    trans = transistion_block(dense, 64, dropout, weight_decay)
    #dense/transistion block 3
    dense = dense_block(trans, 12, dropout, weight_decay)
    #trans = transistion_block(dense, 78, dropout, weight_decay)
    #dense block 4
    #dense = dense_block(trans, 12, dropout, weight_decay)
    x = layers.BatchNormalization(gamma_regularizer=l2(weight_decay), \
            beta_regularizer=l2(weight_decay))(dense)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    if merged_model:
        model = models.Model(inputs=input_layer, outputs=x)
    else: 
        output_layer = layers.Dense(units=output_shape, \
                activation="softmax", kernel_regularizer=l2(weight_decay), \
                bias_regularizer=l2(weight_decay))(x)
        model = models.Model(inputs=input_layer, outputs=output_layer)
    return(model)

def multi_dense_net(input_shape, output_shape, \
        dropout, cropped_size=(31,31), weight_decay=1E-4, \
        merged_model=False):
    input_layer = layers.Input(shape=input_shape)
    c = (((input_layer._keras_shape[1]-cropped_size[0])//2),\
            ((input_layer._keras_shape[2]-cropped_size[1])//2))
    crop_layer = layers.Cropping2D(c)(input_layer)
    init_conv = layers.Conv2D(filters=16, kernel_size=(1,1), \
            padding='same', use_bias=False,
            kernel_regularizer=l2(weight_decay))(crop_layer)
    #dense/transistion block 1
    dense = dense_block(init_conv, 12, dropout, weight_decay)
    trans = transistion_block(dense, 40, dropout, weight_decay)
    #dense/transistion block 2 
    dense = dense_block(trans, 12, dropout, weight_decay)
    # here we utilize the original image to train on also
    init_conv2 = layers.Conv2D(filters=16, kernel_size=(1,1), \
            padding='same', use_bias=False,
            kernel_regularizer=l2(weight_decay))(input_layer)
    dense2 = dense_block(init_conv2, 12, dropout, weight_decay)
    trans2 = transistion_block(dense2, 64, dropout, weight_decay)
    dense2 = dense_block(trans2, 12, dropout, weight_decay)
    x2 = layers.BatchNormalization(gamma_regularizer=l2(weight_decay), \
            beta_regularizer=l2(weight_decay))(dense2)
    x2 = layers.Activation('relu')(x2)
    x2 = layers.GlobalAveragePooling2D()(x2)
    #trans = transistion_block(dense, 78, dropout, weight_decay)
    #dense block 4
    #dense = dense_block(trans, 12, dropout, weight_decay)
    x = layers.BatchNormalization(gamma_regularizer=l2(weight_decay), \
            beta_regularizer=l2(weight_decay))(dense)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    #joining differing spatial views
    x = layers.Concatenate()([x,x2])
    if merged_model:
        model = models.Model(inputs=input_layer, outputs=x)
    else: 
        output_layer = layers.Dense(units=output_shape, \
                activation="softmax", kernel_regularizer=l2(weight_decay), \
                bias_regularizer=l2(weight_decay))(x)
        model = models.Model(inputs=input_layer, outputs=output_layer)
    return(model)


#########################################################
def ts_net(input_shape, output_shape, dropout, merged_model=False):
    input_layer = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 2, activation='relu')(input_layer)
    x = layers.Conv1D(64, 2, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Flatten()(x)
    if merged_model:
        model = models.Model(inputs=input_layer, outputs=x)
    else:
        output_layer = layers.Dense(units=output_shape, activation='softmax')(x)
        model = models.Model(inputs=input_layer, outputs=output_layer)
    return(model)

def m_net(input_shape_1, input_shape_2, output_shape, dropout, cropped_size):
    im_model = dense_net(input_shape_1, output_shape, dropout=dropout, \
            cropped_size=cropped_size, merged_model=True)
    ts_model = ts_net(input_shape_2, output_shape, dropout=dropout,\
            merged_model=True)
    in_a = layers.Input(shape=input_shape_1)
    in_b = layers.Input(shape=input_shape_2)
    out_a = im_model(in_a)
    out_b = ts_model(in_b)
    x = layers.Concatenate()([out_a, out_b])
    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    out = layers.Dense(output_shape, activation='softmax')(x)
    model = models.Model([in_a, in_b], out)
    return(model)
