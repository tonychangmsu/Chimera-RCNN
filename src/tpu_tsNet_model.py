from tensorflow.keras import models, layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, BatchNormalization, Dropout, \
        Flatten, Dense, Concatenate, Input
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import LSTM, GRU
from src.tpu_dense_net import dense_stack
from tensorflow.keras import backend as K

K.set_image_data_format('channels_last')

def rmse(y_true, y_pred):
    return(K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)))

def det_coeff(y_true, y_pred):
    u = K.sum(K.square(y_true - y_pred))
    v = K.sum(K.square(y_true - K.mean(y_true)))
    return(K.ones_like(v) - (u / v))

def td_conv_blk(input_layer, filters):
    '''
    Standard 2D convolutions, no batch normalization or dropout;
    May add later when applying noise
    '''
    x = TimeDistributed(Conv2D(filters, (3,3), activation='relu',padding= 'same'))(input_layer)
    x = TimeDistributed(Conv2D(filters, (3,3), activation='relu',padding= 'same'))(x)
    x = TimeDistributed(MaxPooling2D((2,2), strides=(2,2)))(x)
    return(x) 

def geotsNet(input_shape, output_shape, model_type='merged'):
    '''
    Prototype the Geographic Time-Space network
    if <model_out> parameter is 'classify', use a 'categorical_crossentropy' loss,
        and input only a single y_train output.
    if <model_out> parameter is 'regress', use a 'mean_squared_error' loss
        and input only a single y_train output.
    otherwise, use 'merged' and apply to the y_train list with both loss functions
    '''
    input_layer0 = Input(shape=input_shape[0], name='naip_input') #naip
    input_layer1 = Input(shape=input_shape[1], name='terrain_input') #terrain
    input_layer2 = Input(shape=input_shape[2], name='climate_input') #climate
    input_layer3 = Input(shape=input_shape[3], name='landsat_input') #landsat

    #only naip and landsat will undergo convolutions
    #naip will be convolution only
    #landsat will be convolved and then undergo an LSTM
    #terrain will be a simple MLP that gets concatenated to the end
    #climate is one dimensional, but we can convolve the channels?
    dense0 = dense_stack(input_layer0, n_final_filter=256, \
            dropout=0.1, n_dblocks=5)

    #terrain layer
    x1 = Flatten()(input_layer1)
    dense1 = Dense(16)(x1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
   
    #climate layer
    #x2 = TimeDistributed(Conv2D(64, (1,1), activation='relu',padding= 'same'))(input_layer2)
    #x2 = TimeDistributed(layers.Flatten())(x2)
    #x2 = Dropout(0.1)(x2)

    #rnn_cell0 = LSTM(units=256, return_sequences=False, dropout=0.25)(x2)
    x2 = Flatten()(input_layer2)
    dense2 = Dense(64)(x2)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
   
    #landsat layer 
    input_layer3 = Input(shape=input_shape[3], name='landsat_input') #landsat
    x3 = TimeDistributed(Conv2D(64, (3,3), activation='relu',padding= 'same'))(input_layer3)
    x3 = TimeDistributed(Conv2D(128, (3,3), activation='relu',padding= 'same'))(x3)
    # x3 = TimeDistributed(Conv2D(128, (3,3), activation='relu',padding= 'same'))(x3)
    x3 = TimeDistributed(layers.Flatten())(x3)
    x3 = Dropout(0.1, name='split_branch')(x3)

    #RNN here.
    #rnn for classification and regression
    rnn_cell1 = LSTM(units=64, return_sequences=False, dropout=0.2)(x3)
    dense3 = Dense(64)(rnn_cell1) 

    aux = Concatenate(name='aux_merge')([dense1, dense2])
    aux = Dense(64)(aux)
    #classification fully connected layer (5 outputs)
    full0 = Concatenate(name='merge0')([dense0, dense3])
    full0 = Dense(256)(full0)
    full0 = Dense(128)(full0)
    full0 = Dense(64)(full0)
    full0 = Dense(output_shape[0])(full0)
    full0 = Activation('softmax', \
            name='fc_class_layer')(full0)

    #regression fully connected layer (4 outputs)
    full1 = Concatenate(name='merge1')([dense0, aux])
    full1 = Dense(512)(full1)
    full1 = Dense(256)(full1)
    full1 = Dense(128)(full1)
    full1 = Dense(output_shape[1])(full1)
    full1 = Activation('relu', \
            name='fc_regress_layer')(full1)
    model_inputs = [input_layer0, input_layer1, input_layer2, input_layer3]
    if model_type == 'merged':
        geotsnet = models.Model(inputs=model_inputs,\
                outputs=[full0, full1])
    if model_type == 'regress':
        geotsnet = models.Model(inputs=model_inputs, outputs=full1)
    if model_type == 'classification':
        geotsnet = models.Model(inputs=model_inputs, outputs=full0)
    return(geotsnet)

def geotsNet_full(input_shape, output_shape, model_type='merged'):
    '''
    Prototype the Geographic Time-Space network
    if <model_out> parameter is 'classify', use a 'categorical_crossentropy' loss,
        and input only a single y_train output.
    if <model_out> parameter is 'regress', use a 'mean_squared_error' loss
        and input only a single y_train output.
    otherwise, use 'merged' and apply to the y_train list with both loss functions
    '''
    input_layer0 = Input(shape=input_shape[0], name='naip_input') #naip
    input_layer1 = Input(shape=input_shape[1], name='terrain_input') #terrain
    input_layer2 = Input(shape=input_shape[2], name='climate_input') #climate
    input_layer3 = Input(shape=input_shape[3], name='landsat_input') #landsat

    #only naip and landsat will undergo convolutions
    #naip will be convolution only
    #landsat will be convolved and then undergo an LSTM
    #terrain will be a simple MLP that gets concatenated to the end
    #climate is one dimensional, but we can convolve the channels?
    dense0 = dense_stack(input_layer0, n_final_filter=256, \
            dropout=0.1, n_dblocks=5)

    #terrain layer
    x1 = Flatten()(input_layer1)
    dense1 = Dense(16)(x1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
   
    #climate layer
    x2 = Flatten()(input_layer2)
    dense2 = Dense(64)(x2)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
   
    #landsat layer 
    input_layer3 = Input(shape=input_shape[3], name='landsat_input') #landsat
    x3 = TimeDistributed(Conv2D(64, (3,3), activation='relu',padding= 'same'))(input_layer3)
    x3 = TimeDistributed(Conv2D(64, (3,3), activation='relu',padding= 'same'))(x3)
    x3 = TimeDistributed(Conv2D(128, (3,3), activation='relu',padding= 'same'))(x3)
    x3 = TimeDistributed(layers.Flatten())(x3)
    x3 = Dropout(0.1, name='split_branch')(x3)

    #RNN here.
    #rnn for classification and regression
    rnn_cell1 = LSTM(units=64, return_sequences=False, dropout=0.2)(x3)
    dense3 = Dense(64)(rnn_cell1) 

    aux = Concatenate(name='aux_merge')([dense1, dense2])
    aux = Dense(64)(aux)
    #classification fully connected layer (5 outputs)
    full0 = Concatenate(name='merge0')([dense0, dense3, aux])
    full0 = Dense(256)(full0)
    full0 = Dense(128)(full0)
    full0 = Dense(64)(full0)
    full0 = Dense(output_shape[0])(full0)
    full0 = Activation('softmax', \
            name='fc_class_layer')(full0)

    #regression fully connected layer (4 outputs)
    full1 = Concatenate(name='merge1')([dense0, dense3, aux])
    full1 = Dense(512)(full1)
    full1 = Dense(256)(full1)
    full1 = Dense(128)(full1)
    full1 = Dense(output_shape[1])(full1)
    full1 = Activation('relu', \
            name='fc_regress_layer')(full1)
    model_inputs = [input_layer0, input_layer1, input_layer2, input_layer3]
    if model_type == 'merged':
        geotsnet = models.Model(inputs=model_inputs,\
                outputs=[full0, full1])
    if model_type == 'regress':
        geotsnet = models.Model(inputs=model_inputs, outputs=full1)
    if model_type == 'classification':
        geotsnet = models.Model(inputs=model_inputs, outputs=full0)
    return(geotsnet)


def geotsNet_threepred(input_shape, output_shape):
    '''
    Prototype the Geographic Time-Space network
    if <model_out> parameter is 'classify', use a 'categorical_crossentropy' loss,
        and input only a single y_train output.
    if <model_out> parameter is 'regress', use a 'mean_squared_error' loss
        and input only a single y_train output.
    otherwise, use 'merged' and apply to the y_train list with both loss functions
    '''
    input_layer0 = Input(shape=input_shape[0], name='naip_input') #naip
    input_layer1 = Input(shape=input_shape[1], name='terrain_input') #terrain
    input_layer2 = Input(shape=input_shape[2], name='climate_input') #climate

    dense0 = dense_stack(input_layer0, n_final_filter=256, \
            dropout=0.1, n_dblocks=5)

    #terrain layer
    x1 = Flatten()(input_layer1)
    dense1 = Dense(16)(x1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
 
    #climate layer
    x2 = Flatten()(input_layer2)
    dense2 = Dense(64)(x2)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
  
    aux = Concatenate(name='aux_merge')([dense1, dense2])
    aux = Dense(64)(aux)

    #classification fully connected layer (5 outputs)
    #full0 = Concatenate(name='merge0')([dense0, aux])
    #full0 = Dense(256)(full0)
    full0 = Dense(256)(dense0)
    full0 = Dense(128)(full0)
    full0 = Dense(64)(full0)
    full0 = Dense(output_shape[0])(full0)
    full0 = Activation('softmax', \
            name='fc_class_layer')(full0)

    #regression fully connected layer (4 outputs)
    full1 = Concatenate(name='merge1')([dense0, aux])
    full1 = Dense(512)(full1)
    full1 = Dense(256)(full1)
    full1 = Dense(128)(full1)
    full1 = Dense(output_shape[1])(full1)
    full1 = Activation('relu', \
            name='fc_regress_layer')(full1)
    model_inputs = [input_layer0, input_layer1, input_layer2]
    geotsnet = models.Model(inputs=model_inputs,\
            outputs=[full0, full1])
    return(geotsnet)

def geotsNet_simple(input_shape, output_shape):
    '''
    Prototype the Geographic Time-Space network
    for naip only
    '''
    #input_layer0 = Input(shape=input_shape[0], name='naip_input') #naip
    #try simple with just imagery.
    input_layer0 = Input(shape=input_shape[0], name='naip_input') #naip
    #dense0 = dense_stack(input_layer0, n_final_filter=256, \
    #        dropout=0.25, n_dblocks=3)
    dense1 = dense_stack(input_layer0, n_final_filter=256, \
            dropout=0.1, n_dblocks=4)
    full1 = Dense(256)(dense1)
    full1 = Dense(128)(full1)
    full1 = Dense(64)(full1)
    full1 = Dense(output_shape[0])(full1)
    final1 = Activation('softmax', name='fc_class_layer')(full1)
    #dense2 = dense_stack(input_layer0, n_final_filter=512, \
    #        dropout=0.1, n_dblocks=4)
    full2 = Dense(256)(dense1)
    full2 = Dense(128)(full2)
    full2 = Dense(64)(full2)
    full2 = Dense(output_shape[1])(full2)
    final2= Activation('relu', name='fc_regress_layer')(full2)
    model_inputs = input_layer0
    out = models.Model(inputs=model_inputs, outputs=[final1,final2])
    return(out)

def geotsNet_imagery(input_shape, output_shape):
    '''
    Prototype the Geographic Time-Space network
    for naip and landsat combined for better classification
    '''
    #naip layer 
    input_layer0 = Input(shape=input_shape[0], name='naip_input') #naip
    dense0 = dense_stack(input_layer0, n_final_filter=256, \
            dropout=0.1, n_dblocks=5)
    #dense1 will split into two fully connected layers

    #landsat layer 
    input_layer1 = Input(shape=input_shape[1], name='landsat_input') #landsat
    x1 = TimeDistributed(Conv2D(64, (3,3), activation='relu',padding= 'same'))(input_layer1)
    x1 = TimeDistributed(Conv2D(128, (3,3), activation='relu',padding= 'same'))(x1)
    x1 = TimeDistributed(Conv2D(128, (3,3), activation='relu',padding= 'same'))(x1)
    x1 = TimeDistributed(layers.Flatten())(x1)
    x1 = Dropout(0.1, name='split_branch')(x1)
    #RNN here.
    #rnn for classification and regression
    rnn_cell0 = LSTM(units=64, return_sequences=False, dropout=0.2)(x1)
    dense1 = Dense(64)(rnn_cell0) 
    #dense2 will also split into two fully connected layers

    full0 = Concatenate(name='merge0')([dense0, dense1])
    full0 = Dense(256)(full0)
    full0 = Dense(128)(full0)
    full0 = Dense(128)(full0)
    full0 = Dense(output_shape[0])(full0)

    full1 = Concatenate(name='merge1')([dense0, dense1])
    full1 = Dense(256)(full1)
    full1 = Dense(128)(full1)
    full1 = Dense(128)(full1)
    full1 = Dense(output_shape[1])(full1)

    final0 = Activation('softmax', name='fc_class_layer')(full0)
    final1= Activation('relu', name='fc_regress_layer')(full1)
    model_inputs = [input_layer0, input_layer1]
    out = models.Model(inputs=model_inputs, outputs=[final0,final1])
    return(out)

def tsNet(input_shape, filters=[32,256], n_blks=4, output_shape=[3, 2],\
        model_out='merged'):
    '''
    Prototype Time-Space network
    if <model_out> parameter is 'classify', use a 'categorical_crossentropy' loss,
        and input only a single y_train output.
    if <model_out> parameter is 'regress', use a 'mean_squared_error' loss
        and input only a single y_train output.
    otherwise, use 'merged' and apply to the y_train list with both loss functions
    '''
    input_layer = layers.Input(shape=input_shape)
    x = TimeDistributed(Conv2D(filters[0], (3,3), activation='relu',padding= 'same'))(input_layer)
    x = TimeDistributed(Conv2D(filters[0], (3,3), activation='relu',padding= 'same'))(x)
    x = TimeDistributed(MaxPooling2D((2,2), strides=(2,2)))(x)
    for i in range(n_blks):
        x = td_conv_blk(x, filters[0]*(2**(i+1)))
    x = TimeDistributed(layers.Flatten())(x)
    split_layer = layers.Dropout(0.5)(x)
    #return here as a branch for the two models
    #split to two different types of outputs
    rnn_cell1 = LSTM(units=filters[1], return_sequences=True, dropout=0.5)(split_layer)
    rnn_cell1 = LSTM(units=filters[1], return_sequences=False, dropout=0.5)(rnn_cell1)
    dense1 = layers.Dense(output_shape[0], activation='softmax')(rnn_cell1)
    #return sequences allows connection from one RNN to another. 
    rnn_cell2 = LSTM(units=filters[1], return_sequences=True, dropout=0.5)(split_layer) 
    rnn_cell2 = LSTM(units=filters[1], return_sequences=False, dropout=0.5)(rnn_cell2)
    dense2 = layers.Dense(output_shape[1])(rnn_cell2) 
    if model_out == 'classify':
        tsnet = models.Model(inputs=input_layer, outputs=dense1)
    elif model_out == 'regress':
        tsnet = models.Model(inputs=input_layer, outputs=dense2)
    else:
        tsnet = models.Model(inputs=input_layer, outputs=[dense1, dense2])
    return(tsnet)
