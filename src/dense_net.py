from keras import models, layers
from keras import utils
from keras.regularizers import l2
import keras.backend as K

def dense_block(input_tensor, filters, dropout, weight_decay):
    x = layers.BatchNormalization (gamma_regularizer=l2(weight_decay),\
            beta_regularizer=l2(weight_decay))(input_tensor)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=filters, kernel_size=(3,3),\
            padding='same', use_bias=False,\
            kernel_regularizer=l2(weight_decay))(x)
    conv_1 = layers.Dropout(dropout)(x)
    x = layers.Concatenate()([input_tensor, conv_1])
    x = layers.BatchNormalization(gamma_regularizer=l2(weight_decay), \
            beta_regularizer=l2(weight_decay))(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=filters, kernel_size=(3,3),\
            padding='same', use_bias=False,\
            kernel_regularizer=l2(weight_decay))(x)
    conv_2 = layers.Dropout(dropout)(x)
    output_tensor = layers.Concatenate()([input_tensor, conv_1 ,conv_2])
    return(output_tensor)

def transistion_block(input_tensor, filters, dropout, weight_decay):
    x = layers.BatchNormalization(gamma_regularizer=l2(weight_decay),\
            beta_regularizer=l2(weight_decay))(input_tensor)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=filters, kernel_size=(1,1),\
            padding='same', use_bias=False,\
            kernel_regularizer=l2(weight_decay))(x)
    x = layers.Dropout(dropout)(x)
    output_tensor = layers.AveragePooling2D(pool_size=(2,2), \
            strides=(2,2))(x)
    return(output_tensor)

def dense_net(input_shape, output_shape, dropout, \
        n_dblocks=1, n_filter=[16,12], weight_decay=1E-4):
    input_layer = layers.Input(shape=input_shape)
    trans = layers.Conv2D(filters=n_filter[0], kernel_size=(1,1),\
            padding='same', use_bias=False, \
            kernel_regularizer=l2(weight_decay))(input_layer)
    for n in range(n_dblocks):
        t_filter = n_filter[0] * (2*(n+1)) 
        dense = dense_block(trans, n_filter[1], dropout, weight_decay)
        trans = transistion_block(dense, t_filter, dropout, weight_decay)
    dense = dense_block(trans, n_filter[1], dropout, weight_decay)
    x = layers.BatchNormalization(gamma_regularizer=l2(weight_decay),\
            beta_regularizer=l2(weight_decay))(dense)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    # here is where we can either output the flattened layer or go towards classification
    output_layer = layers.Dense(units=output_shape, \
            kernel_initializer='normal', activation='relu')(x) #this is of shape (nsamples, 1)
    model = models.Model(inputs=input_layer, outputs=output_layer)
    return(model)

def dense_stack(input_layer, n_final_filter, dropout, \
        n_dblocks=1, n_filter=[16,12], weight_decay=1E-4):
    trans = layers.Conv2D(filters=n_filter[0], kernel_size=(1,1),\
            padding='same', use_bias=False, \
            kernel_regularizer=l2(weight_decay))(input_layer)
    for n in range(n_dblocks):
        t_filter = n_filter[0] * (2*(n+1)) 
        dense = dense_block(trans, n_filter[1], dropout, weight_decay)
        trans = transistion_block(dense, t_filter, dropout, weight_decay)
    dense = dense_block(trans, n_filter[1], dropout, weight_decay)
    x = layers.BatchNormalization(gamma_regularizer=l2(weight_decay),\
            beta_regularizer=l2(weight_decay))(dense)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    # here is where we can either output the flattened layer or go towards classification
    output_layer = layers.Dense(units=n_final_filter, \
            kernel_initializer='normal', activation='relu')(x) #this is of shape (nsamples, 1)
    return(output_layer)

