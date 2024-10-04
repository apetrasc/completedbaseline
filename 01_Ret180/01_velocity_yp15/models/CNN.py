import tensorflow as tf
from tensorflow.keras import layers
import sys
import os
from src.activations import thres_relu
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import config_deep as config
app = config.NN_WallRecon

nx = 432
nz = 432 

@tf.function
def configure_padding():
    if app.NET_MODEL == 1:
        # 'same' または 'valid' に設定
        padding = 'same'
        pad_out = 2
        padding_in = 64
        padding_out = 0
    else:
        raise ValueError('NET_MODEL = 1 is the only one implemented so far')
    
    return padding

#padding = configure_padding()
input_shape = (app.N_VARS_IN, nz, nx)
pred_fluct = app.FLUCTUATIONS_PRED

padding = 'same'
pad_out = 2
padding_in = 64
padding_out = 0
'''
def cnn_model():
    input_data = layers.Input(shape=input_shape, name='input_data')
    ini = 30
    dx = 4
    # ------------------------------------------------------------------
    cnv_1 = layers.Conv2D(ini, (3, 3), padding=padding,
                                data_format='channels_first')(input_data)
    bch_1 = layers.BatchNormalization(axis=1)(cnv_1)
    act_1 = layers.Activation('relu')(bch_1)
    # ------------------------------------------------------------------
    cnv_2 = layers.Conv2D(ini+dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_1)
    bch_2 = layers.BatchNormalization(axis=1)(cnv_2)
    act_2 = layers.Activation('relu')(bch_2)
    # ------------------------------------------------------------------
    cnv_3 = layers.Conv2D(ini+2*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_2)
    bch_3 = layers.BatchNormalization(axis=1)(cnv_3)
    act_3 = layers.Activation('relu')(bch_3)
    # ------------------------------------------------------------------
    cnv_4 = layers.Conv2D(ini+3*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_3)
    bch_4 = layers.BatchNormalization(axis=1)(cnv_4)
    act_4 = layers.Activation('relu')(bch_4)
    # ------------------------------------------------------------------
    cnv_5 = layers.Conv2D(ini+4*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_4)
    bch_5 = layers.BatchNormalization(axis=1)(cnv_5)
    act_5 = layers.Activation('relu')(bch_5)
    # ------------------------------------------------------------------
    cnv_6 = layers.Conv2D(ini+5*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_5)
    bch_6 = layers.BatchNormalization(axis=1)(cnv_6)
    act_6 = layers.Activation('relu')(bch_6)
    # ------------------------------------------------------------------
    cnv_7 = layers.Conv2D(ini+6*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_6)
    bch_7 = layers.BatchNormalization(axis=1)(cnv_7)
    act_7 = layers.Activation('relu')(bch_7)
    # ------------------------------------------------------------------
    cnv_8 = layers.Conv2D(ini+7*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_7)
    bch_8 = layers.BatchNormalization(axis=1)(cnv_8)
    act_8 = layers.Activation('relu')(bch_8)
    # ------------------------------------------------------------------
    cnv_9 = layers.Conv2D(ini+8*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_8)
    bch_9 = layers.BatchNormalization(axis=1)(cnv_9)
    act_9 = layers.Activation('relu')(bch_9)
    # ------------------------------------------------------------------
    cnv_10 = layers.Conv2D(ini+9*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_9)
    bch_10 = layers.BatchNormalization(axis=1)(cnv_10)
    act_10 = layers.Activation('relu')(bch_10)
    # ------------------------------------------------------------------
    cnv_11 = layers.Conv2D(ini+10*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_10)
    bch_11 = layers.BatchNormalization(axis=1)(cnv_11)
    act_11 = layers.Activation('relu')(bch_11)
    # ------------------------------------------------------------------
    cnv_12 = layers.Conv2D(ini+11*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_11)
    bch_12 = layers.BatchNormalization(axis=1)(cnv_12)
    act_12 = layers.Activation('relu')(bch_12)
    # ------------------------------------------------------------------
    cnv_13 = layers.Conv2D(ini+12*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_12)
    bch_13 = layers.BatchNormalization(axis=1)(cnv_13)
    act_13 = layers.Activation('relu')(bch_13)
    # ------------------------------------------------------------------
    cnv_14 = layers.Conv2D(ini+13*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_13)
    bch_14 = layers.BatchNormalization(axis=1)(cnv_14)
    act_14 = layers.Activation('relu')(bch_14)
    # ------------------------------------------------------------------
    cnv_15 = layers.Conv2D(ini+14*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_14)
    bch_15 = layers.BatchNormalization(axis=1)(cnv_15)
    act_15 = layers.Activation('relu')(bch_15)
    # ------------------------------------------------------------------
    cnv_16 = layers.Conv2D(ini+14*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_15)
    bch_16 = layers.BatchNormalization(axis=1)(cnv_16)
    act_16 = layers.Activation('relu')(bch_16)
    # ------------------------------------------------------------------
    cnv_17 = layers.Conv2D(ini+13*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_16)
    bch_17 = layers.BatchNormalization(axis=1)(cnv_17)
    act_17 = layers.Activation('relu')(bch_17)
    # ------------------------------------------------------------------
    cnv_18 = layers.Conv2D(ini+12*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_17)
    bch_18 = layers.BatchNormalization(axis=1)(cnv_18)
    act_18 = layers.Activation('relu')(bch_18)
    # ------------------------------------------------------------------
    cnv_19 = layers.Conv2D(ini+11*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_18)
    bch_19 = layers.BatchNormalization(axis=1)(cnv_19)
    act_19 = layers.Activation('relu')(bch_19)
    # ------------------------------------------------------------------
    cnv_20 = layers.Conv2D(ini+10*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_19)
    bch_20 = layers.BatchNormalization(axis=1)(cnv_20)
    act_20 = layers.Activation('relu')(bch_20)
    # ------------------------------------------------------------------
    cnv_21 = layers.Conv2D(ini+9*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_20)
    bch_21 = layers.BatchNormalization(axis=1)(cnv_21)
    act_21 = layers.Activation('relu')(bch_21)
    # ------------------------------------------------------------------
    cnv_22 = layers.Conv2D(ini+8*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_21)
    bch_22 = layers.BatchNormalization(axis=1)(cnv_22)
    act_22 = layers.Activation('relu')(bch_22)
    # ------------------------------------------------------------------
    cnv_23 = layers.Conv2D(ini+7*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_22)
    bch_23 = layers.BatchNormalization(axis=1)(cnv_23)
    act_23 = layers.Activation('relu')(bch_23)
    # ------------------------------------------------------------------
    cnv_24 = layers.Conv2D(ini+6*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_23)
    bch_24 = layers.BatchNormalization(axis=1)(cnv_24)
    act_24 = layers.Activation('relu')(bch_24)
    # ------------------------------------------------------------------
    cnv_25 = layers.Conv2D(ini+5*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_24)
    bch_25 = layers.BatchNormalization(axis=1)(cnv_25)
    act_25 = layers.Activation('relu')(bch_25)
    # ------------------------------------------------------------------
    cnv_26 = layers.Conv2D(ini+4*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_25)
    bch_26 = layers.BatchNormalization(axis=1)(cnv_26)
    act_26 = layers.Activation('relu')(bch_26)
    # ------------------------------------------------------------------
    cnv_27 = layers.Conv2D(ini+3*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_26)
    bch_27 = layers.BatchNormalization(axis=1)(cnv_27)
    act_27 = layers.Activation('relu')(bch_27)
    # ------------------------------------------------------------------
    cnv_28 = layers.Conv2D(ini+2*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_27)
    bch_28 = layers.BatchNormalization(axis=1)(cnv_28)
    act_28 = layers.Activation('relu')(bch_28)
    # ------------------------------------------------------------------
    cnv_29 = layers.Conv2D(ini+dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_28)
    bch_29 = layers.BatchNormalization(axis=1)(cnv_29)
    act_29 = layers.Activation('relu')(bch_29)
    # ------------------------------------------------------------------
    cnv_30 = layers.Conv2D(ini, (3, 3), padding=padding,
                                data_format='channels_first')(act_29)
    bch_30 = layers.BatchNormalization(axis=1)(cnv_30)
    act_30 = layers.Activation('relu')(bch_30)


    # Different branches for different components
    
    # Branch 1
    cnv_b1 = layers.Conv2D(1, (3, 3), padding=padding,
                                data_format='channels_first')(act_30)
    if pred_fluct == True:
        act_b1 = layers.ReLU(threshold=app.RELU_THRESHOLD)(cnv_b1)
        output_b1 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b1')(act_b1)
    else:
        act_b1 = layers.Activation('relu')(cnv_b1)
        output_b1 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b1')(act_b1)

    losses = {'output_b1':'mse'}
    
    if app.N_VARS_OUT == 2:
        # Branch 2
        cnv_b2 = layers.Conv2D(1, (3, 3), padding=padding,
                                data_format='channels_first')(act_5)
        if pred_fluct == True:
            act_b2 = layers.ReLU(threshold=app.RELU_THRESHOLD)(cnv_b2)
            output_b2 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b2')(act_b2)
        else:
            act_b2 = layers.Activation('relu')(cnv_b2)
            output_b2 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b2')(act_b2)

        outputs_model = [output_b1, output_b2]

        losses['output_b2']='mse'
    
    elif app.N_VARS_OUT == 3:
        # Branch 2
        cnv_b2 = layers.Conv2D(1, (3, 3), padding=padding,
                                data_format='channels_first')(act_30)
        if pred_fluct == True:
            act_b2 = layers.ReLU(threshold=app.RELU_THRESHOLD)(cnv_b2)
            output_b2 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b2')(act_b2)
        else:
            act_b2 = layers.Activation('relu')(cnv_b2)
            output_b2 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b2')(act_b2)
    
        losses['output_b2']='mse'

        # Branch 3
        cnv_b3 = layers.Conv2D(1, (3, 3), padding=padding,
                                data_format='channels_first')(act_30)
        if pred_fluct == True:
            act_b3 = layers.ReLU(threshold=app.RELU_THRESHOLD)(cnv_b3)
            output_b3 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b3')(act_b3)
        else:
            act_b3 = layers.Activation('relu')(cnv_b3)
            output_b3 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b3')(act_b3)
    
        outputs_model = [output_b1, output_b2, output_b3]

        losses['output_b3']='mse'
    
    else:
        outputs_model = output_b1
    
    CNN_model = tf.keras.models.Model(inputs=input_data, outputs=outputs_model)
    return CNN_model, losses
    '''
def cnn_model():
    input_data = layers.Input(shape=input_shape, name='input_data')
    ini = 30
    dx = 4
    # ------------------------------------------------------------------
    cnv_1 = layers.Conv2D(ini, (3, 3), padding=padding,
                                data_format='channels_first')(input_data)
    bch_1 = layers.BatchNormalization(axis=1)(cnv_1)
    act_1 = layers.Activation('relu')(bch_1)
    # ------------------------------------------------------------------
    cnv_2 = layers.Conv2D(ini+dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_1)
    bch_2 = layers.BatchNormalization(axis=1)(cnv_2)
    act_2 = layers.Activation('relu')(bch_2)
    # ------------------------------------------------------------------
    cnv_3 = layers.Conv2D(ini+2*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_2)
    bch_3 = layers.BatchNormalization(axis=1)(cnv_3)
    act_3 = layers.Activation('relu')(bch_3)
    # ------------------------------------------------------------------
    cnv_4 = layers.Conv2D(ini+3*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_3)
    bch_4 = layers.BatchNormalization(axis=1)(cnv_4)
    act_4 = layers.Activation('relu')(bch_4)
    # ------------------------------------------------------------------
    cnv_5 = layers.Conv2D(ini+4*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_4)
    bch_5 = layers.BatchNormalization(axis=1)(cnv_5)
    act_5 = layers.Activation('relu')(bch_5)
    # ------------------------------------------------------------------
    cnv_6 = layers.Conv2D(ini+5*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_5)
    bch_6 = layers.BatchNormalization(axis=1)(cnv_6)
    act_6 = layers.Activation('relu')(bch_6)
    # ------------------------------------------------------------------
    cnv_7 = layers.Conv2D(ini+6*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_6)
    bch_7 = layers.BatchNormalization(axis=1)(cnv_7)
    act_7 = layers.Activation('relu')(bch_7)
    # ------------------------------------------------------------------
    cnv_8 = layers.Conv2D(ini+7*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_7)
    bch_8 = layers.BatchNormalization(axis=1)(cnv_8)
    act_8 = layers.Activation('relu')(bch_8)
    # ------------------------------------------------------------------
    cnv_9 = layers.Conv2D(ini+8*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_8)
    bch_9 = layers.BatchNormalization(axis=1)(cnv_9)
    act_9 = layers.Activation('relu')(bch_9)
    # ------------------------------------------------------------------
    cnv_10 = layers.Conv2D(ini+9*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_9)
    bch_10 = layers.BatchNormalization(axis=1)(cnv_10)
    act_10 = layers.Activation('relu')(bch_10)
    # ------------------------------------------------------------------
    cnv_11 = layers.Conv2D(ini+10*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_10)
    bch_11 = layers.BatchNormalization(axis=1)(cnv_11)
    act_11 = layers.Activation('relu')(bch_11)
    # ------------------------------------------------------------------
    cnv_12 = layers.Conv2D(ini+11*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_11)
    bch_12 = layers.BatchNormalization(axis=1)(cnv_12)
    act_12 = layers.Activation('relu')(bch_12)
    # ------------------------------------------------------------------
    cnv_13 = layers.Conv2D(ini+12*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_12)
    bch_13 = layers.BatchNormalization(axis=1)(cnv_13)
    act_13 = layers.Activation('relu')(bch_13)
    # ------------------------------------------------------------------
    cnv_14 = layers.Conv2D(ini+13*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_13)
    bch_14 = layers.BatchNormalization(axis=1)(cnv_14)
    act_14 = layers.Activation('relu')(bch_14)
    # ------------------------------------------------------------------
    cnv_15 = layers.Conv2D(ini+14*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_14)
    bch_15 = layers.BatchNormalization(axis=1)(cnv_15)
    act_15 = layers.Activation('relu')(bch_15)
    # ------------------------------------------------------------------
    cnv_16 = layers.Conv2D(ini+14*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_15)
    bch_16 = layers.BatchNormalization(axis=1)(cnv_16)
    act_16 = layers.Activation('relu')(bch_16)
    # ------------------------------------------------------------------
    cnv_17 = layers.Conv2D(ini+13*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_16)
    bch_17 = layers.BatchNormalization(axis=1)(cnv_17)
    act_17 = layers.Activation('relu')(bch_17)
    # ------------------------------------------------------------------
    cnv_18 = layers.Conv2D(ini+12*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_17)
    bch_18 = layers.BatchNormalization(axis=1)(cnv_18)
    act_18 = layers.Activation('relu')(bch_18)
    # ------------------------------------------------------------------
    cnv_19 = layers.Conv2D(ini+11*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_18)
    bch_19 = layers.BatchNormalization(axis=1)(cnv_19)
    act_19 = layers.Activation('relu')(bch_19)
    # ------------------------------------------------------------------
    cnv_20 = layers.Conv2D(ini+10*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_19)
    bch_20 = layers.BatchNormalization(axis=1)(cnv_20)
    act_20 = layers.Activation('relu')(bch_20)
    # ------------------------------------------------------------------
    cnv_21 = layers.Conv2D(ini+9*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_20)
    bch_21 = layers.BatchNormalization(axis=1)(cnv_21)
    act_21 = layers.Activation('relu')(bch_21)
    # ------------------------------------------------------------------
    cnv_22 = layers.Conv2D(ini+8*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_21)
    bch_22 = layers.BatchNormalization(axis=1)(cnv_22)
    act_22 = layers.Activation('relu')(bch_22)
    # ------------------------------------------------------------------
    cnv_23 = layers.Conv2D(ini+7*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_22)
    bch_23 = layers.BatchNormalization(axis=1)(cnv_23)
    act_23 = layers.Activation('relu')(bch_23)
    # ------------------------------------------------------------------
    cnv_24 = layers.Conv2D(ini+6*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_23)
    bch_24 = layers.BatchNormalization(axis=1)(cnv_24)
    act_24 = layers.Activation('relu')(bch_24)
    # ------------------------------------------------------------------
    cnv_25 = layers.Conv2D(ini+5*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_24)
    bch_25 = layers.BatchNormalization(axis=1)(cnv_25)
    act_25 = layers.Activation('relu')(bch_25)
    # ------------------------------------------------------------------
    cnv_26 = layers.Conv2D(ini+4*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_25)
    bch_26 = layers.BatchNormalization(axis=1)(cnv_26)
    act_26 = layers.Activation('relu')(bch_26)
    # ------------------------------------------------------------------
    cnv_27 = layers.Conv2D(ini+3*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_26)
    bch_27 = layers.BatchNormalization(axis=1)(cnv_27)
    act_27 = layers.Activation('relu')(bch_27)
    # ------------------------------------------------------------------
    cnv_28 = layers.Conv2D(ini+2*dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_27)
    bch_28 = layers.BatchNormalization(axis=1)(cnv_28)
    act_28 = layers.Activation('relu')(bch_28)
    # ------------------------------------------------------------------
    cnv_29 = layers.Conv2D(ini+dx, (3, 3), padding=padding,
                                data_format='channels_first')(act_28)
    bch_29 = layers.BatchNormalization(axis=1)(cnv_29)
    act_29 = layers.Activation('relu')(bch_29)
    # ------------------------------------------------------------------
    cnv_30 = layers.Conv2D(ini, (3, 3), padding=padding,
                                data_format='channels_first')(act_29)
    bch_30 = layers.BatchNormalization(axis=1)(cnv_30)
    act_30 = layers.Activation('relu')(bch_30)


    # Different branches for different components
    
    # Branch 1
    cnv_b1 = layers.Conv2D(1, (3, 3), padding=padding,
                                data_format='channels_first')(act_30)
    if pred_fluct == True:
        act_b1 = layers.Lambda(lambda x: thres_relu(x,app))(cnv_b1)
        output_b1 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b1')(act_b1)
    else:
        act_b1 = layers.Activation('relu')(cnv_b1)
        output_b1 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b1')(act_b1)

    losses = {'output_b1':'mse'}
    
    if app.N_VARS_OUT == 2:
        # Branch 2
        cnv_b2 = layers.Conv2D(1, (3, 3), padding=padding,
                                data_format='channels_first')(act_5)
        if pred_fluct == True:
            act_b2 = layers.Lambda(lambda x: thres_relu(x,app))(cnv_b2)
            output_b2 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b2')(act_b2)
        else:
            act_b2 = layers.Activation('relu')(cnv_b2)
            output_b2 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b2')(act_b2)

        outputs_model = [output_b1, output_b2]

        losses['output_b2']='mse'
    
    elif app.N_VARS_OUT == 3:
        # Branch 2
        cnv_b2 = layers.Conv2D(1, (3, 3), padding=padding,
                                data_format='channels_first')(act_30)
        if pred_fluct == True:
            act_b2 = layers.Lambda(lambda x: thres_relu(x,app))(cnv_b2)
            output_b2 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b2')(act_b2)
        else:
            act_b2 = layers.Activation('relu')(cnv_b2)
            output_b2 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b2')(act_b2)
    
        losses['output_b2']='mse'

        # Branch 3
        cnv_b3 = layers.Conv2D(1, (3, 3), padding=padding,
                                data_format='channels_first')(act_30)
        if pred_fluct == True:
            act_b3 = layers.Lambda(lambda x: thres_relu(x,app))(cnv_b3)
            output_b3 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b3')(act_b3)
        else:
            act_b3 = layers.Activation('relu')(cnv_b3)
            output_b3 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b3')(act_b3)
    
        outputs_model = [output_b1, output_b2, output_b3]

        losses['output_b3']='mse'
    
    else:
        outputs_model = output_b1
    
    CNN_model = tf.keras.models.Model(inputs=input_data, outputs=outputs_model)
    return CNN_model, losses