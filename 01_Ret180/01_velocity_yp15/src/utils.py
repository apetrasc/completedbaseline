import tensorflow as tf
import config_deep as config
import os
import math
prb_def = os.environ.get('MODEL_CNN', None)
app = config.NN_WallRecon
init_lr = app.INIT_LR
lr_drop = app.LR_DROP
lr_epdrop = app.LR_EPDROP
@tf.function
def periodic_padding(tensor, padding):
    """
    Tensorflow function to pad periodically a 2D tensor

    Parameters
    ----------
    tensor : 2D tf.Tensor
        Tensor to be padded
    padding : integer values
        Padding value, same in all directions

    Returns
    -------
    Padded tensor
    """
    lower_pad = tensor[:padding[0][0], :]
    upper_pad = tensor[-padding[0][1]:, :]

    partial_tensor = tf.concat([upper_pad, tensor, lower_pad], axis=0)

    left_pad = partial_tensor[:, -padding[1][0]:]
    right_pad = partial_tensor[:, :padding[1][1]]

    padded_tensor = tf.concat([left_pad, partial_tensor, right_pad], axis=1)

    return padded_tensor

@tf.function
def periodic_padding_z(tensor, padding):
    """
    Tensorflow function to pad periodically a 2D tensor

    Parameters
    ----------
    tensor : 2D tf.Tensor
        Tensor to be padded
    padding : integer values
        Padding value, same in all directions

    Returns
    -------
    Padded tensor

    """
    lower_pad = tensor[:padding[0][0],:]
    upper_pad = tensor[-padding[0][1]:,:]
    
    padded_tensor = tf.concat([upper_pad, tensor, lower_pad], axis=0)
    
    return padded_tensor
@tf.function
def parser(rec, pad):
    '''
    This is a parser function. It defines the template for
    interpreting the examples you're feeding in. Basically,
    this function defines what the labels and data look like
    for your labeled data.
    '''
    features = {
        'i_sample': tf.io.FixedLenFeature([], tf.int64),
        'nx': tf.io.FixedLenFeature([], tf.int64),
        'ny': tf.io.FixedLenFeature([], tf.int64),
        'nz': tf.io.FixedLenFeature([], tf.int64),
        'comp_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'comp_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'comp_raw3': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'comp_out_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'comp_out_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'comp_out_raw3': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
    }

    parsed_rec = tf.io.parse_single_example(rec, features)
    
    nx = 432
    nz = 432
    
    padding = tf.cast(pad / 2, tf.int32)

    nxd = nx + pad
    nzd = nz + pad
    
    inputs = periodic_padding(tf.reshape(parsed_rec['comp_raw1'], (nz, nx)), ((padding, padding), (padding, padding)))
    inputs = tf.reshape(inputs, (1, nzd, nxd))

    for i_comp in range(1, app.N_VARS_IN):
        new_input = tf.reshape(parsed_rec[f'comp_raw{i_comp+1}'], (nz, nx))
        inputs = tf.concat((inputs, tf.reshape(periodic_padding(new_input, ((padding, padding), (padding, padding))), (1, nzd, nxd))), 0)

    output1 = tf.reshape(parsed_rec['comp_out_raw1'], (1, nz, nx))

    if app.N_VARS_OUT == 1:
        return inputs, output1
    else:
        output2 = tf.reshape(parsed_rec['comp_out_raw2'], (1, nz, nx))
        if app.N_VARS_OUT == 2:
            return inputs, (output1, output2)
        else:
            output3 = tf.reshape(parsed_rec['comp_out_raw3'], (1, nz, nx))
            return inputs, (output1, output2, output3)
        
def step_decay(epoch):
   epochs_drop = lr_epdrop
   initial_lrate = init_lr#/5
   drop = lr_drop
   lrate = initial_lrate * math.pow(drop,
           math.floor((epoch)/epochs_drop))
   return lrate

# Final ReLu function for fluctuations
def thres_relu(x):
   return tf.keras.activations.relu(x, threshold=app.RELU_THRESHOLD)

@tf.function
def periodic_padding(tensor, padding):
    """
    Tensorflow function to pad periodically a 2D tensor

    Parameters
    ----------
    tensor : 2D tf.Tensor
        Tensor to be padded
    padding : integer values
        Padding value, same in all directions

    Returns
    -------
    Padded tensor

    """
    lower_pad = tensor[:padding[0][0],:]
    upper_pad = tensor[-padding[0][1]:,:]
    
    partial_tensor = tf.concat([upper_pad, tensor, lower_pad], axis=0)
    
    left_pad = partial_tensor[:,-padding[1][0]:]
    right_pad = partial_tensor[:,:padding[1][1]]
    
    padded_tensor = tf.concat([left_pad, partial_tensor, right_pad], axis=1)
    
    return padded_tensor
    
@tf.function
def periodic_padding_z(tensor, padding):
    """
    Tensorflow function to pad periodically a 2D tensor

    Parameters
    ----------
    tensor : 2D tf.Tensor
        Tensor to be padded
    padding : integer values
        Padding value, same in all directions

    Returns
    -------
    Padded tensor

    """
    lower_pad = tensor[:padding[0][0],:]
    upper_pad = tensor[-padding[0][1]:,:]
    
    padded_tensor = tf.concat([upper_pad, tensor, lower_pad], axis=0)
    
    return padded_tensor

@tf.function
def output_parser(rec):
    '''
    This is a parser function. It defines the template for
    interpreting the examples you're feeding in. Basically, 
    this function defines what the labels and data look like
    for your labeled data. 
    '''        
    features = {
        'i_sample': tf.io.FixedLenFeature([], tf.int64),
        'nx': tf.io.FixedLenFeature([], tf.int64),
        'ny': tf.io.FixedLenFeature([], tf.int64),
        'nz': tf.io.FixedLenFeature([], tf.int64),
        'comp_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'comp_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'comp_raw3': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'comp_out_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'comp_out_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'comp_out_raw3': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
#        'comp_out_raw4': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
        }
        
    parsed_rec = tf.io.parse_single_example(rec, features)
    
    # i_sample = parsed_rec['i_sample']
    nx = tf.cast(parsed_rec['nx'], tf.int32)
    # ny = tf.cast(parsed_rec['ny'], tf.int32)
    nz = tf.cast(parsed_rec['nz'], tf.int32)
    nx = 432
    nz = 432
#    print('nx:'+str(nx))
#    print('nz:'+str(nz))

    output1 = tf.reshape(parsed_rec['comp_out_raw1'],(1,nz, nx))

    if app.N_VARS_OUT == 1:
        return output1
    else:
        output2 = tf.reshape(parsed_rec['comp_out_raw2'],(1,nz, nx))
        if app.N_VARS_OUT == 2:
            return (output1, output2)
        else:
            output3 = tf.reshape(parsed_rec['comp_out_raw3'],(1,nz, nx))
            return (output1, output2, output3)

@tf.function
def input_parser(rec):
    '''
    This is a parser function. It defines the template for
    interpreting the examples you're feeding in. Basically, 
    this function defines what the labels and data look like
    for your labeled data. 
    '''        
    features = {
        'i_sample': tf.io.FixedLenFeature([], tf.int64),
        'nx': tf.io.FixedLenFeature([], tf.int64),
        'ny': tf.io.FixedLenFeature([], tf.int64),
        'nz': tf.io.FixedLenFeature([], tf.int64),
        'comp_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'comp_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'comp_raw3': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
#        'comp_raw4': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'comp_out_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'comp_out_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'comp_out_raw3': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
        }
    
    parsed_rec = tf.io.parse_single_example(rec, features)
    
    # i_sample = parsed_rec['i_sample']
    nx = tf.cast(parsed_rec['nx'], tf.int32)
    # ny = tf.cast(parsed_rec['ny'], tf.int32)
    nz = tf.cast(parsed_rec['nz'], tf.int32)
    nx = 432
    nz = 432
    padding = tf.cast(pad/2, tf.int32)

    nxd = nx + pad
    nzd = nz + pad
    
#    # Input processing --------------------------------------------------------
    inputs = periodic_padding(tf.reshape(parsed_rec['comp_raw1'],(nz, nx)),((padding,padding),(padding,padding)))
    inputs = tf.reshape(inputs,(1,nzd,nxd))

    for i_comp in range(1,app.N_VARS_IN):
        new_input = tf.reshape(parsed_rec[f'comp_raw{i_comp+1}'],(nz,nx))
        inputs = tf.concat((inputs, tf.reshape(periodic_padding(new_input,((padding,padding),(padding,padding))),(1,nzd,nxd))),0)
        
    return inputs

@tf.function
def eval_parser(rec):
    '''
    This is a parser function. It defines the template for
    interpreting the examples you're feeding in. Basically, 
    this function defines what the labels and data look like
    for your labeled data. 
    '''        
    features = {
        'i_sample': tf.io.FixedLenFeature([], tf.int64),
        'nx': tf.io.FixedLenFeature([], tf.int64),
        'ny': tf.io.FixedLenFeature([], tf.int64),
        'nz': tf.io.FixedLenFeature([], tf.int64),
        'comp_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'comp_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'comp_raw3': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'comp_out_raw1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'comp_out_raw2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'comp_out_raw3': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
        }
    
    parsed_rec = tf.io.parse_single_example(rec, features)
    
    # i_sample = parsed_rec['i_sample']
    nx = tf.cast(parsed_rec['nx'], tf.int32)
    # ny = tf.cast(parsed_rec['ny'], tf.int32)
    nz = tf.cast(parsed_rec['nz'], tf.int32)
    
#    print('nx:'+str(nx))
#    print('nz:'+str(nz))
    
    padding = tf.cast(pad/2, tf.int32)

#    nxd = nx + pad
    nzd = nz + pad
    
    # Input processing --------------------------------------------------------
    if norm_input == True:
        inputs = periodic_padding(tf.reshape((parsed_rec['comp_raw1']-avgs_in[0])/std_in[0],(nx, nz)),((padding,padding),(padding,padding)))
    else:
        inputs = periodic_padding(tf.reshape(parsed_rec['comp_raw1'],(nx, nz)),((padding,padding),(padding,padding)))
    inputs = tf.reshape(inputs,(1,nxd,nzd))
    
    for i_comp in range(1,app.N_VARS_IN):
        new_input = parsed_rec[f'comp_raw{i_comp+1}']
        if norm_input == True:
            new_input = (new_input-avgs_in[i_comp])/std_in[i_comp]
        inputs = tf.concat((inputs, tf.reshape(periodic_padding(tf.reshape(new_input,(nx, nz)),((padding,padding),(padding,padding))),(1,nxd,nzd))),0)
    
    # Output processing
    nx_out = nx
    nz_out = nz
    
    output1 = tf.reshape(parsed_rec['comp_out_raw1'],(1,nx_out, nz_out))
    
    if pred_fluct == True:    
        output1 = output1 - avgs[0][ypos_Ret[str(target_yp)]]
    
    if app.N_VARS_OUT == 1:
        return inputs, output1
    else:
        output2 = tf.reshape(parsed_rec['comp_out_raw2'],(1,nx_out, nz_out))
        if pred_fluct == True:    
            output2 = output2 - avgs[1][ypos_Ret[str(target_yp)]]
        
        if scale_output == True:
            scaling_coeff2 = tf.cast(rms[0][ypos_Ret[str(target_yp)]] / rms[1][ypos_Ret[str(target_yp)]], tf.float32)
            output2 = output2 * scaling_coeff2
        if app.N_VARS_OUT == 2:
            return inputs, (output1, output2)
        else:
            output3 = tf.reshape(parsed_rec['comp_out_raw3'],(1,nx_out, nz_out))
            if pred_fluct == True:    
                output3 = output3 - avgs[2][ypos_Ret[str(target_yp)]]
            
            if scale_output == True:
                scaling_coeff3 = tf.cast(rms[0][ypos_Ret[str(target_yp)]] / rms[2][ypos_Ret[str(target_yp)]], tf.float32)
                output3 = output3 * scaling_coeff3
            return inputs, (output1, output2, output3)