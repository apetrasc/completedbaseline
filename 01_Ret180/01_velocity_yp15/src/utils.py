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
        
import os
import typing
import warnings
from urllib import request
from http import client
import io
import pkg_resources
import validators
import numpy as np
import scipy as sp
import cv2

try:
    import PIL
    import PIL.Image
except ImportError:  # pragma: no cover
    PIL = None

ImageInputType = typing.Union[str, np.ndarray, "PIL.Image.Image", io.BytesIO]


def get_imagenet_classes() -> typing.List[str]:
    """Get the list of ImageNet 2012 classes."""
    filepath = pkg_resources.resource_filename("vit_keras", "imagenet2012.txt")
    with open(filepath, encoding="utf-8") as f:
        classes = [l.strip() for l in f.readlines()]
    return classes


def read(filepath_or_buffer: ImageInputType, size, timeout=None):
    """Read a file into an image object
    Args:
        filepath_or_buffer: The path to the file or any object
            with a `read` method (such as `io.BytesIO`)
        size: The size to resize the image to.
        timeout: If filepath_or_buffer is a URL, the timeout to
            use for making the HTTP request.
    """
    if PIL is not None and isinstance(filepath_or_buffer, PIL.Image.Image):
        return np.array(filepath_or_buffer.convert("RGB"))
    if isinstance(filepath_or_buffer, (io.BytesIO, client.HTTPResponse)):
        image = np.asarray(bytearray(filepath_or_buffer.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    elif isinstance(filepath_or_buffer, str) and validators.url(filepath_or_buffer):
        with request.urlopen(filepath_or_buffer, timeout=timeout) as r:
            return read(r, size=size)
    else:
        if not os.path.isfile(typing.cast(str, filepath_or_buffer)):
            raise FileNotFoundError(
                "Could not find image at path: " + filepath_or_buffer
            )
        image = cv2.imread(filepath_or_buffer)
    if image is None:
        raise ValueError(f"An error occurred reading {filepath_or_buffer}.")
    # We use cvtColor here instead of just ret[..., ::-1]
    # in order to ensure that we provide a contiguous
    # array for later processing. Some hashers use ctypes
    # to pass the array and non-contiguous arrays can lead
    # to erroneous results.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (size, size))


def apply_embedding_weights(target_layer, source_weights, num_x_patches, num_y_patches):
    """Apply embedding weights to a target layer.

    Args:
        target_layer: The target layer to which weights will
            be applied.
        source_weights: The source weights, which will be
            resized as necessary.
        num_x_patches: Number of patches in width of image.
        num_y_patches: Number of patches in height of image.
    """
    expected_shape = target_layer.weights[0].shape
    if expected_shape != source_weights.shape:
        token, grid = source_weights[0, :1], source_weights[0, 1:]
        sin = int(np.sqrt(grid.shape[0]))
        sout_x = num_x_patches
        sout_y = num_y_patches
        warnings.warn(
            "Resizing position embeddings from " f"{sin}, {sin} to {sout_x}, {sout_y}",
            UserWarning,
        )
        zoom = (sout_y / sin, sout_x / sin, 1)
        grid = sp.ndimage.zoom(grid.reshape(sin, sin, -1), zoom, order=1).reshape(
            sout_x * sout_y, -1
        )
        source_weights = np.concatenate([token, grid], axis=0)[np.newaxis]
    target_layer.set_weights([source_weights])


def load_weights_numpy(
    model, params_path, pretrained_top, num_x_patches, num_y_patches
):
    """Load weights saved using Flax as a numpy array.

    Args:
        model: A Keras model to load the weights into.
        params_path: Filepath to a numpy archive.
        pretrained_top: Whether to load the top layer weights.
        num_x_patches: Number of patches in width of image.
        num_y_patches: Number of patches in height of image.
    """
    params_dict = np.load(
        params_path, allow_pickle=False
    )  # pylint: disable=unexpected-keyword-arg
    source_keys = list(params_dict.keys())
    pre_logits = any(l.name == "pre_logits" for l in model.layers)
    source_keys_used = []
    n_transformers = len(
        set(
            "/".join(k.split("/")[:2])
            for k in source_keys
            if k.startswith("Transformer/encoderblock_")
        )
    )
    n_transformers_out = sum(
        l.name.startswith("Transformer/encoderblock_") for l in model.layers
    )
    assert n_transformers == n_transformers_out, (
        f"Wrong number of transformers ("
        f"{n_transformers_out} in model vs. {n_transformers} in weights)."
    )

    matches = []
    for tidx in range(n_transformers):
        encoder = model.get_layer(f"Transformer/encoderblock_{tidx}")
        source_prefix = f"Transformer/encoderblock_{tidx}"
        matches.extend(
            [
                {
                    "layer": layer,
                    "keys": [
                        f"{source_prefix}/{norm}/{name}" for name in ["scale", "bias"]
                    ],
                }
                for norm, layer in [
                    ("LayerNorm_0", encoder.layernorm1),
                    ("LayerNorm_2", encoder.layernorm2),
                ]
            ]
            + [
                {
                    "layer": encoder.mlpblock.get_layer(
                        f"{source_prefix}/Dense_{mlpdense}"
                    ),
                    "keys": [
                        f"{source_prefix}/MlpBlock_3/Dense_{mlpdense}/{name}"
                        for name in ["kernel", "bias"]
                    ],
                }
                for mlpdense in [0, 1]
            ]
            + [
                {
                    "layer": layer,
                    "keys": [
                        f"{source_prefix}/MultiHeadDotProductAttention_1/{attvar}/{name}"
                        for name in ["kernel", "bias"]
                    ],
                    "reshape": True,
                }
                for attvar, layer in [
                    ("query", encoder.att.query_dense),
                    ("key", encoder.att.key_dense),
                    ("value", encoder.att.value_dense),
                    ("out", encoder.att.combine_heads),
                ]
            ]
        )
    for layer_name in ["embedding", "head", "pre_logits"]:
        if layer_name == "head" and not pretrained_top:
            source_keys_used.extend(["head/kernel", "head/bias"])
            continue
        if layer_name == "pre_logits" and not pre_logits:
            continue
        matches.append(
            {
                "layer": model.get_layer(layer_name),
                "keys": [f"{layer_name}/{name}" for name in ["kernel", "bias"]],
            }
        )
    matches.append({"layer": model.get_layer("class_token"), "keys": ["cls"]})
    matches.append(
        {
            "layer": model.get_layer("Transformer/encoder_norm"),
            "keys": [f"Transformer/encoder_norm/{name}" for name in ["scale", "bias"]],
        }
    )
    apply_embedding_weights(
        target_layer=model.get_layer("Transformer/posembed_input"),
        source_weights=params_dict["Transformer/posembed_input/pos_embedding"],
        num_x_patches=num_x_patches,
        num_y_patches=num_y_patches,
    )
    source_keys_used.append("Transformer/posembed_input/pos_embedding")
    for match in matches:
        source_keys_used.extend(match["keys"])
        source_weights = [params_dict[k] for k in match["keys"]]
        if match.get("reshape", False):
            source_weights = [
                source.reshape(expected.shape)
                for source, expected in zip(
                    source_weights, match["layer"].get_weights()
                )
            ]
        match["layer"].set_weights(source_weights)
    unused = set(source_keys).difference(source_keys_used)
    if unused:
        warnings.warn(f"Did not use the following weights: {unused}", UserWarning)
    target_keys_set = len(source_keys_used)
    target_keys_all = len(model.weights)
    if target_keys_set < target_keys_all:
        warnings.warn(
            f"Only set {target_keys_set} of {target_keys_all} weights.", UserWarning
        )