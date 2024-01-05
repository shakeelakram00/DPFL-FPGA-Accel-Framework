import math
import numpy as np
import os
import textwrap
import warnings
from enum import Enum, EnumMeta
import binascii
import numpy as np
import os
import sys
from bitstring import BitArray


def calc_wmem(mw, mh, pe, simd):
  """Calculates and returns WMEM."""
  #mw = self.get_nodeattr("MW")
  #mh = self.get_nodeattr("MH")
  #pe = self.get_nodeattr("PE")
  #simd = self.get_nodeattr("SIMD")
  #mw = 36   #1_0_StreamingDataflowpartition_1_MatrixVectorActivation_0   (36, 21) (mw, mh) (756, 21)
  #mh = 21
  assert mh % pe == 0, "Requirement MH divisable by PE is violated."
  assert mw % simd == 0, "Requirement MW divisable by SIMD is violated."
  wmem = mw * mh // (pe * simd)
  return wmem

# calc_wmem()

def interleave_matrix_outer_dim_from_partitions(matrix, n_partitions):
    """Interleave the outermost dimension of a matrix from given
    partitions (n_partitions)."""
    if type(matrix) != np.ndarray or matrix.dtype != np.float32:
        # try to convert to a float numpy array (container dtype is float)
        matrix = np.asarray(matrix, dtype=np.float32)
    shp = matrix.shape
    ndim = matrix.ndim
    # ensure # partitions evenly divide the outermost dimension
    assert (
        shp[0] % n_partitions == 0
    ), """The outermost dimension is not divisable
    by the number of partitions."""
    # only tested for matrices
    assert (
        ndim == 2
    ), """The dimension of the matrix is not 2. Currently this function
    only works for matrices."""
    # interleave rows between PEs using reshape + transpose
    matrix_r = matrix.reshape(-1, n_partitions, shp[1]).transpose((1, 0, 2))
    matrix_r = matrix_r.reshape(n_partitions, -1, shp[1])
    return matrix_r

def get_hls_compatible_weight_tensor(orig_weight_matrix, mw, mh, pe, simd):
        """Convert the original numpy weight matrix orig_weight_matrix into
        a form suitable for passing to the hlslib call:
        * ensure MH % PE == 0 and MW % SIMD == 0
        * for bipolar {-1,+1} weights, convert to binary {0, 1}
        * interleave rows between PEs
        * reshape into (1, PE, WMEM, SIMD) and return
        """
        wmem = calc_wmem(mw, mh, pe, simd)
        assert orig_weight_matrix.shape == (mw, mh,), """Weights matrix doesn't
        have expected shape (mw, mh)"""
        assert mw % simd == 0, "Requirement MH divisable by SIMD is violated."
        assert mh % pe == 0, "Requirement MH divisable by PE is violated."
        # start by transposing the original weight matrix, since ONNX and
        # finn-hlslib use different assumptions
        # ONNX uses (in_features, out_features) and matmul(x, W)
        # finn-hlslib uses (out_features, in_features) and matmul(W, x)
        is_bipolar = np.all(np.isin(orig_weight_matrix, [-1,1]))
        ret = orig_weight_matrix.T
        if is_bipolar:
            # convert bipolar to binary
            ret = (ret + 1) / 2
        # interleave rows between PEs and reshape
        # distribute rows between PEs
        ret = interleave_matrix_outer_dim_from_partitions(ret, pe)
        # create SIMD as innermost dimension and add a dummy outer dim
        ret = ret.reshape(1, pe, wmem, simd)
        # reverse the SIMD dimension
        ret = np.flip(ret, axis=-1)
        return ret
#a = get_hls_compatible_weight_tensor(orig_weight_matrix)
# a.shape

def roundup_to_integer_multiple(x, factor):
    """Round up integer x to the nearest integer multiple of integer factor.
    Returns x if factor is set to -1. Both x and factor must otherwise be
    positive."""
    # ensure integers
    assert int(x) == x, "The input x is not an integer."
    assert int(factor) == factor, "The input factor is not an integer."
    # use -1 to indicate no padding needed
    if factor == -1:
        return x
    # ensure positive values
    assert factor > 0 and x > 0, "Factor and x are <= 0."
    if x < factor:
        return factor
    else:
        if x % factor == 0:
            return x
        else:
            return x + (factor - (x % factor))
        
def array2hexstring(array, dtype, pad_to_nbits, prefix="0x", reverse=False):
    """
    Pack given one-dimensional NumPy array with FINN DataType dtype into a hex
    string.
    Any BIPOLAR values will be converted to a single bit with a 0 representing
    -1.
    pad_to_nbits is used to prepend leading zeros to ensure packed strings of
    fixed width. The minimum value for pad_to_nbits is 4, since a single hex
    digit is four bits. reverse can be used to reverse the array prior to
    packing.

    Examples:

    array2hexstring([1, 1, 1, 0], DataType["BINARY"], 4) = "0xe"

    array2hexstring([1, 1, 1, 0], DataType["BINARY"], 8) = "0x0e"

    array2hexstring([1, 1, 0, 1], DataType["BINARY"], 4, reverse=True) = "0xb"

    array2hexstring([1, 1, 1, 0], DataType["BINARY"], 8, reverse=True) = "0x07"
    """
    if pad_to_nbits < 4:
        pad_to_nbits = 4
    # ensure input is a numpy array with float values
    if type(array) != np.ndarray or array.dtype != np.float32:
        # try to convert to a float numpy array (container dtype is float)
        array = np.asarray(array, dtype=np.float32)
    # ensure one-dimensional array to pack
    assert array.ndim == 1, "The given array is not one-dimensional."
    if dtype == "BIPOLAR":
        # convert bipolar values to binary
        array = (array + 1) / 2
        dtype = "BINARY"
    # reverse prior to packing, if desired
    if reverse:
        array = np.flip(array, -1)
    lineval = BitArray(length=0)
    bw = 1 ###Due to binary
    # special handling for fixed point: rescale, then pack as integers
    for val in array:
        # ensure that this value is permitted by chosen dtype
        #assert dtype.allowed(val), "This value is not permitted by chosen dtype."
        if dtype == "BINARY":
          lineval.append(BitArray(uint=int(val), length=bw))
        else:
          lineval.append(BitArray(float=val, length=bw))
    if pad_to_nbits >= lineval.len:
        # extend to the desired output width (a minimum of 4 bits)
        lineval.prepend(BitArray(length=pad_to_nbits - lineval.len))
    else:
        raise Exception("Number of bits is greater than pad_to_nbits")
    # represent as hex
    return prefix + lineval.hex

# def fun(x):
#   return array2hexstring(
#             x, "BINARY", 32, reverse=False, prefix="0x"
#         )

# ab = np.apply_along_axis(fun, a.ndim - 1, a)

# print(ab)


def pack_innermost_dim_as_hex_string(
    ndarray, dtype, pad_to_nbits, reverse_inner=False, prefix="0x"
):
    """Pack the innermost dimension of the given numpy ndarray into hex
    strings using array2hexstring.

    Examples:

    A = [[1, 1, 1, 0], [0, 1, 1, 0]]

    eA = ["0e", "06"]

    pack_innermost_dim_as_hex_string(A, DataType["BINARY"], 8) == eA

    B = [[[3, 3], [3, 3]], [[1, 3], [3, 1]]]

    eB = [[ "0f", "0f"], ["07", "0d"]]

    pack_innermost_dim_as_hex_string(B, DataType["UINT2"], 8) == eB
    """

    if type(ndarray) != np.ndarray or ndarray.dtype != np.float32:
        # try to convert to a float numpy array (container dtype is float)
        ndarray = np.asarray(ndarray, dtype=np.float32)

    def fun(x):
        return array2hexstring(
            x, dtype, pad_to_nbits, reverse=reverse_inner, prefix=prefix
        )

    return np.apply_along_axis(fun, ndarray.ndim - 1, ndarray)


def get_weightstream_width(pe, simd):
  """Returns weight stream width. Used only in decoupled mode."""
  wp = 1   #because of binary datatype 1 is used, for bipolar 1, for rest accordingly
  w_width = pe * simd * wp
  return w_width
# print(get_weightstream_width())

# ONNX i/o tensor shape assumptions for MatrixVectorActivation:
# input 0 is the input tensor, shape (.., i_size) = (..., MW)
# input 1 is the weight tensor, shape (i_size, o_size) = (MW, MH)
# (optional) input 2 is the thresholds tensor, shape (o_size, n_thres)
# output 0 is the output tensor, shape (.., o_size) = (..., MH)
# the ... here can be any shape (representing groups of vectors)
# make_weight_file(orig_weight_matrix, "decoupled_runtime", w_filename)
def make_weight_file(weights, weight_file_mode, weight_file_name, mw, mh, pe, simd):
        """Produce a file containing given weights in appropriate format for this
        layer. This file can be used for either synthesis or run-time reconfig
        of weights.
        Arguments:
        * weights : numpy array with weights to be put into the file
        * weight_file_mode : one of {hls_header, decoupled_verilog_dat,
          decoupled_runtime}
        * weight_file_name : filename for the weight file to be generated
        """
        # convert weights into hlslib-compatible format
        weight_tensor = get_hls_compatible_weight_tensor(weights, mw, mh, pe, simd)

        export_wdt = "BINARY"
        # we have converted bipolar weights to binary for export,
        # so use it as such for weight generation
        if "decoupled" in weight_file_mode:
            # create a weight stream for various flavors of decoupled mode:
            # transpose weight tensor from (1, PE, WMEM, SIMD) to (1, WMEM, PE, SIMD)
            weight_tensor_unflipped = np.transpose(weight_tensor, (0, 2, 1, 3))
            # reverse SIMD flip for saving weights in .npy
            weight_tensor_simd_flipped = np.flip(weight_tensor_unflipped, axis=-1)
            # PE flip for saving weights in .dat
            weight_tensor_pe_flipped = np.flip(weight_tensor_unflipped, axis=-2)
            # reshape weight tensor (simd_flipped and pe_flipped) to desired shape
            # simd_flipped
            weight_tensor_simd_flipped = weight_tensor_simd_flipped.reshape(
                1, -1, pe * simd
            )
            weight_tensor_simd_flipped = weight_tensor_simd_flipped.copy()
            # flipped
            weight_tensor_pe_flipped = weight_tensor_pe_flipped.reshape(
                1, -1, pe * simd
            )
            weight_tensor_pe_flipped = weight_tensor_pe_flipped.copy()
            # if weight_file_mode == "decoupled_npy":
            #     # save weight stream into npy for cppsim
            #     np.save(weight_file_name, weight_tensor_simd_flipped)
            if weight_file_mode == "decoupled_runtime":
                # memstream axi-lite interface will map each mem line to
                # one or multiple 32-bit words
                weight_width = get_weightstream_width(pe, simd)
                words_per_memwidth = 2 ** math.ceil(math.log2(weight_width / 32))
                if words_per_memwidth < 1:
                    words_per_memwidth = 1
                weight_width_padded = words_per_memwidth * 32
                # first, pack and ensure padding to 32 bits
                weight_tensor_pe_flipped = pack_innermost_dim_as_hex_string(
                    weight_tensor_pe_flipped, export_wdt, weight_width_padded, prefix=""
                )
                weight_stream = weight_tensor_pe_flipped.flatten()
                weight_stream = weight_stream.copy()
                with open(weight_file_name, "w") as f:
                    for val in weight_stream:
                        # split into groups of 8 hex digits (= 32 bits)
                        words_32b = textwrap.wrap(val, 8)
                        words_32b.reverse()
                        for word_32b in words_32b:
                            f.write(word_32b + "\n")
            else:
                raise Exception("Unknown weight_file_mode")

        else:
            raise Exception("Unknown weight_file_mode")