
from __future__ import print_function

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))

TRT_LOGGER = trt.Logger()
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
onnx_file_path="20220121_V18_V21.onnx"
engine_file_path="dete.trt"

with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
    builder.max_workspace_size = 1 << 28 # 256MiB
    builder.max_batch_size = 1
    # Parse model file
    if not os.path.exists(onnx_file_path):
        print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
        exit(0)
    print('Loading ONNX file from path {}...'.format(onnx_file_path))
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        if not parser.parse(model.read()):
            print ('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print (parser.get_error(error))
    network.get_input(0).shape = [1, 3, 1080, 1920]
    print('Completed parsing of ONNX file')
    print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
    engine = builder.build_cuda_engine(network)
    print("Completed creating Engine")
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())

 