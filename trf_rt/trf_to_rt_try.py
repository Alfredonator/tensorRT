#SOURCE: https://developer.nvidia.com/blog/tensorrt-integration-speeds-tensorflow-inference/
import tf as tf
import tensorflow.contrib.tensorrt as trt
from utils.detection_utils import Props
from detection.frozen_to_rt import build_frozen

MEM_FRACTION = 0 #number_between_0_and_1
BATCH_SIZE = 1
CONFIG_FILE = Props.OUTPUT_DIR + '/' + Props.PIPELINE_CONFIG_NAME   # path to model's config file
CHECKPOINT_FILE = Props.OUTPUT_DIR + '/' + Props.CHECKPOINT_NAME    # path to model's checkpoints ((weights))

def tr_to_trt():
    frozen_graph, input, output = build_frozen()

    #apply TensorRT optimizations to the frozen graph with the new create_inference_graph function.
    #TensorRT then takes a frozen TensorFlow graph as input and returns an optimized graph with TensorRT nodes
    trt_graph = trt.create_inference_graph(
                    input_graph_def=frozen_graph,
                    outputs=output, #list of strings with names of output nodes e.g. “resnet_v1_50/predictions/Reshape_1”
                    max_batch_size=BATCH_SIZE,
                    max_workspace_size_bytes=MEM_FRACTION-0.05, #integer, maximum GPU memory size available for TensorRT
                    precision_mode='FP16') #“FP32”, “FP16” or “INT8”


if __name__ == '__main__':
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEM_FRACTION)
    tr_to_trt()


