import tensorflow.contrib.tensorrt as trt
import matplotlib
matplotlib.use('Agg')
from utils.detection_utils import build_detection_graph, Props
from datetime import datetime

CONFIG_FILE = Props.OUTPUT_DIR + '/' + Props.PIPELINE_CONFIG_NAME   # path to model's config file
CHECKPOINT_FILE = Props.OUTPUT_DIR + '/' + Props.CHECKPOINT_NAME    # path to model's checkpoints ((weights))

def build_frozen():
    frozen_graph, input_names, output_names = build_detection_graph(
        config=CONFIG_FILE,
        checkpoint=CHECKPOINT_FILE,
        score_threshold=0.3,
        batch_size=1
    )
    return frozen_graph, input_names, output_names

#RT optimization
def build_rt(frozen_graph, output):
    trt_graph = trt.create_inference_graph(
        input_graph_def=frozen_graph,
        outputs=output,
        max_batch_size=1,
        max_workspace_size_bytes=1 << 25,
        precision_mode='FP16', #TODO check if FP32 is feasible/possible
        minimum_segment_size=50
    )
    return trt_graph

def save_model(trt_graph):
    try:
        with open('./data/saved_model_{}.pb'.format(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]), 'wb') as f:
            f.write(trt_graph.SerializeToString())
    except Exception:
        print('Error saving')


if __name__== '__main__':
    frozen_graph, input, output = build_frozen()
    rt_graph = build_rt(frozen_graph, output)
    save_model(rt_graph)

