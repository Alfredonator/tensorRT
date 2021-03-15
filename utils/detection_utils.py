import datetime
from object_detection.protos import pipeline_pb2
from object_detection import exporter
import os
import subprocess
from collections import namedtuple
from google.protobuf import text_format
import tensorflow as tf
from .graph_utils import force_nms_cpu as f_force_nms_cpu
from .graph_utils import replace_relu6 as f_replace_relu6
from .graph_utils import remove_assert as f_remove_assert

DetectionModel = namedtuple('DetectionModel', ['name', 'url', 'extract_dir'])

class Props:
    INPUT_NAME = 'image_tensor'
    BOXES_NAME = 'detection_boxes'
    CLASSES_NAME = 'detection_classes'
    SCORES_NAME = 'detection_scores'
    MASKS_NAME = 'detection_masks'
    NUM_DETECTIONS_NAME = 'num_detections'
    FROZEN_GRAPH_NAME = 'frozen_inference_graph.pb'
    PIPELINE_CONFIG_NAME = 'pipeline.config'
    CHECKPOINT_NAME = 'model.ckpt'
    OUTPUT_DIR = '../data/generated_model_{}'.format(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])



def build_detection_graph(config, checkpoint,
        batch_size=1,
        score_threshold=None,
        force_nms_cpu=False,
        replace_relu6=False,
        remove_assert=False,
        input_shape=None,
        output_dir=Props.OUTPUT_DIR):

    config_path = config
    checkpoint_path = checkpoint

    # parse config from file
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, 'r') as f:
        text_format.Merge(f.read(), config, allow_unknown_extension=True)

    # override some config parameters
    if config.model.HasField('ssd'):
        config.model.ssd.feature_extractor.override_base_feature_extractor_hyperparams = True
        if score_threshold is not None:
            config.model.ssd.post_processing.batch_non_max_suppression.score_threshold = score_threshold    
        if input_shape is not None:
            config.model.ssd.image_resizer.fixed_shape_resizer.height = input_shape[0]
            config.model.ssd.image_resizer.fixed_shape_resizer.width = input_shape[1]
    elif config.model.HasField('faster_rcnn'):
        if score_threshold is not None:
            config.model.faster_rcnn.second_stage_post_processing.score_threshold = score_threshold
        if input_shape is not None:
            config.model.faster_rcnn.image_resizer.fixed_shape_resizer.height = input_shape[0]
            config.model.faster_rcnn.image_resizer.fixed_shape_resizer.width = input_shape[1]

    if os.path.isdir(output_dir):
        subprocess.call(['rm', '-rf', output_dir])

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    # export inference graph to file (initial)
    with tf.Session(config=tf_config) as tf_sess:
        with tf.Graph().as_default() as tf_graph:
            exporter.export_inference_graph(
                Props.INPUT_NAME,
                config, 
                checkpoint_path, 
                output_dir, 
                input_shape=[batch_size, None, None, 3]
            )

    # read frozen graph from file
    frozen_graph = tf.GraphDef()
    with open(os.path.join(output_dir, Props.FROZEN_GRAPH_NAME), 'rb') as f:
        frozen_graph.ParseFromString(f.read())

    # apply graph modifications
    if force_nms_cpu:
        frozen_graph = f_force_nms_cpu(frozen_graph)
    if replace_relu6:
        frozen_graph = f_replace_relu6(frozen_graph)
    if remove_assert:
        frozen_graph = f_remove_assert(frozen_graph)

    # get input names
    input_names = [Props.INPUT_NAME]
    output_names = [Props.BOXES_NAME, Props.CLASSES_NAME, Props.SCORES_NAME, Props.NUM_DETECTIONS_NAME]

    # remove temporary directory
    subprocess.call(['rm', '-rf', output_dir])

    return frozen_graph, input_names, output_names
