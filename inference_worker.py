import sys
import time
import grpc
import inference_service_pb2
import tensorflow as tf
import inference_service_pb2_grpc
from absl import flags
import dual_net
import go

flags.DEFINE_string("model", "", "Path to the TensorFlow model.")
flags.DEFINE_string("address", "localhost:50051", "Inference server address.")
flags.DEFINE_string("descriptor", "",
                    "Path to the InferenceService proto descriptor.")

FLAGS = flags.FLAGS


def get_server_config():
    while True:
        try:
            # Fetch the server config, used to set batch size.
            channel = grpc.insecure_channel(FLAGS.address)
            stub = inference_service_pb2_grpc.InferenceServiceStub(channel)
            return stub.GetConfig(inference_service_pb2.GetConfigRequest())
        except Exception:
            print("Waiting for server")
            time.sleep(1)


def main():
    server_config = get_server_config()
    print("server_config:\n%s" % server_config)

    if server_config.board_size != go.N:
        raise RuntimeError("Board size mismatch: server=%d, worker=%d" % (
            server_config.board_size, go.N))

    worker_config = dual_net.InferenceWorkerConfig(
        address=FLAGS.address,
        get_features_method="/minigo.InferenceService/GetFeatures",
        put_outputs_method="/minigo.InferenceService/PutOutputs",
        batch_size=server_config.batch_size,
        descriptor_path=FLAGS.descriptor)

    print("building graph")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(graph=tf.Graph(), config=config)
    with sess.graph.as_default():
        loop = dual_net.model_inference_worker_fn(worker_config)
        tf.train.Saver().restore(sess, FLAGS.model)

    print("running graph")
    sess.run(loop)


if __name__ == "__main__":
    flags.FLAGS(sys.argv, known_only=True)
    main()
