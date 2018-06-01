from concurrent import futures
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
        raise RuntimeError('Board size mismatch: server=%d, worker=%d' % (
            server_config.board_size, go.N))

    worker_config = dual_net.InferenceWorkerConfig(
        address=FLAGS.address,
        get_features_method="/minigo.InferenceService/GetFeatures",
        put_outputs_method="/minigo.InferenceService/PutOutputs",
        batch_size=server_config.batch_size,
        descriptor_path=FLAGS.descriptor)

    n = dual_net.DualNetwork(FLAGS.model, worker_config)

    outputs = {
        'features': n.sess.graph.get_tensor_by_name("unflatten_features:0"),
        'dummy': n.sess.graph.get_tensor_by_name("put_outputs:0"),
    }
    output = n.sess.run(outputs)

    for f in output['features']:
        print('---------------------------')
        for r in range(go.N):
            row = ""
            for c in range(go.N):
                if f[r][c][0]:
                    row += ' 0'
                elif f[r][c][1]:
                    row += ' 1'
                else:
                    row += ' .'
            print(row)


if __name__ == '__main__':
    flags.FLAGS(sys.argv, known_only=True)
    main()
