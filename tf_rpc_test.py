from concurrent import futures
import sys
import grpc
import inference_service_pb2
import inference_service_pb2_grpc
from absl import flags
import dual_net

flags.DEFINE_string("model", "", "Path to the TF model.")
flags.DEFINE_string("descriptor", "",
                    "Path to the InferenceService proto descriptor.")

FLAGS = flags.FLAGS


class InferenceService(inference_service_pb2_grpc.InferenceServiceServicer):
    def GetFeatures(self, request, context):
        response = inference_service_pb2.GetFeaturesResponse()
        features = []
        for _ in range(9):
            for _ in range(9):
                for _ in range(16):
                    response.features.append(0)
                response.features.append(1)
        return response

    def PutOutputs(self, request, context):
        return inference_service_pb2.PutOutputsResponse()


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    inference_service_pb2_grpc.add_InferenceServiceServicer_to_server(
        InferenceService(), server)
    server.add_insecure_port('[::]:50051')
    print("### starting server")
    server.start()
    print("### server started")

    rpc_client_info = dual_net.InferenceWorkerConfig(
        "localhost:50051",
        "/minigo.InferenceService/GetFeatures",
        "/minigo.InferenceService/PutOutputs",
        1)
    n = dual_net.DualNetwork(FLAGS.model, rpc_client_info)

    print("### running inference")
    policy = n.sess.run(n.inference_output)["policy_output"][0]
    for row in range(9):
        start = row * 9
        print(" ".join(["%.3f" % x for x in policy[start:start+9]]))
    print("### stopping server")

    server.stop(0)


if __name__ == '__main__':
    flags.FLAGS(sys.argv, known_only=True)
    serve()
