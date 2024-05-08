from concurrent import futures
import logging

import grpc
from distrl.grpc.grpc_agent_pb2_grpc import RPCAgentServicer, add_RPCAgentServicer_to_server
from distrl.grpc.tools import *


class Server(RPCAgentServicer):
    def Send(self, request, context):
        print(proto2tensor(request))

        tensor = torch.rand(2, 2)
        return tensor2proto(tensor)


def serve():
    port = "50051"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_RPCAgentServicer_to_server(Server(), server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    serve()
