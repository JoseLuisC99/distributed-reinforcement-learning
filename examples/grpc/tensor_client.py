import logging

import grpc
from distrl.grpc.grpc_agent_pb2_grpc import RPCAgentStub
from distrl.grpc.tools import *


def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = RPCAgentStub(channel)

        tensor = torch.zeros((3, 3), dtype=torch.int8)
        response = stub.Send(tensor2proto(tensor))
        print(f"Greeter client received: {proto2tensor(response)}")


if __name__ == "__main__":
    logging.basicConfig()
    run()
