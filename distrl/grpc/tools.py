import logging
from concurrent import futures
from typing import Optional
from typing import Any, Callable, Dict, Any, Union

import numpy as np
import torch
from torch import nn

from distrl.a3c.a3c_pb2_grpc import *
from distrl.grpc.tensor_pb2 import *
import threading
import random
import gymnasium as gym

logger = logging.getLogger(__name__)


class SingletonMetaclass(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class ReadPreferringLock:
    def __init__(self):
        self.current_readers = 0
        self.read_lock = threading.Lock()
        self.write_lock = threading.Lock()

    def acquire_read_lock(self):
        self.read_lock.acquire()
        self.current_readers += 1
        if self.current_readers == 1:
            self.write_lock.acquire()
        self.read_lock.release()

    def release_read_lock(self):
        self.read_lock.acquire()
        self.current_readers -= 1
        if self.current_readers == 0:
            self.write_lock.release()
        self.read_lock.release()

    def acquire_write_lock(self):
        self.write_lock.acquire()

    def release_write_lock(self):
        self.write_lock.release()


class WritePreferringLock:
    def __init__(self):
        self.lock = threading.Lock()
        self.read_lock = threading.Lock()
        self.cond = threading.Condition()
        self.readers = 0

    def acquire_read_lock(self):
        with self.read_lock:
            with self.lock:
                self.readers += 1

    def release_read_lock(self):
        with self.lock:
            if self.readers == 0:
                raise RuntimeError("release unlocked lock")

            with self.cond:
                self.readers -= 1
                if self.readers == 0:
                    self.cond.notify_all()

    def acquire_write_lock(self):
        self.read_lock.acquire()
        with self.cond:
            self.cond.wait_for(lambda: self.readers == 0)

    def release_write_lock(self):
        self.read_lock.release()


def tensor2proto(tensor: torch.Tensor, version: Optional[int] = None):
    torch_types = {
        torch.float: (DataType.DT_FLOAT, "float_data"),
        torch.double: (DataType.DT_DOUBLE, "double_data"),
        torch.int64: (DataType.DT_INT64, "int64_data"),
        torch.int32: (DataType.DT_INT32, "int_data"),
        torch.int16: (DataType.DT_INT16, "int_data"),
        torch.int8: (DataType.DT_INT8, "int_data"),
        torch.bool: (DataType.DT_BOOL, "bool_data")
    }

    if tensor.dtype not in torch_types:
        raise Exception(f"datatype {tensor.dtype} is not supported")
    dtype, data_slot = torch_types[tensor.dtype]
    dims = map(lambda x: TensorShape.Dim(size=x), tensor.shape)
    kwargs = {data_slot: tensor.reshape(-1).tolist()}
    if version is not None:
        assert version >= 0
        kwargs["version"] = version

    return Tensor(dtype=dtype, shape=TensorShape(dims=dims), **kwargs)


def proto2tensor(proto: Tensor):
    proto_types = {
        DataType.DT_FLOAT: (torch.float, "float_data"),
        DataType.DT_DOUBLE: (torch.double, "double_data"),
        DataType.DT_INT64: (torch.int64, "int64_data"),
        DataType.DT_INT32: (torch.int32, "int_data"),
        DataType.DT_INT16: (torch.int16, "int_data"),
        DataType.DT_INT8: (torch.int8, "int_data"),
        DataType.DT_BOOL: (torch.bool, "bool_data")
    }

    if proto.dtype not in proto_types:
        raise Exception(f"datatype {proto.dtype} is not supported")
    dtype, data_slot = proto_types[proto.dtype]
    shape = tuple(map(lambda x: x.size, proto.shape.dims))
    data = proto.__getattribute__(data_slot)

    return torch.tensor(np.array(data), dtype=dtype).reshape(shape)


def parse_tensor_parameters(params: Dict[str, Tensor]):
    parsed_params = dict()
    for name in params:
        parsed_params[name] = proto2tensor(params[name])
    return parsed_params


def epsilon_greedy(epsilon: float, env: Union[gym.Env, gym.Wrapper], state: torch.Tensor, q_network: nn.Module) -> Any:
    assert 0.0 <= epsilon <= 1.0
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        with torch.no_grad():
            q_values = q_network(state)
            action = torch.argmax(q_values, dim=1).detach().cpu().item()
    return action


def create_server(servicer: Callable, max_workers: int, port: int, service: Any):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    servicer(service, server)
    server.add_insecure_port(f"[::]:{port}")
    return server
