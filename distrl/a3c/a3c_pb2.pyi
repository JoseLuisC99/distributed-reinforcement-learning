from distrl.grpc import tensor_pb2 as _tensor_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class VersionMessage(_message.Message):
    __slots__ = ("version",)
    VERSION_FIELD_NUMBER: _ClassVar[int]
    version: int
    def __init__(self, version: _Optional[int] = ...) -> None: ...

class ParametersMessage(_message.Message):
    __slots__ = ("actor", "critic", "version", "terminated", "n_steps")
    class ActorEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _tensor_pb2.Tensor
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_tensor_pb2.Tensor, _Mapping]] = ...) -> None: ...
    class CriticEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _tensor_pb2.Tensor
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_tensor_pb2.Tensor, _Mapping]] = ...) -> None: ...
    ACTOR_FIELD_NUMBER: _ClassVar[int]
    CRITIC_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    TERMINATED_FIELD_NUMBER: _ClassVar[int]
    N_STEPS_FIELD_NUMBER: _ClassVar[int]
    actor: _containers.MessageMap[str, _tensor_pb2.Tensor]
    critic: _containers.MessageMap[str, _tensor_pb2.Tensor]
    version: int
    terminated: bool
    n_steps: int
    def __init__(self, actor: _Optional[_Mapping[str, _tensor_pb2.Tensor]] = ..., critic: _Optional[_Mapping[str, _tensor_pb2.Tensor]] = ..., version: _Optional[int] = ..., terminated: bool = ..., n_steps: _Optional[int] = ...) -> None: ...
