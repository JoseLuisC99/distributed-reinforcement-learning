from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DataType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DT_INVALID: _ClassVar[DataType]
    DT_FLOAT: _ClassVar[DataType]
    DT_DOUBLE: _ClassVar[DataType]
    DT_INT64: _ClassVar[DataType]
    DT_INT32: _ClassVar[DataType]
    DT_INT16: _ClassVar[DataType]
    DT_INT8: _ClassVar[DataType]
    DT_BOOL: _ClassVar[DataType]
DT_INVALID: DataType
DT_FLOAT: DataType
DT_DOUBLE: DataType
DT_INT64: DataType
DT_INT32: DataType
DT_INT16: DataType
DT_INT8: DataType
DT_BOOL: DataType

class TensorShape(_message.Message):
    __slots__ = ("dims",)
    class Dim(_message.Message):
        __slots__ = ("size", "name")
        SIZE_FIELD_NUMBER: _ClassVar[int]
        NAME_FIELD_NUMBER: _ClassVar[int]
        size: int
        name: str
        def __init__(self, size: _Optional[int] = ..., name: _Optional[str] = ...) -> None: ...
    DIMS_FIELD_NUMBER: _ClassVar[int]
    dims: _containers.RepeatedCompositeFieldContainer[TensorShape.Dim]
    def __init__(self, dims: _Optional[_Iterable[_Union[TensorShape.Dim, _Mapping]]] = ...) -> None: ...

class Tensor(_message.Message):
    __slots__ = ("dtype", "shape", "version", "data", "float_data", "double_data", "int64_data", "int_data", "bool_data")
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    FLOAT_DATA_FIELD_NUMBER: _ClassVar[int]
    DOUBLE_DATA_FIELD_NUMBER: _ClassVar[int]
    INT64_DATA_FIELD_NUMBER: _ClassVar[int]
    INT_DATA_FIELD_NUMBER: _ClassVar[int]
    BOOL_DATA_FIELD_NUMBER: _ClassVar[int]
    dtype: DataType
    shape: TensorShape
    version: int
    data: bytes
    float_data: _containers.RepeatedScalarFieldContainer[float]
    double_data: _containers.RepeatedScalarFieldContainer[float]
    int64_data: _containers.RepeatedScalarFieldContainer[int]
    int_data: _containers.RepeatedScalarFieldContainer[int]
    bool_data: _containers.RepeatedScalarFieldContainer[bool]
    def __init__(self, dtype: _Optional[_Union[DataType, str]] = ..., shape: _Optional[_Union[TensorShape, _Mapping]] = ..., version: _Optional[int] = ..., data: _Optional[bytes] = ..., float_data: _Optional[_Iterable[float]] = ..., double_data: _Optional[_Iterable[float]] = ..., int64_data: _Optional[_Iterable[int]] = ..., int_data: _Optional[_Iterable[int]] = ..., bool_data: _Optional[_Iterable[bool]] = ...) -> None: ...
