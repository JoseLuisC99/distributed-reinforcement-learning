# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: distrl/a3c/a3c.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from distrl.grpc import tensor_pb2 as distrl_dot_grpc_dot_tensor__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x14\x64istrl/a3c/a3c.proto\x12\x12\x64istributed_rl.a3c\x1a\x18\x64istrl/grpc/tensor.proto\"!\n\x0eVersionMessage\x12\x0f\n\x07version\x18\x01 \x01(\x05\"\xff\x02\n\x11ParametersMessage\x12?\n\x05\x61\x63tor\x18\x01 \x03(\x0b\x32\x30.distributed_rl.a3c.ParametersMessage.ActorEntry\x12\x41\n\x06\x63ritic\x18\x02 \x03(\x0b\x32\x31.distributed_rl.a3c.ParametersMessage.CriticEntry\x12\x0f\n\x07version\x18\x03 \x01(\x05\x12\x17\n\nterminated\x18\x04 \x01(\x08H\x00\x88\x01\x01\x12\x14\n\x07n_steps\x18\x05 \x01(\x03H\x01\x88\x01\x01\x1a\x44\n\nActorEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12%\n\x05value\x18\x02 \x01(\x0b\x32\x16.distributed_rl.Tensor:\x02\x38\x01\x1a\x45\n\x0b\x43riticEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12%\n\x05value\x18\x02 \x01(\x0b\x32\x16.distributed_rl.Tensor:\x02\x38\x01\x42\r\n\x0b_terminatedB\n\n\x08_n_steps2\xc7\x01\n\x0b\x43oordinator\x12\\\n\rPushGradients\x12%.distributed_rl.a3c.ParametersMessage\x1a\".distributed_rl.a3c.VersionMessage\"\x00\x12Z\n\x0bSynchronize\x12\".distributed_rl.a3c.VersionMessage\x1a%.distributed_rl.a3c.ParametersMessage\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'distrl.a3c.a3c_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_PARAMETERSMESSAGE_ACTORENTRY']._options = None
  _globals['_PARAMETERSMESSAGE_ACTORENTRY']._serialized_options = b'8\001'
  _globals['_PARAMETERSMESSAGE_CRITICENTRY']._options = None
  _globals['_PARAMETERSMESSAGE_CRITICENTRY']._serialized_options = b'8\001'
  _globals['_VERSIONMESSAGE']._serialized_start=70
  _globals['_VERSIONMESSAGE']._serialized_end=103
  _globals['_PARAMETERSMESSAGE']._serialized_start=106
  _globals['_PARAMETERSMESSAGE']._serialized_end=489
  _globals['_PARAMETERSMESSAGE_ACTORENTRY']._serialized_start=323
  _globals['_PARAMETERSMESSAGE_ACTORENTRY']._serialized_end=391
  _globals['_PARAMETERSMESSAGE_CRITICENTRY']._serialized_start=393
  _globals['_PARAMETERSMESSAGE_CRITICENTRY']._serialized_end=462
  _globals['_COORDINATOR']._serialized_start=492
  _globals['_COORDINATOR']._serialized_end=691
# @@protoc_insertion_point(module_scope)