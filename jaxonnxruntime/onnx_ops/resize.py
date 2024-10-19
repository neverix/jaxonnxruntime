"""Define ONNX Resize operator."""
# pylint: disable=unused-argument
# pylint: disable=g-explicit-length-test
import functools
from collections.abc import Callable, Sequence
from typing import Any

import jax
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node
from jaxonnxruntime.onnx_ops import onnx_ops_utils


@handler.register_op("Resize")
class Resize(handler.Handler):
  """Implementation of the ONNX Resize operator."""

  @classmethod
  def _prepare(cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any):
    onnx_ops_utils.update_node_attrs_dict(node, onnx_jax_impl)

  @classmethod
  def version_10(cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]) -> Callable[..., Any]:
    """ONNX version_10 Resize op."""
    cls._prepare(node, inputs, onnx_resize)
    return onnx_resize

  @classmethod
  def version_11(cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]) -> Callable[..., Any]:
    """ONNX version_11 Resize op."""
    cls._prepare(node, inputs, onnx_resize)
    return onnx_resize

  @classmethod
  def version_13(cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]) -> Callable[..., Any]:
    """ONNX version_13 Resize op."""
    cls._prepare(node, inputs, onnx_resize)
    return onnx_resize

  @classmethod
  def version_18(cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]) -> Callable[..., Any]:
    """ONNX version_18 Resize op."""
    cls._prepare(node, inputs, onnx_resize)
    return onnx_resize

  @classmethod
  def version_19(cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]) -> Callable[..., Any]:
    """ONNX version_19 Resize op."""
    cls._prepare(node, inputs, onnx_resize)
    return onnx_resize


@functools.partial(jax.jit, static_argnames=())
def onnx_resize(*input_args):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Resize for more details."""
  # TODO(neverix): add the implementation here.
  return input_args
