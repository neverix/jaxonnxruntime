"""Define ONNX Resize operator."""
# pylint: disable=unused-argument
# pylint: disable=g-explicit-length-test
import functools
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import jax
from jax import numpy as jnp
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node
from jaxonnxruntime.onnx_ops import onnx_ops_utils


@handler.register_op("Resize")
class Resize(handler.Handler):
    """Implementation of the ONNX Resize operator."""

    @classmethod
    def _prepare(
        cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
    ):
        onnx_ops_utils.update_node_attrs_dict(node, onnx_jax_impl)
        if len(inputs) >= 3:
            # inputs[2] = tuple(inputs[2].tolist())
            node.attrs_dict["scales"] = tuple(inputs[2].tolist())

    @classmethod
    def version_10(
        cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
    ) -> Callable[..., Any]:
        """ONNX version_10 Resize op."""
        cls._prepare(node, inputs, onnx_resize)
        return onnx_resize

    @classmethod
    def version_11(
        cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
    ) -> Callable[..., Any]:
        """ONNX version_11 Resize op."""
        cls._prepare(node, inputs, onnx_resize)
        return onnx_resize

    @classmethod
    def version_13(
        cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
    ) -> Callable[..., Any]:
        """ONNX version_13 Resize op."""
        cls._prepare(node, inputs, onnx_resize)
        return onnx_resize

    @classmethod
    def version_18(
        cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
    ) -> Callable[..., Any]:
        """ONNX version_18 Resize op."""
        cls._prepare(node, inputs, onnx_resize)
        return onnx_resize

    @classmethod
    def version_19(
        cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
    ) -> Callable[..., Any]:
        """ONNX version_19 Resize op."""
        cls._prepare(node, inputs, onnx_resize)
        return onnx_resize

def is_not_empty(x):
  if x is None:
    return False
  if isinstance(x, jax.typing.ArrayLike):
    return x.size != 0
  return not not x

@functools.partial(jax.jit, static_argnames=("mode", "scales"))
def onnx_resize(x, roi=[], idk=None, sizes=None, mode="nearest", scales=None):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#Resize for more details."""
  if is_not_empty(roi) or is_not_empty(sizes):
    raise NotImplementedError("Resize with roi or sizes is not implemented.")
  if scales is not None:
    x = jax.image.resize(x,
                         (np.array(x.shape) * np.array(scales)).astype(np.int32).tolist(),
                         method=mode)
  return [x]
