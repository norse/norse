import functools
from numbers import Number
import typing

# pytype: disable=import-error
from typing import Any, Callable, Type, TypeVar, NamedTupleMeta

# pytype: enable=import-error

import torch
from torch.utils._pytree import (
    PyTree,
    _namedtuple_flatten,
    _namedtuple_unflatten,
    register_pytree_node,
    tree_map,
)

# Thanks to https://stackoverflow.com/a/50369521
NamedTuple = typing.NamedTuple
if hasattr(typing.NamedTuple, "__mro_entries__"):
    # Python 3.9 fixed and broke multiple inheritance in a different way
    # see https://github.com/python/cpython/issues/88089
    NamedTuple = typing._NamedTuple

T = TypeVar("T")


def map_only(ty: Type[T]) -> Callable[[Callable[[T], Any]], Callable[[Any], Any]]:
    """
    Returns a function that allows mapping on a certain type.

    Usage:
        mapping = map_only(some_type)
        tree_map(mapping(fn), tree)
    """

    def callback(f: Callable[[T], Any]) -> Callable[[Any], Any]:
        @functools.wraps(f)
        def inner(x: T) -> Any:
            return f(x) if isinstance(x, ty) else x

        return inner

    return callback


def tree_map_only(ty: Type[T], fn: Callable[[T], Any], pytree: PyTree) -> PyTree:
    """Maps a pytree of a certain type with a given function"""
    return tree_map(map_only(ty)(fn), pytree)


def register_tuple(typ: Any):
    register_pytree_node(typ, _namedtuple_flatten, _namedtuple_unflatten)


class MultipleInheritanceNamedTupleMeta(NamedTupleMeta):
    """A meta class to instantiate and register named tuples"""

    def __new__(mcls, typename, bases, ns):
        cls_obj = super().__new__(mcls, typename + "_nm_base", (NamedTuple,), ns)
        t = type(typename, bases + (cls_obj,), {})
        register_tuple(t)  # Registers the tuple in the pytree registry
        return t


class StateTuple:
    """
    A base class for state-tuples that allow operations like `cuda`, `to`, `int`, etc.

    Usage:
    >>> from norse.torch.utils import pytree
    >>> class SomeState(pytree.NamedTuple, pytree.StateTuple):
    >>>     x: torch.Tensor = torch.tensor(0.0)
    >>>
    >>> s = SomeState()
    >>> s.x # torch.tensor([0])
    >>> s.cuda() # torch.tensor([0], device='cuda:0')
    """

    def broadcast(self, template: torch.Tensor):
        """
        Broadcasts a state tensor according to a given template, returning a tensor of the same shape on the same device.

        Example
        >>> input_tensor = ...
        >>> state.broadcast(input_tensor)

        Arguments:
            template (torch.Tensor): The tensor template whose shape we broadcast to
        """
        return tree_map_only(
            torch.Tensor,
            functools.partial(broadcast_input, template=template),
            self,
        )

    def cuda(self):
        return self.to("cuda")

    def cpu(self):
        return self.to("cpu")

    def int(self):
        return tree_map_only(torch.Tensor, lambda x: x.int(), self)

    def float(self):
        return tree_map_only(torch.Tensor, lambda x: x.float(), self)

    def to(self, device):
        return tree_map_only(
            torch.Tensor, lambda x: x.to(device).requires_grad_(x.requires_grad), self
        )


def broadcast_input(potential_scalar: Any, template: torch.Tensor):
    if (  # Is single-value tensor
        isinstance(potential_scalar, torch.Tensor) and potential_scalar.numel() == 1
    ) or (  # Is scalar
        isinstance(potential_scalar, Number)
    ):  # Is similarly-shaped tensor
        return torch.full_like(template, potential_scalar.item())
    elif (
        isinstance(potential_scalar, torch.Tensor)
        and potential_scalar.shape == template.shape
    ):
        return potential_scalar
    else:
        raise ValueError(
            f"Cannot broadcast tensor because it is non-scalar (shape {potential_scalar.shape}) with a different shape than the template ({template.shape})"
        )
