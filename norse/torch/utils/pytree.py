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
    tree_map,
)

try:
    from torch.utils._pytree import register_pytree_node
except ImportError:  # Support older versions of PyTorch
    from torch.utils._pytree import _register_pytree_node as register_pytree_node

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

    # Disallow creation of weakrefs (like in NamedTuples)
    # by removing the __weakref__ slot
    __slots__ = []

    def broadcast_to(
        self, template: typing.Union[torch.Tensor, torch.Size, typing.Tuple[int]]
    ):
        """
        Broadcasts a state tensor according to a given template, returning a tensor of the same shape on the same device.

        Example
        >>> input_tensor = ...
        >>> state.broadcast_to(input_tensor)

        Arguments:
            template (torch.Tensor): The tensor template whose shape we broadcast to
        """
        if isinstance(template, typing.Tuple):
            template = torch.Size(template)
        f = (
            functools.partial(broadcast_size, template=template)
            if isinstance(template, torch.Size)
            else functools.partial(broadcast_input, template=template)
        )
        return tree_map_only(torch.Tensor, f, self)

    def clone(self):
        """
        Clones the state tuple.

        Example
        >>> s1 = SomeState()
        >>> s2 = s1.clone()

        Returns:
            StateTuple: A clone of the state tuple
        """
        return tree_map_only(torch.Tensor, lambda x: x.clone(), self)

    def cuda(self):
        return self.to("cuda")

    def cpu(self):
        return self.to("cpu")

    def detach(self):
        return tree_map_only(torch.Tensor, lambda x: x.detach(), self)

    def requires_grad_(self, requires_grad: bool):
        return tree_map_only(
            torch.Tensor, lambda x: x.requires_grad_(requires_grad), self
        )

    def int(self):
        return tree_map_only(torch.Tensor, lambda x: x.int(), self)

    def float(self):
        return tree_map_only(torch.Tensor, lambda x: x.float(), self)

    def to_dense(self):
        return tree_map_only(torch.Tensor, lambda x: x.to_dense(), self)

    def to_sparse(self):
        return tree_map_only(torch.Tensor, lambda x: x.to_sparse(), self)

    def to(self, device):
        return tree_map_only(
            torch.Tensor, lambda x: x.to(device).requires_grad_(x.requires_grad), self
        )


def broadcast_input(potential_scalar: Any, template: torch.Tensor):
    if (  # Is single-value tensor
        isinstance(potential_scalar, torch.Tensor) and potential_scalar.numel() == 1
    ) or (  # Is scalar
        isinstance(potential_scalar, Number)
    ):
        return torch.full_like(template, float(potential_scalar))
    elif (  # Is similarly-shaped tensor
        isinstance(potential_scalar, torch.Tensor)
        and potential_scalar.shape == template.shape
    ):
        return potential_scalar
    else:
        raise ValueError(
            f"Cannot broadcast tensor because it is non-scalar (shape {potential_scalar.shape}) with a different shape than the template ({template.shape})"
        )


def broadcast_size(potential_scalar: Any, template: torch.Size) -> torch.Tensor:
    if (  # Is single-value tensor
        isinstance(potential_scalar, torch.Tensor) and potential_scalar.numel() == 1
    ) or (  # Is scalar
        isinstance(potential_scalar, Number)
    ):
        return torch.full(template, float(potential_scalar))
    elif (  # Is similarly-shaped tensor
        isinstance(potential_scalar, torch.Tensor)
        and potential_scalar.shape == template
    ):
        return potential_scalar
    else:
        raise ValueError(
            f"Cannot broadcast tensor because it is non-scalar (shape {potential_scalar.shape}) with a different shape than the template ({template})"
        )


class StateTupleMeta(NamedTupleMeta):
    """
    A meta class to instantiate and register state classes as named tuples while enriching
    them with the StateTuple methods
    """

    def __new__(mcls, typename, bases, ns):
        t = super().__new__(mcls, typename, bases, ns)
        # Registers the tuple in the pytree registry
        register_tuple(t)
        # Enrich the tuple with the StateTuple methods
        methods = {
            name: fun
            for name, fun in StateTuple.__dict__.items()
            if not name.startswith("_")
        }
        for name, fun in methods.items():
            setattr(t, name, fun)
        return t
