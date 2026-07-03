# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""torch.compile custom-op framework for Transformer Engine."""

from __future__ import annotations
import dataclasses
import warnings
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import torch

from .traceable_utils import _contiguous_stride, _slot_count, _maybe_reassemble_tensor_subclass
from ..quantized_tensor import (
    QuantizedTensor,
    QuantizedTensorStorage,
    Quantizer,
    _STORAGE_REGISTRY,
    _quantized_tensor_passthrough_ops,
    prepare_for_saving,
)

_TE_OP_NAMESPACE = "transformer_engine_compile"


# ``None`` entries in an op's flat ``Tensor[]`` return are smuggled through a
# 0-element uint8 tensor: a non-nullable ``Tensor[]`` schema is required for
# ``register_autograd`` to attach a ``grad_fn`` to the outputs.
#
# TODO: once https://github.com/pytorch/pytorch/pull/187434 lands, a nullable
# ``Tensor?[]`` return schema lets ``None`` pass through directly and this
# sentinel encoding (``_encode_none`` / ``_decode_none``) can be removed.
_NONE_SENTINEL_DTYPE = torch.uint8


def _encode_none(t: Optional[torch.Tensor]) -> torch.Tensor:
    """Replace ``None`` with a 0-element uint8 sentinel tensor."""
    if t is None:
        return torch.empty(0, dtype=_NONE_SENTINEL_DTYPE)
    return t


def _decode_none(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Inverse of :func:`_encode_none`."""
    if t is None:
        return None
    if t.numel() == 0 and t.dtype == _NONE_SENTINEL_DTYPE:
        return None
    return t


# --------------------------------------------------------------------------- #
# OpaqueValueBundle: bundle of simple / value-opaque Python values
# --------------------------------------------------------------------------- #


class OpaqueValueBundle:
    """Opaque value-type bundle of simple Python values.

    Wraps a ``{name: value}`` dict so many small non-Tensor args pass through a
    single custom-op input; registered as a torch.compile *value* opaque type
    (Dynamo specializes the graph on its contents). Allowed values: primitives
    in :attr:`PRIMITIVE_TYPES` (incl. ``torch.Size``), ``enum.Enum``, any
    registered value-opaque type (e.g. TE quantizers), plus nested tuples /
    lists / dicts thereof (so a bundle can carry a ``__tensor_flatten__``
    context verbatim).
    """

    PRIMITIVE_TYPES: Tuple[type, ...] = (
        type(None),
        bool,
        int,
        float,
        str,
        torch.dtype,
        torch.device,
        torch.Size,
    )

    @classmethod
    def is_simple_value(cls, value: Any) -> bool:
        """Whether ``value`` may be stored inside an instance (recursive)."""
        if isinstance(value, cls.PRIMITIVE_TYPES):
            return True
        if isinstance(value, Enum):
            return True
        if _is_opaque_value_type(type(value)):
            return True
        if isinstance(value, dict):
            return all(
                isinstance(k, str) and cls.is_simple_value(v) for k, v in value.items()
            )
        if isinstance(value, (list, tuple)):
            return all(cls.is_simple_value(v) for v in value)
        return False

    @classmethod
    def _to_hashable(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return tuple(sorted((k, cls._to_hashable(v)) for k, v in value.items()))
        if isinstance(value, (list, tuple, torch.Size)):
            return tuple(cls._to_hashable(v) for v in value)
        return value

    @classmethod
    def _fmt_simple(cls, value: Any) -> str:
        """Repr for a value, evaluable in a context with ``torch`` globals."""
        if isinstance(value, torch.dtype):
            return f"__import__('torch').{str(value).split('.')[-1]}"
        if isinstance(value, torch.device):
            return f"__import__('torch').device({str(value)!r})"
        if isinstance(value, torch.Size):
            return f"__import__('torch').Size({list(value)!r})"
        # Enum before primitives: IntEnum is also ``int`` but must render as
        # ``EnumName.MEMBER`` (the Enum class is added to globals by ``_collect``).
        if isinstance(value, Enum):
            return f"{type(value).__name__}.{value.name}"
        if isinstance(value, dict):
            body = ", ".join(f"{k!r}: {cls._fmt_simple(v)}" for k, v in value.items())
            return f"{{{body}}}"
        if isinstance(value, list):
            return "[" + ", ".join(cls._fmt_simple(v) for v in value) + "]"
        if isinstance(value, tuple):
            body = ", ".join(cls._fmt_simple(v) for v in value)
            return f"({body},)" if len(value) == 1 else f"({body})"
        if _is_opaque_value_type(type(value)):
            return value.__fx_repr__()[0]
        return repr(value)

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        data = dict(data) if data else {}
        for k, v in data.items():
            if not OpaqueValueBundle.is_simple_value(v):
                raise TypeError(
                    f"OpaqueValueBundle field '{k}' has unsupported type "
                    f"{type(v).__name__}; only simple primitives, Enum, "
                    "torch.Size, registered value-opaque types and nested "
                    "tuples / lists / dicts thereof are allowed."
                )
        self._data: Dict[str, Any] = data
        self._frozen: Tuple[Tuple[str, Any], ...] = tuple(
            (k, OpaqueValueBundle._to_hashable(v)) for k, v in sorted(data.items())
        )

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def get(self, key: str, default: Any = None) -> Any:
        """Return ``self._data.get(key, default)``."""
        return self._data.get(key, default)

    def as_dict(self) -> Dict[str, Any]:
        """Return a shallow copy of the stored mapping."""
        return dict(self._data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OpaqueValueBundle):
            return NotImplemented
        return self._frozen == other._frozen

    def __hash__(self) -> int:
        return hash(self._frozen)

    def __fx_repr__(self) -> Tuple[str, Dict[str, Any]]:
        items = ", ".join(
            f"{k!r}: {OpaqueValueBundle._fmt_simple(v)}" for k, v in self._data.items()
        )
        globals_: Dict[str, Any] = {"OpaqueValueBundle": OpaqueValueBundle}

        def _collect(value: Any) -> None:
            if isinstance(value, dict):
                for v in value.values():
                    _collect(v)
                return
            if isinstance(value, (list, tuple)):
                for v in value:
                    _collect(v)
                return
            if isinstance(value, Enum):
                globals_[type(value).__name__] = type(value)
                return
            if isinstance(value, OpaqueValueBundle.PRIMITIVE_TYPES):
                return
            if _is_opaque_value_type(type(value)):
                _, extra = value.__fx_repr__()
                globals_.update(extra)

        for v in self._data.values():
            _collect(v)
        return (f"OpaqueValueBundle({{{items}}})", globals_)


try:
    from torch._library.opaque_object import (  # pylint: disable=import-outside-toplevel
        get_opaque_type_name,
        is_opaque_value_type as _is_opaque_value_type,
        is_opaque_reference_type as _is_opaque_reference_type,
        register_opaque_type,
    )

    register_opaque_type(OpaqueValueBundle, typ="value")
    _OPAQUE_VALUE_BUNDLE_TYPE_NAME: Optional[str] = get_opaque_type_name(OpaqueValueBundle)
except Exception:  # pylint: disable=broad-exception-caught  # pragma: no cover - older torch without opaque_object
    _is_opaque_value_type = None
    _is_opaque_reference_type = None
    _OPAQUE_VALUE_BUNDLE_TYPE_NAME = None


def _pg_pickle_stub(*args: Any) -> None:  # pragma: no cover
    raise RuntimeError("ProcessGroup cannot be unpickled — cache-key use only")


def _ensure_distributed_opaque_types() -> None:
    """Register ``torch.distributed.ProcessGroup`` as a *reference* opaque type.

    A process group is live distributed state: unlike a value-opaque quantizer
    (which Dynamo bakes into the graph as a constant), it must be carried through
    the custom op as a graph *input*. PyTorch supports this via
    ``register_opaque_type(ProcessGroup, typ="reference")`` but only auto-runs it
    when ``torch.distributed.tensor`` (DTensor) is imported; TE may not import
    that, so trigger the same idempotent registration here. Best-effort: on
    builds without the opaque-object / distributed APIs this is a no-op and the
    process-group field simply falls back to eager under torch.compile.

    Also registers a ``copyreg`` reducer that lets ``FxGraphCachePickler`` hash
    graphs containing a ``ProcessGroup`` input without crashing.  Without this,
    inductor logs "Failed to pickle cache key" warnings and bypasses the FX
    graph disk cache for every distributed compiled call.  The reducer encodes
    the group as (world_size, rank, backend) — enough to distinguish configs —
    and raises on reconstruct since deserialization is never needed for hashing.
    """
    if _is_opaque_reference_type is None:
        return
    try:  # pylint: disable=import-outside-toplevel
        from torch.distributed.device_mesh import _register_distributed_opaque_types

        _register_distributed_opaque_types()
    except Exception:  # pylint: disable=broad-exception-caught
        pass

    # Workaround for PyTorch issue: FxGraphCachePickler handles FakeScriptObject
    # but not the real ProcessGroup that appears in example_inputs at inductor
    # compile time.  Register a copyreg reducer so the pickler can hash the key.
    try:  # pylint: disable=import-outside-toplevel
        import copyreg
        import torch.distributed as dist
        from torch._C._distributed_c10d import ProcessGroup

        if ProcessGroup not in copyreg.dispatch_table:

            def _pg_reduce(pg: ProcessGroup) -> tuple:  # type: ignore[valid-type]
                try:
                    return _pg_pickle_stub, (
                        dist.get_world_size(pg),
                        dist.get_rank(pg),
                        dist.get_backend(pg),
                    )
                except Exception:  # pylint: disable=broad-exception-caught
                    return _pg_pickle_stub, (id(pg),)

            copyreg.pickle(ProcessGroup, _pg_reduce)
    except Exception:  # pylint: disable=broad-exception-caught
        pass


_ensure_distributed_opaque_types()


# --------------------------------------------------------------------------- #
# Storage flatten / unflatten (value-opaque quantizer; no ProcessGroup)
# --------------------------------------------------------------------------- #


def _storage_flatten(value: Any) -> Tuple["OpaqueValueBundle", List[torch.Tensor]]:
    """Split a ``QuantizedTensor`` / bare storage into ``(meta, Tensor[])``.

    The flatten context (embedding the value-opaque quantizer) plus inner names
    and -- for a wrapper subclass -- the outer geometry are stashed in the bundle
    so :func:`_storage_unflatten` can rebuild without PyTorch's ``outer_size``.
    """
    inner_names, ctx = value.__tensor_flatten__()
    meta = dict(ctx)
    meta["_inner_names"] = list(inner_names)
    if isinstance(value, torch.Tensor):
        meta["_outer_shape"] = torch.Size(value.shape)
    tensors = [getattr(value, name) for name in inner_names]
    return OpaqueValueBundle(meta), tensors


def _storage_unflatten(meta: Any, tensors: List[torch.Tensor]) -> Any:
    """Inverse of :func:`_storage_flatten`."""
    meta_dict = meta.as_dict() if isinstance(meta, OpaqueValueBundle) else dict(meta)
    inner_names = meta_dict["_inner_names"]
    inner = dict(zip(inner_names, tensors))
    outer_shape = meta_dict.get("_outer_shape")
    stride = _contiguous_stride(tuple(outer_shape)) if outer_shape is not None else None
    return QuantizedTensorStorage.__tensor_unflatten__(inner, meta_dict, outer_shape, stride)


# --------------------------------------------------------------------------- #
# Field buckets: dataclass field <-> flat torch.library slot(s)
# --------------------------------------------------------------------------- #


def _strip_optional(annot: Any) -> Tuple[Any, bool]:
    """If ``annot`` is ``Optional[X]`` return ``(X, True)``; else ``(annot, False)``."""
    if get_origin(annot) is Union:
        args = get_args(annot)
        if type(None) in args:
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return non_none[0], True
    return annot, False


class _Bucket:
    """Maps one (or, for the aggregating bucket, several) dataclass field(s)
    to/from a contiguous run of custom-op schema *slots*.

    A custom op only takes flat, simply-typed arguments, but a TE op takes a
    single ``@dataclass`` of mixed fields. Each bucket knows how to translate
    its kind of field both ways. ``try_build`` and ``schema_slots`` run once at
    registration (to build the op's schema); ``pack`` and ``unpack`` run on each
    call and must agree on the slot layout that ``schema_slots`` declares.
    """

    @classmethod
    def try_build(cls, name: str, annot: Any) -> Optional["_Bucket"]:
        """Decide whether this bucket type handles the field ``name`` given its
        type annotation ``annot``; return a configured bucket if so, else
        ``None`` so the next candidate is tried.

        Called once per field at registration, in :data:`_FIELD_BUCKETS`
        priority order.
        """
        raise NotImplementedError

    def schema_slots(self) -> List[Tuple[str, str]]:
        """Declare the schema slots this field occupies, each as a
        ``(slot_name, schema_type)`` pair (e.g. ``("bias", "Tensor?")``).

        Concatenated across all buckets to form the op's schema string.
        """
        raise NotImplementedError

    def pack(self, owner: Any) -> List[Tuple[str, Any]]:
        """Read this field from the dataclass ``owner`` and produce the concrete
        value for each of its schema slots, as ``(slot_name, value)`` pairs.

        Composite values are flattened to fit the (tensor-only) slots: e.g. a
        quantized tensor is split into its plain inner buffers plus a metadata
        bundle. Inverse of :meth:`unpack`.
        """
        raise NotImplementedError

    def unpack(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        """Read this field's slots back from the op arguments ``args`` and write
        the reconstructed field value into ``kwargs`` (rebuilding any flattened
        composite). The filled ``kwargs`` are then used to rebuild the original
        dataclass for the eager implementation. Inverse of :meth:`pack`.
        """
        raise NotImplementedError

    def grad_slot(self) -> Optional[int]:
        """Index (within this bucket's :meth:`schema_slots`) of the slot that
        carries a gradient, or ``None`` if the field is not differentiable.

        Used to map ``input_tensors_for_grad`` names onto backward grad-output
        positions. Non-tensor buckets (quantizers, metadata) return ``None``.
        """
        return None


class _UniversalKind(Enum):
    """What a universal-tensor slot group carries, tagged in its ``__meta``."""

    NONE = "none"
    TENSOR = "tensor"
    STORAGE = "storage"


class _UniversalTensorBucket(_Bucket):
    """``Tensor | QuantizedTensorStorage`` (also subclass tensor) field.

    Three slots regardless of value: ``<name>`` (``Tensor?`` -- plain / subclass
    tensor passes through, ``None`` for bare storage), ``<name>__tensors``
    (``Tensor[]`` flat inner tensors when flattened), ``<name>__meta``
    (``OpaqueValueBundle`` flatten metadata + a ``__kind__`` marker).
    """

    KIND_KEY = "__kind__"

    def __init__(self, name: str) -> None:
        self.name = name

    def slot_name(self) -> str:
        """Primary slot name for a plain / subclass tensor."""
        return self.name

    def slot_tensors(self) -> str:
        """Flat inner-tensor slot name."""
        return self.name + "__tensors"

    def slot_meta(self) -> str:
        """Flatten-metadata slot name."""
        return self.name + "__meta"

    def schema_slots(self) -> List[Tuple[str, str]]:
        return [
            (self.slot_name(), "Tensor?"),
            (self.slot_tensors(), "Tensor[]"),
            (self.slot_meta(), _OPAQUE_VALUE_BUNDLE_TYPE_NAME),
        ]

    @staticmethod
    def _is_tensor_storage_union(annot: Any) -> bool:
        if get_origin(annot) is not Union:
            return False
        members = [a for a in get_args(annot) if a is not type(None)]
        if torch.Tensor not in members:
            return False
        return any(
            isinstance(m, type) and issubclass(m, QuantizedTensorStorage) for m in members
        )

    @classmethod
    def try_build(cls, name: str, annot: Any) -> Optional["_UniversalTensorBucket"]:
        if cls._is_tensor_storage_union(annot):
            return cls(name)
        return None

    def pack(self, owner: Any) -> List[Tuple[str, Any]]:
        value = getattr(owner, self.name)
        if value is None:
            return [
                (self.slot_name(), None),
                (self.slot_tensors(), []),
                (self.slot_meta(), OpaqueValueBundle({self.KIND_KEY: _UniversalKind.NONE})),
            ]
        if isinstance(value, torch.Tensor):
            # Plain tensor *and* subclass (e.g. Float8Tensor) pass through the
            # ``Tensor?`` slot; subclass flattening (if any) is done by the
            # outer op's ``register_torch_dispatch`` rule.
            return [
                (self.slot_name(), value),
                (self.slot_tensors(), []),
                (self.slot_meta(), OpaqueValueBundle({self.KIND_KEY: _UniversalKind.TENSOR})),
            ]
        if isinstance(value, QuantizedTensorStorage):
            meta, tensors = _storage_flatten(value)
            meta._data[self.KIND_KEY] = _UniversalKind.STORAGE
            return [
                (self.slot_name(), None),
                (self.slot_tensors(), list(tensors)),
                (self.slot_meta(), meta),
            ]
        raise TypeError(
            f"field {self.name!r} expected None, torch.Tensor, or "
            f"QuantizedTensorStorage, got {type(value).__name__}"
        )

    def unpack(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        meta = args[self.slot_meta()]
        kind = meta.get(self.KIND_KEY)
        if kind == _UniversalKind.NONE:
            kwargs[self.name] = None
        elif kind == _UniversalKind.TENSOR:
            kwargs[self.name] = args[self.slot_name()]
        else:
            kwargs[self.name] = _storage_unflatten(meta, args[self.slot_tensors()])

    def grad_slot(self) -> Optional[int]:
        # Gradient flows to the plain / subclass tensor slot (``slot_name``,
        # the first of the three).
        return 0


class _TensorBucket(_Bucket):
    """``Tensor`` / ``Optional[Tensor]`` -> single ``Tensor`` / ``Tensor?`` slot."""

    def __init__(self, name: str, is_optional: bool) -> None:
        self.name = name
        self.type_str = "Tensor?" if is_optional else "Tensor"

    @classmethod
    def try_build(cls, name: str, annot: Any) -> Optional["_TensorBucket"]:
        stripped, is_optional = _strip_optional(annot)
        if stripped is torch.Tensor:
            return cls(name, is_optional)
        return None

    def schema_slots(self) -> List[Tuple[str, str]]:
        return [(self.name, self.type_str)]

    def pack(self, owner: Any) -> List[Tuple[str, Any]]:
        return [(self.name, getattr(owner, self.name))]

    def unpack(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        kwargs[self.name] = args[self.name]

    def grad_slot(self) -> Optional[int]:
        return 0


class _QuantizerBucket(_Bucket):
    """``Quantizer`` / ``Optional[Quantizer]`` -> one own ``OpaqueValueBundle`` slot.

    Each quantizer gets its own dedicated slot. The field is annotated with the
    base ``Quantizer`` (not itself a registered opaque type), so the simple
    bundle would not claim it.
    """

    KEY = "q"

    def __init__(self, name: str) -> None:
        self.name = name

    def slot(self) -> str:
        """Opaque quantizer metadata slot name."""
        return self.name + "__q"

    @classmethod
    def try_build(cls, name: str, annot: Any) -> Optional["_QuantizerBucket"]:
        stripped, _ = _strip_optional(annot)
        if not isinstance(stripped, type):
            return None
        if issubclass(stripped, Quantizer):
            return cls(name)
        return None

    def schema_slots(self) -> List[Tuple[str, str]]:
        return [(self.slot(), _OPAQUE_VALUE_BUNDLE_TYPE_NAME)]

    def pack(self, owner: Any) -> List[Tuple[str, Any]]:
        return [(self.slot(), OpaqueValueBundle({self.KEY: getattr(owner, self.name)}))]

    def unpack(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        kwargs[self.name] = args[self.slot()][self.KEY]


class _ReferenceOpaqueBucket(_Bucket):
    """``ProcessGroup`` (or any reference-opaque type) -> one own opaque slot.

    A reference-opaque object is live, stateful black-box data (e.g. a
    ``torch.distributed.ProcessGroup``): it cannot be specialized on or baked
    into the graph as a constant the way a value-opaque quantizer is. torch.compile
    instead carries it through as a graph *input*, so it passes straight through
    its own schema slot (no ``OpaqueValueBundle`` wrapper). The field is annotated
    with a concrete type registered via ``register_opaque_type(..., typ="reference")``.

    On the fake / setup-context path the slot holds a ``FakeScriptObject`` (or
    ``None``); it is assigned to the field verbatim, so the fake impl must never
    read the object's contents.
    """

    def __init__(self, name: str, type_name: str, is_optional: bool) -> None:
        self.name = name
        self.type_str = f"{type_name}?" if is_optional else type_name

    @classmethod
    def try_build(cls, name: str, annot: Any) -> Optional["_ReferenceOpaqueBucket"]:
        if _is_opaque_reference_type is None:
            return None
        stripped, is_optional = _strip_optional(annot)
        if not isinstance(stripped, type):
            return None
        if _is_opaque_reference_type(stripped):
            return cls(name, get_opaque_type_name(stripped), is_optional)
        return None

    def schema_slots(self) -> List[Tuple[str, str]]:
        return [(self.name, self.type_str)]

    def pack(self, owner: Any) -> List[Tuple[str, Any]]:
        return [(self.name, getattr(owner, self.name))]

    def unpack(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        kwargs[self.name] = args[self.name]


class _SimpleBundleBucket(_Bucket):
    """Aggregates every simple-typed field into a single OpaqueValueBundle."""

    SLOT = "_simple_meta"

    def __init__(self, names: List[str]) -> None:
        self.names = list(names)

    @classmethod
    def matches_field(cls, annot: Any) -> bool:
        """Whether ``annot`` (Optional-aware, recursive) is bundle-simple."""
        annot, _ = _strip_optional(annot)
        if annot in OpaqueValueBundle.PRIMITIVE_TYPES:
            return True
        if isinstance(annot, type) and issubclass(annot, Enum):
            return True
        if (
            isinstance(annot, type)
            and _is_opaque_value_type is not None
            and _is_opaque_value_type(annot)
        ):
            return True
        if get_origin(annot) in (tuple, list):
            inner = [a for a in get_args(annot) if a is not Ellipsis]
            return bool(inner) and all(cls.matches_field(a) for a in inner)
        return False

    def schema_slots(self) -> List[Tuple[str, str]]:
        return [(self.SLOT, _OPAQUE_VALUE_BUNDLE_TYPE_NAME)]

    def pack(self, owner: Any) -> List[Tuple[str, Any]]:
        return [(self.SLOT, OpaqueValueBundle({n: getattr(owner, n) for n in self.names}))]

    def unpack(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        if self.SLOT not in args:
            return
        meta = args[self.SLOT]
        for n in self.names:
            kwargs[n] = meta[n]


class _UnknownBucket(_Bucket):
    """Fallback for fields no other bucket claims.

    Emits no slot; pack rejects non-trivial values (anything other than
    ``None`` / all-``None`` sequence); unpack restores the field as ``None``.
    """

    def __init__(self, name: str, owner_cls_name: str) -> None:
        self.name = name
        self.owner_cls_name = owner_cls_name

    @staticmethod
    def _is_trivial(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, (list, tuple)):
            return all(v is None for v in value)
        return False

    def schema_slots(self) -> List[Tuple[str, str]]:
        return []

    def pack(self, owner: Any) -> List[Tuple[str, Any]]:
        value = getattr(owner, self.name, None)
        if not self._is_trivial(value):
            raise TypeError(
                f"{self.owner_cls_name} field {self.name!r} has a type not "
                "supported by torch.compile (not Tensor, simple, or Quantizer) "
                "and carries a non-trivial value; add a matching bucket in "
                "dynamo.py to handle it."
            )
        return []

    def unpack(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        kwargs[self.name] = None


# Buckets, in priority order, owning ``try_build`` for a single field.
_FIELD_BUCKETS: Tuple[type, ...] = (
    _UniversalTensorBucket,
    _TensorBucket,
    _ReferenceOpaqueBucket,
    _QuantizerBucket,
)


def _resolved_field_annotations(cls: type) -> List[Tuple[str, Any]]:
    """Return ``[(field_name, resolved_type), ...]`` for a dataclass."""
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls.__name__} must be a @dataclass to be a TE op arg container.")
    try:
        hints = get_type_hints(cls)
    except Exception:  # pylint: disable=broad-exception-caught
        hints = {}
    return [(f.name, hints.get(f.name, f.type)) for f in dataclasses.fields(cls)]


def _get_buckets(cls: type) -> List[_Bucket]:
    """Build the bucket list for a dataclass from its field annotations."""
    if _OPAQUE_VALUE_BUNDLE_TYPE_NAME is None:
        raise RuntimeError(
            f"{cls.__name__} cannot be turned into a TE custom op: OpaqueValueBundle "
            "is not registered as a torch._library value-opaque type (PyTorch build "
            "without opaque-object support)."
        )
    buckets: List[_Bucket] = []
    simple_names: List[str] = []
    for name, annot in _resolved_field_annotations(cls):
        built: Optional[_Bucket] = None
        for bucket_cls in _FIELD_BUCKETS:
            built = bucket_cls.try_build(name, annot)
            if built is not None:
                break
        if built is not None:
            buckets.append(built)
        elif _SimpleBundleBucket.matches_field(annot):
            simple_names.append(name)
        else:
            buckets.append(_UnknownBucket(name, cls.__name__))
    if simple_names:
        buckets.append(_SimpleBundleBucket(simple_names))
    return buckets


def _build_schema(buckets: List[_Bucket]) -> Tuple[str, List[str]]:
    """Return ``(schema_arg_str, slot_names)`` for a bucket list."""
    spec = [slot for b in buckets for slot in b.schema_slots()]
    names = [name for name, _ in spec]
    schema_str = "(" + ", ".join(f"{type_str} {name}" for name, type_str in spec) + ")"
    return schema_str, names


def _pack(obj: Any, buckets: List[_Bucket]) -> Dict[str, Any]:
    """Build the op's flat ``{slot_name: value}`` argument dict from an args
    dataclass ``obj`` (e.g. ``LinearFwdArgs``), by collecting every bucket's
    packed slot(s). Inverse of :func:`_unpack`.
    """
    out: Dict[str, Any] = {}
    for bucket in buckets:
        for name, value in bucket.pack(obj):
            out[name] = value
    return out


def _unpack(cls: type, args: Dict[str, Any], buckets: List[_Bucket]) -> Any:
    """Rebuild a fresh args dataclass ``cls`` (e.g. ``LinearFwdArgs``) from the
    op's flat slot ``args`` dict, by letting every bucket restore its field(s).
    Inverse of :func:`_pack`.
    """
    kwargs: Dict[str, Any] = {}
    for bucket in buckets:
        bucket.unpack(args, kwargs)
    obj = cls.__new__(cls)
    for k, v in kwargs.items():
        object.__setattr__(obj, k, v)
    return obj

# --------------------------------------------------------------------------- #
# Op outputs <-> flat ``Tensor[]`` payload: this is how an op returns / saves
# quantized tensors (and wrapper subclasses). Outputs are flattened to their
# inner buffers on the way out and rebuilt via ``__tensor_unflatten__`` on the
# way back; the fake impl returns actual tensors whose __tensor_flatten__
# provides the template for reassembly.
# --------------------------------------------------------------------------- #


def _value_to_flat_tensors(
    value: Optional[Union[torch.Tensor, QuantizedTensorStorage]],
) -> List[torch.Tensor]:
    """Return the flat ``Tensor[]`` slots that represent one op output ``value``.

    Inverse of :func:`_maybe_reassemble_tensor_subclass`; the slot count matches
    :func:`_slot_count`.
    """
    if value is None:
        return [_encode_none(None)]
    if isinstance(value, torch.Tensor):
        if type(value) is not torch.Tensor and hasattr(  # pylint: disable=unidiomatic-typecheck
            value, "__tensor_flatten__"
        ):
            inner_names, _ = value.__tensor_flatten__()
            return [_encode_none(getattr(value, n)) for n in inner_names]
        return [_encode_none(value)]
    if hasattr(value, "__tensor_flatten__"):
        inner_names, _ = value.__tensor_flatten__()
        return [_encode_none(getattr(value, n)) for n in inner_names]
    raise TypeError(
        f"unsupported value type {type(value).__name__}; expected None / "
        "torch.Tensor / tensor subclass / bare storage."
    )


# Trailing slots in every fwd-impl return: ``tensors_to_save, tensor_objects,
# ctx_attrs``. User-output count is ``len(result) - this``.
_FWD_TRAILING_SLOTS = 3


def _format_fwd_result(result: Any) -> List[torch.Tensor]:
    """Pack a fwd-impl return tuple into the op's ``Tensor[]`` payload.

    User outputs first, then saved-for-backward tensors in declaration order.
    """
    num_outputs = len(result) - _FWD_TRAILING_SLOTS
    flat: List[torch.Tensor] = []
    for value in result[:num_outputs]:
        flat.extend(_value_to_flat_tensors(value))
    saved = result[num_outputs] or ()
    for value in saved:
        flat.extend(_value_to_flat_tensors(value))
    return flat


def _format_bwd_result(grads: Any, num_grad_inputs: int, op_qualname: str) -> List[torch.Tensor]:
    """Pack a backward-impl return tuple into the op's ``Tensor[]`` payload.

    Each grad occupies exactly one slot (validated against ``num_grad_inputs``).
    """
    grads = list(grads)
    if len(grads) != num_grad_inputs:
        raise RuntimeError(
            f"{op_qualname} expected backward_impl to return {num_grad_inputs} grads "
            f"(one per input_tensors_for_grad entry), got {len(grads)}"
        )
    return [_encode_none(g) for g in grads]


def _split_fwd_fake_result(
    result: Tuple[Any, ...],
) -> Tuple[List[Any], List[Any], Dict[str, Any]]:
    """Slice a fwd fake-impl return into ``(user_fakes, saved_fakes, ctx_attrs)``."""
    num_outputs = len(result) - _FWD_TRAILING_SLOTS
    saved = result[num_outputs]
    ctx_attrs = result[num_outputs + 2]
    user_fakes = list(result[:num_outputs])
    saved_fakes = list(saved) if saved is not None else []
    ctx_attrs = dict(ctx_attrs) if ctx_attrs else {}
    return user_fakes, saved_fakes, ctx_attrs


# --------------------------------------------------------------------------- #
# Op registration
# --------------------------------------------------------------------------- #


def _resolve_grad_targets(
    fwd_buckets: List[_Bucket],
    fwd_arg_type: type,
    input_tensors_for_grad: List[str],
) -> Tuple[List[Any], List[int]]:
    """Validate ``input_tensors_for_grad`` and resolve the grad-output layout.

    Returns ``(fwd_slot_defaults, grad_targets)``: the per-slot no-grad template
    (``[]`` for ``Tensor[]`` slots, ``None`` otherwise) and, for each requested
    input name, the schema-slot index its gradient maps to.
    """
    fwd_slot_defaults: List[Any] = []
    name_to_slot: Dict[str, int] = {}
    slot_offset = 0
    for bucket in fwd_buckets:
        slots = bucket.schema_slots()
        for _, type_str in slots:
            fwd_slot_defaults.append([] if type_str.endswith("[]") else None)
        grad_slot = bucket.grad_slot()
        if grad_slot is not None:
            name_to_slot[bucket.name] = slot_offset + grad_slot
        slot_offset += len(slots)

    unknown = [n for n in input_tensors_for_grad if n not in name_to_slot]
    if unknown:
        raise ValueError(
            f"input_tensors_for_grad contains names not in {fwd_arg_type.__name__} "
            f"schema: {unknown}"
        )
    grad_targets = [name_to_slot[n] for n in input_tensors_for_grad]
    return fwd_slot_defaults, grad_targets


def _register_kernel(
    *,
    op_name: str,
    schema_str: str,
    arg_type: type,
    arg_names: List[str],
    buckets: List[_Bucket],
    impl: Callable[[Any], Any],
    fake_impl: Callable[[Any], Any],
    format_result: Callable[[Any], List[torch.Tensor]],
) -> Any:
    """Define the op via ``torch.library.custom_op`` with the real ``impl`` + the
    ``fake_impl``, returning the ``CustomOpDef``.

    The real kernel rebuilds the dataclass and runs ``impl``; the fake kernel
    runs the fake impl directly on the unpacked object. Both go through
    ``format_result``.
    """

    def _impl(*flat: Any) -> List[torch.Tensor]:
        kwargs = dict(zip(arg_names, flat))
        obj = _unpack(arg_type, kwargs, buckets)
        return format_result(impl(obj))

    def _fake(*flat: Any) -> List[torch.Tensor]:
        kwargs = dict(zip(arg_names, flat))
        obj = _unpack(arg_type, kwargs, buckets)
        return format_result(fake_impl(obj))

    op = torch.library.custom_op(
        f"{_TE_OP_NAMESPACE}::{op_name}", _impl, mutates_args=(), schema=schema_str
    )
    op.register_fake(_fake)
    return op


def _register_autograd_for_op(
    *,
    fwd_op: Any,
    bwd_op_name: str,
    fwd_arg_type: type,
    fwd_arg_names: List[str],
    fwd_buckets: List[_Bucket],
    bwd_arg_names: List[str],
    bwd_buckets: List[_Bucket],
    fwd_slot_defaults: List[Any],
    grad_targets: List[int],
    setup_context_user: Callable[..., Any],
    backward_obj_type: type,
    fwd_fake_impl: Callable[[Any], Tuple[Any, ...]],
) -> None:
    """Wire ``register_autograd`` on a forward op so its backward calls ``bwd_op_name``.

    ``setup_context`` re-runs the fwd fake impl to recover output / saved
    templates, reassembles each flat output chunk, and hands the saved tuple +
    ``ctx_attrs`` to the module's ``setup_context``.
    """

    def _setup_context(ctx, inputs, output):
        ctx._te_fwd_tensor_list_lengths = {
            i: len(value) for i, value in enumerate(inputs) if isinstance(value, list)
        }
        kwargs = dict(zip(fwd_arg_names, inputs))
        fwd_obj = _unpack(fwd_arg_type, kwargs, fwd_buckets)

        user_fakes, saved_fakes, ctx_attrs = _split_fwd_fake_result(fwd_fake_impl(fwd_obj))

        cursor = 0
        user_outputs: List[Any] = []
        for template in user_fakes:
            n = _slot_count(template)
            chunk = [_decode_none(t) for t in output[cursor : cursor + n]]
            cursor += n
            user_outputs.append(_maybe_reassemble_tensor_subclass(template, chunk))

        saved_list: List[Any] = []
        for template in saved_fakes:
            n = _slot_count(template)
            chunk = [_decode_none(t) for t in output[cursor : cursor + n]]
            cursor += n
            saved_list.append(_maybe_reassemble_tensor_subclass(template, chunk))

        bwd_obj = backward_obj_type()
        tensors_to_save_from_setup = setup_context_user(
            bwd_obj,
            fwd_obj,
            user_outputs[0] if len(user_fakes) == 1 else tuple(user_outputs),
            ctx_attrs,
            tuple(saved_list),
        )
        tensors_to_save, tensor_objects = prepare_for_saving(
            *(tensors_to_save_from_setup or ())
        )
        ctx.tensor_objects = tensor_objects
        ctx.save_for_backward(*tensors_to_save)
        ctx.bwd_obj = bwd_obj

    def _autograd_backward(ctx, *grad_outputs):
        bwd_obj = ctx.bwd_obj
        if hasattr(bwd_obj, "setup_saved_tensors"):
            bwd_obj.setup_saved_tensors(ctx)
        ctx.tensor_objects = None
        per_output_grads = grad_outputs[0]
        bwd_obj.grad_output = _decode_none(per_output_grads[0])
        kwargs = _pack(bwd_obj, bwd_buckets)
        bwd_args_flat = [kwargs[name] for name in bwd_arg_names]
        bwd_op = getattr(getattr(torch.ops, _TE_OP_NAMESPACE), bwd_op_name)
        grads = [_decode_none(g) for g in bwd_op(*bwd_args_flat)]
        out: List[Any] = list(fwd_slot_defaults)
        tensor_list_lengths = getattr(ctx, "_te_fwd_tensor_list_lengths", {})
        for pos, length in tensor_list_lengths.items():
            if isinstance(out[pos], list):
                out[pos] = [None] * length
        for pos, g in zip(grad_targets, grads):
            out[pos] = g
        return tuple(out)

    fwd_op.register_autograd(_autograd_backward, setup_context=_setup_context)


def _collect_universal_slot_offsets(buckets: List[_Bucket]) -> List[int]:
    """Start index of each ``_UniversalTensorBucket`` group in the flat args."""
    offsets: List[int] = []
    pos = 0
    for bucket in buckets:
        if isinstance(bucket, _UniversalTensorBucket):
            offsets.append(pos)
        pos += len(bucket.schema_slots())
    return offsets


def _flatten_subclass_into_slots(
    new_args: List[Any], slot_offsets: List[int], subclass: type
) -> None:
    """Rewrite each universal-bucket group whose ``Tensor?`` slot holds an
    instance of ``subclass`` into the storage layout (3 slots: name / tensors / meta).
    """
    for offset in slot_offsets:
        val = new_args[offset]
        if val is None or not isinstance(val, subclass):
            continue
        meta, tensors = _storage_flatten(val)
        meta._data[_UniversalTensorBucket.KIND_KEY] = _UniversalKind.STORAGE
        new_args[offset] = None
        new_args[offset + 1] = list(tensors)
        new_args[offset + 2] = meta


def _register_outer_forwarder(
    *,
    outer_op_name: str,
    schema_str: str,
    inner_op_name: str,
    buckets: Optional[List[_Bucket]] = None,
    subclass_list: Optional[List[type]] = None,
) -> Any:
    """Define the outer op via ``torch.library.custom_op``: forward to the inner
    op, optionally flattening registered subclass inputs in place first. Returns
    the ``CustomOpDef``.
    """
    inner_op = getattr(getattr(torch.ops, _TE_OP_NAMESPACE), inner_op_name)
    input_flatten_enabled = bool(subclass_list) and buckets is not None
    slot_offsets = _collect_universal_slot_offsets(buckets) if input_flatten_enabled else []

    def _forward(*flat: Any) -> List[torch.Tensor]:
        if not input_flatten_enabled:
            return inner_op(*flat)
        new_args = list(flat)
        for sub in subclass_list:
            _flatten_subclass_into_slots(new_args, slot_offsets, sub)
        return inner_op(*new_args)

    op = torch.library.custom_op(
        f"{_TE_OP_NAMESPACE}::{outer_op_name}", _forward, mutates_args=(), schema=schema_str
    )
    op.register_fake(_forward)
    return op


def _all_quantized_tensor_subclasses() -> List[type]:
    """Return every imported ``QuantizedTensor`` wrapper subclass."""
    import transformer_engine.pytorch.tensor  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import
    return [cls for cls in _STORAGE_REGISTRY.values() if issubclass(cls, QuantizedTensor)]


def register_custom_op(
    *,
    op_name: str,
    input_tensors_for_grad: List[str],
    fwd_arg_type: type,
    fwd_impl: Callable[[Any], Any],
    setup_context: Callable[..., Any],
    backward_arg_type: type,
    backward_obj: type,
    backward_impl: Callable[[Any], Any],
    fwd_fake_impl: Callable[[Any], Tuple[Any, ...]],
    bwd_fake_impl: Callable[[Any], Tuple[Any, ...]],
) -> Optional[Callable[..., Any]]:
    """Register a TE module's forward + backward as torch custom ops.

    Always two-tier: an inner ``<op>_base`` op carries the real schema /
    autograd, and an outer ``<op>`` op forwards to it, flattening any
    quantized-tensor wrapper inputs first via ``register_torch_dispatch`` (an
    empty subclass list simply makes the outer op a pass-through, so a pure
    plain-tensor / bf16 call goes straight through).

    Returns ``forward_fn(fwd_arg_type_instance)`` -- a drop-in for
    ``Function.apply`` under ``torch.compiler.is_compiling()`` that dispatches
    through the outer op and returns the user-facing outputs.

    Registration touches experimental ``torch.library`` / opaque-object APIs
    that may be missing on older PyTorch. If it fails, this warns once and
    returns ``None`` instead of raising, so callers can fall back to eager under
    ``torch.compile`` (a graph break) rather than breaking import.
    """
    try:
        return _register_custom_op_impl(
            op_name=op_name,
            input_tensors_for_grad=input_tensors_for_grad,
            fwd_arg_type=fwd_arg_type,
            fwd_impl=fwd_impl,
            setup_context=setup_context,
            backward_arg_type=backward_arg_type,
            backward_obj=backward_obj,
            backward_impl=backward_impl,
            fwd_fake_impl=fwd_fake_impl,
            bwd_fake_impl=bwd_fake_impl,
        )
    except (ImportError, AttributeError, RuntimeError, TypeError) as e:
        warnings.warn(
            f"Could not register the torch.compile custom op '{op_name}' "
            f"({type(e).__name__}: {e}); modules using it will fall back to eager "
            "execution under torch.compile (a graph break, incompatible with "
            "fullgraph=True)."
        )
        return None


def _register_custom_op_impl(
    *,
    op_name: str,
    input_tensors_for_grad: List[str],
    fwd_arg_type: type,
    fwd_impl: Callable[[Any], Any],
    setup_context: Callable[..., Any],
    backward_arg_type: type,
    backward_obj: type,
    backward_impl: Callable[[Any], Any],
    fwd_fake_impl: Callable[[Any], Tuple[Any, ...]],
    bwd_fake_impl: Callable[[Any], Tuple[Any, ...]],
) -> Callable[..., Any]:
    """Body of :func:`register_custom_op`; see it for semantics."""
    outer_fwd_name = op_name
    outer_bwd_name = f"{op_name}_backward"
    inner_fwd_name = f"{op_name}_base"
    inner_bwd_name = f"{outer_bwd_name}_base"
    subclass_list = _all_quantized_tensor_subclasses()

    fwd_buckets = _get_buckets(fwd_arg_type)
    bwd_buckets = _get_buckets(backward_arg_type)

    fwd_schema_args, fwd_arg_names = _build_schema(fwd_buckets)
    bwd_schema_args, bwd_arg_names = _build_schema(bwd_buckets)

    num_grad_inputs = len(input_tensors_for_grad)
    fwd_slot_defaults, grad_targets = _resolve_grad_targets(
        fwd_buckets, fwd_arg_type, input_tensors_for_grad
    )

    fwd_schema = f"{fwd_schema_args} -> Tensor[]"
    bwd_schema = f"{bwd_schema_args} -> Tensor[]"

    inner_bwd_qualname = f"{_TE_OP_NAMESPACE}::{inner_bwd_name}"

    inner_fwd_def = _register_kernel(
        op_name=inner_fwd_name,
        schema_str=fwd_schema,
        arg_type=fwd_arg_type,
        arg_names=fwd_arg_names,
        buckets=fwd_buckets,
        impl=fwd_impl,
        fake_impl=fwd_fake_impl,
        format_result=_format_fwd_result,
    )
    _register_kernel(
        op_name=inner_bwd_name,
        schema_str=bwd_schema,
        arg_type=backward_arg_type,
        arg_names=bwd_arg_names,
        buckets=bwd_buckets,
        impl=backward_impl,
        fake_impl=bwd_fake_impl,
        format_result=lambda g: _format_bwd_result(g, num_grad_inputs, inner_bwd_qualname),
    )

    outer_fwd_def = _register_outer_forwarder(
        outer_op_name=outer_fwd_name,
        schema_str=fwd_schema,
        inner_op_name=inner_fwd_name,
        buckets=fwd_buckets,
        subclass_list=list(subclass_list),
    )
    outer_bwd_def = _register_outer_forwarder(
        outer_op_name=outer_bwd_name, schema_str=bwd_schema, inner_op_name=inner_bwd_name
    )

    autograd_common = {
        "fwd_arg_type": fwd_arg_type,
        "fwd_arg_names": fwd_arg_names,
        "fwd_buckets": fwd_buckets,
        "bwd_arg_names": bwd_arg_names,
        "bwd_buckets": bwd_buckets,
        "fwd_slot_defaults": fwd_slot_defaults,
        "grad_targets": grad_targets,
        "setup_context_user": setup_context,
        "backward_obj_type": backward_obj,
        "fwd_fake_impl": fwd_fake_impl,
    }
    _register_autograd_for_op(
        fwd_op=inner_fwd_def, bwd_op_name=inner_bwd_name, **autograd_common
    )
    _register_autograd_for_op(
        fwd_op=outer_fwd_def, bwd_op_name=outer_bwd_name, **autograd_common
    )

    inner_fwd_op = getattr(getattr(torch.ops, _TE_OP_NAMESPACE), inner_fwd_name)
    inner_bwd_op = getattr(getattr(torch.ops, _TE_OP_NAMESPACE), inner_bwd_name)
    outer_fwd_op = getattr(getattr(torch.ops, _TE_OP_NAMESPACE), outer_fwd_name)
    outer_bwd_op = getattr(getattr(torch.ops, _TE_OP_NAMESPACE), outer_bwd_name)

    fwd_slot_offsets = _collect_universal_slot_offsets(fwd_buckets)
    bwd_slot_offsets = _collect_universal_slot_offsets(bwd_buckets)

    def _fwd_rule(mode, func, types, args, kwargs):
        del mode, func, types, kwargs
        new_args = list(args)
        for sub in subclass_list:
            _flatten_subclass_into_slots(new_args, fwd_slot_offsets, sub)
        return inner_fwd_op(*new_args)

    def _bwd_rule(mode, func, types, args, kwargs):
        del mode, func, types, kwargs
        new_args = list(args)
        for sub in subclass_list:
            _flatten_subclass_into_slots(new_args, bwd_slot_offsets, sub)
        return inner_bwd_op(*new_args)

    for sub in subclass_list:
        outer_fwd_def.register_torch_dispatch(sub, _fwd_rule)
        outer_bwd_def.register_torch_dispatch(sub, _bwd_rule)

    _quantized_tensor_passthrough_ops.add(outer_fwd_op.default)
    _quantized_tensor_passthrough_ops.add(outer_bwd_op.default)
    _quantized_tensor_passthrough_ops.add(inner_fwd_op.default)
    _quantized_tensor_passthrough_ops.add(inner_bwd_op.default)

    def forward_fn(fwd_args):
        user_fakes, _saved_fakes, _ctx_attrs = _split_fwd_fake_result(fwd_fake_impl(fwd_args))
        kwargs = _pack(fwd_args, fwd_buckets)
        flat_in = [kwargs[name] for name in fwd_arg_names]
        result = outer_fwd_op(*flat_in)

        cursor = 0
        outputs: List[Any] = []
        for template in user_fakes:
            n = _slot_count(template)
            chunk = [_decode_none(t) for t in result[cursor : cursor + n]]
            cursor += n
            outputs.append(_maybe_reassemble_tensor_subclass(template, chunk))

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    return forward_fn
