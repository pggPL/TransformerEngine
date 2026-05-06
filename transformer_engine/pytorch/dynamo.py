# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""torch.compile (Dynamo) integration for TransformerEngine modules."""
from __future__ import annotations

import copy
import dataclasses
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import torch


__all__ = [
    "ArgObject",
    "OpaqueSimpleMetadata",
    "_te_register_custom_op",
]


# Sentinel for ``None`` entries inside the op's flat ``Tensor[]`` return.
# Used by :func:`_te_register_custom_op` to support ``None`` outputs (e.g.
# an FP8 weight workspace returned only on the cache-miss path) on a
# non-nullable schema -- ``Tensor?[]`` returns are not picked up by
# ``torch.library.register_autograd``, so the registered backward never
# attaches a ``grad_fn`` to the op's outputs.
_NONE_SENTINEL_DTYPE = torch.uint8


# Name of the synthetic int slot appended to every TE custom op's
# schema. :meth:`ArgObject.torch_compile_pack` populates it with
# ``hash(tuple(<bucket key parts>))`` so :meth:`torch_compile_unpack`
# can key the skeleton cache off a single dict lookup. Under
# ``torch.compile`` the value folds to a constant during tracing
# (every input feeding the hash is either an interned
# OpaqueSimpleMetadata's ``_hash`` or ``id(pg)`` of a guarded
# ProcessGroup), so no per-call hashing happens at runtime; under
# eager pack still runs per call but pays a single ``hash(tuple)`` in
# place of the bucket-driven tuple-build + dict lookup the unpack side
# would otherwise do. NOTE: collisions on this single int are
# *accepted* (Option A of the original interning design): two distinct
# ``(meta._hash, id(pg))`` configurations that happen to share the
# same combined hash would be conflated. With well-distributed Python
# int hashes and a typical handful of distinct configs per process,
# the probability is well below the noise floor of any other
# correctness risk in this stack.
_CACHE_KEY_SLOT = "__te_cache_key__"


# Sentinel returned by :meth:`_Bucket.try_bake_skeleton` to flag
# "this field is fully baked into the cached skeleton; no per-call
# injector is needed". A unique object instance keeps the hot-path
# dispatch in :meth:`ArgObject.torch_compile_unpack` to identity
# comparisons (no hashing / str compares).
_SKELETON_BAKED: Any = object()


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
# OpaqueSimpleMetadata
# --------------------------------------------------------------------------- #

class OpaqueSimpleMetadata:
    """Opaque value-type bundle of simple Python values.

    Wraps a ``{name: value}`` dict so that many small non-Tensor arguments
    of a TE custom op can be passed as a single op input. Registered as a
    torch.compile *value* opaque type, meaning Dynamo specializes the
    traced graph on the bundle's contents: ``__eq__`` installs a guard,
    and any change to a wrapped value triggers a recompile.

    Allowed value types: primitives in :attr:`PRIMITIVE_TYPES`,
    :class:`enum.Enum`, :class:`torch.Size`, plus arbitrarily nested
    tuples/lists thereof.
    """

    # Primitive Python types we are willing to bundle into a single op
    # input. The bundle is registered as a torch.compile *value* opaque
    # type, so its contents must be hashable, comparable for equality,
    # and round-trippable through ``__fx_repr__``.
    PRIMITIVE_TYPES: Tuple[type, ...] = (
        type(None),
        bool,
        int,
        float,
        str,
        torch.dtype,
        torch.device,
    )

    @classmethod
    def _is_opaque_value(cls, value: Any) -> bool:
        """Whether ``value``'s class is registered as a value-opaque type."""
        try:
            from torch._library.opaque_object import is_opaque_value_type
        except Exception:  # pragma: no cover - older torch
            return False
        return is_opaque_value_type(type(value))

    @classmethod
    def is_simple_value(cls, value: Any) -> bool:
        """Whether ``value`` is allowed inside an instance.

        Accepts simple primitives (see :attr:`PRIMITIVE_TYPES`),
        :class:`enum.Enum`, :class:`torch.Size`, instances of any class
        registered as a torch.compile *value*-opaque type (the latter
        already supplies ``__eq__`` / ``__hash__`` / ``__fx_repr__`` as
        a registration prerequisite), and arbitrarily nested
        tuples / lists thereof.
        """
        if isinstance(value, cls.PRIMITIVE_TYPES):
            return True
        if isinstance(value, Enum):
            return True
        if isinstance(value, torch.Size):
            return True
        if cls._is_opaque_value(value):
            return True
        if isinstance(value, (list, tuple)):
            return all(cls.is_simple_value(v) for v in value)
        return False

    @classmethod
    def _to_hashable(cls, value: Any) -> Any:
        """Convert a simple value into something hashable (lists -> tuples)."""
        if isinstance(value, (list, tuple, torch.Size)):
            return tuple(cls._to_hashable(v) for v in value)
        # Opaque-value instances already supply ``__hash__`` (required
        # by registration) so they can stay as-is.
        return value

    @classmethod
    def _fmt_simple(cls, value: Any) -> str:
        """Repr for a simple value, evaluable in a context with ``torch`` globals."""
        if isinstance(value, torch.dtype):
            return f"__import__('torch').{str(value).split('.')[-1]}"
        if isinstance(value, torch.device):
            return f"__import__('torch').device({str(value)!r})"
        if isinstance(value, torch.Size):
            return f"__import__('torch').Size({list(value)!r})"
        if isinstance(value, Enum):
            return f"{type(value).__name__}.{value.name}"
        if isinstance(value, list):
            return "[" + ", ".join(cls._fmt_simple(v) for v in value) + "]"
        if isinstance(value, tuple):
            body = ", ".join(cls._fmt_simple(v) for v in value)
            return f"({body},)" if len(value) == 1 else f"({body})"
        if cls._is_opaque_value(value):
            # Opaque-value types declare their FX reconstruction via
            # ``__fx_repr__``; just splice their expression in here.
            return value.__fx_repr__()[0]
        return repr(value)

    # Process-wide intern cache: ``(cls, _frozen) -> instance``. Hot
    # callers (e.g. inductor's ``Runner.call`` literally re-constructs
    # the same ``OpaqueSimpleMetadata({...})`` on every iteration) get
    # back the same Python object, so:
    #
    # * follow-up ``__eq__`` checks (used by torch.compile guard
    #   verification on every call) short-circuit on tuple-identity
    #   instead of paying for an O(n) tuple compare;
    # * the cached instance's ``_data`` / ``_frozen`` allocations are
    #   amortized across all callers with the same content.
    #
    # Keyed on the subclass so a hypothetical subclass with different
    # behavior won't be accidentally aliased to a base-class entry.
    _INTERN_CACHE: Dict[
        Tuple[type, Tuple[Tuple[str, Any], ...]],
        "OpaqueSimpleMetadata",
    ] = {}

    # Mirror cache keyed by ``hash(_frozen)`` (cheap ``int -> instance``
    # lookup) for the FX-codegen fast path: ``__fx_repr__`` emits a call
    # to :meth:`try_create_from_cache` so Inductor's per-iteration
    # ``Runner.call`` skips the ``OpaqueSimpleMetadata({...})``
    # constructor entirely on cache hit.
    #
    # Hash collisions between *different*-content ``_frozen``s are
    # blacklisted -- the bucket is dropped from the cache and never
    # re-populated, so the FX fallback's full literal reconstructs the
    # right instance on every call. Collisions are extremely rare for
    # the small number of distinct metadatas a real workload produces.
    _HASH_CACHE: Dict[int, "OpaqueSimpleMetadata"] = {}
    _HASH_BLACKLIST: Set[int] = set()

    def __new__(
        cls,
        data: Optional[Dict[str, Any]] = None,
        /,
        **kwargs: Any,
    ) -> "OpaqueSimpleMetadata":
        merged: Dict[str, Any] = dict(data) if data else {}
        merged.update(kwargs)
        for k, v in merged.items():
            if not cls.is_simple_value(v):
                raise TypeError(
                    f"OpaqueSimpleMetadata field '{k}' has unsupported "
                    f"type {type(v).__name__}; only simple primitives "
                    f"({', '.join(t.__name__ for t in cls.PRIMITIVE_TYPES)}, "
                    f"Enum, torch.Size, registered torch.compile value-"
                    f"opaque types) and tuples/lists thereof are allowed."
                )
        frozen: Tuple[Tuple[str, Any], ...] = tuple(
            (k, cls._to_hashable(v)) for k, v in sorted(merged.items())
        )
        key = (cls, frozen)
        cached = OpaqueSimpleMetadata._INTERN_CACHE.get(key)
        if cached is not None:
            return cached
        instance = super().__new__(cls)
        instance._data = merged
        instance._frozen = frozen
        instance._hash = hash(frozen)
        canonical = OpaqueSimpleMetadata._INTERN_CACHE.setdefault(key, instance)
        # Maintain the hash-mirror only on (effective) miss. On hit the
        # bucket was already populated when the canonical instance was
        # first created.
        h = canonical._hash
        if h not in OpaqueSimpleMetadata._HASH_BLACKLIST:
            prior = OpaqueSimpleMetadata._HASH_CACHE.get(h)
            if prior is None:
                OpaqueSimpleMetadata._HASH_CACHE[h] = canonical
            elif prior is not canonical:
                # Different content with the same Python hash: disable
                # the hash-only fast path for this bucket so the FX
                # fallback (full literal) is always used here.
                OpaqueSimpleMetadata._HASH_CACHE.pop(h, None)
                OpaqueSimpleMetadata._HASH_BLACKLIST.add(h)
        return canonical

    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
        /,
        **kwargs: Any,
    ) -> None:
        # All construction (validation, freezing, interning, hash
        # precompute) lives in ``__new__``; ``__init__`` must not
        # re-run on cached instances or it would clobber their state.
        del data, kwargs

    @classmethod
    def try_create_from_cache(
        cls, h: int
    ) -> Optional["OpaqueSimpleMetadata"]:
        """Return the canonical interned instance for hash ``h``, or
        ``None`` if no cache entry exists or the hash bucket has been
        invalidated by a collision.

        Used by :meth:`__fx_repr__`: Inductor's generated wrapper
        evaluates ``cls.try_create_from_cache(<h>) or OpaqueSimpleMetadata({...full...})``,
        so the costly ``{...full...}`` literal is only ever built on a
        true cache miss (cold start, cross-process hash-randomization
        change, or collision blacklist).
        """
        if h in cls._HASH_BLACKLIST:
            return None
        return cls._HASH_CACHE.get(h)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __getattr__(self, name: str) -> Any:
        # Only called when normal attribute lookup fails, so ``_data`` /
        # ``_frozen`` won't recurse here once set in ``__init__``.
        try:
            return self._data[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def keys(self) -> List[str]:
        return list(self._data.keys())

    def values(self) -> List[Any]:
        return list(self._data.values())

    def items(self) -> List[Tuple[str, Any]]:
        return list(self._data.items())

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    def __eq__(self, other: object) -> bool:
        # Fast path leveraging interning (see ``__new__`` /
        # ``_INTERN_CACHE``): equal-content instances are guaranteed to
        # be the *same* Python object, so identity is sufficient when
        # both operands have been allocated via ``__new__``. The
        # ``_frozen`` tuple compare below stays as a correctness
        # backstop for callers that bypass interning (tests).
        if self is other:
            return True
        if not isinstance(other, OpaqueSimpleMetadata):
            return NotImplemented
        return self._frozen == other._frozen

    def __hash__(self) -> int:
        return self._hash

    def __fx_repr__(self) -> Tuple[str, Dict[str, Any]]:
        cls = type(self)
        items = ", ".join(
            f"{k!r}: {cls._fmt_simple(v)}" for k, v in self._data.items()
        )
        # Collect every type referenced by a nested opaque-value's
        # ``__fx_repr__`` so the FX codegen can resolve those names.
        globals_: Dict[str, Any] = {
            "OpaqueSimpleMetadata": OpaqueSimpleMetadata,
        }

        def _collect(value: Any) -> None:
            if isinstance(value, (list, tuple)):
                for v in value:
                    _collect(v)
                return
            # Skip plain Python / torch primitives up-front: they're
            # rendered as literals by ``_fmt_simple`` and need no
            # globals entry.
            if isinstance(value, cls.PRIMITIVE_TYPES):
                return
            if isinstance(value, torch.Size):
                return
            if isinstance(value, Enum):
                # ``_fmt_simple`` emits ``EnumName.MEMBER``; the Enum
                # class must be in scope when the source string is
                # later ``exec``d (e.g. by ``GraphModule.print_readable``
                # or by Inductor's runtime wrapper).
                t = type(value)
                globals_[t.__name__] = t
                return
            if cls._is_opaque_value(value):
                _, extra = value.__fx_repr__()
                globals_.update(extra)

        for v in self._data.values():
            _collect(v)
        # Fast path: try the hash-keyed cache before falling back to a
        # full ``OpaqueSimpleMetadata({...})`` constructor call. Python's
        # ``or`` short-circuits, so on cache hit (the common case in a
        # steady-state compiled loop) the costly dict literal isn't
        # even built, let alone passed through ``__new__``'s validate
        # + freeze + setattr work. The fallback keeps the source
        # robust against cold start, hash blacklisting, and
        # cross-process PYTHONHASHSEED randomization.
        return (
            f"(OpaqueSimpleMetadata.try_create_from_cache({self._hash}) "
            f"or OpaqueSimpleMetadata({{{items}}}))",
            globals_,
        )

    def __repr__(self) -> str:
        # ``__repr__`` is on hot diagnostic paths (Inductor error
        # formatters, FX node printers, ...) and must never raise:
        # treating any embedded value's ``repr`` failure as a soft
        # placeholder keeps those error reporters from masking the
        # actual root-cause exception with a crash inside our repr.
        parts: List[str] = []
        for k, v in self._data.items():
            try:
                v_repr = repr(v)
            except Exception as e:  # pylint: disable=broad-except
                v_repr = f"<{type(v).__name__}: repr failed: {e!s}>"
            parts.append(f"{k!r}: {v_repr}")
        return f"OpaqueSimpleMetadata({{{', '.join(parts)}}})"


# Register OpaqueSimpleMetadata as a torch.compile value-opaque type, and
# resolve the schema name of ``torch.distributed.ProcessGroup`` (registered
# upstream as a *reference* opaque type via
# ``torch.distributed.device_mesh._register_distributed_opaque_types``).
# Both are done at module import so that any TE op declared via
# ``_te_register_custom_op`` can immediately reference them in its schema.
# Older PyTorch versions without these APIs are tolerated: the eager path
# keeps working, only torch.compile tracing of TE custom ops is unavailable.
try:
    from torch._library.opaque_object import (
        get_opaque_type_name,
        register_opaque_type,
    )

    register_opaque_type(OpaqueSimpleMetadata, typ="value")
    _OPAQUE_SIMPLE_META_TYPE_NAME: Optional[str] = get_opaque_type_name(
        OpaqueSimpleMetadata
    )

    _PROCESS_GROUP_TYPE_NAME: Optional[str] = None
    try:
        from torch.distributed import ProcessGroup
        from torch.distributed.device_mesh import (
            _register_distributed_opaque_types,
        )

        _register_distributed_opaque_types()
        _PROCESS_GROUP_TYPE_NAME = get_opaque_type_name(ProcessGroup)
    except Exception:  # pragma: no cover - distributed not built / disabled
        _PROCESS_GROUP_TYPE_NAME = None
except Exception:  # pragma: no cover - older torch without opaque_object
    _OPAQUE_SIMPLE_META_TYPE_NAME = None
    _PROCESS_GROUP_TYPE_NAME = None


# --------------------------------------------------------------------------- #
# Field buckets
# --------------------------------------------------------------------------- #

# Each dataclass field of an :class:`ArgObject` is mapped to exactly one
# bucket. A bucket owns the full per-field "vocabulary" -- which schema
# slots it emits, how its packed value(s) are produced from the dataclass
# instance, and how the unpacked value is re-injected into the
# reconstructed instance. ``ArgObject`` then becomes three trivial loops
# over a list of buckets, instead of three parallel branch ladders.
#
# Five bucket kinds are used:
#
# * :class:`_TensorBucket` -- :class:`torch.Tensor` /
#   :class:`Optional[torch.Tensor] <typing.Optional>` -> one ``Tensor`` /
#   ``Tensor?`` slot.
# * :class:`_TensorListBucket` -- ``List[torch.Tensor]`` /
#   ``Tuple[torch.Tensor, ...]`` -> one ``Tensor[]`` slot. Used for
#   variable-length tensor sequences such as ``ctx.saved_tensors``.
# * :class:`_ProcessGroupBucket` -- :class:`torch.distributed.ProcessGroup`
#   (already registered upstream as a value-opaque type) -> one direct
#   slot.
# * :class:`_FlattenableBucket` -- a field whose type implements the
#   ``_flatten`` / ``_unflatten`` protocol (today: any
#   :class:`Quantizer` or :class:`Recipe` subclass) -> three slots
#   ``<name>__fmeta`` / ``<name>__fpg`` / ``<name>__ftensors``. Bases
#   are discovered via :func:`_flattenable_bases`, lazily imported to
#   avoid an import cycle.
# * :class:`_SimpleBundleBucket` -- aggregator over **all** simple-typed
#   fields of the dataclass; emits a single ``_simple_meta`` slot
#   carrying an :class:`OpaqueSimpleMetadata` bundle.
# * :class:`_UnknownBucket` -- a field whose annotation matches none of
#   the above. Emits no schema slot; pack raises if the field holds a
#   non-``None`` value, unpack restores it as ``None``.


def _strip_optional(annot: Any) -> Tuple[Any, bool]:
    """If ``annot`` is ``Optional[X]`` return ``(X, True)``; else ``(annot, False)``.

    Shared by all bucket matchers below.
    """
    if get_origin(annot) is Union:
        args = get_args(annot)
        if type(None) in args:
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return non_none[0], True
    return annot, False


class _Bucket:
    """Per-field handler for translating between a dataclass field and the
    flat ``{slot_name: slot_value}`` view consumed by ``torch.library``.

    Each concrete bucket owns:

    * a :meth:`try_build` classmethod that decides whether a given
      ``(name, annotation)`` pair belongs to this bucket (returning an
      instance, or ``None`` to defer to the next bucket);
    * the runtime :meth:`schema_slots` / :meth:`pack` / :meth:`unpack`
      logic for that field.

    :class:`_SimpleBundleBucket` is the exception: it aggregates many
    simple-typed fields into a single op input, so it does not implement
    ``try_build``. It exposes :meth:`matches_field` for the per-field
    membership test, and is constructed once at the end of dispatch
    with the collected names.
    """

    @classmethod
    def try_build(cls, name: str, annot: Any) -> Optional["_Bucket"]:
        """Return an instance handling ``(name, annot)``, or ``None``."""
        raise NotImplementedError

    def schema_slots(self) -> List[Tuple[str, str]]:
        """Return ``[(slot_name, schema_type_str), ...]`` for this field."""
        raise NotImplementedError

    def pack(self, owner: "ArgObject") -> List[Tuple[str, Any]]:
        """Return ``[(slot_name, value), ...]`` extracted from ``owner``."""
        raise NotImplementedError

    def unpack(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        """Read this field's slots from ``args`` and write the
        reconstructed dataclass attribute(s) into ``kwargs``."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Hooks for the ``ArgObject._UNPACK_CACHE`` skeleton-cache fast path
    # ------------------------------------------------------------------ #

    def is_skeleton_stable(self) -> bool:
        """Whether this bucket's :meth:`unpack` output is fully
        determined by *opaque inputs* (i.e. the inputs that participate
        in the unpack cache key) and therefore safe to bake into the
        cached skeleton.

        Defaults to ``False``: most buckets carry tensors or other
        per-call references and must run every iteration.
        :class:`_SimpleBundleBucket` overrides this to ``True`` because
        its output is fully derived from a single
        :class:`OpaqueSimpleMetadata` whose identity participates in
        the cache key.
        """
        return False

    def try_bake_skeleton(
        self,
        args: Dict[str, Any],
        skeleton_kwargs: Dict[str, Any],
    ) -> Any:
        """Attempt to populate this field on the cached skeleton.

        Called once per ``(cls, cache_key)`` on the cache-miss path.
        The base implementation honours :meth:`is_skeleton_stable`:
        stable buckets fully bake; everything else falls back to
        per-call :meth:`unpack`.

        Return values:

        * :data:`_SKELETON_BAKED` -- field is fully in
          ``skeleton_kwargs``; no per-call work needed.
        * ``None`` -- no caching; the caller will run
          :meth:`unpack` every call (the default for
          tensor-carrying buckets).
        * a callable ``f(args, obj_dict) -> None`` -- the bucket
          baked a partial value (e.g. a storage shell) into
          ``skeleton_kwargs`` and provided an injector that, given
          a fresh ``args`` dict and the per-call object's
          ``__dict__``, finishes reconstruction (e.g. shallow-copies
          the cached storage and replaces tensor attrs from
          ``args``).
        """
        if self.is_skeleton_stable():
            self.unpack(args, skeleton_kwargs)
            return _SKELETON_BAKED
        return None

    def opaque_cache_key_parts(
        self, args: Dict[str, Any]
    ) -> Tuple[int, ...]:
        """Return ``id``\\ s of any opaque-typed inputs this bucket
        consumes from ``args``. The aggregate of these (across all
        buckets) is the cache key used by
        :meth:`ArgObject.torch_compile_unpack`'s skeleton cache.

        Default: empty tuple (no opaque inputs).
        """
        del args
        return ()


class _TensorOrStorageBucket(_Bucket):
    """``Tensor | QuantizedTensorStorage`` -> meta / pg / Tensor[] slots.

    Plain tensors are carried as a single-element ``Tensor[]``. Quantized
    tensor wrappers and storage shells are carried through their
    ``_torch_compile_flatten`` protocol so the backward op receives the same
    structured object type that eager restoration produced.
    """

    SUFFIX_META = "__tsmeta"
    SUFFIX_PG = "__tspg"
    SUFFIX_TENSORS = "__tstensors"

    KIND_KEY = "_te_tensor_storage_kind"
    KIND_NONE = "none"
    KIND_TENSOR = "tensor"

    def __init__(self, name: str) -> None:
        if _OPAQUE_SIMPLE_META_TYPE_NAME is None or _PROCESS_GROUP_TYPE_NAME is None:
            raise RuntimeError(
                f"Tensor/storage field {name!r} requires both "
                "OpaqueSimpleMetadata and torch.distributed.ProcessGroup "
                "to be registered as torch._library opaque types; one or "
                "both are unavailable in this PyTorch build."
            )
        self.name = name

    @staticmethod
    def _is_tensor_storage_union(annot: Any) -> bool:
        origin = get_origin(annot)
        if origin is not Union:
            return False
        members = [a for a in get_args(annot) if a is not type(None)]
        if torch.Tensor not in members:
            return False
        try:
            from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage
        except Exception:  # pragma: no cover - partial init
            return False
        return any(
            isinstance(member, type) and issubclass(member, QuantizedTensorStorage)
            for member in members
        )

    @classmethod
    def try_build(cls, name: str, annot: Any) -> Optional["_TensorOrStorageBucket"]:
        if cls._is_tensor_storage_union(annot):
            return cls(name)
        return None

    def _slot_meta(self) -> str:
        return self.name + self.SUFFIX_META

    def _slot_pg(self) -> str:
        return self.name + self.SUFFIX_PG

    def _slot_tensors(self) -> str:
        return self.name + self.SUFFIX_TENSORS

    def schema_slots(self) -> List[Tuple[str, str]]:
        return [
            (self._slot_meta(), _OPAQUE_SIMPLE_META_TYPE_NAME),
            (self._slot_pg(), _PROCESS_GROUP_TYPE_NAME + "?"),
            (self._slot_tensors(), "Tensor[]"),
        ]

    def pack(self, owner: "ArgObject") -> List[Tuple[str, Any]]:
        value = getattr(owner, self.name)
        if value is None:
            meta = OpaqueSimpleMetadata({self.KIND_KEY: self.KIND_NONE})
            pg: Any = None
            tensors: List[torch.Tensor] = []
        else:
            from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage

            if isinstance(value, QuantizedTensorStorage):
                meta, pg, tensors = value._torch_compile_flatten()
            elif isinstance(value, torch.Tensor):
                meta = OpaqueSimpleMetadata({self.KIND_KEY: self.KIND_TENSOR})
                pg = None
                tensors = [value]
            else:
                raise TypeError(
                    f"{type(owner).__name__} field {self.name!r} expected "
                    "None, torch.Tensor, or QuantizedTensorStorage, got "
                    f"{type(value).__name__}"
                )
        return [
            (self._slot_meta(), meta),
            (self._slot_pg(), pg),
            (self._slot_tensors(), list(tensors)),
        ]

    def unpack(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        meta = args[self._slot_meta()]
        kind = meta.get(self.KIND_KEY)
        if kind == self.KIND_NONE:
            kwargs[self.name] = None
            return
        tensors = args[self._slot_tensors()]
        if kind == self.KIND_TENSOR:
            kwargs[self.name] = tensors[0]
            return

        from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage

        kwargs[self.name] = QuantizedTensorStorage._torch_compile_unflatten(
            meta,
            args[self._slot_pg()],
            tensors,
        )

    def opaque_cache_key_parts(
        self, args: Dict[str, Any]
    ) -> Tuple[int, ...]:
        # ``meta._hash`` is content-deterministic for any
        # :class:`OpaqueSimpleMetadata` and survives the interning
        # falling back to a fresh constructor (cold start, blacklisted
        # hash bucket, ...). ``pg`` is a reference-opaque PG which is
        # not interned -- identity is the only stable handle.
        return (args[self._slot_meta()]._hash, id(args.get(self._slot_pg())))

    def try_bake_skeleton(
        self,
        args: Dict[str, Any],
        skeleton_kwargs: Dict[str, Any],
    ) -> Any:
        meta = args[self._slot_meta()]
        kind = meta.get(self.KIND_KEY)
        if kind == self.KIND_NONE:
            # Field is just ``None``: bake.
            skeleton_kwargs[self.name] = None
            return _SKELETON_BAKED
        if kind == self.KIND_TENSOR:
            # Bare ``torch.Tensor`` -- the value is the new tensor on
            # every call and there is no shell to cache. Fall back to
            # per-call :meth:`unpack`.
            return None

        # ``QuantizedTensorStorage`` branch.
        from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage

        pg = args[self._slot_pg()]
        tensors = args[self._slot_tensors()]
        storage = QuantizedTensorStorage._torch_compile_unflatten(meta, pg, tensors)

        if not tensors:
            # Storage carries no tensor state -- bake the whole
            # instance. Subsequent calls with matching ``meta._hash``
            # / ``id(pg)`` reuse it via the shallow-copy fast path.
            skeleton_kwargs[self.name] = storage
            return _SKELETON_BAKED

        # Storage with tensors. Try to bake the *shell*: stash the
        # reconstructed instance in the skeleton, build an injector
        # that on each call shallow-copies the shell and overwrites
        # the tensor-typed attributes from the per-call ``args``.
        if isinstance(storage, torch.Tensor):
            # ``torch.Tensor`` subclass: shallow-copy semantics are
            # tangled (``cls.__new__(cls)`` does not produce a valid
            # tensor instance, ``copy.copy`` would call into the
            # tensor copy paths). Bail out and rebuild per call.
            return None

        # Identify which top-level ``__dict__`` entries hold one of
        # ``tensors``. Match by identity: the unflatten just walked
        # ``tensors`` and assigned slices of it to its own attributes,
        # so ``id(v)`` for any tensor attr is one of those we passed
        # in.
        tensor_ids: Dict[int, int] = {id(t): i for i, t in enumerate(tensors)}
        storage_dict = storage.__dict__
        slot_map: List[Tuple[str, int]] = []
        found: Set[int] = set()
        for attr, value in storage_dict.items():
            tid = tensor_ids.get(id(value))
            if tid is None:
                continue
            slot_map.append((attr, tid))
            found.add(tid)

        if len(found) != len(tensors):
            # Some tensors are nested deeper (e.g. inside a
            # tensor-bearing quantizer attached to the storage).
            # Mutating only top-level attrs would leave stale
            # tensors elsewhere -- safer to rebuild per call.
            return None

        skeleton_kwargs[self.name] = storage
        name = self.name
        slot_tensors_name = self._slot_tensors()

        def injector(
            call_args: Dict[str, Any],
            obj_dict: Dict[str, Any],
            _slot_map: List[Tuple[str, int]] = slot_map,
            _name: str = name,
            _slot_tensors_name: str = slot_tensors_name,
            _object_new: Callable[..., Any] = object.__new__,
        ) -> None:
            # ``obj_dict`` already contains the cached shell (set by
            # :meth:`copy.copy` of the skeleton). Shallow-copy it so
            # we don't mutate the cached instance, then overwrite the
            # tensor attrs with this call's tensors.
            #
            # Use ``object.__new__`` rather than ``cls.__new__`` to
            # bypass custom ``__new__`` signatures that require
            # constructor kwargs (e.g. ``Float8TensorStorage.__new__``
            # mandates ``data`` / ``fp8_scale_inv`` / ``fp8_dtype``);
            # we restore identical state by copying ``__dict__``.
            baseline = obj_dict[_name]
            fresh = _object_new(type(baseline))
            fresh.__dict__.update(baseline.__dict__)
            new_tensors = call_args[_slot_tensors_name]
            fresh_dict = fresh.__dict__
            for attr, idx in _slot_map:
                fresh_dict[attr] = new_tensors[idx]
            obj_dict[_name] = fresh

        return injector


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

    def pack(self, owner: "ArgObject") -> List[Tuple[str, Any]]:
        return [(self.name, getattr(owner, self.name))]

    def unpack(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        kwargs[self.name] = args[self.name]


class _TensorListBucket(_Bucket):
    """``List[Tensor]`` / ``Tuple[Tensor, ...]`` -> single ``Tensor[]`` slot.

    Used for fields like ``LinearBwdArgs.saved_tensors`` that carry an
    arbitrary-length sequence of tensors (typically the
    ``ctx.saved_tensors`` payload restored before invoking the backward
    op). The slot itself is non-nullable, but individual ``None``
    elements are smuggled through using :func:`_encode_none` /
    :func:`_decode_none` sentinels (matching what the forward op return
    list already does). An empty sequence is valid.
    """

    def __init__(self, name: str, container: type) -> None:
        self.name = name
        # Remember the original container type so unpack returns the
        # exact same Python type the dataclass annotation declared.
        self.container = container

    @classmethod
    def try_build(cls, name: str, annot: Any) -> Optional["_TensorListBucket"]:
        stripped, _ = _strip_optional(annot)
        origin = get_origin(stripped)
        if origin is None:
            return None
        args = get_args(stripped)
        if not args:
            return None
        # ``Tuple[Tensor, ...]`` -> args = (Tensor, Ellipsis); other forms
        # like ``Tuple[Tensor, Tensor]`` or ``List[Tensor]`` only have
        # type entries.
        if origin is tuple:
            if len(args) == 2 and args[1] is Ellipsis:
                elem = args[0]
            else:
                elem = args[0] if all(a is args[0] for a in args) else None
        elif origin is list:
            elem = args[0]
        else:
            return None
        if elem is not torch.Tensor:
            return None
        return cls(name, list if origin is list else tuple)

    def schema_slots(self) -> List[Tuple[str, str]]:
        return [(self.name, "Tensor[]")]

    def pack(self, owner: "ArgObject") -> List[Tuple[str, Any]]:
        value = getattr(owner, self.name) or ()
        return [(self.name, [_encode_none(t) for t in value])]

    def unpack(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        kwargs[self.name] = self.container(_decode_none(t) for t in args[self.name])


class _ProcessGroupBucket(_Bucket):
    """``ProcessGroup`` / ``Optional[ProcessGroup]`` -> one direct opaque-ref slot.

    PG is registered upstream (in ``torch.distributed.device_mesh``) as
    a value-opaque type, so torch.library carries it without help.
    """

    def __init__(self, name: str, is_optional: bool) -> None:
        if _PROCESS_GROUP_TYPE_NAME is None:
            raise RuntimeError(
                f"ProcessGroup field {name!r} requires torch.distributed "
                "and the opaque-type registration in "
                "torch.distributed.device_mesh; neither is available in "
                "this PyTorch build."
            )
        self.name = name
        self.type_str = _PROCESS_GROUP_TYPE_NAME + ("?" if is_optional else "")

    @classmethod
    def try_build(cls, name: str, annot: Any) -> Optional["_ProcessGroupBucket"]:
        stripped, is_optional = _strip_optional(annot)
        if not isinstance(stripped, type):
            return None
        try:
            from torch.distributed import ProcessGroup
        except Exception:  # pragma: no cover - distributed not built
            return None
        if not issubclass(stripped, ProcessGroup):
            return None
        return cls(name, is_optional)

    def schema_slots(self) -> List[Tuple[str, str]]:
        return [(self.name, self.type_str)]

    def pack(self, owner: "ArgObject") -> List[Tuple[str, Any]]:
        return [(self.name, getattr(owner, self.name))]

    def unpack(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        kwargs[self.name] = args[self.name]

    def is_skeleton_stable(self) -> bool:
        # The PG identity is part of the unpack cache key, so the
        # field is stable for all calls that hit a given cache slot.
        return True

    def opaque_cache_key_parts(
        self, args: Dict[str, Any]
    ) -> Tuple[int, ...]:
        # PG is reference-opaque -- the only stable handle is identity.
        return (id(args.get(self.name)),)


def _flattenable_bases() -> Tuple[type, ...]:
    """Return the list of base classes whose subclasses are routed
    through :class:`_FlattenableBucket`.

    A "flattenable" type implements the duck-typed pair

    * instance method ``_flatten() -> (OpaqueSimpleMetadata, ref, list[Tensor])``
    * classmethod ``_unflatten(meta, ref, tensors)`` (dispatches by an
      identifier stamped into ``meta``)

    Lazy import keeps ``dynamo`` importable before the modules that
    define these bases (avoid import cycles).
    """
    bases: List[type] = []
    try:
        from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage, Quantizer

        bases.append(Quantizer)
        bases.append(QuantizedTensorStorage)
    except Exception:  # pragma: no cover - partial init
        pass
    try:
        from transformer_engine.common.recipe import Recipe

        bases.append(Recipe)
    except Exception:  # pragma: no cover - partial init
        pass
    return tuple(bases)


class _FlattenableBucket(_Bucket):
    """Three-slot expansion (``meta`` / ``ref`` / ``tensors``) for any
    field whose type implements the ``_flatten`` / ``_unflatten``
    protocol (see :func:`_flattenable_bases`). Used today for
    :class:`~transformer_engine.pytorch.quantized_tensor.Quantizer` and
    :class:`~transformer_engine.common.recipe.Recipe`.
    """

    SUFFIX_META = "__fmeta"
    SUFFIX_PG = "__fpg"
    SUFFIX_TENSORS = "__ftensors"

    # Stored under ``_qcls`` in the metadata bundle to encode ``None``
    # without making any of the three slots nullable.
    NONE_MARKER_KEY = "_qcls"
    NONE_MARKER_VAL = ""

    def __init__(self, name: str, base_cls: type) -> None:
        if _OPAQUE_SIMPLE_META_TYPE_NAME is None or _PROCESS_GROUP_TYPE_NAME is None:
            raise RuntimeError(
                f"Flattenable field {name!r} requires both "
                "OpaqueSimpleMetadata and torch.distributed.ProcessGroup "
                "to be registered as torch._library opaque types; one or "
                "both are unavailable in this PyTorch build."
            )
        self.name = name
        self.base_cls = base_cls

    @classmethod
    def try_build(cls, name: str, annot: Any) -> Optional["_FlattenableBucket"]:
        stripped, _ = _strip_optional(annot)
        if not isinstance(stripped, type):
            return None
        for base in _flattenable_bases():
            if issubclass(stripped, base):
                return cls(name, base)
        return None

    def _slot_meta(self) -> str:
        return self.name + self.SUFFIX_META

    def _slot_pg(self) -> str:
        return self.name + self.SUFFIX_PG

    def _slot_tensors(self) -> str:
        return self.name + self.SUFFIX_TENSORS

    def schema_slots(self) -> List[Tuple[str, str]]:
        return [
            (self._slot_meta(), _OPAQUE_SIMPLE_META_TYPE_NAME),
            (self._slot_pg(), _PROCESS_GROUP_TYPE_NAME + "?"),
            (self._slot_tensors(), "Tensor[]"),
        ]

    def pack(self, owner: "ArgObject") -> List[Tuple[str, Any]]:
        value = getattr(owner, self.name)
        if value is None:
            meta = OpaqueSimpleMetadata({self.NONE_MARKER_KEY: self.NONE_MARKER_VAL})
            pg: Any = None
            tensors: List[torch.Tensor] = []
        else:
            if hasattr(value, "_flatten"):
                meta, pg, tensors = value._flatten()
            else:
                meta, pg, tensors = value._torch_compile_flatten()
        return [
            (self._slot_meta(), meta),
            (self._slot_pg(), pg),
            (self._slot_tensors(), list(tensors)),
        ]

    def unpack(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        meta = args[self._slot_meta()]
        if meta.get(self.NONE_MARKER_KEY) == self.NONE_MARKER_VAL:
            kwargs[self.name] = None
            return
        if hasattr(self.base_cls, "_unflatten"):
            kwargs[self.name] = self.base_cls._unflatten(
                meta, args[self._slot_pg()], args[self._slot_tensors()]
            )
        else:
            kwargs[self.name] = self.base_cls._torch_compile_unflatten(
                meta, args[self._slot_pg()], args[self._slot_tensors()]
            )

    def opaque_cache_key_parts(
        self, args: Dict[str, Any]
    ) -> Tuple[int, ...]:
        # ``meta._hash`` is content-deterministic and survives intern
        # cache misses; ``pg`` is reference-opaque, identity is the
        # only stable handle. Tensors are not keyed -- they're injected
        # per call via the standard :meth:`unpack` path on the copied
        # skeleton.
        return (args[self._slot_meta()]._hash, id(args.get(self._slot_pg())))

    def try_bake_skeleton(
        self,
        args: Dict[str, Any],
        skeleton_kwargs: Dict[str, Any],
    ) -> Any:
        # Reconstruct the field on the cache-miss path. ``unpack`` may
        # produce ``None`` (when the field carried the ``NONE_MARKER``)
        # or a concrete instance.
        self.unpack(args, skeleton_kwargs)
        value = skeleton_kwargs.get(self.name)
        if value is None:
            return _SKELETON_BAKED
        # Cacheable iff the *concrete* class declared via
        # ``_TORCH_COMPILE_UNFLATTEN_USES_TENSORS = False`` that its
        # ``_unflatten`` ignores the tensors arg. Such instances carry
        # no tensor state and are safe to share across iterations.
        if not getattr(
            type(value), "_TORCH_COMPILE_UNFLATTEN_USES_TENSORS", True
        ):
            return _SKELETON_BAKED
        # Not cacheable -- drop the field and let the caller run
        # :meth:`unpack` per call.
        del skeleton_kwargs[self.name]
        return None


class _SimpleBundleBucket(_Bucket):
    """Aggregator: bundles every simple-typed field of the dataclass
    into a single :class:`OpaqueSimpleMetadata` slot.

    Does not implement :meth:`try_build` because membership is decided
    per-field via :meth:`matches_field`, with construction deferred
    until all simple field names are collected.
    """

    SLOT = "_simple_meta"

    def __init__(self, names: List[str]) -> None:
        if _OPAQUE_SIMPLE_META_TYPE_NAME is None:
            raise RuntimeError(
                "OpaqueSimpleMetadata could not be registered with "
                "torch._library.opaque_object; cannot bundle simple fields "
                f"{names}. Upgrade PyTorch or drop the simple fields."
            )
        self.names = list(names)

    @classmethod
    def matches_field(cls, annot: Any) -> bool:
        """Whether ``annot`` (Optional-aware, recursive on tuple/list) is
        bundled-simple, i.e. eligible for this aggregator.

        Accepts simple primitives, :class:`enum.Enum`, :class:`torch.Size`,
        any class registered as a torch.compile *value*-opaque type, and
        nested tuples / lists thereof.
        """
        annot, _ = _strip_optional(annot)
        if annot in OpaqueSimpleMetadata.PRIMITIVE_TYPES:
            return True
        if isinstance(annot, type) and issubclass(annot, Enum):
            return True
        if annot is torch.Size:
            return True
        # Any registered value-opaque class is hashable / FX-reproducible
        # and therefore safe to embed in the OpaqueSimpleMetadata bundle.
        if isinstance(annot, type):
            try:
                from torch._library.opaque_object import is_opaque_value_type
            except Exception:  # pragma: no cover - older torch
                is_opaque_value_type = None
            if is_opaque_value_type is not None and is_opaque_value_type(annot):
                return True
        origin = get_origin(annot)
        if origin in (tuple, list):
            # Inner args may contain Ellipsis (e.g. ``Tuple[int, ...]``);
            # only require the *concrete* inner annotations to be simple.
            inner = [a for a in get_args(annot) if a is not Ellipsis]
            return bool(inner) and all(cls.matches_field(a) for a in inner)
        return False

    def schema_slots(self) -> List[Tuple[str, str]]:
        return [(self.SLOT, _OPAQUE_SIMPLE_META_TYPE_NAME)]

    def pack(self, owner: "ArgObject") -> List[Tuple[str, Any]]:
        bundle = OpaqueSimpleMetadata({n: getattr(owner, n) for n in self.names})
        return [(self.SLOT, bundle)]

    def unpack(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        if self.SLOT not in args:
            return
        meta = args[self.SLOT]
        for n in self.names:
            kwargs[n] = meta[n]

    def is_skeleton_stable(self) -> bool:
        # Bundle output is fully derived from the single ``OpaqueSimpleMetadata``
        # whose hash participates in the cache key; safe to bake into
        # the cached skeleton.
        return True

    def opaque_cache_key_parts(
        self, args: Dict[str, Any]
    ) -> Tuple[int, ...]:
        meta = args.get(self.SLOT)
        # ``meta`` is missing only on edge-case dataclasses with zero
        # simple-typed fields; emit a sentinel so the rest of the key
        # still matters.
        return (meta._hash if meta is not None else 0,)


class _UnknownBucket(_Bucket):
    """Fallback for fields whose annotation no other bucket claims.
    Emits no schema slot; pack rejects non-trivial values to avoid silent
    data loss; unpack restores the field as ``None``.

    A "trivial" value is one that carries no observable information --
    ``None`` itself or a sequence of all-``None`` entries (e.g. a
    ``tensor_objects`` payload from :func:`prepare_for_saving` over a
    bag of plain bf16 tensors). Such values are dropped on the way into
    the op and reconstructed from companion fields (``saved_tensors``,
    quantizer metadata, ...) on the way out.

    Constructed directly by :meth:`ArgObject._buckets` (it has no
    annotation-based ``try_build`` -- it's the explicit "no match" case).
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

    def pack(self, owner: "ArgObject") -> List[Tuple[str, Any]]:
        value = getattr(owner, self.name, None)
        if not self._is_trivial(value):
            raise TypeError(
                f"{self.owner_cls_name} field {self.name!r} has a type not "
                "supported by torch.compile (not Tensor, simple, "
                "ProcessGroup, or Quantizer) and carries "
                "a non-trivial value; override "
                f"{self.owner_cls_name}.torch_compile_pack to handle it."
            )
        return []

    def unpack(self, args: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        kwargs[self.name] = None

    def is_skeleton_stable(self) -> bool:
        # Always None -- safe to bake into the cached skeleton.
        return True


# Buckets, in priority order, that own ``try_build`` for a single field.
_FIELD_BUCKETS: Tuple[type, ...] = (
    _TensorOrStorageBucket,
    _TensorBucket,
    _TensorListBucket,
    _ProcessGroupBucket,
    _FlattenableBucket,
)


# --------------------------------------------------------------------------- #
# ArgObject
# --------------------------------------------------------------------------- #


class ArgObject:
    """Base class for structured argument containers passed to TE custom ops.

    Subclassed by per-module forward / backward dataclasses
    (e.g. ``LinearFwdArgs``, ``LinearBwdArgs``). Provides the pack /
    unpack / schema hooks consumed by :func:`_te_register_custom_op`
    when wiring the dataclass into a ``torch.library`` schema.

    The default pack / unpack / schema implementations dispatch on
    dataclass field annotations. Each field is mapped to exactly one
    :class:`_Bucket` (see module-level docstring); the three methods
    then become trivial iterations over the bucket list.
    """

    # Skeleton cache used by :meth:`torch_compile_unpack`. Maps the
    # single int :data:`_CACHE_KEY_SLOT` value (computed once in
    # :meth:`torch_compile_pack` from each bucket's
    # :meth:`_Bucket.opaque_cache_key_parts`) to a
    # ``(skeleton, per_call_actions)`` pair. The skeleton is a
    # partially populated instance: stable buckets, cacheable
    # quantizer fields, and storage shells whose tensor attrs can be
    # re-injected via shallow-copy are baked in. ``per_call_actions``
    # is a flat list of callables ``f(args, obj_dict) -> None``:
    # either a pre-bound ``bucket.unpack`` method (the common case)
    # or a custom injector returned by
    # :meth:`_Bucket.try_bake_skeleton` (e.g. the storage-shell
    # shallow-copy + tensor-attr swap). The hot path therefore runs
    # a single tight loop with no branching. Class-level so each
    # ArgObject subclass has its own bucket of cached skeletons --
    # different dataclasses naturally have different field sets.
    _UNPACK_CACHE: Dict[
        int,
        Tuple["ArgObject", List[Callable[[Dict[str, Any], Dict[str, Any]], None]]],
    ] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Each subclass gets its own ``_UNPACK_CACHE``; sharing the
        # base class's dict would conflate skeletons of unrelated
        # ArgObject types whose key tuples could collide.
        cls._UNPACK_CACHE = {}

    @classmethod
    def _resolved_field_annotations(cls) -> List[Tuple[str, Any]]:
        if not dataclasses.is_dataclass(cls):
            raise TypeError(
                f"{cls.__name__} must be a @dataclass to use the default "
                f"ArgObject torch_compile_* implementations."
            )
        # ``get_type_hints`` resolves forward references and PEP 563
        # ``from __future__ import annotations`` strings.
        try:
            hints = get_type_hints(cls)
        except Exception:
            hints = {}
        return [
            (f.name, hints.get(f.name, f.type)) for f in dataclasses.fields(cls)
        ]

    @classmethod
    def _buckets(cls) -> List[_Bucket]:
        """Build the bucket list for this dataclass from field annotations.

        Dispatch order per field: try each bucket in :data:`_FIELD_BUCKETS`
        (Tensor, ProcessGroup, Quantizer); if none claims the field, route
        it to :class:`_SimpleBundleBucket` if its annotation is bundle-able,
        else to :class:`_UnknownBucket`.

        Intentionally **not** cached. Caching on ``cls`` (e.g. by writing
        ``cls.__te_buckets__``) tickles Dynamo: subsequent reads of
        ``cls.__dict__`` from a compiled function trigger
        "mappingproxy affected by dictionary mutation" graph breaks.
        Hot paths must instead capture the bucket list once at op
        registration time and pass it explicitly to :meth:`torch_compile_pack`
        / :meth:`torch_compile_unpack`.
        """
        buckets: List[_Bucket] = []
        simple_names: List[str] = []
        for name, annot in cls._resolved_field_annotations():
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

    @classmethod
    def torch_compile_get_schema(cls) -> List[Tuple[str, str]]:
        """Default: derive the schema from dataclass annotations.

        See :class:`_Bucket` subclasses for the per-field-kind layout
        (Tensor, ProcessGroup, Quantizer, and the
        aggregated ``_simple_meta`` bundle of simple fields). A single
        synthetic ``int`` slot named :data:`_CACHE_KEY_SLOT` is
        appended unconditionally; see :meth:`torch_compile_pack` /
        :meth:`torch_compile_unpack` for how it is used.
        """
        slots = [slot for b in cls._buckets() for slot in b.schema_slots()]
        slots.append((_CACHE_KEY_SLOT, "int"))
        return slots

    def torch_compile_pack(
        self, buckets: Optional[List[_Bucket]] = None
    ) -> Dict[str, Any]:
        """Default: ask each bucket to extract its slot(s) from ``self``.

        ``buckets`` is the precomputed bucket list (from
        :meth:`_buckets`). Hot paths -- e.g. the closures created by
        :func:`_te_register_custom_op` -- must pass it to avoid recomputing
        and, critically, to keep Dynamo away from ``cls.__dict__`` while
        tracing. When ``None``, this method recomputes the buckets
        (eager-only fallback intended for ad-hoc / test usage).

        After packing, computes the synthetic
        :data:`_CACHE_KEY_SLOT` integer by hashing the concatenated
        per-bucket :meth:`_Bucket.opaque_cache_key_parts` -- this is
        the cache key consumed by :meth:`torch_compile_unpack`. Under
        ``torch.compile`` Dynamo folds this entire computation to a
        constant during tracing because every input is either an
        interned OSM ``_hash`` or ``id(pg)`` of a guarded ProcessGroup.
        """
        if buckets is None:
            buckets = type(self)._buckets()
        out: Dict[str, Any] = {}
        key_parts: List[int] = []
        for bucket in buckets:
            for name, value in bucket.pack(self):
                out[name] = value
            key_parts.extend(bucket.opaque_cache_key_parts(out))
        out[_CACHE_KEY_SLOT] = hash(tuple(key_parts))
        return out

    @classmethod
    def torch_compile_unpack(
        cls,
        args: Dict[str, Any],
        buckets: Optional[List[_Bucket]] = None,
    ) -> "ArgObject":
        """Default: build a freshly populated dataclass instance from
        the flat ``{slot_name: slot_value}`` dict torch.library hands
        us. The hot path is two-tier:

        1. **Skeleton cache** (:attr:`_UNPACK_CACHE`). Keyed on a
           tuple gathered from each bucket's
           :meth:`_Bucket.opaque_cache_key_parts` -- conceptually
           ``hash`` of every interned :class:`OpaqueSimpleMetadata`
           bundle in ``args`` and ``id`` of every reference-opaque
           input (``ProcessGroup``). On hit, we own a partially
           populated dataclass with every *stable* field already set:
           simple-bundle scalars, PGs, ``None``-valued unknowns, and
           any quantizer whose concrete subclass declared
           ``_TORCH_COMPILE_UNFLATTEN_USES_TENSORS = False`` (the
           reconstructed instance is reused identically across
           iterations because ``_do_unflatten`` ignores its
           ``tensors`` arg).
        2. **Per-call buckets** run on a shallow copy of the skeleton
           and write tensors, tensor-storage payloads, and
           non-cacheable quantizers directly into the copy's
           ``__dict__``. The cache stores the precomputed indices of
           those buckets so the hit path is a flat
           ``for i in per_call_indices: buckets[i].unpack(args, ...)``.

        ``buckets`` semantics match :meth:`torch_compile_pack`: hot
        paths pass the precomputed list, eager-only callers may omit
        it.
        """
        if buckets is None:
            buckets = cls._buckets()
        # Cache key: a single int precomputed by ``torch_compile_pack``
        # -- under ``torch.compile`` it folds to a constant in the FX
        # graph (no runtime hashing); under eager it was just hashed
        # once on the way in. ``cache.get`` is therefore a single dict
        # probe with no per-call list/tuple construction.
        cache_key = args[_CACHE_KEY_SLOT]

        cache = cls._UNPACK_CACHE
        cached = cache.get(cache_key)
        if cached is None:
            skeleton_kwargs: Dict[str, Any] = {}
            per_call_actions: List[
                Callable[[Dict[str, Any], Dict[str, Any]], None]
            ] = []
            for bucket in buckets:
                result = bucket.try_bake_skeleton(args, skeleton_kwargs)
                if result is _SKELETON_BAKED:
                    continue
                # ``None`` -> pre-bind the bucket's ``unpack`` method
                # (saves per-call descriptor lookup).
                # callable -> per-call custom injector (e.g. a
                # storage-shell shallow-copy + tensor-attr swap).
                per_call_actions.append(
                    bucket.unpack if result is None else result
                )
            skeleton = cls.__new__(cls)
            skeleton.__dict__.update(skeleton_kwargs)
            cache[cache_key] = (skeleton, per_call_actions)
        else:
            skeleton, per_call_actions = cached

        # Shallow copy: ``copy.copy`` on a plain (non-slots) instance
        # is just ``cls.__new__(cls); new.__dict__.update(old.__dict__)``,
        # i.e. a single C-level dict copy.
        obj = copy.copy(skeleton)
        obj_dict = obj.__dict__
        for action in per_call_actions:
            # Either a pre-bound ``bucket.unpack`` (writes straight
            # into the copy's ``__dict__``) or a custom injector
            # finalising a cached storage shell.
            action(args, obj_dict)
        return obj

    @classmethod
    def torch_compile_get_input_tensors_for_grad(cls) -> List[str]:
        """Names of forward inputs (from :meth:`torch_compile_get_schema`)
        for which the corresponding ``backward_impl`` produces gradients,
        in the exact order ``backward_impl`` returns them.

        Only meaningful on the forward arg type. Default is ``[]`` (no
        gradients, e.g. for inference-only ops). The wrapper uses this
        to pad the autograd return tuple with ``None`` for every input
        not listed here, so torch sees one slot per forward input as
        required by ``register_autograd``.
        """
        return []


def _te_register_custom_op(
    *,
    linear_impl: Callable[[Any], Any],
    linear_arg_type: type,
    setup_context: Callable[..., None],
    backward_impl: Callable[[Any], Any],
    backward_obj: type,
    backward_arg_type: type,
    num_outputs: int,
    linear_fake_impl: Optional[Callable[[Any], Any]] = None,
    backward_fake_impl: Optional[Callable[[Any], Any]] = None,
    op_namespace: str = "transformer_engine",
    op_name: str = "linear",
) -> Callable[..., Any]:
    """Register a TE module's forward + backward as a single torch custom op.

    Parameters
    ----------
    linear_impl
        Eager forward implementation. Receives a single argument of type
        ``linear_arg_type`` and must return a tuple of the form
        ``(*output_tensors, tensors_to_save, tensor_objects, ctx_attrs)``
        where:

        * ``output_tensors`` -- one or more :class:`torch.Tensor` outputs
          returned to the caller.
        * ``tensors_to_save`` -- flat list of :class:`torch.Tensor` to be
          stashed via ``ctx.save_for_backward``.
        * ``tensor_objects`` -- the metadata object produced by
          :func:`prepare_for_saving`, paired with ``tensors_to_save`` to
          let the backward reconstruct quantized / structured tensors.
        * ``ctx_attrs`` -- non-tensor state to attach to the autograd
          context, restricted to values that cannot be derived from the
          forward args inside ``setup_context``.
    linear_arg_type
        Dataclass type aggregating all forward inputs (e.g.
        :class:`LinearFwdArgs`). Used to (re)build the structured argument
        from the flat tensor / non-tensor inputs accepted by the custom op.
    setup_context
        Eager autograd ``setup_context`` analogue. Receives a freshly
        constructed ``backward_obj`` instance, the forward args, the
        forward output, and ``ctx_attrs`` produced by ``linear_impl``;
        is responsible for populating the backward-state object so that
        ``backward_impl`` can later consume it.
    backward_impl
        Eager backward implementation. Receives a single argument of type
        ``backward_arg_type`` and returns the gradient tuple.
    backward_obj
        Dataclass / class used to instantiate a fresh backward-state
        container at the end of the forward pass (typically the same as
        ``backward_arg_type``).
    backward_arg_type
        Type accepted by ``backward_impl``. May differ from ``backward_obj``
        if the backward op needs a wrapped / opaque view of the state.
    num_outputs
        Number of user-facing tensor outputs returned by ``linear_impl``.
        The op concatenates ``[*output_tensors, *tensors_to_save]`` into
        a single ``Tensor[]`` return; the wrapper uses ``num_outputs`` to
        split the two halves on the way back out.

        The list of forward inputs that receive gradients is declared on
        the forward arg type itself, via
        :meth:`ArgObject.torch_compile_get_input_tensors_for_grad`.
        ``backward_impl`` must return its gradients in that exact order.
    linear_fake_impl
        Optional fake (shape inference) counterpart of ``linear_impl``,
        registered via ``torch.library.register_fake``. Returns the same
        tuple shape as ``linear_impl`` -- ``(*output_tensors,
        tensors_to_save, tensor_objects, ctx_attrs)`` -- but every
        ``torch.Tensor`` is a fake tensor (allocated via
        ``quantizer.make_empty`` or ``torch.empty``) carrying only the
        correct shape / dtype / device, with no real storage or
        computation. ``tensor_objects`` and ``ctx_attrs`` must be
        structurally identical to those produced by ``linear_impl`` so
        that ``setup_context`` and ``backward_impl`` see the same
        non-tensor state in eager and traced modes.
    backward_fake_impl
        Optional fake counterpart of ``backward_impl``. Returns the same
        gradient tuple as ``backward_impl``, with fake tensors in place
        of the real gradients.
    op_namespace, op_name
        Library namespace / op name used when registering with
        ``torch.library``.

    Returns
    -------
    Callable
        A function ``forward_fn(linear_arg_type_instance)`` that dispatches
        through the registered custom op, returning the user-facing
        outputs (single tensor if ``num_outputs == 1``, otherwise a
        tuple). Use under ``torch.compiler.is_compiling()`` as a drop-in
        for ``Function.apply``.
    """

    fwd_qualname = f"{op_namespace}::{op_name}"
    bwd_op_name = f"{op_name}_backward"
    bwd_qualname = f"{op_namespace}::{bwd_op_name}"

    # Precompute the bucket list for both arg types and capture them in
    # the closures below. Critical for the compiled path: re-deriving
    # buckets at call time would force ``ArgObject._buckets`` to read
    # ``cls.__dict__`` from inside a Dynamo-traced function, which
    # triggers a "mappingproxy affected by dictionary mutation" graph
    # break under ``fullgraph=True``.
    fwd_buckets: List[_Bucket] = linear_arg_type._buckets()
    bwd_buckets: List[_Bucket] = backward_arg_type._buckets()

    def _build_schema(buckets: List[_Bucket]) -> Tuple[str, List[str]]:
        spec = [slot for b in buckets for slot in b.schema_slots()]
        # Synthetic int slot consumed by ``torch_compile_unpack`` to
        # key the skeleton cache in O(1). Kept in sync with
        # :meth:`ArgObject.torch_compile_get_schema`.
        spec.append((_CACHE_KEY_SLOT, "int"))
        names = [name for name, _ in spec]
        schema_str = "(" + ", ".join(f"{type_str} {name}" for name, type_str in spec) + ")"
        return schema_str, names

    fwd_schema_args, fwd_arg_names = _build_schema(fwd_buckets)
    bwd_schema_args, bwd_arg_names = _build_schema(bwd_buckets)

    # ``torch.library.register_autograd`` requires the backward to return
    # one grad slot per forward input, with the same Python tree
    # structure as the input itself: a ``Tensor[]`` slot must get back a
    # ``list``, not a bare ``None``. Precompute the per-slot "no-grad"
    # value so the autograd return matches.
    fwd_slot_defaults: List[Any] = []
    for bucket in fwd_buckets:
        for _, type_str in bucket.schema_slots():
            fwd_slot_defaults.append([] if type_str.endswith("[]") else None)
    # Synthetic ``int`` cache-key slot appended in ``_build_schema``;
    # ints carry no grad, default is ``None``.
    fwd_slot_defaults.append(None)

    # Validate ``input_tensors_for_grad`` references real forward inputs
    # and precompute the positions where backward grads land in the
    # autograd return tuple. Some logical fields (e.g. Tensor-or-storage
    # fields) expand to a ``Tensor[]`` slot; their gradient must be returned
    # as a list matching that input tree.
    input_tensors_for_grad = linear_arg_type.torch_compile_get_input_tensors_for_grad()
    fwd_grad_targets: Dict[str, Tuple[int, bool]] = {}
    slot_offset = 0
    for bucket in fwd_buckets:
        slots = bucket.schema_slots()
        if isinstance(bucket, _TensorBucket):
            fwd_grad_targets[bucket.name] = (slot_offset, False)
        elif isinstance(bucket, _TensorListBucket):
            fwd_grad_targets[bucket.name] = (slot_offset, True)
        elif isinstance(bucket, _TensorOrStorageBucket):
            for i, (slot_name, _) in enumerate(slots):
                if slot_name == bucket._slot_tensors():
                    fwd_grad_targets[bucket.name] = (slot_offset + i, True)
                    break
        slot_offset += len(slots)
    unknown_grad_names = [n for n in input_tensors_for_grad if n not in fwd_grad_targets]
    if unknown_grad_names:
        raise ValueError(
            f"{linear_arg_type.__name__}.torch_compile_get_input_tensors_for_grad() "
            f"contains names not present in "
            f"{linear_arg_type.__name__}.torch_compile_get_schema(): "
            f"{unknown_grad_names}"
        )
    grad_targets = [fwd_grad_targets[n] for n in input_tensors_for_grad]
    num_grad_inputs = len(input_tensors_for_grad)

    lib = torch.library.Library(op_namespace, "FRAGMENT")
    # Forward op concatenates user outputs and tensors_to_save into a
    # single ``Tensor[]`` return so that autograd's ``setup_context`` can
    # stash the saved-for-backward tensors without re-running the eager
    # impl. The schema is non-nullable (``Tensor[]``, not ``Tensor?[]``)
    # because ``torch.library.register_autograd`` does not propagate
    # ``grad_fn`` to a nullable list output. ``None`` entries on either
    # side are smuggled through via :func:`_encode_none` /
    # :func:`_decode_none` sentinels (see below).
    lib.define(f"{op_name}{fwd_schema_args} -> Tensor[]")
    lib.define(f"{bwd_op_name}{bwd_schema_args} -> Tensor[]")

    def _outputs_for_setup(outputs: List[torch.Tensor]) -> Any:
        return outputs[0] if num_outputs == 1 else tuple(outputs)

    def _prepare_for_saving(tensors: Any) -> Tuple[List[Optional[torch.Tensor]], Any]:
        from transformer_engine.pytorch.quantized_tensor import prepare_for_saving

        return prepare_for_saving(*(tensors or ()))

    def _restore_from_saved(tensor_objects: Any, saved_tensors: List[Any]) -> Any:
        from transformer_engine.pytorch.quantized_tensor import restore_from_saved

        return restore_from_saved(tensor_objects, saved_tensors)

    def _fwd_impl(*flat: Any) -> List[torch.Tensor]:
        kwargs = dict(zip(fwd_arg_names, flat))
        obj = linear_arg_type.torch_compile_unpack(kwargs, fwd_buckets)
        result = linear_impl(obj)
        outputs = list(result[:num_outputs])
        tensors_to_save, _ = _prepare_for_saving(result[num_outputs])
        return [_encode_none(t) for t in outputs + tensors_to_save]

    lib.impl(op_name, _fwd_impl, "CompositeExplicitAutograd")

    if linear_fake_impl is not None:

        def _fwd_fake(*flat: Any) -> List[torch.Tensor]:
            kwargs = dict(zip(fwd_arg_names, flat))
            obj = linear_arg_type.torch_compile_unpack(kwargs, fwd_buckets)
            result = linear_fake_impl(obj)
            outputs = list(result[:num_outputs])
            tensors_to_save, _ = _prepare_for_saving(result[num_outputs])
            return [_encode_none(t) for t in outputs + tensors_to_save]

        torch.library.register_fake(fwd_qualname, _fwd_fake, lib=lib)

    def _check_bwd_len(grads):
        if len(grads) != num_grad_inputs:
            raise RuntimeError(
                f"{op_namespace}::{bwd_op_name} expected backward_impl to "
                f"return {num_grad_inputs} grads (one per "
                f"input_tensors_for_grad entry), got {len(grads)}"
            )

    def _bwd_impl(*flat: Any) -> List[torch.Tensor]:
        kwargs = dict(zip(bwd_arg_names, flat))
        obj = backward_arg_type.torch_compile_unpack(kwargs, bwd_buckets)
        grads = list(backward_impl(obj))
        _check_bwd_len(grads)
        return [_encode_none(g) for g in grads]

    lib.impl(bwd_op_name, _bwd_impl, "CompositeExplicitAutograd")

    if backward_fake_impl is not None:

        def _bwd_fake(*flat: Any) -> List[torch.Tensor]:
            kwargs = dict(zip(bwd_arg_names, flat))
            obj = backward_arg_type.torch_compile_unpack(kwargs, bwd_buckets)
            grads = list(backward_fake_impl(obj))
            _check_bwd_len(grads)
            return [_encode_none(g) for g in grads]

        torch.library.register_fake(bwd_qualname, _bwd_fake, lib=lib)

    # Re-run fake (or real) impl in setup_context to recover
    # tensor_objects / ctx_attrs, which are not part of the op's return.
    fake_for_setup = linear_fake_impl if linear_fake_impl is not None else linear_impl

    def _setup_context(ctx, inputs, output):
        ctx._te_fwd_tensor_list_lengths = {
            i: len(value) for i, value in enumerate(inputs) if isinstance(value, list)
        }
        kwargs = dict(zip(fwd_arg_names, inputs))
        fwd_obj = linear_arg_type.torch_compile_unpack(kwargs, fwd_buckets)
        fake_result = fake_for_setup(fwd_obj)
        _, tensor_objects = _prepare_for_saving(fake_result[num_outputs])
        ctx_attrs = fake_result[num_outputs + 2]

        # Split op output: first num_outputs are user-facing tensors,
        # the rest are tensors_to_save. ``output`` is a flat ``Tensor[]``
        # with our None-sentinels in place; decode here so downstream
        # eager code sees the original ``None``\ s.
        user_outputs = [_decode_none(t) for t in output[:num_outputs]]
        op_saved_tensors = [_decode_none(t) for t in output[num_outputs:]]
        tensors_to_save_from_forward = _restore_from_saved(
            tensor_objects,
            op_saved_tensors,
        )

        bwd_obj = backward_obj()
        tensors_to_save_from_setup = setup_context(
            bwd_obj,
            fwd_obj,
            _outputs_for_setup(user_outputs),
            ctx_attrs,
            tensors_to_save_from_forward,
        )
        tensors_to_save, tensor_objects = _prepare_for_saving(tensors_to_save_from_setup)
        ctx.tensor_objects = tensor_objects
        ctx.save_for_backward(*tensors_to_save)
        ctx.bwd_obj = bwd_obj

    def _autograd_backward(ctx, *grad_outputs):
        bwd_obj = ctx.bwd_obj
        if hasattr(bwd_obj, "setup_saved_tensors"):
            bwd_obj.setup_saved_tensors(ctx.saved_tensors, ctx.tensor_objects)
        ctx.tensor_objects = None
        # The forward op returns a single ``Tensor[]`` (concatenation of
        # user outputs and saved tensors), so ``grad_outputs`` is a
        # 1-tuple containing the per-element grad list. Only the first
        # ``num_outputs`` of those correspond to user-facing outputs;
        # ``grad_output`` for the backward is the grad of the primary
        # output.
        per_output_grads = grad_outputs[0]
        bwd_obj.grad_output = _decode_none(per_output_grads[0])
        kwargs = backward_arg_type.torch_compile_pack(bwd_obj, bwd_buckets)
        bwd_args_flat = [kwargs[name] for name in bwd_arg_names]
        bwd_op = getattr(getattr(torch.ops, op_namespace), bwd_op_name)
        grads = [_decode_none(g) for g in bwd_op(*bwd_args_flat)]
        # ``register_autograd`` requires one grad slot per forward input
        # with the same tree structure as the input (a ``Tensor[]`` slot
        # must get back a list, never a bare ``None``). Start from the
        # precomputed per-slot defaults and overlay the produced grads
        # at the positions declared by ``input_tensors_for_grad``.
        out: List[Any] = list(fwd_slot_defaults)
        tensor_list_lengths = getattr(ctx, "_te_fwd_tensor_list_lengths", {})
        for (pos, as_list), g in zip(grad_targets, grads):
            if as_list:
                length = tensor_list_lengths.get(pos, 1)
                out[pos] = [g] + [None] * (length - 1)
            else:
                out[pos] = g
        return tuple(out)

    torch.library.register_autograd(
        fwd_qualname,
        _autograd_backward,
        setup_context=_setup_context,
        lib=lib,
    )

    fwd_op = getattr(getattr(torch.ops, op_namespace), op_name)

    def forward_fn(fwd_args):
        # Bind ``lib`` here so its registrations (impl / register_fake /
        # register_autograd) outlive ``_te_register_custom_op`` even if
        # all other references to it are dropped: ``torch.library`` uses
        # the ``Library`` instance lifetime for all attached registrations.
        _ = lib  # noqa: F841 -- closure-captured for lifetime only
        kwargs = linear_arg_type.torch_compile_pack(fwd_args, fwd_buckets)
        flat = [kwargs[name] for name in fwd_arg_names]
        result = fwd_op(*flat)
        outputs = [_decode_none(t) for t in result[:num_outputs]]
        if num_outputs == 1:
            return outputs[0]
        return tuple(outputs)

    return forward_fn
