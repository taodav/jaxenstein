"""Small pytree dataclass helper used for JAX environment state."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax


field = dataclasses.field


def dataclass(cls: type[Any] | None = None, **kwargs: Any):
    """Create a frozen dataclass registered as a JAX pytree."""

    kwargs.setdefault("frozen", True)
    kwargs.setdefault("eq", False)

    def wrap(cls: type[Any]) -> type[Any]:
        dataclass_cls = dataclasses.dataclass(**kwargs)(cls)
        data_fields = tuple(field.name for field in dataclasses.fields(dataclass_cls))

        def replace(self, **updates: Any):
            return dataclasses.replace(self, **updates)

        dataclass_cls.replace = replace
        return jax.tree_util.register_dataclass(
            dataclass_cls,
            data_fields=data_fields,
            meta_fields=(),
        )

    if cls is None:
        return wrap
    return wrap(cls)
