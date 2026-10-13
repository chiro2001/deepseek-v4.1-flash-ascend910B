"""Deduplicate cache backing-storage metadata without importing a device SDK."""
from collections.abc import Mapping

from .topology import integer


def backing_storages(caches):
    found = {}

    def visit(value):
        if isinstance(value, Mapping):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
        elif value is not None:
            if not hasattr(value, "device") or not callable(getattr(value, "untyped_storage", None)):
                raise ValueError("Unknown cache backing object; cannot claim allocated pool bytes")
            storage = value.untyped_storage()
            pointer = integer(storage.data_ptr(), "storage.data_ptr", minimum=1)
            size = integer(storage.nbytes(), "storage.nbytes", minimum=1)
            key = (str(value.device), pointer)
            row = {"device": key[0], "data_ptr": pointer, "size_bytes": size}
            if key in found and found[key] != row:
                raise ValueError("Conflicting sizes for the same backing storage")
            found[key] = row

    visit(caches)
    if not found:
        raise ValueError("No actual cache backing metadata observed")
    return list(found.values())
