"""Worker-local token mirrors required before a CED DRAM continuation."""

from __future__ import annotations

from weakref import WeakSet

_histories = WeakSet()


def register_history(history):
    _histories.add(history)


def restore_history_pages(pages, block_size):
    if not pages:
        return
    if not _histories:
        raise RuntimeError("CED DRAM resumed without a registered Engram token history")
    for history in tuple(_histories):
        history.restore_token_pages(pages, block_size)
    print("[CED-DRAM] restored Engram history pages=%d block_size=%d" %
          (len(pages), block_size), flush=True)
