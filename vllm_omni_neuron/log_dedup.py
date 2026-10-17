# SPDX-License-Identifier: Apache-2.0
"""Suppress exact-repeat dependency logs while exempting package-owned loggers."""

import logging

_EXEMPT_PREFIX = "vllm_omni_neuron"
_seen: dict[tuple[str, int, str], int] = {}
_installed = False
_original_handle = logging.Logger.handle


def _is_exempt(logger_name: str) -> bool:
    return logger_name == _EXEMPT_PREFIX or logger_name.startswith(_EXEMPT_PREFIX + ".")


def _deduped_handle(self: logging.Logger, record: logging.LogRecord) -> None:
    if _is_exempt(record.name):
        _original_handle(self, record)
        return
    key = (record.name, record.lineno, record.getMessage())
    count = _seen.get(key, 0)
    _seen[key] = count + 1
    if count == 0:
        _original_handle(self, record)
    elif count == 1:
        record.msg = f"{record.getMessage()} (further exact repeats suppressed)"
        record.args = ()
        _original_handle(self, record)


def install() -> None:
    """Idempotently patch ``logging.Logger.handle`` for this process."""
    global _installed
    if _installed:
        return
    logging.Logger.handle = _deduped_handle
    _installed = True
