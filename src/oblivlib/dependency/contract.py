"""The data contract at the library's public entry points: ORAM keys are int addresses in
``[0, num_data)``, OMAP keys are ``bytes`` of at most ``key_size``, and values are ``bytes`` of at most
``data_size``. Each check raises ``ContractError`` prefixed with ``owner``, the scheme's identity."""

from typing import Any

from oblivlib.dependency.errors import ContractError


def require_oram_key(owner: str, key: Any, num_data: int) -> int:
    if type(key) is not int or not 0 <= key < num_data:
        raise ContractError(f"{owner}: a key must be an int in [0, {num_data}), got {key!r}.")
    return key


def require_omap_key(owner: str, key: Any, key_size: int) -> bytes:
    if type(key) is not bytes:
        raise ContractError(f"{owner}: a key must be bytes, got {type(key).__name__}.")
    if len(key) > key_size:
        raise ContractError(f"{owner}: a key is {len(key)} bytes, key_size is {key_size}.")
    return key


def require_value(owner: str, value: Any, data_size: int) -> bytes:
    if type(value) is not bytes:
        raise ContractError(f"{owner}: a value must be bytes, got {type(value).__name__}.")
    if len(value) > data_size:
        raise ContractError(f"{owner}: a value is {len(value)} bytes, data_size is {data_size}.")
    return value
