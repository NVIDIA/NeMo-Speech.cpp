import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

VERSION = "1.2.2"


def install(protocol):
    import msgpack
    import msgpack._cmsgpack

    if msgpack.__version__ != VERSION or msgpack.Packer.__module__ != "msgpack._cmsgpack":
        raise RuntimeError("required native msgpack version is " + VERSION)
    if getattr(protocol, "_r11_native_serializer", None):
        return protocol._r11_native_serializer
    original = protocol.packb
    native = msgpack.packb
    error_type = protocol.ProtocolError
    limit = protocol.MAX_COLLECTION_ENTRIES

    def validate(value, depth=0):
        if depth > 64:
            raise error_type("MessagePack nesting exceeds 64")
        kind = type(value)
        if value is None or kind in (bool, str, bytes, bytearray, memoryview):
            return
        if kind is int:
            if not -(1 << 63) <= value <= (1 << 64) - 1:
                raise error_type("integer is outside MessagePack int64/uint64")
            return
        if kind is float:
            if not math.isfinite(value):
                raise error_type("non-finite floats are forbidden")
            return
        if kind is dict:
            if len(value) > limit:
                raise error_type("MessagePack map is unreasonably large")
            for key, child in value.items():
                if type(key) is not str:
                    raise error_type("protocol map keys must be strings")
                validate(key, depth + 1)
                validate(child, depth + 1)
            return
        if kind in (list, tuple):
            if len(value) > limit:
                raise error_type("MessagePack array is unreasonably large")
            for child in value:
                validate(child, depth + 1)
            return
        if isinstance(value, (Mapping, Sequence, int, float, str)):
            raise TypeError("legacy compatible value")
        raise error_type("unsupported MessagePack type: " + kind.__name__)

    def packb(value):
        try:
            validate(value)
        except TypeError:
            return original(value)
        try:
            return native(value, use_bin_type=True, use_single_float=False)
        except (ValueError, TypeError, OverflowError, BufferError):
            return original(value)

    path = Path(msgpack._cmsgpack.__file__)
    record = {
        "version": VERSION,
        "implementation": msgpack.Packer.__module__,
        "extension": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "wire_contract": "legacy exact bytes, validation and frame limits preserved",
    }
    protocol.packb = packb
    from .. import r31

    if r31.enabled("native_unpack"):
        protocol.unpackb = native_unpackb(protocol)
        record["unpack"] = (
            "native msgpack.unpackb with the pure-Python validation contract; "
            "float32 encodings decode instead of being rejected (never sent by the frontend)"
        )
    protocol._r11_native_serializer = record
    return record


def native_unpackb(protocol):
    import msgpack

    error_type = protocol.ProtocolError
    limit = protocol.MAX_COLLECTION_ENTRIES

    def pairs(items):
        result = {}
        for key, child in items:
            if type(key) is not str:
                raise error_type("protocol map keys must be strings")
            if key in result:
                raise error_type(f"duplicate protocol map key {key!r}")
            result[key] = child
        return result

    def check(value, depth):
        if depth > 64:
            raise error_type("MessagePack nesting exceeds 64")
        kind = type(value)
        if kind is float:
            if not math.isfinite(value):
                raise error_type("non-finite floats are forbidden")
        elif kind is dict or kind is list:
            if value and depth + 1 > 64:
                raise error_type("MessagePack nesting exceeds 64")
            for child in value.values() if kind is dict else value:
                if type(child) in (dict, list, float):
                    check(child, depth + 1)

    def unpackb(payload):
        try:
            value = msgpack.unpackb(
                payload,
                raw=False,
                use_list=True,
                strict_map_key=False,
                object_pairs_hook=pairs,
                max_array_len=limit,
                max_map_len=limit,
                max_ext_len=0,
                max_str_len=len(payload),
                max_bin_len=len(payload),
            )
        except error_type:
            raise
        except msgpack.ExtraData as error:
            raise error_type("trailing bytes after MessagePack value") from error
        except (
            ValueError,
            TypeError,
            UnicodeDecodeError,
            msgpack.FormatError,
            msgpack.StackError,
        ) as error:
            raise error_type("invalid MessagePack value: " + str(error)) from error
        check(value, 0)
        return value

    return unpackb
