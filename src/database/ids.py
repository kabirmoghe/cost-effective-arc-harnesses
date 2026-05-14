"""UUID v7 generator (RFC 9562). Stdlib gains this in Python 3.14."""

import secrets
import time
import uuid


def uuid7() -> uuid.UUID:
    ts_ms = int(time.time() * 1000) & ((1 << 48) - 1)
    rand_a = secrets.randbits(12)
    rand_b = secrets.randbits(62)
    n = ts_ms << 80
    n |= 0x7 << 76
    n |= rand_a << 64
    n |= 0x2 << 62
    n |= rand_b
    return uuid.UUID(int=n)
