import random
import struct
import sys

n = int(sys.argv[1])
if not 0 <= n <= 537000000:
    raise ValueError("Invalid array size")

sys.stdout.buffer.write(struct.pack("<i", n))
sys.stdout.buffer.write(random.Random(42).randbytes(n))
