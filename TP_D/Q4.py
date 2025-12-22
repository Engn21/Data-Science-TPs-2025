import numpy as np
from collections import Counter
from dahuffman import HuffmanCodec
import math

print("\n DATA GENERATION ")

# Parameters
n = 10000
theta = 0.1

# Generate Bernoulli samples
X = np.random.choice([0,1], size=n, p=[0.9, 0.1])

# Group into 5-bit symbols
symbols = X.reshape(-1,5)
symbol_strings = [''.join(map(str, s)) for s in symbols]

# Count frequencies of each 5-bit symbol
freqs = Counter(symbol_strings)

print("Generated 10,000 samples.")
print("Total symbols:", len(symbol_strings))
print("Example symbols:", symbol_strings[:10])
print("Example frequencies:", list(freqs.items())[:5])

print("\nHUFFMAN CODING (dahuffman) ")

# Build Huffman code from frequencies
codec = HuffmanCodec.from_frequencies(freqs)

# Encode the whole symbol sequence
encoded = codec.encode(symbol_strings)

# 'encoded' is a bytes object -> convert to bits
encoded_bits = len(encoded) * 8

print("Huffman code created using dahuffman.")
print("Total encoded length (bits):", encoded_bits)

# Get code table and show a few codes
code_table = codec.get_code_table()   # dict: symbol -> bitarray (or similar)
print("Example Huffman codes:")
for s in list(freqs.keys())[:5]:
    print(f"  {s} -> {code_table[s]}")

print("\n===== 3. ENTROPY & EFFICIENCY ")

# Theoretical entropy of X
H_X = -0.9 * math.log2(0.9) - 0.1 * math.log2(0.1)
H5 = 5 * H_X  # for 5 samples per symbol

# Average code length per symbol
L = encoded_bits / len(symbol_strings)

print("H(X) =", H_X)
print("Entropy per 5-bit symbol H5 =", H5)
print("Average Huffman code length L =", L)

print("\nSUMMARY ")
if H5 <= L < H5 + 1:
    print("Compression efficiency is good: Huffman is close to optimal.")
else:
    print("Unexpected result: L not within Shannon bounds.")
