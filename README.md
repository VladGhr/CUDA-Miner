## Building the Merkle Root

The `construct_merkle_root` function builds the Merkle tree through a binary reduction: level 0 contains the SHA256 of each transaction, and each higher level combines pairs of hashes until a single hash remains (the root).

### The `merkle_level0_kernel` kernel

It uses one thread per transaction. Each thread computes `index = blockIdx.x * blockDim.x + threadIdx.x`, and if the index exceeds the number of transactions `n`, the thread stops. Otherwise it applies `apply_sha256` to the transaction at offset `index * transaction_size` and writes the result into the hash vector on the GPU.

After this kernel we obtain `n` hashes, each of length `SHA256_HASH_SIZE`, stored directly in the GPU's memory.

### The `merkle_reduce_kernel` kernel

This kernel performs a single reduction level and is launched repeatedly from the host until a single hash remains. One thread processes a pair of hashes, so `ceil(n / 2)` threads are needed.

Each thread:

Computes the indices `i = tid * 2` and `j = i + 1`. If `j` exceeds `n` (an odd number of hashes on the current level), `j` becomes equal to `i`, which duplicates the last hash. It then concatenates the two hashes into a local `combined` buffer of `2 * 64` hex characters, followed by the null terminator. The concatenation is done through two loops marked with `#pragma unroll`, after which it applies `apply_sha256` to the concatenated buffer and writes the result into the output vector.

### The `construct_merkle_root` host function

The function manages memory and orchestrates the kernel launches.

It uses persistent buffers. The three GPU buffers (`d_transactions`, `d_hashes_a`, `d_hashes_b`) are declared `static` and allocated only once, on the first call, with the size given by `max_transactions_in_a_block`. This avoids a `cudaMalloc` and a `cudaFree` on every transaction block, which would have introduced significant overhead since the function is called for every block in the blockchain.

Execution steps:

1. Copies the current block's transactions to the GPU through a single host-to-device `cudaMemcpy`.
2. Launches `merkle_level0_kernel` with `ceil(n / 256)` blocks of 256 threads each.
3. Applies the reduction iteratively. On each iteration it launches `merkle_reduce_kernel`, then swaps the input buffer with the output buffer. This way the result of one level becomes the input of the next level without any additional copies. The number of hashes halves at each step (`cur_n = ceil(cur_n / 2)`).
4. When a single hash remains, it copies it back to the host through a device-to-host `cudaMemcpy`. Because `cudaMemcpy` is synchronous, it implicitly waits for all previously launched kernels to finish, so an explicit `cudaDeviceSynchronize` is not needed.


## Nonce Search

The `find_nonce` function searches for the smallest nonce for which `SHA256(prev_block_hash || merkle_root || nonce)`, expressed as a hexadecimal string, begins with the required number of zeros. The search space `[0, max_nonce)` is partitioned among the threads.

### The optimization insight: midstate caching

The prefix to which SHA256 is applied is `prev_block_hash || merkle_root`, that is, two hexadecimal strings of 64 characters each, exactly 128 bytes in total. SHA256 processes the message in internal blocks of 64 bytes, so the prefix corresponds to exactly two transformations (`sha256_transform`).

An important aspect of this implementation is that these two transformations are identical for all nonces tested within a block, because the prefix does not change. Only the last SHA256 block — the one containing the nonce digits and the padding — differs from one nonce to another.

Conclusion: instead of recomputing the entire SHA256 (3 transformations) for each of the tens of thousands of nonces, we compute the intermediate state after the first 128 bytes once, on the host, and reuse it. In the kernel, each thread then performs a single transformation instead of three.

### Host-side preparation (in `find_nonce`)

Before launching the kernel, the host does three things, once per block:

1. Computing the midstate. It calls `sha256_init`, then `sha256_trans` on the first 64 bytes of the prefix and once more on the next 64. The resulting state (`SHA256_CTX.state`, 8 words of 32 bits) is copied into constant memory `c_midstate` via `cudaMemcpyToSymbol`.
2. Parsing the difficulty: `difficulty` is received as a 64-character hexadecimal string. It is converted into 32 raw bytes (using the `hex_nibble` helper) and copied into constant memory `c_difficulty_bytes`. This way the comparison in the kernel is performed on 32 bytes instead of 64 hex characters.
3. Initializing the result. The `d_valid_nonce` variable on the GPU is set to `UINT32_MAX`, a value that signals that the nonce has not been found yet.

Constant memory is used for the midstate and difficulty because this data is read-only, small, and identical for all threads — a case in which the hardware provides a very efficient broadcast mechanism.

### The `find_nonce_kernel` kernel

2048 blocks of 256 threads each are launched, i.e. 524288 threads. Each thread starts from `nonce = tid` and advances in steps of `stride = gridDim.x * blockDim.x`, so that the entire space `[0, max_nonce)` is covered regardless of how large `max_nonce` is.

For each candidate nonce, the thread:

1. Checks whether `valid_nonce` (read through a `volatile` pointer, to force a re-read from global memory) is already less than or equal to the current nonce; if so, the thread stops. The reason: we are looking for the global minimum, and all the nonces this thread would still test are larger than the current one, so they cannot improve the result.
2. Converts the nonce into a decimal string with `intToString`.
3. Manually builds the last 64-byte SHA256 block: the nonce digits, followed by the padding byte `0x80`, then zeros, and on the last 8 bytes the total message length in bits, in big-endian format. The total length is `(128 + nonce_len) * 8`. This manual construction avoids the overhead of the `sha256_update` and `sha256_final` functions.
4. Initializes a `SHA256_CTX` with the state taken from `c_midstate` and calls `sha256_trans` a single time on the constructed block.
5. Extracts the final hash as 32 raw bytes directly from `ctx.state` (with big-endian conversion) and compares it byte by byte with `c_difficulty_bytes`. The comparison stops at the first differing byte: if the hash's byte is smaller, the nonce is valid; if it is larger, it is not.
6. If the nonce is valid, it calls `atomicMin(valid_nonce, nonce)`.

Using `atomicMin` guarantees that, regardless of the order in which threads find valid nonces, `valid_nonce` retains the smallest of them.

### Host-side finalization

After the kernel finishes (implicitly awaited by `cudaMemcpy`), the host retrieves the value of `d_valid_nonce`. If it is still `UINT32_MAX`, no nonce was found and the function returns 1. Otherwise, it writes the found nonce into `valid_nonce` and recomputes `block_hash` once on the host, with `apply_sha256`, to populate the output expected by the rest of the program.
