#include <stdio.h>
#include <stdint.h>
#include "utils.h"
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>


// CUDA sprintf alternative for nonce finding. Converts integer to its string representation. Returns string's length.

#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))
#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))

#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x,2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x) (ROTRIGHT(x,6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x) (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

__host__ __device__ void sha256_trans(SHA256_CTX *ctx, const BYTE data[])
{
	WORD a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];
	WORD k[64] = {
		0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
		0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
		0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
		0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
		0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
		0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
		0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
		0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
	};

	for (i = 0, j = 0; i < 16; ++i, j += 4)
		m[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);
	for ( ; i < 64; ++i)
		m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];

	a = ctx->state[0];
	b = ctx->state[1];
	c = ctx->state[2];
	d = ctx->state[3];
	e = ctx->state[4];
	f = ctx->state[5];
	g = ctx->state[6];
	h = ctx->state[7];

	for (i = 0; i < 64; ++i) {
		t1 = h + EP1(e) + CH(e,f,g) + k[i] + m[i];
		t2 = EP0(a) + MAJ(a,b,c);
		h = g;
		g = f;
		f = e;
		e = d + t1;
		d = c;
		c = b;
		b = a;
		a = t1 + t2;
	}

	ctx->state[0] += a;
	ctx->state[1] += b;
	ctx->state[2] += c;
	ctx->state[3] += d;
	ctx->state[4] += e;
	ctx->state[5] += f;
	ctx->state[6] += g;
	ctx->state[7] += h;
}


__device__ int intToString(uint64_t num, char *out)
{
    if (num == 0)
    {
        out[0] = '0';
        out[1] = '\0';
        return 1;
    }

    int i = 0;
    while (num != 0)
    {
        int digit = num % 10;
        num /= 10;
        out[i++] = '0' + digit;
    }

    // Reverse the string
    for (int j = 0; j < i / 2; j++)
    {
        char temp = out[j];
        out[j] = out[i - j - 1];
        out[i - j - 1] = temp;
    }
    out[i] = '\0';
    return i;
}

// CUDA strlen implementation.
__host__ __device__ size_t d_strlen(const char *str)
{
    size_t len = 0;
    while (str[len] != '\0')
    {
        len++;
    }
    return len;
}

// CUDA strcpy implementation.
__device__ void d_strcpy(char *dest, const char *src)
{
    int i = 0;
    while ((dest[i] = src[i]) != '\0')
    {
        i++;
    }
}

// CUDA strcat implementation.
__device__ void d_strcat(char *dest, const char *src)
{
    while (*dest != '\0')
    {
        dest++;
    }
    while (*src != '\0')
    {
        *dest = *src;
        dest++;
        src++;
    }
    *dest = '\0';
}

// Compute SHA256 and convert to hex
__host__ __device__ void apply_sha256(const BYTE *input, BYTE *output)
{
    size_t input_length = d_strlen((const char *)input);
    SHA256_CTX ctx;
    BYTE buf[SHA256_BLOCK_SIZE];
    const char hex_chars[] = "0123456789abcdef";

    sha256_init(&ctx);
    sha256_update(&ctx, input, input_length);
    sha256_final(&ctx, buf);

    for (size_t i = 0; i < SHA256_BLOCK_SIZE; i++)
    {
        output[i * 2] = hex_chars[(buf[i] >> 4) & 0x0F]; // High nibble
        output[i * 2 + 1] = hex_chars[buf[i] & 0x0F];    // Low nibble
    }
    output[SHA256_BLOCK_SIZE * 2] = '\0'; // Null-terminate
}

// Compare two hashes
__host__ __device__ int compare_hashes(BYTE *hash1, BYTE *hash2)
{
    for (int i = 0; i < SHA256_HASH_SIZE - 1; i++)
    {
        if (hash1[i] < hash2[i])
        {
            return -1; // hash1 is lower
        }
        else if (hash1[i] > hash2[i])
        {
            return 1; // hash2 is lower
        }
    }
    return 0; // hashes are equal
}

__global__ void merkle_level0_kernel(const BYTE *transactions, int transaction_size, int n, BYTE (*hashes)[SHA256_HASH_SIZE])
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n)
        return;
    apply_sha256(transactions + tid * transaction_size, hashes[tid]);
}

__global__ void merkle_reduce_kernel(const BYTE (*input)[SHA256_HASH_SIZE], BYTE (*output)[SHA256_HASH_SIZE], int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int pair_count = (n + 1) / 2;
    if (tid >= pair_count)
        return;

    int i = tid * 2;
    int j = (i + 1 < n) ? (i + 1) : i;

    BYTE combined[SHA256_HASH_SIZE * 2];

#pragma unroll
    for (int k = 0; k < SHA256_HASH_SIZE - 1; k++)
    {
        combined[k] = input[i][k];
    }
#pragma unroll
    for (int k = 0; k < SHA256_HASH_SIZE - 1; k++)
    {
        combined[(SHA256_HASH_SIZE - 1) + k] = input[j][k];
    }
    combined[2 * (SHA256_HASH_SIZE - 1)] = '\0';

    apply_sha256(combined, output[tid]);
}

void construct_merkle_root(int transaction_size, BYTE *transactions,int max_transactions_in_a_block,int n, BYTE merkle_root[SHA256_HASH_SIZE])
{
   
    static BYTE *d_transactions = NULL;
    static BYTE(*d_hashes_a)[SHA256_HASH_SIZE] = NULL;
    static BYTE(*d_hashes_b)[SHA256_HASH_SIZE] = NULL;
    static int allocated_max = 0;
    static int allocated_tx_size = 0;

    if (max_transactions_in_a_block > allocated_max || transaction_size > allocated_tx_size)
    {
        if (d_transactions)
            cudaFree(d_transactions);
        if (d_hashes_a)
            cudaFree(d_hashes_a);
        if (d_hashes_b)
            cudaFree(d_hashes_b);

        cudaMalloc((void **)&d_transactions, (size_t)max_transactions_in_a_block * transaction_size);
        cudaMalloc((void **)&d_hashes_a, (size_t)max_transactions_in_a_block * SHA256_HASH_SIZE);
        cudaMalloc((void **)&d_hashes_b, (size_t)max_transactions_in_a_block * SHA256_HASH_SIZE);

        allocated_max = max_transactions_in_a_block;
        allocated_tx_size = transaction_size;
    }

    cudaMemcpy(d_transactions, transactions, (size_t)n * transaction_size, cudaMemcpyHostToDevice);

    const int TPB = 256;
    int blocks0 = (n + TPB - 1) / TPB;
    merkle_level0_kernel<<<blocks0, TPB>>>(d_transactions, transaction_size, n, d_hashes_a);

    int cur_n = n;
    BYTE(*in_buf)
    [SHA256_HASH_SIZE] = d_hashes_a;
    BYTE(*out_buf)
    [SHA256_HASH_SIZE] = d_hashes_b;

    while (cur_n > 1)
    {
        int next_n = (cur_n + 1) / 2;
        int blocks_r = (next_n + TPB - 1) / TPB;
        merkle_reduce_kernel<<<blocks_r, TPB>>>(in_buf, out_buf, cur_n);

        BYTE(*tmp)
        [SHA256_HASH_SIZE] = in_buf;
        in_buf = out_buf;
        out_buf = tmp;

        cur_n = next_n;
    }

    cudaMemcpy(merkle_root, in_buf[0], SHA256_HASH_SIZE, cudaMemcpyDeviceToHost);
}



__constant__ WORD c_midstate[8];         
__constant__ BYTE c_difficulty_bytes[32]; 

__global__ void find_nonce_kernel(uint32_t max_nonce, uint32_t *valid_nonce)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;

    volatile uint32_t *vn = valid_nonce;

    uint32_t nonce = tid;
    while (nonce < max_nonce)
    {
        if (*vn <= nonce)
            return;

        char nonce_str[12];
        int nonce_len = intToString((uint64_t)nonce, nonce_str);

        BYTE block[64];
        for (int i = 0; i < nonce_len; i++)
        {
            block[i] = (BYTE)nonce_str[i];
        }
        block[nonce_len] = 0x80;
        for (int i = nonce_len + 1; i < 56; i++)
        {
            block[i] = 0;
        }
        uint32_t bitlen = (uint32_t)(128 + nonce_len) * 8;
        block[56] = 0;
        block[57] = 0;
        block[58] = 0;
        block[59] = 0;
        block[60] = (BYTE)(bitlen >> 24);
        block[61] = (BYTE)(bitlen >> 16);
        block[62] = (BYTE)(bitlen >> 8);
        block[63] = (BYTE)bitlen;

        SHA256_CTX ctx;
        ctx.state[0] = c_midstate[0];
        ctx.state[1] = c_midstate[1];
        ctx.state[2] = c_midstate[2];
        ctx.state[3] = c_midstate[3];
        ctx.state[4] = c_midstate[4];
        ctx.state[5] = c_midstate[5];
        ctx.state[6] = c_midstate[6];
        ctx.state[7] = c_midstate[7];
        sha256_trans(&ctx, block);

        bool valid = true;
#pragma unroll
        for (int i = 0; i < 32; i++)
        {
            BYTE hb = (BYTE)((ctx.state[i >> 2] >> (24 - (i & 3) * 8)) & 0xff);
            BYTE db = c_difficulty_bytes[i];
            if (hb < db)
                break;
            if (hb > db)
            {
                valid = false;
                break;
            }
        }

        if (valid)
        {
            atomicMin(valid_nonce, nonce);
            return;
        }

        if (UINT32_MAX - stride < nonce)
            break;
        nonce += stride;
    }
}

static inline BYTE hex_nibble(BYTE c)
{
    if (c >= '0' && c <= '9')
        return c - '0';
    if (c >= 'a' && c <= 'f')
        return c - 'a' + 10;
    if (c >= 'A' && c <= 'F')
        return c - 'A' + 10;
    return 0;
}

int find_nonce(BYTE *difficulty, uint32_t max_nonce,BYTE *block_content, size_t current_length, BYTE *block_hash, uint32_t *valid_nonce)
{
    static uint32_t *d_valid_nonce = NULL;
    if (!d_valid_nonce)
    {
        cudaMalloc((void **)&d_valid_nonce, sizeof(uint32_t));
    }

    SHA256_CTX midctx;
    sha256_init(&midctx);
    sha256_trans(&midctx, block_content);
    sha256_trans(&midctx, block_content + 64);
    cudaMemcpyToSymbol(c_midstate, midctx.state, 8 * sizeof(WORD));

    BYTE diff_bytes[32];
    for (int i = 0; i < 32; i++)
    {
        diff_bytes[i] = (BYTE)((hex_nibble(difficulty[2 * i]) << 4) |
                               hex_nibble(difficulty[2 * i + 1]));
    }
    cudaMemcpyToSymbol(c_difficulty_bytes, diff_bytes, 32);

    uint32_t init_val = UINT32_MAX;
    cudaMemcpy(d_valid_nonce, &init_val, sizeof(uint32_t), cudaMemcpyHostToDevice);

    const int TPB = 256;
    const int NBLOCKS = 2048;
    find_nonce_kernel<<<NBLOCKS, TPB>>>(max_nonce, d_valid_nonce);

    uint32_t found_nonce;
    cudaMemcpy(&found_nonce, d_valid_nonce, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    if (found_nonce == UINT32_MAX)
    {
        return 1;
    }

    *valid_nonce = found_nonce;

    char nonce_str[NONCE_SIZE];
    sprintf(nonce_str, "%u", found_nonce);
    strcpy((char *)block_content + current_length, nonce_str);
    apply_sha256(block_content, block_hash);

    return 0;
}

__global__ void dummy_kernel() {}

// Warm-up function
void warm_up_gpu()
{
    BYTE *dummy_data;
    cudaMalloc((void **)&dummy_data, 256);
    dummy_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    cudaFree(dummy_data);
}
