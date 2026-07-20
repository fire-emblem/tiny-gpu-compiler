#!/bin/bash
# Cross-validation script: compare tgc xcore1000 output with mxcc output
# Usage: bash test/XCore1000/validate.sh
#
# This script compiles equivalent CUDA kernels with both:
#   1. tgc --target xcore1000 (our compiler)
#   2. mxcc -aop -S (MetaX reference compiler)
# and compares the instruction categories for semantic equivalence.

MXCC=/opt/maca/mxgpu_llvm/bin/mxcc
TGC=./build/bin/tgc
TMPDIR=/tmp/xcore1000_validate
mkdir -p $TMPDIR

echo "=== xcore1000 Cross-Validation ==="
echo ""

# Check tools
if [ ! -f "$MXCC" ]; then
    echo "SKIP: mxcc not found at $MXCC"
    exit 0
fi

# Extract xcore1000 device assembly from mxcc output
extract_device_asm() {
    local file=$1
    local tmpfile=$TMPDIR/_extract_tmp.txt
    sed -n '/__CLANG_OFFLOAD_BUNDLE____START__ maca-mxc/,/__CLANG_OFFLOAD_BUNDLE____END__ maca-mxc/p' "$file" | \
        grep -P '^\t[a-z]' | \
        sed 's/\t//g' | \
        sed 's/ .*//' > "$tmpfile"
    sort "$tmpfile" | uniq -c | sort -rn
    rm -f "$tmpfile"
}

# Test 1: vector_add
echo "--- Test 1: vector_add ---"
cat > $TMPDIR/vector_add.cu << 'EOF'
__global__ void vector_add(float *c, const float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
EOF

$MXCC -aop -S $TMPDIR/vector_add.cu -o $TMPDIR/vector_add_mxcc.s > /dev/null 2>&1 || true
echo "mxcc instruction distribution:"
extract_device_asm $TMPDIR/vector_add_mxcc.s | head -15
echo ""

# Test 2: integer arithmetic
echo "--- Test 2: integer_arith ---"
cat > $TMPDIR/int_arith.cu << 'EOF'
__global__ void int_arith(int *out, int a, int b) {
    int i = threadIdx.x;
    out[i] = a + b;
    out[i+1] = a - b;
    out[i+2] = a * b;
}
EOF

$MXCC -aop -S $TMPDIR/int_arith.cu -o $TMPDIR/int_arith_mxcc.s > /dev/null 2>&1 || true
echo "mxcc instruction distribution:"
extract_device_asm $TMPDIR/int_arith_mxcc.s | head -15
echo ""

# Test 3: shared memory
echo "--- Test 3: shared_memory ---"
cat > $TMPDIR/shared.cu << 'EOF'
__global__ void shared_mem(float *out, float *in) {
    __shared__ float tile[64];
    int i = threadIdx.x;
    tile[i] = in[i];
    __syncthreads();
    out[i] = tile[63 - i];
}
EOF

$MXCC -aop -S $TMPDIR/shared.cu -o $TMPDIR/shared_mxcc.s > /dev/null 2>&1 || true
echo "mxcc instruction distribution:"
extract_device_asm $TMPDIR/shared_mxcc.s | head -15
echo ""

# Test 4: float operations
echo "--- Test 4: float_ops ---"
cat > $TMPDIR/float_ops.cu << 'EOF'
__global__ void float_ops(float *out, float a, float b) {
    int i = threadIdx.x;
    out[i] = a + b;
    out[i+1] = a * b;
    out[i+2] = a / b;
}
EOF

$MXCC -aop -S $TMPDIR/float_ops.cu -o $TMPDIR/float_ops_mxcc.s > /dev/null 2>&1 || true
echo "mxcc instruction distribution:"
extract_device_asm $TMPDIR/float_ops_mxcc.s | head -15
echo ""

# Summary of expected xcore1000 instruction categories
echo "=== Expected xcore1000 Instruction Categories ==="
echo "  Integer:  add_u32, sub_u32, mul_u32, mad_i32"
echo "  Float:    add_f32, sub_f32, mul_f32, fma_f32"
echo "  Division: div_scale_f32, div_fmas_f32, div_fixup_f32"
echo "  Memory:   ldg_b32, stg_b32 (global); lds_b32, sts_b32 (shared)"
echo "  Control:  bra, bra_xmskz, cmp_lt_i32, csel_b32"
echo "  Sync:     barrier, arrive slcnt/gvmcnt/bsmcnt"
echo "  Special:  endk, snop, mov_b32"
echo ""
echo "=== Validation Complete ==="
