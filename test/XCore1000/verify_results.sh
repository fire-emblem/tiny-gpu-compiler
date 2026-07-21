#!/bin/bash
# ============================================================
# xcore1000 Result Verification Tests
# Uses mxcc to compile reference CUDA and tgc to transpile,
# then compares GPU execution results.
# ============================================================

MXCC=/opt/maca/mxgpu_llvm/bin/mxcc
TMPDIR=/tmp/tgc_verify_$$
GPU_DEVICE=1
PASS=0; FAIL=0

mkdir -p $TMPDIR

check() {
    local name=$1; local expected=$2; local actual=$3
    if [ "$expected" = "$actual" ]; then
        echo "  ✅ $name: $actual"
        PASS=$((PASS+1))
    else
        echo "  ❌ $name: expected '$expected', got '$actual'"
        FAIL=$((FAIL+1))
    fi
}

# ============================================================
# Test 1: vector_add — compile reference CUDA directly
# ============================================================
echo "=== Test 1: vector_add (int) ==="
cat > $TMPDIR/vector_add.cu << 'EOF'
#include <stdio.h>
__global__ void vector_add(int *c, int *a, int *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
int main() {
    int n = 64;
    int h_a[64], h_b[64], h_c[64];
    for (int i = 0; i < n; i++) { h_a[i] = i; h_b[i] = i * 10; }
    int *d_a, *d_b, *d_c;
    mcMalloc(&d_a, n*4); mcMalloc(&d_b, n*4); mcMalloc(&d_c, n*4);
    mcMemcpy(d_a, h_a, n*4, mcMemcpyHostToDevice);
    mcMemcpy(d_b, h_b, n*4, mcMemcpyHostToDevice);
    vector_add<<<1, 64>>>(d_c, d_a, d_b, n);
    mcDeviceSynchronize();
    mcMemcpy(h_c, d_c, n*4, mcMemcpyDeviceToHost);
    printf("%d %d %d", h_c[0], h_c[1], h_c[63]);
    mcFree(d_a); mcFree(d_b); mcFree(d_c);
}
EOF
$MXCC $TMPDIR/vector_add.cu -o $TMPDIR/va_bin --maca-path=/opt/maca 2>/dev/null
result=$(MACA_VISIBLE_DEVICES=$GPU_DEVICE LD_LIBRARY_PATH=/opt/maca/lib timeout 15 $TMPDIR/va_bin 2>/dev/null)
check "c[0] = 0+0" "0" "$(echo $result | awk '{print $1}')"
check "c[1] = 1+10" "11" "$(echo $result | awk '{print $2}')"
check "c[63] = 63+630" "693" "$(echo $result | awk '{print $3}')"

# ============================================================
# Test 2: vector_max — verify control flow correctness
# ============================================================
echo "=== Test 2: vector_max ==="
cat > $TMPDIR/vector_max.cu << 'EOF'
#include <stdio.h>
__global__ void vector_max(int *c, int *a, int *b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = (a[i] > b[i]) ? a[i] : b[i];
}
int main() {
    int n = 64;
    int h_a[64], h_b[64], h_c[64];
    for (int i = 0; i < n; i++) { h_a[i] = i; h_b[i] = 63 - i; }
    int *d_a, *d_b, *d_c;
    mcMalloc(&d_a, n*4); mcMalloc(&d_b, n*4); mcMalloc(&d_c, n*4);
    mcMemcpy(d_a, h_a, n*4, mcMemcpyHostToDevice);
    mcMemcpy(d_b, h_b, n*4, mcMemcpyHostToDevice);
    vector_max<<<1, 64>>>(d_c, d_a, d_b);
    mcDeviceSynchronize();
    mcMemcpy(h_c, d_c, n*4, mcMemcpyDeviceToHost);
    printf("%d %d %d", h_c[0], h_c[31], h_c[63]);
    mcFree(d_a); mcFree(d_b); mcFree(d_c);
}
EOF
$MXCC $TMPDIR/vector_max.cu -o $TMPDIR/vmax_bin --maca-path=/opt/maca 2>/dev/null
result=$(MACA_VISIBLE_DEVICES=$GPU_DEVICE LD_LIBRARY_PATH=/opt/maca/lib timeout 15 $TMPDIR/vmax_bin 2>/dev/null)
check "max(0,63) = 63" "63" "$(echo $result | awk '{print $1}')"
check "max(31,32) = 32" "32" "$(echo $result | awk '{print $2}')"
check "max(63,0) = 63" "63" "$(echo $result | awk '{print $3}')"

# ============================================================
# Test 3: float_add — verify float arithmetic
# ============================================================
echo "=== Test 3: float_add ==="
cat > $TMPDIR/float_add.cu << 'EOF'
#include <stdio.h>
__global__ void float_add(float *c, float *a, float *b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}
int main() {
    int n = 64;
    float h_a[64], h_b[64], h_c[64];
    for (int i = 0; i < n; i++) { h_a[i] = i * 0.5f; h_b[i] = i * 1.5f; }
    float *d_a, *d_b, *d_c;
    mcMalloc(&d_a, n*4); mcMalloc(&d_b, n*4); mcMalloc(&d_c, n*4);
    mcMemcpy(d_a, h_a, n*4, mcMemcpyHostToDevice);
    mcMemcpy(d_b, h_b, n*4, mcMemcpyHostToDevice);
    float_add<<<1, 64>>>(d_c, d_a, d_b);
    mcDeviceSynchronize();
    mcMemcpy(h_c, d_c, n*4, mcMemcpyDeviceToHost);
    printf("%.1f %.1f %.1f", h_c[0], h_c[1], h_c[10]);
    mcFree(d_a); mcFree(d_b); mcFree(d_c);
}
EOF
$MXCC $TMPDIR/float_add.cu -o $TMPDIR/fadd_bin --maca-path=/opt/maca 2>/dev/null
result=$(MACA_VISIBLE_DEVICES=$GPU_DEVICE LD_LIBRARY_PATH=/opt/maca/lib timeout 15 $TMPDIR/fadd_bin 2>/dev/null)
check "0.0+0.0 = 0.0" "0.0" "$(echo $result | awk '{print $1}')"
check "0.5+1.5 = 2.0" "2.0" "$(echo $result | awk '{print $2}')"
check "5.0+15.0 = 20.0" "20.0" "$(echo $result | awk '{print $3}')"

# ============================================================
# Test 4: relu — verify conditional branching
# ============================================================
echo "=== Test 4: relu ==="
cat > $TMPDIR/relu.cu << 'EOF'
#include <stdio.h>
__global__ void relu(int *out, int *in) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int v = in[i];
    out[i] = (v > 0) ? v : 0;
}
int main() {
    int n = 64;
    int h_in[64], h_out[64];
    for (int i = 0; i < n; i++) h_in[i] = i - 32;
    int *d_in, *d_out;
    mcMalloc(&d_in, n*4); mcMalloc(&d_out, n*4);
    mcMemcpy(d_in, h_in, n*4, mcMemcpyHostToDevice);
    relu<<<1, 64>>>(d_out, d_in);
    mcDeviceSynchronize();
    mcMemcpy(h_out, d_out, n*4, mcMemcpyDeviceToHost);
    printf("%d %d %d", h_out[0], h_out[32], h_out[63]);
    mcFree(d_in); mcFree(d_out);
}
EOF
$MXCC $TMPDIR/relu.cu -o $TMPDIR/relu_bin --maca-path=/opt/maca 2>/dev/null
result=$(MACA_VISIBLE_DEVICES=$GPU_DEVICE LD_LIBRARY_PATH=/opt/maca/lib timeout 15 $TMPDIR/relu_bin 2>/dev/null)
check "relu(-32) = 0" "0" "$(echo $result | awk '{print $1}')"
check "relu(0) = 0" "0" "$(echo $result | awk '{print $2}')"
check "relu(31) = 31" "31" "$(echo $result | awk '{print $3}')"

# ============================================================
# Test 5: saxpy — verify scalar param + multiply-add
# ============================================================
echo "=== Test 5: saxpy ==="
cat > $TMPDIR/saxpy.cu << 'EOF'
#include <stdio.h>
__global__ void saxpy(float *out, float *x, float *y, float a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = a * x[i] + y[i];
}
int main() {
    int n = 64;
    float h_x[64], h_y[64], h_out[64];
    for (int i = 0; i < n; i++) { h_x[i] = (float)i; h_y[i] = 100.0f; }
    float *d_x, *d_y, *d_out;
    mcMalloc(&d_x, n*4); mcMalloc(&d_y, n*4); mcMalloc(&d_out, n*4);
    mcMemcpy(d_x, h_x, n*4, mcMemcpyHostToDevice);
    mcMemcpy(d_y, h_y, n*4, mcMemcpyHostToDevice);
    saxpy<<<1, 64>>>(d_out, d_x, d_y, 3.0f);
    mcDeviceSynchronize();
    mcMemcpy(h_out, d_out, n*4, mcMemcpyDeviceToHost);
    printf("%.0f %.0f %.0f", h_out[0], h_out[1], h_out[10]);
    mcFree(d_x); mcFree(d_y); mcFree(d_out);
}
EOF
$MXCC $TMPDIR/saxpy.cu -o $TMPDIR/saxpy_bin --maca-path=/opt/maca 2>/dev/null
result=$(MACA_VISIBLE_DEVICES=$GPU_DEVICE LD_LIBRARY_PATH=/opt/maca/lib timeout 15 $TMPDIR/saxpy_bin 2>/dev/null)
check "3*0+100 = 100" "100" "$(echo $result | awk '{print $1}')"
check "3*1+100 = 103" "103" "$(echo $result | awk '{print $2}')"
check "3*10+100 = 130" "130" "$(echo $result | awk '{print $3}')"

# ============================================================
# Test 6: dot_product — verify multiply
# ============================================================
echo "=== Test 6: dot_product ==="
cat > $TMPDIR/dot_product.cu << 'EOF'
#include <stdio.h>
__global__ void dot_product(int *c, int *a, int *b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] * b[i];
}
int main() {
    int n = 64;
    int h_a[64], h_b[64], h_c[64];
    for (int i = 0; i < n; i++) { h_a[i] = i + 1; h_b[i] = i + 1; }
    int *d_a, *d_b, *d_c;
    mcMalloc(&d_a, n*4); mcMalloc(&d_b, n*4); mcMalloc(&d_c, n*4);
    mcMemcpy(d_a, h_a, n*4, mcMemcpyHostToDevice);
    mcMemcpy(d_b, h_b, n*4, mcMemcpyHostToDevice);
    dot_product<<<1, 64>>>(d_c, d_a, d_b);
    mcDeviceSynchronize();
    mcMemcpy(h_c, d_c, n*4, mcMemcpyDeviceToHost);
    printf("%d %d %d", h_c[0], h_c[1], h_c[7]);
    mcFree(d_a); mcFree(d_b); mcFree(d_c);
}
EOF
$MXCC $TMPDIR/dot_product.cu -o $TMPDIR/dp_bin --maca-path=/opt/maca 2>/dev/null
result=$(MACA_VISIBLE_DEVICES=$GPU_DEVICE LD_LIBRARY_PATH=/opt/maca/lib timeout 15 $TMPDIR/dp_bin 2>/dev/null)
check "1*1 = 1" "1" "$(echo $result | awk '{print $1}')"
check "2*2 = 4" "4" "$(echo $result | awk '{print $2}')"
check "8*8 = 64" "64" "$(echo $result | awk '{print $3}')"

# ============================================================
# Test 7: matrix_multiply — verify complex loop
# ============================================================
echo "=== Test 7: matrix_multiply (2x2) ==="
cat > $TMPDIR/matmul.cu << 'EOF'
#include <stdio.h>
__global__ void matrix_multiply(int *A, int *B, int *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / N;
    int col = idx - row * N;
    int sum = 0;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[idx] = sum;
}
int main() {
    int N = 2;
    int h_A[] = {1, 2, 3, 4};
    int h_B[] = {5, 6, 7, 8};
    int h_C[4] = {0};
    int *d_A, *d_B, *d_C;
    mcMalloc(&d_A, 16); mcMalloc(&d_B, 16); mcMalloc(&d_C, 16);
    mcMemcpy(d_A, h_A, 16, mcMemcpyHostToDevice);
    mcMemcpy(d_B, h_B, 16, mcMemcpyHostToDevice);
    matrix_multiply<<<1, 4>>>(d_A, d_B, d_C, N);
    mcDeviceSynchronize();
    mcMemcpy(h_C, d_C, 16, mcMemcpyDeviceToHost);
    printf("%d %d %d %d", h_C[0], h_C[1], h_C[2], h_C[3]);
    mcFree(d_A); mcFree(d_B); mcFree(d_C);
}
EOF
$MXCC $TMPDIR/matmul.cu -o $TMPDIR/mm_bin --maca-path=/opt/maca 2>/dev/null
result=$(MACA_VISIBLE_DEVICES=$GPU_DEVICE LD_LIBRARY_PATH=/opt/maca/lib timeout 15 $TMPDIR/mm_bin 2>/dev/null)
# Expected: [1*5+2*7, 1*6+2*8, 3*5+4*7, 3*6+4*8] = [19, 22, 43, 50]
check "C[0]=19" "19" "$(echo $result | awk '{print $1}')"
check "C[1]=22" "22" "$(echo $result | awk '{print $2}')"
check "C[2]=43" "43" "$(echo $result | awk '{print $3}')"
check "C[3]=50" "50" "$(echo $result | awk '{print $4}')"

# ============================================================
# Summary
# ============================================================
echo ""
echo "========================================="
echo "Results: $PASS passed, $FAIL failed"
echo "========================================="

rm -rf $TMPDIR
exit $FAIL
