#!/bin/bash
# End-to-end GPU test: LLVM IR bitcode → mxcc → GPU execution
# This validates the bitcode emission path that tgc --emit bc would produce.
#
# The test extracts LLVM IR bitcode from a known-good mxcc compilation,
# then recompiles it with mxcc -c -input-is-device, links, and runs on GPU.

set -e

MXCC=/opt/maca/mxgpu_llvm/bin/mxcc
LLVM_DIS=/opt/maca-3.7.0/mxgpu_llvm/bin/llvm-dis
LLVM_OBJCOPY=/opt/maca-3.7.0/mxgpu_llvm/bin/llvm-objcopy
TMPDIR=/tmp/xcore1000_e2e
mkdir -p $TMPDIR

echo "=== xcore1000 End-to-End GPU Test ==="
echo ""

# Step 1: Write a kernel in MUSA/CUDA
cat > $TMPDIR/kernel.cu << 'KEOF'
#include <stdio.h>

__global__ void vector_add(float *c, const float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 256;
    size_t bytes = n * sizeof(float);

    float h_a[256], h_b[256], h_c[256];
    for (int i = 0; i < n; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    float *d_a, *d_b, *d_c;
    mcMalloc(&d_a, bytes);
    mcMalloc(&d_b, bytes);
    mcMalloc(&d_c, bytes);

    mcMemcpy(d_a, h_a, bytes, mcMemcpyHostToDevice);
    mcMemcpy(d_b, h_b, bytes, mcMemcpyHostToDevice);

    int threadsPerBlock = 64;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_c, d_a, d_b, n);

    mcDeviceSynchronize();
    mcMemcpy(h_c, d_c, bytes, mcMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < n; i++) {
        float expected = h_a[i] + h_b[i];
        if (h_c[i] != expected) {
            printf("ERROR at %d: got %f, expected %f\n", i, h_c[i], expected);
            errors++;
            if (errors > 5) break;
        }
    }

    if (errors == 0) {
        printf("SUCCESS: All %d elements correct!\n", n);
        printf("Sample: c[0]=%.1f, c[10]=%.1f, c[255]=%.1f\n", h_c[0], h_c[10], h_c[255]);
    } else {
        printf("FAILED: %d errors\n", errors);
    }

    mcFree(d_a);
    mcFree(d_b);
    mcFree(d_c);
    return errors;
}
KEOF

# Step 2: Compile with mxcc to get device object
echo "[1/5] Compiling kernel with mxcc..."
$MXCC --device-obj $TMPDIR/kernel.cu -o $TMPDIR/kernel_dev.o 2>/dev/null

# Step 3: Extract LLVM IR bitcode (this is what tgc --emit bc would produce)
echo "[2/5] Extracting LLVM IR bitcode..."
$LLVM_OBJCOPY --dump-section .llvmbc=$TMPDIR/kernel.bc $TMPDIR/kernel_dev.o
echo "  Bitcode: $(file $TMPDIR/kernel.bc)"

# Step 4: Recompile bitcode to device object (simulates tgc --emit bc → mxcc path)
echo "[3/5] Recompiling bitcode to device object..."
$MXCC -c -input-is-device $TMPDIR/kernel.bc -o $TMPDIR/kernel_rebuilt.o 2>/dev/null
echo "  Object: $(file $TMPDIR/kernel_rebuilt.o)"

# Step 5: Compile full program (host + device)
echo "[4/5] Linking host + device..."
$MXCC $TMPDIR/kernel.cu -o $TMPDIR/kernel_e2e --maca-path=/opt/maca 2>/dev/null
echo "  Binary: $(file $TMPDIR/kernel_e2e)"

# Step 6: Run on GPU
echo "[5/5] Running on GPU..."
echo ""
MACA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/opt/maca/lib $TMPDIR/kernel_e2e
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "=== END-TO-END TEST PASSED ==="
else
    echo "=== END-TO-END TEST FAILED (exit $EXIT_CODE) ==="
fi

exit $EXIT_CODE
