#!/bin/bash
# ============================================================
# tgc 编译运行脚本 — MetaX xcore1000 GPU
# ============================================================
# 用法:
#   bash scripts/build_and_run.sh              # 构建 tgc
#   bash scripts/build_and_run.sh compile      # 编译 kernel
#   bash scripts/build_and_run.sh run          # 在 GPU 上运行
#   bash scripts/build_and_run.sh all          # 完整流程
# ============================================================

set -e

# ---- 配置 ----
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
TGC="$BUILD_DIR/bin/tgc"
MXCC=/opt/maca/mxgpu_llvm/bin/mxcc
EXAMPLES_DIR="$PROJECT_DIR/examples"
TMPDIR=/tmp/tgc_gpu_run
GPU_DEVICE=1  # GPU 0 被 pytest 占用，使用 GPU 1

# ---- 颜色 ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ============================================================
# 1. 构建 tgc 编译器
# ============================================================
build_tgc() {
    info "构建 tgc 编译器..."

    # 检查 MLIR 18
    if [ ! -d "/tmp/llvm-18-build/lib/cmake/mlir" ]; then
        error "MLIR 18 未找到。请先构建 LLVM/MLIR 18:"
        echo "  git clone --depth 1 --branch llvmorg-18.1.8 https://github.com/llvm/llvm-project.git /tmp/llvm-18"
        echo "  cmake -G Ninja -S /tmp/llvm-18/llvm -B /tmp/llvm-18-build \\"
        echo "    -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \\"
        echo "    -DCMAKE_ASM_COMPILER=gcc -DLLVM_ENABLE_PROJECTS=mlir \\"
        echo "    -DLLVM_TARGETS_TO_BUILD=host -DCMAKE_INSTALL_PREFIX=/opt/llvm-18"
        echo "  ninja -C /tmp/llvm-18-build -j\$(nproc) mlir-libraries mlir-headers"
        return 1
    fi

    # 配置
    cmake -G Ninja -S "$PROJECT_DIR" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=/usr/bin/gcc \
        -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
        -DMLIR_DIR=/tmp/llvm-18-build/lib/cmake/mlir \
        -DLLVM_DIR=/tmp/llvm-18-build/lib/cmake/llvm \
        2>&1 | tail -3

    # 构建
    ninja -C "$BUILD_DIR" -j$(nproc) 2>&1 | tail -5

    if [ -f "$TGC" ]; then
        info "tgc 构建成功: $TGC"
    else
        error "tgc 构建失败"
        return 1
    fi
}

# ============================================================
# 2. 编译 .tgc kernel
# ============================================================
compile_kernel() {
    local kernel_file="${1:-$EXAMPLES_DIR/vector_add.tgc}"
    local target="${2:-xcore1000}"
    local format="${3:-asm}"

    mkdir -p "$TMPDIR"

    info "编译 kernel: $kernel_file"
    info "目标: $target, 格式: $format"

    case "$format" in
        asm)
            local output="$TMPDIR/$(basename "$kernel_file" .tgc)_${target}.s"
            $TGC --target "$target" --emit asm "$kernel_file" > "$output" 2>&1
            info "汇编输出: $output"
            cat "$output"
            ;;
        bc)
            local output="$TMPDIR/$(basename "$kernel_file" .tgc).bc"
            $TGC --target "$target" --emit bc "$kernel_file" > "$output" 2>&1
            info "Bitcode 输出: $output"
            file "$output"
            ;;
        trace)
            local output="$TMPDIR/$(basename "$kernel_file" .tgc).json"
            $TGC --target "$target" --emit trace "$kernel_file" > "$output" 2>&1
            info "JSON trace: $output"
            head -20 "$output"
            ;;
        mlir)
            $TGC --target "$target" --emit mlir "$kernel_file" 2>&1
            ;;
    esac
}

# ============================================================
# 3. 用 mxcc 编译 CUDA kernel 并在 GPU 上运行
# ============================================================
run_on_gpu() {
    mkdir -p "$TMPDIR"

    # 写一个简单的 CUDA kernel
    cat > "$TMPDIR/kernel.cu" << 'KEOF'
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

    info "编译 CUDA kernel..."
    $MXCC "$TMPDIR/kernel.cu" -o "$TMPDIR/kernel_bin" --maca-path=/opt/maca 2>/dev/null
    info "二进制: $(file "$TMPDIR/kernel_bin")"

    info "在 MetaX C500 GPU $GPU_DEVICE 上运行..."
    MACA_VISIBLE_DEVICES=$GPU_DEVICE LD_LIBRARY_PATH=/opt/maca/lib "$TMPDIR/kernel_bin"
}

# ============================================================
# 4. 列出可用 example kernels
# ============================================================
list_examples() {
    info "可用的 example kernels:"
    for f in "$EXAMPLES_DIR"/*.tgc; do
        echo "  $(basename "$f")"
    done
}

# ============================================================
# 5. 对比 tgc 和 mxcc 输出
# ============================================================
compare_output() {
    mkdir -p "$TMPDIR"

    info "对比 tgc xcore1000 汇编 vs mxcc 输出..."

    # tgc 输出
    $TGC --target xcore1000 --emit asm "$EXAMPLES_DIR/vector_add.tgc" > "$TMPDIR/tgc_asm.s" 2>&1

    # mxcc 输出
    cat > "$TMPDIR/compare.cu" << 'KEOF'
__global__ void vector_add(float *c, const float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
KEOF
    $MXCC -aop -S "$TMPDIR/compare.cu" -o "$TMPDIR/mxcc_asm.s" 2>/dev/null

    echo ""
    echo "=== tgc xcore1000 输出 ==="
    grep -E "^\t[a-z]" "$TMPDIR/tgc_asm.s" | head -15
    echo ""
    echo "=== mxcc 参考输出 ==="
    sed -n '/__CLANG_OFFLOAD_BUNDLE____START__ maca-mxc/,/__CLANG_OFFLOAD_BUNDLE____END__ maca-mxc/p' "$TMPDIR/mxcc_asm.s" | grep -E "^\t[a-z]" | head -15
}

# ============================================================
# 主入口
# ============================================================
case "${1:-help}" in
    build)
        build_tgc
        ;;
    compile)
        compile_kernel "$2" "${3:-xcore1000}" "${4:-asm}"
        ;;
    run)
        run_on_gpu
        ;;
    compare)
        compare_output
        ;;
    list)
        list_examples
        ;;
    all)
        build_tgc
        echo ""
        compile_kernel "$EXAMPLES_DIR/vector_add.tgc" xcore1000 asm
        echo ""
        run_on_gpu
        ;;
    help|*)
        echo "用法: $0 <command> [args]"
        echo ""
        echo "命令:"
        echo "  build                    构建 tgc 编译器"
        echo "  compile <file> [tgt] [fmt]  编译 kernel (tgt=xcore1000|tinygpu, fmt=asm|bc|trace|mlir)"
        echo "  run                      用 mxcc 编译并在 GPU 上运行"
        echo "  compare                  对比 tgc 和 mxcc 输出"
        echo "  list                     列出可用的 example kernels"
        echo "  all                      完整流程: 构建 + 编译 + 运行"
        echo ""
        echo "示例:"
        echo "  $0 build"
        echo "  $0 compile examples/vector_add.tgc xcore1000 asm"
        echo "  $0 compile examples/vector_add.tgc tinygpu asm"
        echo "  $0 run"
        echo "  $0 all"
        ;;
esac
