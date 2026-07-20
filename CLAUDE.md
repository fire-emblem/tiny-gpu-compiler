# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An MLIR-based GPU compiler that compiles `.tgc` kernels (C-like DSL) through a custom `tinygpu` MLIR dialect down to 16-bit binary instructions targeting the [tiny-gpu](https://github.com/adam-maj/tiny-gpu) open-source Verilog hardware. Includes a browser-based interactive visualizer with compilation pipeline viewer, GPU simulator, and performance analysis.

## Build Commands

### Docker (recommended, includes LLVM/MLIR pre-built)
```bash
docker build -t tgc .
docker run -v $(pwd)/examples:/workspace tgc --emit asm /workspace/vector_add.tgc
```

### Native Build (requires LLVM/MLIR 18)
```bash
cmake -G Ninja -S . -B build \
  -DMLIR_DIR=/path/to/llvm-install/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-install/lib/cmake/llvm
cmake --build build
```

### Run Tests
```bash
cmake --build build --target check-tgc   # LLVM lit tests
```

### Run Compiler
```bash
./build/bin/tgc --emit asm examples/vector_add.tgc    # annotated assembly
./build/bin/tgc --emit mlir examples/vector_add.tgc   # MLIR IR
./build/bin/tgc --emit hex examples/vector_add.tgc    # hex for Verilog testbench
./build/bin/tgc --emit trace examples/vector_add.tgc  # JSON trace for web UI
```

### Web Visualizer
```bash
cd web && npm install && npm run dev
```

## Architecture

### Compilation Pipeline (5 stages)

```
.tgc source → Lexer/Parser → MLIRGen → Optimizations → Register Alloc → Binary Emitter
```

Each stage's output is captured as a string in `CompilationTrace` for the web visualizer.

### C++ Backend (lib/)

- **Frontend** (`lib/Frontend/`): Lexer → Parser (recursive descent + precedence climbing) → AST → MLIRGen. The MLIRGen walks the AST and emits `tinygpu` dialect ops directly. All values are `i8` (8-bit unsigned). Pointer params map to fixed 64-byte memory regions (param 0 → addr 0, param 1 → addr 64, etc.). Scalar params start at addr 192.

- **Dialect** (`include/.../Dialect/TinyGPU/`): Custom MLIR dialect defined in TableGen ODS (`.td` files). 18 operations mapping 1:1 to hardware capabilities. The dialect and ops headers are generated from the `.td` files at build time.

- **Passes** (`lib/Passes/`): Four optimization passes run iteratively (3 iterations): constant folding, strength reduction, CSE (per-block), dead code elimination. Each pass modifies the MLIR module in-place.

- **Register Allocation** (`lib/CodeGen/RegisterAllocator.cpp`): Linear scan over 13 GPRs (R0-R12). R13/R14/R15 are reserved for blockIdx/blockDim/threadIdx. Annotates MLIR ops with `rd`, `rs`, `rt` integer attributes.

- **Binary Emission** (`lib/CodeGen/TinyGPUEmitter.cpp`): Two-pass — first computes block addresses, then encodes each MLIR op into a 16-bit word. Encoding: `opcode[15:12] | rd[11:8] | rs[7:4] | rt[3:0]`. Branch encoding uses `nzp[11:9] | target[7:0]`. Special register reads (thread_id, block_id, block_dim) emit no instructions — they reference hardware registers directly.

- **Pipeline** (`lib/Pipeline/Pipeline.cpp`): Orchestrates all stages. Entry point: `tgc::compile()`. The `tgc` CLI tool (`tools/tgc/tgc.cpp`) is a thin wrapper around this.

### Web Frontend (web/src/)

TypeScript/Vite app with a complete in-browser reimplementation of the compiler:

- `compiler/TGCCompiler.ts` — browser-side compiler (mirrors the C++ pipeline)
- `simulator/TinyGPUSim.ts` — cycle-accurate GPU simulator with shared memory, warp divergence tracking
- `components/` — Monaco editor, pipeline viewer, binary view, GPU simulator, analysis panel

### Key Design Constraints

- **All values are i8** (8-bit unsigned). The hardware data path is 8 bits wide.
- **Instruction width is 16 bits** — fixed encoding, 4 fields of 4 bits each.
- **16 registers total**: R0-R12 GPRs, R13=blockIdx, R14=blockDim, R15=threadIdx.
- **256-byte data memory**, **64-byte shared memory** per block.
- **SIMD lockstep** within blocks, sequential block dispatch.

## Test Structure

Tests use LLVM's `lit` framework (`test/`). Test files are `.tgc` or `.mlir`. The `%tgc` substitution expands to the compiler binary. Run with `ninja check-tgc` from the build directory.

## MetaX xcore1000 Backend (In Progress)

Reverse-engineered ISA via `mxcc -aop -S`. See `docs/xcore1000-isa-analysis.md` for the complete reference.

Key facts:
- **Target**: `mxc-metax-macahca`, CPU=`xcore1000`, base=ARCv2.3 64-bit
- **Warp size**: 64 threads, max block 512, max 8 warps/PEU
- **Compiler**: `mxcc` at `/opt/maca/mxgpu_llvm/bin/mxcc` (MACA SDK 3.7.0)
- **Get device assembly**: `mxcc -aop -S kernel.cu -o kernel.s` (extract between `__CLANG_OFFLOAD_BUNDLE____START__ maca-mxc` markers)
- **Get device object**: `mxcc --device-obj kernel.cu -o kernel_dev.o`
- **Get LLVM IR**: `llvm-objcopy --dump-section .llvmbc=out.bc kernel_dev.o && llvm-dis out.bc`
- **Register classes**: `r0`-`rN` (VGPR/vector), `s0`-`sN` (SGPR/scalar), `xmsk` (exec mask), `cmsk` (compare mask)
- **Key instructions**: `ldg_b32` (global load), `stg_b32 ... devc` (global store), `lds_b32`/`sts_b32` (shared), `barrier`, `endk`, `sm_bperm_b32` (warp shuffle)
- **Division**: 3-step: `div_scale` → `div_fmas` → `div_fixup` (no single FDIV)
- **Dependency control**: `arrive slcnt(N)` / `arrive gvmcnt(N)` / `arrive bsmcnt(N)`
- **Shared memory**: addrspace(3), `fence syncscope("block") release/acquire`
