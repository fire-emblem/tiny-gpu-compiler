# MetaX xcore1000 ISA Reference (Reverse-Engineered via `mxcc -aop -S`)

> **Method**: Compiled known CUDA kernels with `mxcc -aop -S`, analyzed device-side GPU assembly.
> **Toolchain**: MACA SDK 3.7.0, `mxcc` v1.0.0 at `/opt/maca/mxgpu_llvm/bin/mxcc`
> **Command**: `mxcc -aop -S kernel.cu -o kernel.s` (extract device section between `__CLANG_OFFLOAD_BUNDLE____START__ maca-mxc-metax-macahca--xcore1000` and `__END__`)

---

## 1. Architecture Overview

| Property | Value |
|----------|-------|
| **Target** | MetaX xcore1000 (MACA = MetaX Architecture for Compute Acceleration) |
| **Target triple** | `mxc-metax-macahca` |
| **Base ISA** | Synopsys ARCv2.3 64-bit (heavily extended for GPU) |
| **Warp size** | **64 threads** |
| **Max block size** | 512 threads |
| **Max warps per PEU** | 8 |
| **MT registers** | 1–3 (per-warp temp regs, allocated dynamically) |
| **ST registers** | 1–2 (scalar regs, shared across warp) |
| **User ST regs** | 4 (for kernel args etc.) |
| **Float mode** | 240 (IEEE754, no flush-to-zero) |
| **BSM (shared mem)** | Per-block, size in `.macahca_bsm_size` directive |

### Register Classes

| Prefix | Type | Description |
|--------|------|-------------|
| `r0`–`rN` | Vector (VGPR) | Per-thread registers, used for computation |
| `s0`–`sN` | Scalar (SGPR) | Per-wavefront registers, used for addresses/constants |
| `cmsk` | Compare mask | Implicit result of `cmp_lg_u32` |
| `xmsk` | Execution mask | Active thread mask for divergence |

### Special Registers

| Register | Description |
|----------|-------------|
| `xmsk` | Execution mask — bits set for active lanes |
| `cmsk` | Compare mask — result of scalar comparisons |

---

## 2. Instruction Categories

### 2.1 Kernel Lifecycle

| Instruction | Syntax | Description |
|-------------|--------|-------------|
| `arrive 0` | `arrive 0` | **Kernel entry** — marks start of execution |
| `endk` | `endk` | **Kernel end** — signals thread completion |
| `trap` | `trap N` | Trap/abort with code N |
| `snop` | `snop N` | Scalar NOP with dependency hint (0–3) |

### 2.2 Integer Arithmetic (32-bit)

| Instruction | Syntax | Description |
|-------------|--------|-------------|
| `add_u32` | `add_u32 rd, rs, rt/imm` | 32-bit unsigned add |
| `add_co_u32` | `add_co_u32 rd, rs, rt` | Add with carry-out |
| `addc_co_u32` | `addc_co_u32 rd, rs, rt/imm` | Add with carry-in and carry-out |
| `sub_u32` | `sub_u32 rd, rs, rt/imm` | 32-bit unsigned subtract |
| `mul_u32` | `mul_u32 rd, rs, rt` | 32-bit unsigned multiply (low 32 bits) |
| `mul_hi_u32` | `mul_hi_u32 rd, rs, rt` | 32-bit unsigned multiply (high 32 bits) |
| `mul_i32` | `mul_i32 rd, rs, rt` | 32-bit signed multiply |
| `smul_i32` | `smul_i32 sd, ss, st` | Scalar signed multiply |
| `mad_i32` | `mad_i32 rd, rs, rt, ru` | Multiply-add: `rd = rs * rt + ru` |
| `smax_i32` | `smax_i32 sd, ss, st` | Scalar signed maximum |
| `sadd_co_i32` | `sadd_co_i32 sd, ss, st/imm` | Scalar add with carry-out (signed) |
| `ssub_co_i32` | `ssub_co_i32 sd, ss, st/imm` | Scalar subtract with carry-out (signed) |
| `sadd_co_u32` | `sadd_co_u32 sd, ss, st/imm` | Scalar add with carry-out (unsigned) |
| `ssub_co_u32` | `ssub_co_u32 sd, ss, st/imm` | Scalar subtract with carry-out (unsigned) |
| `saddc_co_u32` | `saddc_co_u32 sd, ss, st/imm` | Scalar add-with-carry |
| `ssubb_co_u32` | `ssubb_co_u32 sd, ss, st/imm` | Scalar subtract-with-borrow |

### 2.3 Bitwise Operations (32-bit)

| Instruction | Syntax | Description |
|-------------|--------|-------------|
| `and_b32` | `and_b32 rd, imm/rs, rt` | Bitwise AND |
| `sand_b32` | `sand_b32 sd, imm/ss, st` | Scalar bitwise AND |
| `sor_b32` | `sor_b32 sd, ss, st` | Scalar bitwise OR |
| `xor_b32` | `xor_b32 rd, rs, rt/imm` | Bitwise XOR |
| `sxor_b32` | `sxor_b32 sd, imm/ss, st` | Scalar bitwise XOR |
| `shl_b32` | `shl_b32 rd, imm/rs, rt` | Shift left (32-bit) |
| `sshl_b32` | `sshl_b32 sd, imm, ss` | Scalar shift left |
| `sshr_b32` | `sshr_b32 sd, imm, ss` | Scalar shift right (logical) |
| `sashr_i32` | `sashr_i32 sd, imm, ss` | Scalar arithmetic shift right |
| `sclz_b32` | `sclz_b32 sd, ss` | Scalar count leading zeros |
| `sbcs_b32` | `sbcs_b32 sd, ss` | Scalar bit count (popcount) |
| `perm_b32` | `perm_b32 rd, rs, rt, ru` | Byte permute |
| `and_xmsk` | `and_xmsk sd, ss` | AND with execution mask |

### 2.4 64-bit Integer Operations

| Instruction | Syntax | Description |
|-------------|--------|-------------|
| `shl_b64` | `shl_b64 rd:rd+1, imm, rs:rs+1` | 64-bit shift left |
| `sshl_b64` | `sshl_b64 sd:sd+1, imm, ss:ss+1` | Scalar 64-bit shift left |
| `cmp_eq_u64` | `cmp_eq_u64 sd, rs:rs+1, rt:rt+1` | 64-bit unsigned equality compare |
| `scmp_eq_u64` | `scmp_eq_u64 sd, ss:ss+1` | Scalar 64-bit equality compare |
| `scmp_ne_u64` | `scmp_ne_u64 sd, ss:ss+1` | Scalar 64-bit not-equal compare |
| `smov_b64` | `smov_b64 sd:sd+1, imm` | Scalar 64-bit move immediate |

### 2.5 Float Arithmetic (32-bit IEEE754)

| Instruction | Syntax | Description |
|-------------|--------|-------------|
| `add_f32` | `add_f32 rd, rs, rt` | FP32 add |
| `sub_f32` | `sub_f32 rd, rs, rt` | FP32 subtract |
| `mul_f32` | `mul_f32 rd, rs/imm, rt` | FP32 multiply |
| `fma_f32` | `fma_f32 rd, rs, rt, ru` | Fused multiply-add: `rd = rs*rt + ru` |
| `fmac_f32` | `fmac_f32 rd, rs, rt` | Fused multiply-accumulate: `rd += rs*rt` |
| `neg()` | `fma_f32 rd, neg(rs), rt, ru` | Negate modifier on source operand |
| `rcp_f32` | `rcp_f32 rd, rs` | Reciprocal approximation: `rd = 1/rs` |
| `rcpi_f32` | `rcpi_f32 rd, rs` | Reciprocal (integer result?) |
| `max_f32` | `max_f32 rd, rs, rt` | FP32 maximum |
| `min_f32` | `min_f32 rd, rs, rt` | FP32 minimum |

### 2.6 Float Division (Multi-Step)

Division is NOT a single instruction. It uses a 3-step sequence:
```
div_scale_f32 rd, rs, rt, ru    ; Scale step: compute initial approximation
div_fmas_f32  rd, rs, rt, ru    ; FMA step: iterative refinement
div_fixup_f32 rd, rs, rt, ru    ; Fixup: handle special cases (Inf, NaN, 0)
```

### 2.7 Double Arithmetic (64-bit IEEE754)

| Instruction | Syntax | Description |
|-------------|--------|-------------|
| `add_f64` | `add_f64 rd:rd+1, rs:rs+1, rt:rt+1` | FP64 add |
| `mul_f64` | `mul_f64 rd:rd+1, rs:rs+1, rt:rt+1` | FP64 multiply |
| `fma_f64` | `fma_f64 rd, neg(rs), rt, ru` | FP64 fused multiply-add |
| `fmac_f64` | `fmac_f64 rd:rd+1, rs:rs+1, rt:rt+1` | FP64 fused multiply-accumulate |
| `div_scale_f64` | `div_scale_f64 rd, rs, rt, ru` | FP64 division scale |
| `div_fmas_f64` | `div_fmas_f64 rd, rs, rt, ru` | FP64 division FMA step |
| `div_fixup_f64` | `div_fixup_f64 rd, rs, rt, ru` | FP64 division fixup |
| `get_mant_f64` | `get_mant_f64 rd, rs:rs+1` | Get mantissa |
| `get_exp_f64` | `get_exp_f64 rd, rs:rs+1` | Get exponent |
| `expadd_f64` | `expadd_f64 rd:rd+1, rs:rs+1, rt` | Add to exponent |

### 2.8 Type Conversion

| Instruction | Syntax | Description |
|-------------|--------|-------------|
| `cvt_i32tof32` | `cvt_i32tof32 rd, rs` | int32 → float32 |
| `cvt_u32tof32` | `cvt_u32tof32 rd, rs` | uint32 → float32 |
| `cvt_f32toi32` | `cvt_f32toi32 rd, rs` | float32 → int32 |
| `cvt_f32tou32` | `cvt_f32tou32 rd, rs` | float32 → uint32 |
| `cvt_f32tof64` | `cvt_f32tof64 rd:rd+1, rs` | float32 → double |
| `cvt_f64tof32` | `cvt_f64tof32 rd, rs:rs+1` | double → float32 |

### 2.9 Comparison

| Instruction | Syntax | Description |
|-------------|--------|-------------|
| `cmp_lt_i32` | `cmp_lt_i32 sd, rs, rt/imm` | Signed less-than → scalar flag |
| `cmp_gt_i32` | `cmp_gt_i32 sd, rs, rt` | Signed greater-than → scalar flag |
| `cmp_ge_u32` | `cmp_ge_u32 sd, rs, rt` | Unsigned greater-or-equal → scalar flag |
| `cmp_eq_u32` | `cmp_eq_u32 sd, rs, rt/imm` | Equality → scalar flag |
| `cmp_lg_u32` | `cmp_lg_u32 sd/cmsk, rs, rt` | Not-equal → scalar flag or cmsk |
| `scmp_lt_i32` | `scmp_lt_i32 sd, ss, imm` | Scalar signed less-than |
| `scmp_lt_u32` | `scmp_lt_u32 sd, ss, imm` | Scalar unsigned less-than |

### 2.10 Conditional Select

| Instruction | Syntax | Description |
|-------------|--------|-------------|
| `csel_b32` | `csel_b32 rd, rs_true, rs_false, cond_sd` | `rd = cond_sd ? rs_true : rs_false` |

### 2.11 Memory Operations

#### Global Memory

| Instruction | Syntax | Description |
|-------------|--------|-------------|
| `ldg_b32` | `ldg_b32 rd, addr_base, offset, 0x0` | Load 32-bit from global memory |
| `stg_b32` | `stg_b32 addr_base, offset, rs devc` | Store 32-bit to global memory |
| `stg_b64` | `stg_b64 addr_base, offset, rs:rs+1 devc` | Store 64-bit to global memory |
| `stg_b96` | `stg_b96 addr_base, offset, rs devc` | Store 96-bit to global memory |
| `stg_b128` | `stg_b128 addr_base, offset, rs devc` | Store 128-bit to global memory |

#### Shared Memory (BSM)

| Instruction | Syntax | Description |
|-------------|--------|-------------|
| `lds_b32` | `lds_b32 rd, addr, offset` | Load 32-bit from shared memory |
| `sts_b32` | `sts_b32 addr, offset, rs` | Store 32-bit to shared memory |

#### Kernel Arguments (Kernarg Segment)

| Instruction | Syntax | Description |
|-------------|--------|-------------|
| `ldu_b32` | `ldu_b32 sd, base, offset` | Load 32-bit from kernarg segment |
| `ldu_b64` | `ldu_b64 sd:sd+1, base, offset` | Load 64-bit from kernarg segment |
| `ldu_b128` | `ldu_b128 sd:sd+3, base, offset` | Load 128-bit from kernarg segment |
| `ldu_b256` | `ldu_b256 sd:sd+7, base, offset` | Load 256-bit from kernarg segment |

### 2.12 Atomic Operations

| Instruction | Syntax | Description |
|-------------|--------|-------------|
| `atom_global_add_i32` | `atom_global_add_i32 addr_base, rs, offset, rt` | Atomic add (32-bit, global) |

### 2.13 Synchronization & Barriers

| Instruction | Syntax | Description |
|-------------|--------|-------------|
| `barrier` | `barrier` | Block-level thread barrier |
| `arrive` | `arrive slcnt(N)` | **Arrive with scalar load count** — wait until ≤N scalar loads pending |
| `arrive` | `arrive gvmcnt(N)` | **Arrive with global vector memory count** — wait until ≤N global loads pending |
| `arrive` | `arrive bsmcnt(N)` | **Arrive with BSM (shared memory) count** — wait until ≤N BSM ops pending |
| `arrive` | `arrive 0` | Kernel entry marker |

The `arrive` instruction is the primary dependency/synchronization mechanism:
- `slcnt(N)` = scalar load count — ensures kernel arg loads are complete
- `gvmcnt(N)` = global vector memory count — ensures global loads are complete
- `bsmcnt(N)` = block shared memory count — ensures shared memory ops are complete

### 2.14 Control Flow

| Instruction | Syntax | Description |
|-------------|--------|-------------|
| `bra` | `bra label` | Unconditional branch |
| `bra_smsks` | `bra_smsks label` | Branch if scalar mask set (non-zero) |
| `bra_smskz` | `bra_smskz label` | Branch if scalar mask zero |
| `bra_xmskz` | `bra_xmskz label` | Branch if execution mask zero (all lanes inactive) |

### 2.15 Move & Data Movement

| Instruction | Syntax | Description |
|-------------|--------|-------------|
| `mov_b32` | `mov_b32 rd, rs/imm` | 32-bit move (vector reg) |
| `smov_b32` | `smov_b32 sd, ss/imm` | 32-bit move (scalar reg) |
| `smov_b64` | `smov_b64 sd:sd+1, imm` | 64-bit move immediate (scalar) |

### 2.16 Warp-Level Operations

| Instruction | Syntax | Description |
|-------------|--------|-------------|
| `mbcnt_lo_b32` | `mbcnt_lo_b32 rd, mask, rs` | Warp thread count (low 32 lanes) |
| `mbcnt_hi_b32` | `mbcnt_hi_b32 rd, mask, rs` | Warp thread count (high 32 lanes) |
| `sm_bperm_b32` | `sm_bperm_b32 rd, idx, 0x0, rs` | **Warp shuffle (BSM permute)** — read value from lane `idx` |
| `and_xmsk` | `and_xmsk sd, ss` | AND with execution mask |

---

## 3. Instruction Encoding

### 3.1 Instruction Width

Instructions appear to be **32-bit fixed-width** (4 bytes each), little-endian.

Evidence: all kernel code sizes are multiples of 4, and `.p2align 8` (256-byte alignment) is used for kernel entry points.

### 3.2 Operand Types

| Prefix | Meaning | Example |
|--------|---------|---------|
| `r0`–`rN` | Vector register (VGPR) | `r0`, `r15` |
| `s0`–`sN` | Scalar register (SGPR) | `s0`, `s16` |
| `imm` | Immediate value | `0x3ff`, `4`, `-1` |
| `label` | Branch target | `LBB5_3` |
| `devc` | Global memory modifier | `stg_b32 s0, r0, 0x0, r1 devc` |
| `neg()` | Negate source operand | `neg(r3)`, `neg(s2)` |

### 3.3 Addressing Modes

```
; Global load: base_reg + index_reg * stride + offset
ldg_b32 rd, base_sreg, index_rreg, offset

; Global store: base_reg + index_reg * stride + offset
stg_b32 base_sreg, offset, rs devc
stg_b32 base_rreg, offset, rs devc       ; 64-bit address via register pair

; Shared load/store: base + offset
lds_b32 rd, addr_rreg, offset
sts_b32 addr_rreg, offset, rs

; Kernarg load: base + constant offset
ldu_b32 sd, kernarg_sreg, const_offset

; MAD (multiply-add for address computation)
mad_i32 rd, rs, imm_scale, base_rs       ; rd = rs * imm_scale + base_rs
```

---

## 4. Kernel Metadata

Each kernel is followed by a `.macahca_kernel` metadata block:

```asm
.macahca_kernel kernel_name
    .macahca_mtreg_size N          ; VGPR count (per-warp temp registers)
    .macahca_streg_size N          ; SGPR count (scalar registers)
    .macahca_user_streg_count N    ; User-visible SGPRs (kernel args)
    .macahca_bsm_size N            ; Shared memory enabled (0 or 1)
    .macahca_warps_per_dpc 416     ; Warps per dispatch
    .macahca_blocks_per_ap 16      ; Blocks per AP
    .macahca_streg_block_id_x_on 1 ; Hardware block_id.x enabled
    .macahca_streg_kernarg_segm_ptr_on 1  ; Kernarg pointer enabled
    .macahca_streg_dispatch_ptr_on N      ; Dispatch pointer enabled
    .macahca_float_mode 240        ; IEEE754 mode
    .macahca_ieee_mode 1           ; Full IEEE compliance
.end_macahca_kernel
```

### Resource Summary (from probes)

| Kernel | Code Size | MT Regs | ST Regs | MaxWarps |
|--------|-----------|---------|---------|----------|
| k_iarith (11 int ops) | 440B | 2 | 2 | 8 |
| k_farith (9 float ops) | 272B | 12 | 10 | 8 |
| k_cmp (6 branches) | 200B | 5 | 12 | 8 |
| k_loop (for loop) | 568B | 2 | 28 | 8 |
| k_shared (shared mem) | 152B | 3 | 8 | 8 |
| k_atomic (atomicAdd) | 80B | 2 | 8 | 8 |
| k_conv (type conv) | 184B | 2 | 12 | 8 |
| k_double (FP64) | 296B | 18 | 12 | 8 |
| k_bits (clz/popc/perm) | 136B | 5 | 12 | 8 |
| k_shuffle (warp shfl) | 304B | 8 | 10 | 8 |

---

## 5. Compilation Workflow

```bash
# 1. Compile CUDA to device assembly
mxcc -aop -S kernel.cu -o kernel.s

# 2. Extract device-only section
sed -n '/__CLANG_OFFLOAD_BUNDLE____START__ maca-mxc/,/__CLANG_OFFLOAD_BUNDLE____END__ maca-mxc/p' kernel.s > kernel_device.s

# 3. Compile to device object (for binary analysis)
mxcc --device-obj kernel.cu -o kernel_dev.o

# 4. Extract LLVM bitcode from device object
llvm-objcopy --dump-section .llvmbc=kernel.bc kernel_dev.o
llvm-dis kernel.bc -o kernel.ll
```

---

## 6. Key Differences from NVIDIA CUDA

| Feature | NVIDIA (PTX/SASS) | MetaX xcore1000 |
|---------|-------------------|-----------------|
| Warp size | 32 | **64** |
| Base ISA | Custom | **ARCv2.3 64-bit** |
| Division | Single `FDIV` | **3-step**: `div_scale` → `div_fmas` → `div_fixup` |
| Memory fence | `membar` | **`arrive gvmcnt(N)`** / `arrive bsmcnt(N)` |
| Warp shuffle | `shfl.sync` | **`sm_bperm_b32`** (BSM permute) |
| Shared mem load | `lds` | `lds_b32` (same mnemonic!) |
| Global store | `st.global` | `stg_b32 ... devc` |
| Kernel end | `exit` / `ret` | **`endk`** |
| Branch on exec mask | `@%p bra` | **`bra_xmskz`** / `bra_smskz` |
| Conditional select | `selp` | **`csel_b32`** |
| MAD (multiply-add) | `mad` / `mad.lo` | **`mad_i32 rd, rs, imm, base`** |
| Bit count | `popc` | **`sbcs_b32`** |
| Leading zeros | `clz` | **`sclz_b32`** |

---

## 7. Example: vector_add Full Assembly

```asm
; vector_add(float *c, const float *a, const float *b, int n)
_Z10vector_addPfPKfS1_i:
    ldu_b32 s0, s0, 0x4           ; load dispatch ptr field (blockDim.x)
    ldu_b32 s1, s2, 0x18          ; load kernel arg: n
    and_b32 r0, 0x3ff, r0         ; threadIdx.x = r0 & 0x3FF (10-bit mask)
    arrive slcnt(0)               ; wait for kernarg loads
    sand_b32 s0, 0xffff, s0       ; mask blockDim to 16 bits
    smul_i32 s0, s6, s0           ; s0 = blockIdx.x * blockDim.x
    add_u32 r0, s0, r0            ; r0 = global thread index
    cmp_lt_i32 s0, r0, s1         ; s0 = (idx < n)
    snop 3                        ; scalar pipeline delay
    and_xmsk s4, s0               ; set execution mask for active lanes
    snop 0
    bra_xmskz LBB2_2              ; if no active lanes, skip
; if.then:
    ldu_b256 s0, s2, 0x0          ; load all 3 pointer args (256-bit)
    arrive slcnt(0)               ; wait for pointer loads
    mad_i32 r6, r0, 4, s2         ; r6 = &a[idx] = idx*4 + base_a
    mad_i32 r4, r0, 4, s4         ; r4 = &b[idx]
    mad_i32 r2, r0, 4, s0         ; r2 = &c[idx]
    ldg_b32 r1, r6, 0x0           ; r1 = a[idx]
    ldg_b32 r0, r4, 0x0           ; r0 = b[idx]
    arrive gvmcnt(0)              ; wait for global loads
    add_f32 r0, r1, r0            ; r0 = a[idx] + b[idx]
    snop 1
    stg_b32 r2, 0x0, r0 devc      ; c[idx] = r0
LBB2_2:
    snop 2
    endk                          ; kernel done
```
