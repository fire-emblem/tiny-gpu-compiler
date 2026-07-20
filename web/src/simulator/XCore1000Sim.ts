/**
 * xcore1000 GPU Simulator — Browser-based execution engine
 *
 * Simulates the MetaX xcore1000 GPU architecture:
 * - 64-thread warps with VGPR (r0-r31) + SGPR (s0-s31) register files
 * - 32-bit instruction execution
 * - Global memory (Float32Array) and shared memory (BSM)
 * - Barrier synchronization, arrive/snop dependency tracking
 * - Warp divergence detection
 */

// ============================================================================
// Types
// ============================================================================

export interface XCore1000ThreadState {
  threadId: number;
  blockId: number;
  pc: number;
  vgprs: Int32Array;       // r0-r31, 32-bit each
  sgprs: Int32Array;       // s0-s31, shared within warp (but we model per-thread for simplicity)
  done: boolean;
  currentInstruction: string;
  divergent: boolean;
  waitingFor: string | null;  // 'slcnt', 'gvmcnt', 'bsmcnt', 'barrier', null
}

export interface XCore1000SimulationState {
  cycle: number;
  warpSize: number;
  threads: XCore1000ThreadState[];
  globalMemory: Float32Array;
  sharedMemory: Float32Array;
  currentBlock: number;
  totalBlocks: number;
  blockDim: number;
}

interface ParsedInstruction {
  mnemonic: string;
  operands: string;
  raw: string;
  addr: number;
}

// ============================================================================
// xcore1000 Simulator
// ============================================================================

export class XCore1000Sim {
  private program: ParsedInstruction[] = [];
  private globalMemory: Float32Array;
  private sharedMemory: Float32Array;
  private threads: XCore1000ThreadState[] = [];
  private numBlocks: number;
  private blockDim: number;
  private warpSize: number;
  private currentBlock = 0;
  private cycle = 0;

  // Pending memory operations tracking
  private pendingGlobalLoads = 0;
  private pendingSharedOps = 0;
  private pendingScalarLoads = 0;
  private barrierReached = 0;

  constructor(
    assemblyText: string,
    initialMemory: Float32Array,
    numBlocks: number,
    blockDim: number,
  ) {
    this.numBlocks = numBlocks;
    this.blockDim = blockDim;
    this.warpSize = 64; // xcore1000 warp size

    // Parse assembly text into instructions
    this.program = this.parseAssembly(assemblyText);

    // Initialize memory
    this.globalMemory = new Float32Array(1024); // 4KB global memory
    for (let i = 0; i < initialMemory.length && i < this.globalMemory.length; i++) {
      this.globalMemory[i] = initialMemory[i];
    }

    this.sharedMemory = new Float32Array(256); // 1KB shared memory per block

    this.initBlock(0);
  }

  // ==========================================================================
  // Assembly Parser
  // ==========================================================================

  private parseAssembly(text: string): ParsedInstruction[] {
    const instructions: ParsedInstruction[] = [];
    const lines = text.split('\n');
    let addr = 0;

    for (const line of lines) {
      const trimmed = line.trim();
      // Skip comments, labels, directives, empty lines
      if (!trimmed || trimmed.startsWith(';') || trimmed.startsWith('#') ||
          trimmed.startsWith('.') || trimmed.startsWith('//') ||
          trimmed.startsWith('__CLANG')) {
        continue;
      }

      // Parse instruction: mnemonic operands
      const parts = trimmed.split(/\s+/);
      const mnemonic = parts[0];
      const operands = parts.slice(1).join(' ');

      // Skip non-instruction lines
      if (['.text', '.type', '.globl', '.protected', '.p2align',
           '.size', '.section', '.fill', '.macahca_kernel', '.end_macahca_kernel',
           '.macahca_'].some(p => mnemonic.startsWith(p))) {
        continue;
      }

      instructions.push({ mnemonic, operands, raw: trimmed, addr });
      addr++;
    }

    return instructions;
  }

  // ==========================================================================
  // Block Initialization
  // ==========================================================================

  private initBlock(blockId: number) {
    this.currentBlock = blockId;
    this.threads = [];
    this.sharedMemory = new Float32Array(256);
    this.pendingGlobalLoads = 0;
    this.pendingSharedOps = 0;
    this.pendingScalarLoads = 0;
    this.barrierReached = 0;

    const numThreads = Math.min(this.blockDim, this.warpSize);

    for (let t = 0; t < numThreads; t++) {
      const vgprs = new Int32Array(32);
      const sgprs = new Int32Array(32);

      // r0 = threadIdx.x (hardware-provided)
      vgprs[0] = t;

      // s0 = kernarg pointer (simulated as 0)
      // s1 = reserved
      sgprs[0] = 0;
      sgprs[1] = 0;

      this.threads.push({
        threadId: t,
        blockId,
        pc: 0,
        vgprs,
        sgprs,
        done: false,
        currentInstruction: '',
        divergent: false,
        waitingFor: null,
      });
    }
  }

  // ==========================================================================
  // State Access
  // ==========================================================================

  getState(): XCore1000SimulationState {
    return {
      cycle: this.cycle,
      warpSize: this.warpSize,
      threads: this.threads.map(t => ({
        ...t,
        vgprs: new Int32Array(t.vgprs),
        sgprs: new Int32Array(t.sgprs),
      })),
      globalMemory: new Float32Array(this.globalMemory),
      sharedMemory: new Float32Array(this.sharedMemory),
      currentBlock: this.currentBlock,
      totalBlocks: this.numBlocks,
      blockDim: this.blockDim,
    };
  }

  isDone(): boolean {
    return this.currentBlock >= this.numBlocks;
  }

  // ==========================================================================
  // Step Execution
  // ==========================================================================

  step(): XCore1000SimulationState {
    if (this.isDone()) return this.getState();

    // Check if all threads in current block are done
    if (this.threads.every(t => t.done)) {
      this.currentBlock++;
      if (this.currentBlock < this.numBlocks) {
        this.initBlock(this.currentBlock);
      }
      this.cycle++;
      return this.getState();
    }

    // Check barrier synchronization
    if (this.barrierReached > 0) {
      const activeThreads = this.threads.filter(t => !t.done);
      if (this.barrierReached >= activeThreads.length) {
        // All active threads reached barrier — release them
        for (const thread of this.threads) {
          if (thread.waitingFor === 'barrier') {
            thread.waitingFor = null;
            thread.pc++;
          }
        }
        this.barrierReached = 0;
        this.pendingSharedOps = 0;
      }
      this.cycle++;
      return this.getState();
    }

    // Resolve pending memory operations
    if (this.pendingGlobalLoads > 0) {
      this.pendingGlobalLoads = 0; // Instant resolve in simulation
      for (const thread of this.threads) {
        if (thread.waitingFor === 'gvmcnt') {
          thread.waitingFor = null;
        }
      }
    }
    if (this.pendingScalarLoads > 0) {
      this.pendingScalarLoads = 0;
      for (const thread of this.threads) {
        if (thread.waitingFor === 'slcnt') {
          thread.waitingFor = null;
        }
      }
    }

    // Track divergence
    const activeThreads = this.threads.filter(t => !t.done && !t.waitingFor);
    const activePCs = new Set(activeThreads.map(t => t.pc));
    if (activePCs.size > 1) {
      const pcCounts: Record<number, number> = {};
      activeThreads.forEach(t => { pcCounts[t.pc] = (pcCounts[t.pc] || 0) + 1; });
      const majorityPC = parseInt(Object.entries(pcCounts).sort((a, b) => b[1] - a[1])[0]?.[0] ?? '0');
      this.threads.forEach(t => {
        t.divergent = !t.done && !t.waitingFor && t.pc !== majorityPC;
      });
    } else {
      this.threads.forEach(t => { t.divergent = false; });
    }

    // Execute one instruction per thread
    for (const thread of this.threads) {
      if (thread.done || thread.waitingFor) continue;
      this.executeThread(thread);
    }

    this.cycle++;
    return this.getState();
  }

  // ==========================================================================
  // Thread Execution
  // ==========================================================================

  private executeThread(thread: XCore1000ThreadState) {
    if (thread.pc >= this.program.length) {
      thread.done = true;
      return;
    }

    const inst = this.program[thread.pc];
    thread.currentInstruction = inst.raw;

    const { mnemonic, operands } = inst;
    const vgprs = thread.vgprs;
    const sgprs = thread.sgprs;

    // Parse operand registers: rN for VGPR, sN for SGPR
    const parseReg = (s: string): { isVgpr: boolean; num: number } | null => {
      const vMatch = s.match(/^r(\d+)$/);
      if (vMatch) return { isVgpr: true, num: parseInt(vMatch[1]) };
      const sMatch = s.match(/^s(\d+)$/);
      if (sMatch) return { isVgpr: false, num: parseInt(sMatch[1]) };
      return null;
    };

    const getVal = (s: string): number => {
      const reg = parseReg(s);
      if (reg) return reg.isVgpr ? vgprs[reg.num] : sgprs[reg.num];
      const imm = parseInt(s);
      if (!isNaN(imm)) return imm;
      // Handle hex
      if (s.startsWith('0x') || s.startsWith('0X')) return parseInt(s, 16);
      return 0;
    };

    const setVgpr = (num: number, val: number) => {
      vgprs[num] = val | 0;
    };

    const setSgpr = (num: number, val: number) => {
      sgprs[num] = val | 0;
    };

    // Split operands by comma
    const ops = operands.split(',').map(s => s.trim()).filter(s => s.length > 0);

    try {
      switch (mnemonic) {
        // --- Kernel lifecycle ---
        case 'endk':
          thread.done = true;
          break;

        case 'trap':
          thread.done = true;
          break;

        case 'snop':
          // NOP with pipeline hint — just advance
          thread.pc++;
          break;

        // --- Integer arithmetic ---
        case 'add_u32': {
          const rd = parseReg(ops[0]);
          const lhs = getVal(ops[1]);
          const rhs = getVal(ops[2]);
          if (rd) setVgpr(rd.num, lhs + rhs);
          thread.pc++;
          break;
        }

        case 'sub_u32': {
          const rd = parseReg(ops[0]);
          const lhs = getVal(ops[1]);
          const rhs = getVal(ops[2]);
          if (rd) setVgpr(rd.num, lhs - rhs);
          thread.pc++;
          break;
        }

        case 'mul_u32': {
          const rd = parseReg(ops[0]);
          const lhs = getVal(ops[1]);
          const rhs = getVal(ops[2]);
          if (rd) setVgpr(rd.num, Math.imul(lhs, rhs));
          thread.pc++;
          break;
        }

        case 'mul_hi_u32': {
          const rd = parseReg(ops[0]);
          const lhs = getVal(ops[1]);
          const rhs = getVal(ops[2]);
          // High 32 bits of 64-bit multiply
          const result = (BigInt(lhs) * BigInt(rhs)) >> 32n;
          if (rd) setVgpr(rd.num, Number(result));
          thread.pc++;
          break;
        }

        case 'mad_i32': {
          const rd = parseReg(ops[0]);
          const a = getVal(ops[1]);
          const b = getVal(ops[2]);
          const c = getVal(ops[3]);
          if (rd) setVgpr(rd.num, Math.imul(a, b) + c);
          thread.pc++;
          break;
        }

        case 'sadd_co_i32':
        case 'sadd_co_u32': {
          const rd = parseReg(ops[0]);
          const lhs = getVal(ops[1]);
          const rhs = getVal(ops[2]);
          if (rd) setSgpr(rd.num, lhs + rhs);
          thread.pc++;
          break;
        }

        case 'ssub_co_i32':
        case 'ssub_co_u32': {
          const rd = parseReg(ops[0]);
          const lhs = getVal(ops[1]);
          const rhs = getVal(ops[2]);
          if (rd) setSgpr(rd.num, lhs - rhs);
          thread.pc++;
          break;
        }

        case 'saddc_co_u32': {
          const rd = parseReg(ops[0]);
          const lhs = getVal(ops[1]);
          const rhs = getVal(ops[2]);
          if (rd) setSgpr(rd.num, lhs + rhs);
          thread.pc++;
          break;
        }

        case 'smul_i32': {
          const rd = parseReg(ops[0]);
          const lhs = getVal(ops[1]);
          const rhs = getVal(ops[2]);
          if (rd) setSgpr(rd.num, Math.imul(lhs, rhs));
          thread.pc++;
          break;
        }

        case 'smax_i32': {
          const rd = parseReg(ops[0]);
          const lhs = getVal(ops[1]);
          const rhs = getVal(ops[2]);
          if (rd) setSgpr(rd.num, Math.max(lhs, rhs));
          thread.pc++;
          break;
        }

        // --- Bitwise operations ---
        case 'and_b32': {
          const rd = parseReg(ops[0]);
          const lhs = getVal(ops[1]);
          const rhs = getVal(ops[2]);
          if (rd) setVgpr(rd.num, lhs & rhs);
          thread.pc++;
          break;
        }

        case 'sand_b32': {
          const rd = parseReg(ops[0]);
          const lhs = getVal(ops[1]);
          const rhs = getVal(ops[2]);
          if (rd) setSgpr(rd.num, lhs & rhs);
          thread.pc++;
          break;
        }

        case 'sor_b32': {
          const rd = parseReg(ops[0]);
          const lhs = getVal(ops[1]);
          const rhs = getVal(ops[2]);
          if (rd) setSgpr(rd.num, lhs | rhs);
          thread.pc++;
          break;
        }

        case 'xor_b32': {
          const rd = parseReg(ops[0]);
          const lhs = getVal(ops[1]);
          const rhs = getVal(ops[2]);
          if (rd) setVgpr(rd.num, lhs ^ rhs);
          thread.pc++;
          break;
        }

        case 'sxor_b32': {
          const rd = parseReg(ops[0]);
          const lhs = getVal(ops[1]);
          const rhs = getVal(ops[2]);
          if (rd) setSgpr(rd.num, lhs ^ rhs);
          thread.pc++;
          break;
        }

        case 'shl_b32': {
          const rd = parseReg(ops[0]);
          const lhs = getVal(ops[1]);
          const rhs = getVal(ops[2]);
          if (rd) setVgpr(rd.num, lhs << rhs);
          thread.pc++;
          break;
        }

        case 'sshr_b32': {
          const rd = parseReg(ops[0]);
          const lhs = getVal(ops[1]);
          const rhs = getVal(ops[2]);
          if (rd) setSgpr(rd.num, lhs >>> rhs);
          thread.pc++;
          break;
        }

        case 'sashr_i32': {
          const rd = parseReg(ops[0]);
          const lhs = getVal(ops[1]);
          const rhs = getVal(ops[2]);
          if (rd) setSgpr(rd.num, lhs >> rhs);
          thread.pc++;
          break;
        }

        case 'sclz_b32': {
          const rd = parseReg(ops[0]);
          const val = getVal(ops[1]);
          if (rd) {
            let count = 0;
            for (let i = 31; i >= 0; i--) {
              if (val & (1 << i)) break;
              count++;
            }
            setSgpr(rd.num, count);
          }
          thread.pc++;
          break;
        }

        // --- Float arithmetic ---
        case 'add_f32': {
          const rd = parseReg(ops[0]);
          const lhs = this.intToFloat(getVal(ops[1]));
          const rhs = this.intToFloat(getVal(ops[2]));
          if (rd) setVgpr(rd.num, this.floatToInt(lhs + rhs));
          thread.pc++;
          break;
        }

        case 'sub_f32': {
          const rd = parseReg(ops[0]);
          const lhs = this.intToFloat(getVal(ops[1]));
          const rhs = this.intToFloat(getVal(ops[2]));
          if (rd) setVgpr(rd.num, this.floatToInt(lhs - rhs));
          thread.pc++;
          break;
        }

        case 'mul_f32': {
          const rd = parseReg(ops[0]);
          const lhs = this.intToFloat(getVal(ops[1]));
          const rhs = this.intToFloat(getVal(ops[2]));
          if (rd) setVgpr(rd.num, this.floatToInt(lhs * rhs));
          thread.pc++;
          break;
        }

        case 'fma_f32': {
          const rd = parseReg(ops[0]);
          const a = this.intToFloat(getVal(ops[1]));
          const b = this.intToFloat(getVal(ops[2]));
          const c = this.intToFloat(getVal(ops[3]));
          if (rd) setVgpr(rd.num, this.floatToInt(a * b + c));
          thread.pc++;
          break;
        }

        case 'fmac_f32': {
          const rd = parseReg(ops[0]);
          const rs = this.intToFloat(getVal(ops[1]));
          const rt = this.intToFloat(getVal(ops[2]));
          const current = this.intToFloat(vgprs[rd?.num ?? 0]);
          if (rd) setVgpr(rd.num, this.floatToInt(current + rs * rt));
          thread.pc++;
          break;
        }

        case 'rcp_f32': {
          const rd = parseReg(ops[0]);
          const val = this.intToFloat(getVal(ops[1]));
          if (rd) setVgpr(rd.num, this.floatToInt(val !== 0 ? 1.0 / val : 0));
          thread.pc++;
          break;
        }

        case 'min_f32': {
          const rd = parseReg(ops[0]);
          const lhs = this.intToFloat(getVal(ops[1]));
          const rhs = this.intToFloat(getVal(ops[2]));
          if (rd) setVgpr(rd.num, this.floatToInt(Math.min(lhs, rhs)));
          thread.pc++;
          break;
        }

        case 'max_f32': {
          const rd = parseReg(ops[0]);
          const lhs = this.intToFloat(getVal(ops[1]));
          const rhs = this.intToFloat(getVal(ops[2]));
          if (rd) setVgpr(rd.num, this.floatToInt(Math.max(lhs, rhs)));
          thread.pc++;
          break;
        }

        // --- Float division (multi-step) ---
        case 'div_scale_f32':
        case 'div_fmas_f32':
        case 'div_fixup_f32': {
          // Simplified: just do the division
          const rd = parseReg(ops[0]);
          const lhs = this.intToFloat(getVal(ops[1]));
          const rhs = this.intToFloat(getVal(ops[2] ?? ops[1]));
          if (rd) setVgpr(rd.num, this.floatToInt(rhs !== 0 ? lhs / rhs : 0));
          thread.pc++;
          break;
        }

        // --- Type conversion ---
        case 'cvt_i32tof32': {
          const rd = parseReg(ops[0]);
          const val = getVal(ops[1]);
          if (rd) setVgpr(rd.num, this.floatToInt(val));
          thread.pc++;
          break;
        }

        case 'cvt_f32toi32': {
          const rd = parseReg(ops[0]);
          const val = this.intToFloat(getVal(ops[1]));
          if (rd) setVgpr(rd.num, Math.trunc(val));
          thread.pc++;
          break;
        }

        case 'cvt_u32tof32': {
          const rd = parseReg(ops[0]);
          const val = getVal(ops[1]) >>> 0; // unsigned
          if (rd) setVgpr(rd.num, this.floatToInt(val));
          thread.pc++;
          break;
        }

        case 'cvt_f32tou32': {
          const rd = parseReg(ops[0]);
          const val = this.intToFloat(getVal(ops[1]));
          if (rd) setVgpr(rd.num, Math.trunc(val) >>> 0);
          thread.pc++;
          break;
        }

        // --- Comparison ---
        case 'cmp_lt_i32': {
          const rd = parseReg(ops[0]);
          const lhs = getVal(ops[1]);
          const rhs = getVal(ops[2]);
          if (rd) setSgpr(rd.num, lhs < rhs ? 1 : 0);
          thread.pc++;
          break;
        }

        case 'cmp_gt_i32': {
          const rd = parseReg(ops[0]);
          const lhs = getVal(ops[1]);
          const rhs = getVal(ops[2]);
          if (rd) setSgpr(rd.num, lhs > rhs ? 1 : 0);
          thread.pc++;
          break;
        }

        case 'cmp_ge_u32': {
          const rd = parseReg(ops[0]);
          const lhs = getVal(ops[1]) >>> 0;
          const rhs = getVal(ops[2]) >>> 0;
          if (rd) setSgpr(rd.num, lhs >= rhs ? 1 : 0);
          thread.pc++;
          break;
        }

        case 'cmp_eq_u32': {
          const rd = parseReg(ops[0]);
          const lhs = getVal(ops[1]);
          const rhs = getVal(ops[2]);
          if (rd) setSgpr(rd.num, lhs === rhs ? 1 : 0);
          thread.pc++;
          break;
        }

        case 'cmp_lg_u32': {
          const rd = parseReg(ops[0]);
          const lhs = getVal(ops[1]);
          const rhs = getVal(ops[2]);
          if (rd) setSgpr(rd.num, lhs !== rhs ? 1 : 0);
          thread.pc++;
          break;
        }

        // --- Conditional select ---
        case 'csel_b32': {
          const rd = parseReg(ops[0]);
          const trueVal = getVal(ops[1]);
          const falseVal = getVal(ops[2]);
          const cond = getVal(ops[3]);
          if (rd) setVgpr(rd.num, cond ? trueVal : falseVal);
          thread.pc++;
          break;
        }

        // --- Memory: Global ---
        case 'ldg_b32': {
          const rd = parseReg(ops[0]);
          const base = getVal(ops[1]);
          const offset = parseInt(ops[2]) || 0;
          const addr = (base + offset) / 4; // byte address → float index
          if (rd && addr >= 0 && addr < this.globalMemory.length) {
            // Store as int bits for later retrieval
            const floatVal = this.globalMemory[addr];
            setVgpr(rd.num, this.floatToInt(floatVal));
          }
          this.pendingGlobalLoads++;
          thread.pc++;
          break;
        }

        case 'stg_b32': {
          const addrReg = parseReg(ops[0]);
          const offset = parseInt(ops[1]) || 0;
          const valReg = parseReg(ops[2]);
          if (addrReg && valReg) {
            const addr = (getVal(ops[0]) + offset) / 4;
            const val = this.intToFloat(getVal(ops[2]));
            if (addr >= 0 && addr < this.globalMemory.length) {
              this.globalMemory[addr] = val;
            }
          }
          thread.pc++;
          break;
        }

        // --- Memory: Shared ---
        case 'lds_b32': {
          const rd = parseReg(ops[0]);
          const addr = getVal(ops[1]) / 4;
          if (rd && addr >= 0 && addr < this.sharedMemory.length) {
            const floatVal = this.sharedMemory[addr];
            setVgpr(rd.num, this.floatToInt(floatVal));
          }
          this.pendingSharedOps++;
          thread.pc++;
          break;
        }

        case 'sts_b32': {
          const addr = getVal(ops[0]) / 4;
          const val = this.intToFloat(getVal(ops[1]));
          if (addr >= 0 && addr < this.sharedMemory.length) {
            this.sharedMemory[addr] = val;
          }
          this.pendingSharedOps++;
          thread.pc++;
          break;
        }

        // --- Kernarg loads ---
        case 'ldu_b32': {
          const rd = parseReg(ops[0]);
          // Simulated: load from kernarg segment
          const offset = parseInt(ops[2]) || 0;
          if (rd) {
            // Simulate dispatch pointer fields
            if (offset === 0x4) setSgpr(rd.num, this.blockDim); // blockDim.x
            else if (offset === 0x10) setSgpr(rd.num, this.currentBlock); // blockIdx.x
            else setSgpr(rd.num, 0);
          }
          this.pendingScalarLoads++;
          thread.pc++;
          break;
        }

        case 'ldu_b64': {
          const rd = parseReg(ops[0]);
          if (rd) setSgpr(rd.num, 0); // Simulated pointer
          this.pendingScalarLoads++;
          thread.pc++;
          break;
        }

        case 'ldu_b128': {
          const rd = parseReg(ops[0]);
          if (rd) setSgpr(rd.num, 0); // Simulated kernarg base
          this.pendingScalarLoads++;
          thread.pc++;
          break;
        }

        case 'ldu_b256': {
          const rd = parseReg(ops[0]);
          if (rd) setSgpr(rd.num, 0); // Simulated bulk load
          this.pendingScalarLoads++;
          thread.pc++;
          break;
        }

        // --- Synchronization ---
        case 'barrier':
          thread.waitingFor = 'barrier';
          this.barrierReached++;
          break;

        case 'arrive': {
          const counter = operands.match(/(\w+)\((\d+)\)/);
          if (counter) {
            const kind = counter[1];
            const count = parseInt(counter[2]);
            if (kind === 'slcnt') {
              if (this.pendingScalarLoads > count) {
                thread.waitingFor = 'slcnt';
              } else {
                thread.pc++;
              }
            } else if (kind === 'gvmcnt') {
              if (this.pendingGlobalLoads > count) {
                thread.waitingFor = 'gvmcnt';
              } else {
                thread.pc++;
              }
            } else if (kind === 'bsmcnt') {
              if (this.pendingSharedOps > count) {
                thread.waitingFor = 'bsmcnt';
              } else {
                thread.pc++;
              }
            } else {
              thread.pc++;
            }
          } else {
            thread.pc++; // arrive 0 — kernel entry marker
          }
          break;
        }

        // --- Control flow ---
        case 'bra': {
          const label = ops[0];
          const target = this.findLabel(label);
          if (target >= 0) thread.pc = target;
          else thread.pc++;
          break;
        }

        case 'bra_smsks': {
          const cond = sgprs[0]; // Check scalar mask
          const label = ops[0];
          if (cond) {
            const target = this.findLabel(label);
            if (target >= 0) thread.pc = target;
            else thread.pc++;
          } else {
            thread.pc++;
          }
          break;
        }

        case 'bra_smskz': {
          const cond = sgprs[0];
          const label = ops[0];
          if (!cond) {
            const target = this.findLabel(label);
            if (target >= 0) thread.pc = target;
            else thread.pc++;
          } else {
            thread.pc++;
          }
          break;
        }

        case 'bra_xmskz': {
          // Branch if all lanes inactive (simplified: check if any thread active)
          const label = ops[0];
          const activeCount = this.threads.filter(t => !t.done && !t.waitingFor).length;
          if (activeCount === 0) {
            const target = this.findLabel(label);
            if (target >= 0) thread.pc = target;
            else thread.pc++;
          } else {
            thread.pc++;
          }
          break;
        }

        // --- Move ---
        case 'mov_b32': {
          const rd = parseReg(ops[0]);
          const val = getVal(ops[1]);
          if (rd) {
            if (rd.isVgpr) setVgpr(rd.num, val);
            else setSgpr(rd.num, val);
          }
          thread.pc++;
          break;
        }

        case 'smov_b32': {
          const rd = parseReg(ops[0]);
          const val = getVal(ops[1]);
          if (rd) setSgpr(rd.num, val);
          thread.pc++;
          break;
        }

        case 'smov_b64': {
          const rd = parseReg(ops[0]);
          const val = getVal(ops[1]);
          if (rd) setSgpr(rd.num, val);
          thread.pc++;
          break;
        }

        // --- Warp operations ---
        case 'mbcnt_lo_b32': {
          const rd = parseReg(ops[0]);
          if (rd) setVgpr(rd.num, thread.threadId); // Simplified
          thread.pc++;
          break;
        }

        case 'mbcnt_hi_b32': {
          const rd = parseReg(ops[0]);
          if (rd) setVgpr(rd.num, 0); // Simplified (high 32 lanes)
          thread.pc++;
          break;
        }

        case 'sm_bperm_b32': {
          // Warp shuffle — read value from another lane
          const rd = parseReg(ops[0]);
          const laneIdx = getVal(ops[1]) / 4; // byte offset → lane index
          const srcLane = Math.min(Math.max(laneIdx, 0), this.threads.length - 1);
          const srcThread = this.threads[srcLane];
          if (rd && srcThread) {
            setVgpr(rd.num, srcThread.vgprs[1]); // Read r1 from source lane
          }
          thread.pc++;
          break;
        }

        case 'and_xmsk': {
          const rd = parseReg(ops[0]);
          const val = getVal(ops[1]);
          // Simplified: just pass through
          if (rd) setSgpr(rd.num, val ? 1 : 0);
          thread.pc++;
          break;
        }

        // --- Atomic ---
        case 'atom_global_add_i32': {
          const addrReg = parseReg(ops[0]);
          const valReg = parseReg(ops[1]);
          if (addrReg && valReg) {
            const addr = getVal(ops[0]) / 4;
            const val = getVal(ops[1]);
            if (addr >= 0 && addr < this.globalMemory.length) {
              const old = this.globalMemory[addr];
              this.globalMemory[addr] = old + val;
              // Store old value in result register
              const rd = parseReg(ops[3]);
              if (rd) setVgpr(rd.num, this.floatToInt(old));
            }
          }
          thread.pc++;
          break;
        }

        // --- Fallback ---
        default:
          // Unknown instruction — skip
          thread.pc++;
          break;
      }
    } catch (e) {
      // Execution error — mark thread done
      thread.done = true;
    }

    // Handle endk specially (doesn't increment PC)
    if (mnemonic === 'endk' || mnemonic === 'trap') {
      // Already handled above
    }
  }

  // ==========================================================================
  // Helpers
  // ==========================================================================

  private findLabel(label: string): number {
    // In the parsed program, labels appear as ".Lxxx:" or just "label:"
    for (let i = 0; i < this.program.length; i++) {
      if (this.program[i].raw === label || this.program[i].raw === label + ':') {
        return i;
      }
    }
    return -1;
  }

  private floatToInt(val: number): number {
    const buf = new ArrayBuffer(4);
    new Float32Array(buf)[0] = val;
    return new Int32Array(buf)[0];
  }

  private intToFloat(val: number): number {
    const buf = new ArrayBuffer(4);
    new Int32Array(buf)[0] = val;
    return new Float32Array(buf)[0];
  }

  /** Run all blocks to completion */
  runToEnd(maxCycles = 100000): XCore1000SimulationState[] {
    const history: XCore1000SimulationState[] = [this.getState()];
    while (!this.isDone() && this.cycle < maxCycles) {
      history.push(this.step());
    }
    return history;
  }
}
