/** Represents a single 16-bit instruction (tiny-gpu) */
export interface Instruction {
  addr: number;
  hex: string;
  asm: string;
  bits: string;
}

/** Represents a single 32-bit xcore1000 instruction */
export interface XCore1000Instruction {
  addr: number;
  mnemonic: string;
  operands: string;
  asm: string;
}

/** Target architecture */
export type TargetArch = 'tinygpu' | 'xcore1000';

/** A single compilation stage showing IR at that point */
export interface CompilationStage {
  name: string;
  ir: string;
}

/** Warp divergence information for a branch instruction */
export interface DivergenceInfo {
  instructionAddr: number;
  type: 'branch' | 'converge';
  branchTaken: boolean[];  // per-thread: did they take the branch?
  description: string;
}

/** Memory coalescing analysis for a memory instruction */
export interface CoalescingInfo {
  instructionAddr: number;
  accessPattern: 'coalesced' | 'strided' | 'scattered';
  addresses: number[];       // per-thread addresses accessed
  transactionsNeeded: number;
  description: string;
}

/** Performance profiling metrics */
export interface PerformanceMetrics {
  totalInstructions: number;
  registersUsed: number;
  sharedMemoryBytes: number;
  branchInstructions: number;
  memoryInstructions: number;
  computeInstructions: number;
  barrierCount: number;
  estimatedCycles: number;
  computeToMemoryRatio: number;
  optimizationSummary: string;
}

/** Full analysis results from the compiler */
export interface AnalysisResult {
  divergence: DivergenceInfo[];
  coalescing: CoalescingInfo[];
  metrics: PerformanceMetrics;
}

/** Full compilation trace from the compiler */
export interface CompilationTrace {
  source: string;
  stages: CompilationStage[];
  binary: {
    instructions: Instruction[];
  };
  analysis?: AnalysisResult;
  target?: TargetArch;
}

/** xcore1000 compilation trace (JSON format from tgc --target xcore1000 --emit trace) */
export interface XCore1000CompilationTrace {
  target: 'xcore1000';
  source: string;
  assembly: string;
}

/** Opcode definitions matching tiny-gpu's decoder.sv */
export enum Opcode {
  NOP   = 0b0000,
  BRnzp = 0b0001,
  CMP   = 0b0010,
  ADD   = 0b0011,
  SUB   = 0b0100,
  MUL   = 0b0101,
  DIV   = 0b0110,
  LDR   = 0b0111,
  STR   = 0b1000,
  CONST = 0b1001,
  SLDR  = 0b1010,
  SSTR  = 0b1011,
  BAR   = 0b1100,
  RET   = 0b1111,
}

export const OPCODE_NAMES: Record<number, string> = {
  [Opcode.NOP]: 'NOP',
  [Opcode.BRnzp]: 'BRnzp',
  [Opcode.CMP]: 'CMP',
  [Opcode.ADD]: 'ADD',
  [Opcode.SUB]: 'SUB',
  [Opcode.MUL]: 'MUL',
  [Opcode.DIV]: 'DIV',
  [Opcode.LDR]: 'LDR',
  [Opcode.STR]: 'STR',
  [Opcode.CONST]: 'CONST',
  [Opcode.SLDR]: 'SLDR',
  [Opcode.SSTR]: 'SSTR',
  [Opcode.BAR]: 'BAR',
  [Opcode.RET]: 'RET',
};

/** xcore1000 instruction categories */
export enum XCore1000InstCategory {
  INTEGER_ARITH = 'integer_arith',
  BITWISE = 'bitwise',
  FLOAT_ARITH = 'float_arith',
  CONVERSION = 'conversion',
  COMPARISON = 'comparison',
  SELECT = 'select',
  MEMORY_GLOBAL = 'memory_global',
  MEMORY_SHARED = 'memory_shared',
  ATOMIC = 'atomic',
  BARRIER = 'barrier',
  CONTROL_FLOW = 'control_flow',
  WARP = 'warp',
  SPECIAL = 'special',
}

/** xcore1000 instruction mnemonic → category mapping */
export const XCORE1000_MNEMONIC_CATEGORIES: Record<string, XCore1000InstCategory> = {
  // Integer arithmetic
  add_u32: XCore1000InstCategory.INTEGER_ARITH,
  sub_u32: XCore1000InstCategory.INTEGER_ARITH,
  mul_u32: XCore1000InstCategory.INTEGER_ARITH,
  mul_hi_u32: XCore1000InstCategory.INTEGER_ARITH,
  mad_i32: XCore1000InstCategory.INTEGER_ARITH,
  smul_i32: XCore1000InstCategory.INTEGER_ARITH,
  sadd_co_i32: XCore1000InstCategory.INTEGER_ARITH,
  ssub_co_i32: XCore1000InstCategory.INTEGER_ARITH,
  // Bitwise
  and_b32: XCore1000InstCategory.BITWISE,
  sand_b32: XCore1000InstCategory.BITWISE,
  sor_b32: XCore1000InstCategory.BITWISE,
  xor_b32: XCore1000InstCategory.BITWISE,
  sxor_b32: XCore1000InstCategory.BITWISE,
  shl_b32: XCore1000InstCategory.BITWISE,
  sshr_b32: XCore1000InstCategory.BITWISE,
  sashr_i32: XCore1000InstCategory.BITWISE,
  sclz_b32: XCore1000InstCategory.BITWISE,
  sbcs_b32: XCore1000InstCategory.BITWISE,
  perm_b32: XCore1000InstCategory.BITWISE,
  // Float arithmetic
  add_f32: XCore1000InstCategory.FLOAT_ARITH,
  sub_f32: XCore1000InstCategory.FLOAT_ARITH,
  mul_f32: XCore1000InstCategory.FLOAT_ARITH,
  fma_f32: XCore1000InstCategory.FLOAT_ARITH,
  fmac_f32: XCore1000InstCategory.FLOAT_ARITH,
  rcp_f32: XCore1000InstCategory.FLOAT_ARITH,
  rcpi_f32: XCore1000InstCategory.FLOAT_ARITH,
  min_f32: XCore1000InstCategory.FLOAT_ARITH,
  max_f32: XCore1000InstCategory.FLOAT_ARITH,
  div_scale_f32: XCore1000InstCategory.FLOAT_ARITH,
  div_fmas_f32: XCore1000InstCategory.FLOAT_ARITH,
  div_fixup_f32: XCore1000InstCategory.FLOAT_ARITH,
  // Double arithmetic
  add_f64: XCore1000InstCategory.FLOAT_ARITH,
  mul_f64: XCore1000InstCategory.FLOAT_ARITH,
  fma_f64: XCore1000InstCategory.FLOAT_ARITH,
  fmac_f64: XCore1000InstCategory.FLOAT_ARITH,
  div_scale_f64: XCore1000InstCategory.FLOAT_ARITH,
  div_fmas_f64: XCore1000InstCategory.FLOAT_ARITH,
  div_fixup_f64: XCore1000InstCategory.FLOAT_ARITH,
  // Conversion
  cvt_i32tof32: XCore1000InstCategory.CONVERSION,
  cvt_f32toi32: XCore1000InstCategory.CONVERSION,
  cvt_u32tof32: XCore1000InstCategory.CONVERSION,
  cvt_f32tou32: XCore1000InstCategory.CONVERSION,
  cvt_f32tof64: XCore1000InstCategory.CONVERSION,
  cvt_f64tof32: XCore1000InstCategory.CONVERSION,
  // Comparison
  cmp_lt_i32: XCore1000InstCategory.COMPARISON,
  cmp_gt_i32: XCore1000InstCategory.COMPARISON,
  cmp_ge_u32: XCore1000InstCategory.COMPARISON,
  cmp_eq_u32: XCore1000InstCategory.COMPARISON,
  cmp_lg_u32: XCore1000InstCategory.COMPARISON,
  cmp_eq_u64: XCore1000InstCategory.COMPARISON,
  scmp_lt_i32: XCore1000InstCategory.COMPARISON,
  scmp_eq_u64: XCore1000InstCategory.COMPARISON,
  scmp_ne_u64: XCore1000InstCategory.COMPARISON,
  // Select
  csel_b32: XCore1000InstCategory.SELECT,
  // Global memory
  ldg_b32: XCore1000InstCategory.MEMORY_GLOBAL,
  stg_b32: XCore1000InstCategory.MEMORY_GLOBAL,
  stg_b64: XCore1000InstCategory.MEMORY_GLOBAL,
  stg_b96: XCore1000InstCategory.MEMORY_GLOBAL,
  stg_b128: XCore1000InstCategory.MEMORY_GLOBAL,
  // Shared memory
  lds_b32: XCore1000InstCategory.MEMORY_SHARED,
  sts_b32: XCore1000InstCategory.MEMORY_SHARED,
  // Kernarg loads
  ldu_b32: XCore1000InstCategory.MEMORY_GLOBAL,
  ldu_b64: XCore1000InstCategory.MEMORY_GLOBAL,
  ldu_b128: XCore1000InstCategory.MEMORY_GLOBAL,
  ldu_b256: XCore1000InstCategory.MEMORY_GLOBAL,
  // Atomic
  atom_global_add_i32: XCore1000InstCategory.ATOMIC,
  // Barrier
  barrier: XCore1000InstCategory.BARRIER,
  arrive: XCore1000InstCategory.BARRIER,
  // Control flow
  bra: XCore1000InstCategory.CONTROL_FLOW,
  bra_smsks: XCore1000InstCategory.CONTROL_FLOW,
  bra_smskz: XCore1000InstCategory.CONTROL_FLOW,
  bra_xmskz: XCore1000InstCategory.CONTROL_FLOW,
  // Warp
  mbcnt_lo_b32: XCore1000InstCategory.WARP,
  mbcnt_hi_b32: XCore1000InstCategory.WARP,
  sm_bperm_b32: XCore1000InstCategory.WARP,
  and_xmsk: XCore1000InstCategory.WARP,
  // Special
  mov_b32: XCore1000InstCategory.SPECIAL,
  smov_b32: XCore1000InstCategory.SPECIAL,
  snop: XCore1000InstCategory.SPECIAL,
  endk: XCore1000InstCategory.SPECIAL,
  trap: XCore1000InstCategory.SPECIAL,
};

/** Pipeline stages for each core's state machine */
export enum PipelineStage {
  FETCH   = 'FETCH',
  DECODE  = 'DECODE',
  REQUEST = 'REQUEST',
  WAIT    = 'WAIT',
  EXECUTE = 'EXECUTE',
  UPDATE  = 'UPDATE',
  BARRIER = 'BARRIER',
  DONE    = 'DONE',
}

/** Per-thread execution state (tiny-gpu) */
export interface ThreadState {
  threadId: number;
  blockId: number;
  pc: number;
  registers: number[];      // R0-R15
  nzp: number;              // 3-bit NZP flags
  stage: PipelineStage;
  done: boolean;
  currentInstruction: string;
  divergent?: boolean;       // true if this thread diverged from majority
}

/** Per-thread execution state (xcore1000) */
export interface XCore1000ThreadState {
  threadId: number;
  blockId: number;
  pc: number;
  vgprs: number[];           // Vector GPRs (r0-r31), 32-bit each
  sgprs: number[];           // Scalar GPRs (s0-s31), 32-bit each (shared within warp)
  execMask: bigint;          // 64-bit execution mask
  cmpFlag: boolean;          // Scalar comparison flag
  done: boolean;
  currentInstruction: string;
  divergent?: boolean;
}

/** Full GPU simulation state at one cycle */
export interface SimulationState {
  cycle: number;
  threads: ThreadState[];
  memory: number[];          // 256-byte data memory
  sharedMemory: number[];    // 64-byte shared memory per block
  currentBlock: number;
  totalBlocks: number;
}

/** Full GPU simulation state for xcore1000 */
export interface XCore1000SimulationState {
  cycle: number;
  warpSize: number;          // 64 threads per warp
  warps: XCore1000ThreadState[][];
  globalMemory: Float32Array;
  sharedMemory: Float32Array; // BSM per block
  currentBlock: number;
  totalBlocks: number;
}
