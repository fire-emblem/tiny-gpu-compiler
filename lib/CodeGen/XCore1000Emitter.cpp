#include "tiny-gpu-compiler/CodeGen/XCore1000Emitter.h"
#include "tiny-gpu-compiler/Dialect/XCore1000/XCore1000Ops.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Format.h"

using namespace mlir;

namespace tgc {

//===----------------------------------------------------------------------===//
// Register name helpers
//===----------------------------------------------------------------------===//

/// Convert a register number to xcore1000 assembly name.
/// VGPRs: r0-r31, SGPRs: s0-s31
/// The register class is determined by the "reg_class" attribute on the op.
static std::string vgprName(int reg) {
  return "r" + std::to_string(reg);
}

static std::string sgprName(int reg) {
  return "s" + std::to_string(reg);
}

/// Get integer attribute value from an operation, or default.
static int getAttrInt(Operation *op, StringRef name, int defaultVal = 0) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(name))
    return attr.getInt();
  return defaultVal;
}

/// Get register class for an operation's result.
/// Returns "vgpr" or "sgpr" based on the operation type.
enum class RegClass { VGPR, SGPR };

static RegClass getRegClass(Operation *op) {
  // SGPR operations: constants, kernel arg loads, block/thread ID, comparisons
  if (isa<xcore::ConstIOp, xcore::ConstFOp>(op))
    return RegClass::SGPR;
  // VGPR operations: arithmetic, memory results, conversions
  return RegClass::VGPR;
}

//===----------------------------------------------------------------------===//
// Instruction emission for individual operations
//===----------------------------------------------------------------------===//

static void emitOp(Operation *op, std::vector<XCore1000Instruction> &insts,
                   int &addr, llvm::DenseMap<Value, int> &regMap,
                   int &nextVgpr, int &nextSgpr) {

  auto addInst = [&](const std::string &asm_str) {
    XCore1000Instruction inst;
    inst.address = addr;
    inst.assembly = asm_str;
    inst.binary = 0; // encoding not implemented yet
    addr += 4;
    insts.push_back(inst);
  };

  auto getVgpr = [&](Value v) -> int {
    auto it = regMap.find(v);
    return (it != regMap.end()) ? it->second : 0;
  };

  auto allocVgpr = [&](Value v) -> int {
    int r = nextVgpr++;
    regMap[v] = r;
    return r;
  };

  auto allocSgpr = [&](Value v) -> int {
    int s = nextSgpr++;
    regMap[v] = s;
    return s;
  };

  // --- Kernel function ---
  if (auto funcOp = dyn_cast<xcore::FuncOp>(op)) {
    // emitOp is called per-op from the flat walk, FuncOp itself is handled
    // in emitXCore1000(). Skip here.
    return;
  }

  // --- Thread/Block ID ---
  if (auto tidOp = dyn_cast<xcore::ThreadIdOp>(op)) {
    // Thread ID comes from hardware r0 register, masked to 10 bits
    int rd = allocVgpr(tidOp.getResult());
    addInst("and_b32 " + vgprName(rd) + ", 0x3ff, r0");
    return;
  }

  if (auto bidOp = dyn_cast<xcore::BlockIdOp>(op)) {
    // Block ID loaded from dispatch pointer (s0+0x10 for blockIdx.x)
    int sd = allocSgpr(bidOp.getResult());
    addInst("ldu_b32 " + sgprName(sd) + ", s0, 0x10");
    return;
  }

  if (auto bdimOp = dyn_cast<xcore::BlockDimOp>(op)) {
    // Block dimension loaded from dispatch pointer (s0+0x4 for blockDim.x)
    int sd = allocSgpr(bdimOp.getResult());
    addInst("ldu_b32 " + sgprName(sd) + ", s0, 0x4");
    return;
  }

  // --- Constants ---
  if (auto constOp = dyn_cast<xcore::ConstIOp>(op)) {
    int sd = allocSgpr(constOp.getResult());
    int val = constOp.getValue();
    addInst("mov_b32 " + sgprName(sd) + ", " + std::to_string(val));
    return;
  }

  if (auto constOp = dyn_cast<xcore::ConstFOp>(op)) {
    int sd = allocSgpr(constOp.getResult());
    float val = constOp.getValue().convertToFloat();
    // Emit as hex for IEEE754 encoding
    uint32_t bits;
    memcpy(&bits, &val, 4);
    std::string hexStr;
    llvm::raw_string_ostream hexOs(hexStr);
    hexOs << llvm::format_hex_no_prefix(bits, 8);
    addInst("mov_b32 " + sgprName(sd) + ", 0x" + hexOs.str());
    return;
  }

  // --- Integer arithmetic ---
  if (auto addOp = dyn_cast<xcore::AddIOp>(op)) {
    int rd = allocVgpr(addOp.getResult());
    int rs = getVgpr(addOp.getLhs());
    int rt = getVgpr(addOp.getRhs());
    addInst("add_u32 " + vgprName(rd) + ", " + sgprName(rs) + ", " +
            sgprName(rt));
    return;
  }

  if (auto subOp = dyn_cast<xcore::SubIOp>(op)) {
    int rd = allocVgpr(subOp.getResult());
    int rs = getVgpr(subOp.getLhs());
    int rt = getVgpr(subOp.getRhs());
    addInst("sub_u32 " + vgprName(rd) + ", " + sgprName(rs) + ", " +
            sgprName(rt));
    return;
  }

  if (auto mulOp = dyn_cast<xcore::MulIOp>(op)) {
    int rd = allocVgpr(mulOp.getResult());
    int rs = getVgpr(mulOp.getLhs());
    int rt = getVgpr(mulOp.getRhs());
    addInst("mul_u32 " + vgprName(rd) + ", " + sgprName(rs) + ", " +
            sgprName(rt));
    return;
  }

  if (auto madOp = dyn_cast<xcore::MadIOp>(op)) {
    int rd = allocVgpr(madOp.getResult());
    int rs = getVgpr(madOp.getLhs());
    int rt = getVgpr(madOp.getRhs());
    int ru = getVgpr(madOp.getAddend());
    addInst("mad_i32 " + vgprName(rd) + ", " + sgprName(rs) + ", " +
            sgprName(rt) + ", " + sgprName(ru));
    return;
  }

  if (auto divOp = dyn_cast<xcore::DivIOp>(op)) {
    int rd = allocVgpr(divOp.getResult());
    int rs = getVgpr(divOp.getLhs());
    int rt = getVgpr(divOp.getRhs());
    // Integer division uses float conversion path
    addInst("cvt_u32tof32 r_tmp, " + sgprName(rs));
    addInst("rcpi_f32 r_tmp, r_tmp");
    addInst("mul_f32 r_tmp, 0x4f7ffffe, r_tmp");
    addInst("cvt_f32tou32 " + vgprName(rd) + ", r_tmp");
    // Simplified — real implementation would need the full multi-step sequence
    return;
  }

  // --- Bitwise operations ---
  if (auto andOp = dyn_cast<xcore::AndIOp>(op)) {
    int rd = allocVgpr(andOp.getResult());
    int rs = getVgpr(andOp.getLhs());
    int rt = getVgpr(andOp.getRhs());
    addInst("and_b32 " + vgprName(rd) + ", " + sgprName(rs) + ", " +
            sgprName(rt));
    return;
  }

  if (auto orOp = dyn_cast<xcore::OrIOp>(op)) {
    int rd = allocVgpr(orOp.getResult());
    int rs = getVgpr(orOp.getLhs());
    int rt = getVgpr(orOp.getRhs());
    addInst("sor_b32 " + vgprName(rd) + ", " + sgprName(rs) + ", " +
            sgprName(rt));
    return;
  }

  if (auto xorOp = dyn_cast<xcore::XorIOp>(op)) {
    int rd = allocVgpr(xorOp.getResult());
    int rs = getVgpr(xorOp.getLhs());
    int rt = getVgpr(xorOp.getRhs());
    addInst("xor_b32 " + vgprName(rd) + ", " + sgprName(rs) + ", " +
            sgprName(rt));
    return;
  }

  if (auto shlOp = dyn_cast<xcore::ShlIOp>(op)) {
    int rd = allocVgpr(shlOp.getResult());
    int rs = getVgpr(shlOp.getLhs());
    int rt = getVgpr(shlOp.getRhs());
    addInst("shl_b32 " + vgprName(rd) + ", " + sgprName(rs) + ", " +
            sgprName(rt));
    return;
  }

  if (auto shrOp = dyn_cast<xcore::ShrIOp>(op)) {
    int rd = allocVgpr(shrOp.getResult());
    int rs = getVgpr(shrOp.getLhs());
    int rt = getVgpr(shrOp.getRhs());
    addInst("sshr_b32 " + vgprName(rd) + ", " + sgprName(rs) + ", " +
            sgprName(rt));
    return;
  }

  if (auto sarOp = dyn_cast<xcore::SarIOp>(op)) {
    int rd = allocVgpr(sarOp.getResult());
    int rs = getVgpr(sarOp.getLhs());
    int rt = getVgpr(sarOp.getRhs());
    addInst("sashr_i32 " + vgprName(rd) + ", " + sgprName(rs) + ", " +
            sgprName(rt));
    return;
  }

  // --- Float arithmetic ---
  if (auto addOp = dyn_cast<xcore::AddFOp>(op)) {
    int rd = allocVgpr(addOp.getResult());
    int rs = getVgpr(addOp.getLhs());
    int rt = getVgpr(addOp.getRhs());
    addInst("add_f32 " + vgprName(rd) + ", " + sgprName(rs) + ", " +
            sgprName(rt));
    return;
  }

  if (auto subOp = dyn_cast<xcore::SubFOp>(op)) {
    int rd = allocVgpr(subOp.getResult());
    int rs = getVgpr(subOp.getLhs());
    int rt = getVgpr(subOp.getRhs());
    addInst("sub_f32 " + vgprName(rd) + ", " + sgprName(rs) + ", " +
            sgprName(rt));
    return;
  }

  if (auto mulOp = dyn_cast<xcore::MulFOp>(op)) {
    int rd = allocVgpr(mulOp.getResult());
    int rs = getVgpr(mulOp.getLhs());
    int rt = getVgpr(mulOp.getRhs());
    addInst("mul_f32 " + vgprName(rd) + ", " + sgprName(rs) + ", " +
            sgprName(rt));
    return;
  }

  if (auto divOp = dyn_cast<xcore::DivFOp>(op)) {
    int rd = allocVgpr(divOp.getResult());
    int rs = getVgpr(divOp.getLhs());
    int rt = getVgpr(divOp.getRhs());
    // xcore1000 float division: 3-step sequence
    addInst("div_scale_f32 " + vgprName(rd) + ", " + sgprName(rs) + ", " +
            sgprName(rs) + ", " + sgprName(rt));
    addInst("rcp_f32 r_rcp, " + vgprName(rd));
    addInst("fmac_f32 r_rcp, neg(" + vgprName(rd) + "), r_rcp");
    addInst("fmac_f32 r_rcp, r_rcp, r_rcp");
    addInst("div_fmas_f32 " + vgprName(rd) + ", " + vgprName(rd) + ", r_rcp, " +
            vgprName(rd));
    addInst("div_fixup_f32 " + vgprName(rd) + ", " + vgprName(rd) + ", " +
            sgprName(rs) + ", " + sgprName(rt));
    return;
  }

  if (auto fmaOp = dyn_cast<xcore::FmaFOp>(op)) {
    int rd = allocVgpr(fmaOp.getResult());
    int ra = getVgpr(fmaOp.getA());
    int rb = getVgpr(fmaOp.getB());
    int rc = getVgpr(fmaOp.getC());
    addInst("fma_f32 " + vgprName(rd) + ", " + sgprName(ra) + ", " +
            sgprName(rb) + ", " + sgprName(rc));
    return;
  }

  if (auto negOp = dyn_cast<xcore::NegFOp>(op)) {
    int rd = allocVgpr(negOp.getResult());
    int rs = getVgpr(negOp.getOperand());
    addInst("sxor_b32 " + vgprName(rd) + ", 0x80000000, " + sgprName(rs));
    return;
  }

  if (auto rcpOp = dyn_cast<xcore::RcpFOp>(op)) {
    int rd = allocVgpr(rcpOp.getResult());
    int rs = getVgpr(rcpOp.getOperand());
    addInst("rcp_f32 " + vgprName(rd) + ", " + sgprName(rs));
    return;
  }

  if (auto minOp = dyn_cast<xcore::MinFOp>(op)) {
    int rd = allocVgpr(minOp.getResult());
    int rs = getVgpr(minOp.getLhs());
    int rt = getVgpr(minOp.getRhs());
    addInst("min_f32 " + vgprName(rd) + ", " + sgprName(rs) + ", " +
            sgprName(rt));
    return;
  }

  if (auto maxOp = dyn_cast<xcore::MaxFOp>(op)) {
    int rd = allocVgpr(maxOp.getResult());
    int rs = getVgpr(maxOp.getLhs());
    int rt = getVgpr(maxOp.getRhs());
    addInst("max_f32 " + vgprName(rd) + ", " + sgprName(rs) + ", " +
            sgprName(rt));
    return;
  }

  // --- Type conversion ---
  if (auto convOp = dyn_cast<xcore::IntToFloatOp>(op)) {
    int rd = allocVgpr(convOp.getResult());
    int rs = getVgpr(convOp.getOperand());
    addInst("cvt_i32tof32 " + vgprName(rd) + ", " + sgprName(rs));
    return;
  }

  if (auto convOp = dyn_cast<xcore::FloatToIntOp>(op)) {
    int rd = allocVgpr(convOp.getResult());
    int rs = getVgpr(convOp.getOperand());
    addInst("cvt_f32toi32 " + vgprName(rd) + ", " + sgprName(rs));
    return;
  }

  // --- Comparison ---
  if (auto cmpOp = dyn_cast<xcore::CmpIOp>(op)) {
    int sd = allocSgpr(cmpOp.getResult());
    int rs = getVgpr(cmpOp.getLhs());
    int rt = getVgpr(cmpOp.getRhs());
    // Default to signed less-than; real implementation would use comparison
    // type attribute
    addInst("cmp_lt_i32 " + sgprName(sd) + ", " + sgprName(rs) + ", " +
            sgprName(rt));
    return;
  }

  // --- Conditional select ---
  if (auto cselOp = dyn_cast<xcore::CselOp>(op)) {
    int rd = allocVgpr(cselOp.getResult());
    int cond = getVgpr(cselOp.getCond());
    int tv = getVgpr(cselOp.getTrueVal());
    int fv = getVgpr(cselOp.getFalseVal());
    addInst("csel_b32 " + vgprName(rd) + ", " + sgprName(tv) + ", " +
            sgprName(fv) + ", " + sgprName(cond));
    return;
  }

  // --- Memory: Global load ---
  if (auto loadOp = dyn_cast<xcore::LoadGlobalOp>(op)) {
    int rd = allocVgpr(loadOp.getResult());
    int addrReg = getVgpr(loadOp.getAddr());
    addInst("ldg_b32 " + vgprName(rd) + ", " + sgprName(addrReg) + ", 0x0");
    addInst("arrive gvmcnt(0)");
    return;
  }

  if (auto loadOp = dyn_cast<xcore::LoadGlobalFOp>(op)) {
    int rd = allocVgpr(loadOp.getResult());
    int addrReg = getVgpr(loadOp.getAddr());
    addInst("ldg_b32 " + vgprName(rd) + ", " + sgprName(addrReg) + ", 0x0");
    addInst("arrive gvmcnt(0)");
    return;
  }

  // --- Memory: Global store ---
  if (auto storeOp = dyn_cast<xcore::StoreGlobalOp>(op)) {
    int addrReg = getVgpr(storeOp.getAddr());
    int valReg = getVgpr(storeOp.getValue());
    addInst("stg_b32 " + sgprName(addrReg) + ", 0x0, " + vgprName(valReg) +
            " devc");
    return;
  }

  if (auto storeOp = dyn_cast<xcore::StoreGlobalFOp>(op)) {
    int addrReg = getVgpr(storeOp.getAddr());
    int valReg = getVgpr(storeOp.getValue());
    addInst("stg_b32 " + sgprName(addrReg) + ", 0x0, " + vgprName(valReg) +
            " devc");
    return;
  }

  // --- Memory: Shared load/store ---
  if (auto loadOp = dyn_cast<xcore::LoadSharedOp>(op)) {
    int rd = allocVgpr(loadOp.getResult());
    int addrReg = getVgpr(loadOp.getAddr());
    addInst("lds_b32 " + vgprName(rd) + ", " + vgprName(addrReg) + ", 0x0");
    return;
  }

  if (auto storeOp = dyn_cast<xcore::StoreSharedOp>(op)) {
    int addrReg = getVgpr(storeOp.getAddr());
    int valReg = getVgpr(storeOp.getValue());
    addInst("sts_b32 " + vgprName(addrReg) + ", 0x0, " + vgprName(valReg));
    return;
  }

  // --- Address calculation ---
  if (auto addrOp = dyn_cast<xcore::AddrCalcOp>(op)) {
    int rd = allocVgpr(addrOp.getResult());
    int base = getVgpr(addrOp.getBase());
    int index = getVgpr(addrOp.getIndex());
    int stride = getVgpr(addrOp.getStride());
    addInst("mad_i32 " + vgprName(rd) + ", " + sgprName(index) + ", " +
            sgprName(stride) + ", " + sgprName(base));
    return;
  }

  // --- Atomic ---
  if (auto atomOp = dyn_cast<xcore::AtomicAddOp>(op)) {
    int rd = allocVgpr(atomOp.getResult());
    int addrReg = getVgpr(atomOp.getAddr());
    int valReg = getVgpr(atomOp.getValue());
    addInst("atom_global_add_i32 " + sgprName(addrReg) + ", " +
            vgprName(valReg) + ", 0x0, " + vgprName(rd));
    return;
  }

  // --- Synchronization ---
  if (isa<xcore::BarrierOp>(op)) {
    addInst("arrive bsmcnt(0)");
    addInst("barrier");
    addInst("arrive bsmcnt(0)");
    return;
  }

  // --- Warp shuffle ---
  if (auto shflOp = dyn_cast<xcore::WarpShuffleOp>(op)) {
    int rd = allocVgpr(shflOp.getResult());
    int val = getVgpr(shflOp.getValue());
    int lane = getVgpr(shflOp.getSrcLane());
    addInst("shl_b32 " + vgprName(rd) + ", 2, " + sgprName(lane));
    addInst("and_b32 " + vgprName(rd) + ", 0x80, " + vgprName(rd));
    addInst("sm_bperm_b32 " + vgprName(rd) + ", " + vgprName(rd) +
            ", 0x0, " + vgprName(val));
    return;
  }

  // --- Control flow ---
  if (auto branchOp = dyn_cast<xcore::BranchOp>(op)) {
    int cond = getVgpr(branchOp.getCondition());
    addInst("snop 1");
    addInst("and_xmsk s_xmsk, " + sgprName(cond));
    addInst("bra_xmskz .Lskip"); // placeholder label
    return;
  }

  if (auto jumpOp = dyn_cast<xcore::JumpOp>(op)) {
    addInst("bra .Ltarget"); // placeholder label
    return;
  }

  // --- Return ---
  if (isa<xcore::ReturnOp>(op)) {
    addInst("snop 2");
    addInst("endk");
    return;
  }

  // Fallback: unknown op
  addInst("; UNKNOWN_OP: " + op->getName().getStringRef().str());
}

//===----------------------------------------------------------------------===//
// Main emission function
//===----------------------------------------------------------------------===//

std::vector<XCore1000Instruction> emitXCore1000(Operation *funcOp) {
  std::vector<XCore1000Instruction> insts;
  llvm::DenseMap<Value, int> regMap;
  int nextVgpr = 1; // r0 is used for threadIdx
  int nextSgpr = 2; // s0 = kernarg ptr, s1 = dispatch ptr

  int addr = 0;

  // Walk all operations in the function
  funcOp->walk([&](Operation *op) {
    if (op == funcOp)
      return;
    emitOp(op, insts, addr, regMap, nextVgpr, nextSgpr);
  });

  return insts;
}

//===----------------------------------------------------------------------===//
// Assembly output
//===----------------------------------------------------------------------===//

void emitXCore1000Assembly(Operation *funcOp, llvm::raw_ostream &os) {
  auto insts = emitXCore1000(funcOp);

  // Emit function label
  if (auto funcOp2 = dyn_cast<xcore::FuncOp>(funcOp)) {
    os << "\t.protected\t" << funcOp2.getName() << "\n";
    os << "\t.globl\t" << funcOp2.getName() << "\n";
    os << "\t.p2align\t8\n";
    os << "\t.type\t" << funcOp2.getName() << ",@function\n";
    os << funcOp2.getName() << ":\n";
  }

  // Emit prologue: kernarg load
  os << "; %bb.0:\t; %entry\n";
  os << "\tldu_b128 s0, s0, 0x0\n"; // Load kernarg pointers
  os << "\tarrive slcnt(0)\n";

  // Emit instructions
  for (auto &inst : insts) {
    os << "\t" << inst.assembly << "\n";
  }

  // Emit epilogue size
  os << "\t.fill 4, 8, 0x2320\n";
  os << "\t.p2align 8\n";
}

//===----------------------------------------------------------------------===//
// Kernel metadata
//===----------------------------------------------------------------------===//

void emitKernelMeta(const XCore1000KernelMeta &meta, llvm::raw_ostream &os) {
  os << "\t.section\t.rodata,#alloc\n";
  os << "\t.p2align\t6, 0x0\n";
  os << "\t.macahca_kernel " << meta.name << "\n";
  os << "\t\t.macahca_mtreg_size " << meta.mtregSize << "\n";
  os << "\t\t.macahca_streg_size " << meta.stregSize << "\n";
  os << "\t\t.macahca_priority 0\n";
  os << "\t\t.macahca_float_mode 240\n";
  os << "\t\t.macahca_privilege 0\n";
  os << "\t\t.macahca_debug_mode 0\n";
  os << "\t\t.macahca_ieee_mode 1\n";
  os << "\t\t.macahca_bulky_mode 0\n";
  os << "\t\t.macahca_fp16_overflow_mode 0\n";
  os << "\t\t.macahca_perf_on 0\n";
  os << "\t\t.macahca_priv_segm_on 0\n";
  os << "\t\t.macahca_user_streg_count " << meta.userStregCount << "\n";
  os << "\t\t.macahca_user_trap_handler_on 0\n";
  os << "\t\t.macahca_streg_block_id_x_on "
     << (meta.blockIdXOn ? 1 : 0) << "\n";
  os << "\t\t.macahca_streg_block_id_y_on 0\n";
  os << "\t\t.macahca_streg_block_id_z_on 0\n";
  os << "\t\t.macahca_streg_block_info_on 0\n";
  os << "\t\t.macahca_enable_mtreg_thread_id 0\n";
  os << "\t\t.macahca_exception_address_watch_on 0\n";
  os << "\t\t.macahca_exception_memory_violation_on 0\n";
  os << "\t\t.macahca_bsm_size " << meta.bsmSize << "\n";
  os << "\t\t.macahca_exception_on 0\n";
  os << "\t\t.macahca_streg_kernarg_segm_ptr_on "
     << (meta.kernargPtrOn ? 1 : 0) << "\n";
  os << "\t\t.macahca_streg_dispatch_ptr_on "
     << (meta.dispatchPtrOn ? 1 : 0) << "\n";
  os << "\t\t.macahca_streg_prvt_init_on 0\n";
  os << "\t\t.macahca_streg_prvt_segm_size_on 0\n";
  os << "\t\t.macahca_unorder_dispatch_on 0\n";
  os << "\t\t.macahca_warps_per_dpc 416\n";
  os << "\t\t.macahca_blocks_per_ap 16\n";
  os << "\t.end_macahca_kernel\n";
}

//===----------------------------------------------------------------------===//
// Full assembly output with bundle markers
//===----------------------------------------------------------------------===//

void emitXCore1000FullAssembly(mlir::ModuleOp module, llvm::raw_ostream &os) {
  os << "\n# __CLANG_OFFLOAD_BUNDLE____START__ maca-mxc-metax-macahca--"
        "xcore1000\n";
  os << "\t.text\n";

  for (auto &op : module.getBody()->getOperations()) {
    if (isa<xcore::FuncOp>(&op)) {
      emitXCore1000Assembly(&op, os);

      // Count VGPRs and SGPRs used
      int maxVgpr = 0, maxSgpr = 0;
      op.walk([&](Operation *child) {
        if (auto rd = child->getAttrOfType<IntegerAttr>("vgpr_rd")) {
          int rdInt = static_cast<int>(rd.getInt());
          if (rdInt > maxVgpr) maxVgpr = rdInt;
        }
        if (auto sd = child->getAttrOfType<IntegerAttr>("sgpr_sd")) {
          int sdInt = static_cast<int>(sd.getInt());
          if (sdInt > maxSgpr) maxSgpr = sdInt;
        }
      });

      XCore1000KernelMeta meta;
      if (auto funcOp2 = dyn_cast<xcore::FuncOp>(&op))
        meta.name = funcOp2.getName().str();
      meta.mtregSize = maxVgpr + 1;
      meta.stregSize = maxSgpr + 1;
      meta.userStregCount = 4;
      meta.bsmSize = 0;
      meta.codeLenInBytes = 0; // computed after emission
      meta.blockIdXOn = true;
      meta.kernargPtrOn = true;
      meta.dispatchPtrOn = false;
      emitKernelMeta(meta, os);
    }
  }

  os << "\n# __CLANG_OFFLOAD_BUNDLE____END__ maca-mxc-metax-macahca--"
        "xcore1000\n";
}

} // namespace tgc
