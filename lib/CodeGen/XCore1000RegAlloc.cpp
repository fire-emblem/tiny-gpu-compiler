#include "tiny-gpu-compiler/CodeGen/XCore1000RegAlloc.h"
#include "tiny-gpu-compiler/Dialect/XCore1000/XCore1000Ops.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;

namespace tgc {

constexpr int NUM_VGPRS = 32;  // r0-r31
constexpr int NUM_SGPRS = 32;  // s0-s31

/// Determine which register class a value should use based on the defining op.
/// SGPR: constants, kernel arg loads, block/thread IDs, comparisons, addresses
/// VGPR: arithmetic results, memory load results, conversions
static XCoreRegClass getRegClassForValue(Value val) {
  Operation *defOp = val.getDefiningOp();
  if (!defOp)
    return XCoreRegClass::VGPR;

  // SGPR ops: constants, thread/block IDs, comparisons
  if (isa<xcore::ConstIOp, xcore::ConstFOp>(defOp))
    return XCoreRegClass::SGPR;
  if (isa<xcore::ThreadIdOp, xcore::BlockIdOp, xcore::BlockDimOp>(defOp))
    return XCoreRegClass::VGPR; // Thread ID is per-thread → VGPR
  if (isa<xcore::CmpIOp, xcore::CmpFOp>(defOp))
    return XCoreRegClass::SGPR; // Comparison result is scalar flag

  // VGPR ops: arithmetic, memory loads, conversions
  return XCoreRegClass::VGPR;
}

/// Determine if an operation's result needs a register pair (64-bit).
static bool isPairType(Type t) {
  return t.isInteger(64) || t.isF64();
}

bool allocateXCore1000Registers(Operation *funcOp) {
  // Collect all operations in order
  std::vector<Operation *> ops;
  funcOp->walk([&](Operation *child) {
    if (child != funcOp)
      ops.push_back(child);
  });

  // Build last-use map
  llvm::DenseMap<Value, int> lastUse;
  for (int i = 0; i < (int)ops.size(); i++) {
    for (Value operand : ops[i]->getOperands()) {
      lastUse[operand] = i;
    }
  }

  // Track register assignments: value → (class, number)
  llvm::DenseMap<Value, XCoreRegAlloc> regMap;

  // Track register availability
  std::vector<bool> vgprInUse(NUM_VGPRS, false);
  std::vector<bool> sgprInUse(NUM_SGPRS, false);

  // Reserve s0 for kernarg pointer, s1 for dispatch pointer
  sgprInUse[0] = true;
  sgprInUse[1] = true;

  auto allocReg = [&](XCoreRegClass cls, bool needPair) -> int {
    auto &inUse = (cls == XCoreRegClass::VGPR) ? vgprInUse : sgprInUse;
    int limit = (cls == XCoreRegClass::VGPR) ? NUM_VGPRS : NUM_SGPRS;

    if (needPair) {
      // Find two consecutive free registers
      for (int r = 0; r < limit - 1; r++) {
        if (!inUse[r] && !inUse[r + 1]) {
          inUse[r] = true;
          inUse[r + 1] = true;
          return r;
        }
      }
      return -1; // No consecutive pair available
    }

    for (int r = 0; r < limit; r++) {
      if (!inUse[r]) {
        inUse[r] = true;
        return r;
      }
    }
    return -1;
  };

  auto freeReg = [&](XCoreRegClass cls, int reg, bool isPair) {
    auto &inUse = (cls == XCoreRegClass::VGPR) ? vgprInUse : sgprInUse;
    if (reg >= 0 && reg < (int)inUse.size()) {
      inUse[reg] = false;
      if (isPair && reg + 1 < (int)inUse.size())
        inUse[reg + 1] = false;
    }
  };

  // Allocate registers for each operation's results
  for (int i = 0; i < (int)ops.size(); i++) {
    Operation *currOp = ops[i];

    // Assign registers to results
    for (Value result : currOp->getResults()) {
      XCoreRegClass cls = getRegClassForValue(result);
      bool needPair = isPairType(result.getType());

      // Special cases
      if (isa<xcore::ThreadIdOp>(currOp)) {
        // Thread ID uses r0 (hardware-provided)
        XCoreRegAlloc alloc{XCoreRegClass::VGPR, 0, false, 0};
        regMap[result] = alloc;
        currOp->setAttr("vgpr_rd",
            IntegerAttr::get(IntegerType::get(currOp->getContext(), 32), 0));
        continue;
      }

      int reg = allocReg(cls, needPair);
      if (reg < 0) {
        currOp->emitError("register allocation failed: no free "
            + std::string(cls == XCoreRegClass::VGPR ? "VGPR" : "SGPR"));
        return false;
      }

      XCoreRegAlloc alloc{cls, reg, needPair, needPair ? reg + 1 : 0};
      regMap[result] = alloc;

      // Annotate the operation
      std::string attrName = (cls == XCoreRegClass::VGPR) ? "vgpr_rd" : "sgpr_sd";
      currOp->setAttr(attrName,
          IntegerAttr::get(IntegerType::get(currOp->getContext(), 32), reg));
    }

    // Annotate operand registers
    for (int opIdx = 0; opIdx < (int)currOp->getNumOperands(); opIdx++) {
      Value operand = currOp->getOperand(opIdx);
      auto it = regMap.find(operand);
      if (it != regMap.end()) {
        std::string attrName;
        if (it->second.regClass == XCoreRegClass::VGPR) {
          attrName = (opIdx == 0) ? "vgpr_rs" : "vgpr_rt";
        } else {
          attrName = (opIdx == 0) ? "sgpr_ss" : "sgpr_st";
        }
        currOp->setAttr(attrName,
            IntegerAttr::get(IntegerType::get(currOp->getContext(), 32),
                             it->second.regNum));
      }
    }

    // Free registers whose values are no longer needed
    for (Value operand : currOp->getOperands()) {
      auto useIt = lastUse.find(operand);
      if (useIt != lastUse.end() && useIt->second == i) {
        auto regIt = regMap.find(operand);
        if (regIt != regMap.end()) {
          freeReg(regIt->second.regClass, regIt->second.regNum,
                  regIt->second.isPair);
        }
      }
    }
  }

  return true;
}

std::pair<int, int> getXCore1000RegPressure(Operation *funcOp) {
  int maxVgpr = 0, maxSgpr = 0;

  funcOp->walk([&](Operation *op) {
    if (auto rd = op->getAttrOfType<IntegerAttr>("vgpr_rd"))
      maxVgpr = std::max(maxVgpr, rd.getInt());
    if (auto sd = op->getAttrOfType<IntegerAttr>("sgpr_sd"))
      maxSgpr = std::max(maxSgpr, sd.getInt());
  });

  return {maxVgpr, maxSgpr};
}

} // namespace tgc
