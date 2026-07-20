#ifndef XCORE1000_REG_ALLOC_H
#define XCORE1000_REG_ALLOC_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"

#include <string>
#include <vector>

namespace tgc {

/// Register class for xcore1000
enum class XCoreRegClass {
  VGPR,  // Vector GPR (r0-r31) — per-thread, for computation
  SGPR,  // Scalar GPR (s0-s31) — per-warp, for addresses/constants
};

/// A register allocation result for one value
struct XCoreRegAlloc {
  XCoreRegClass regClass;
  int regNum;
  bool isPair;    // true if this is a 64-bit value using two registers
  int regNumHi;   // high register for pair (regNum is low)
};

/// Allocate VGPRs and SGPRs for an XCore1000 kernel function.
/// Annotates each operation with "vgpr_rd", "sgpr_sd", "vgpr_rs", "sgpr_ss"
/// attributes indicating which registers to use.
/// Returns true on success.
bool allocateXCore1000Registers(mlir::Operation *funcOp);

/// Get the maximum VGPR and SGPR used after allocation.
/// Returns {maxVgpr, maxSgpr}.
std::pair<int, int> getXCore1000RegPressure(mlir::Operation *funcOp);

} // namespace tgc

#endif // XCORE1000_REG_ALLOC_H
