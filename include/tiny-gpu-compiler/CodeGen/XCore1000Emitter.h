#ifndef XCORE1000_EMITTER_H
#define XCORE1000_EMITTER_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>

namespace tgc {

/// xcore1000 instruction representation
struct XCore1000Instruction {
  int address;              // Byte address within kernel
  std::string mnemonic;     // Assembly mnemonic (e.g., "add_u32", "ldg_b32")
  std::string operands;     // Full operand string
  std::string assembly;     // Full assembly line (mnemonic + operands)
  uint32_t binary;          // 32-bit encoded instruction (0 if not encodable)
};

/// Kernel metadata (maps to .macahca_kernel directives)
struct XCore1000KernelMeta {
  std::string name;
  int mtregSize;            // VGPR count
  int stregSize;            // SGPR count
  int userStregCount;       // User-visible SGPRs (kernel args)
  int bsmSize;              // Shared memory enabled (0 or 1)
  int codeLenInBytes;       // Total code size
  bool blockIdXOn;          // Hardware block_id.x enabled
  bool kernargPtrOn;        // Kernarg segment pointer enabled
  bool dispatchPtrOn;       // Dispatch pointer enabled
};

/// Emit xcore1000 assembly for a kernel function
void emitXCore1000Assembly(Operation *funcOp, llvm::raw_ostream &os);

/// Emit xcore1000 assembly as a vector of instructions
std::vector<XCore1000Instruction> emitXCore1000(Operation *funcOp);

/// Emit kernel metadata
void emitKernelMeta(const XCore1000KernelMeta &meta, llvm::raw_ostream &os);

/// Emit full .s file with device bundle markers
void emitXCore1000FullAssembly(mlir::ModuleOp module, llvm::raw_ostream &os);

} // namespace tgc

#endif // XCORE1000_EMITTER_H
