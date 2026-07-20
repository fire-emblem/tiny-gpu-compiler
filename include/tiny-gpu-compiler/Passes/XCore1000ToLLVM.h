#ifndef TGC_PASSES_XCORE1000_TO_LLVM_H
#define TGC_PASSES_XCORE1000_TO_LLVM_H

#include "mlir/IR/BuiltinOps.h"

namespace tgc {

/// Lower XCore1000 dialect ops to LLVM dialect ops.
/// After this pass, the module contains only LLVM dialect operations
/// that can be translated to LLVM IR bitcode via mlir-translate.
/// Returns true on success.
bool lowerXCore1000ToLLVM(mlir::ModuleOp module);

/// Emit LLVM IR bitcode from an LLVM dialect module to a file.
/// Uses mlir-translate to convert MLIR LLVM dialect → LLVM IR bitcode.
/// Returns true on success.
bool emitLLVMBitcode(mlir::ModuleOp module, llvm::raw_ostream &os);

} // namespace tgc

#endif // TGC_PASSES_XCORE1000_TO_LLVM_H
