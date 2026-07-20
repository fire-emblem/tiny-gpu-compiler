#ifndef TGC_PASSES_XCORE1000_LOWERING_H
#define TGC_PASSES_XCORE1000_LOWERING_H

#include "mlir/IR/BuiltinOps.h"

namespace tgc {

/// Lower a TinyGPU dialect module to XCore1000 dialect.
/// This converts all tinygpu.* ops to xcore.* ops, widening i8 values to i32.
/// Returns true on success, false on failure.
bool lowerTinyGPUToXCore1000(mlir::ModuleOp module);

} // namespace tgc

#endif // TGC_PASSES_XCORE1000_LOWERING_H
