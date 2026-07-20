#ifndef TGC_PIPELINE_PIPELINE_H
#define TGC_PIPELINE_PIPELINE_H

#include "tiny-gpu-compiler/CodeGen/TinyGPUEmitter.h"

#include "mlir/IR/BuiltinOps.h"

#include <string>

namespace tgc {

enum class OutputFormat {
  MLIR,       // Dump the dialect MLIR
  Assembly,   // Human-readable annotated assembly
  Hex,        // Hex text (one instruction per line)
  Binary,     // Raw binary
  Bitcode,    // LLVM IR text (for mxcc compilation)
  CUDA,       // CUDA source (for mxcc compilation and GPU execution)
  JsonTrace,  // Full compilation trace for visualizer
};

enum class Target {
  TinyGPU,    // tiny-gpu 16-bit ISA (default)
  XCore1000,  // MetaX xcore1000 32-bit ISA
};

struct CompilerOptions {
  OutputFormat format = OutputFormat::Assembly;
  Target target = Target::TinyGPU;
  bool dumpAfterEachPass = false;
};

/// Run the full compilation pipeline: parse .tgc source, generate MLIR,
/// allocate registers, and emit binary.
/// Returns the compilation trace (populated if format == JsonTrace).
CompilationTrace compile(const std::string &source,
                         const CompilerOptions &opts,
                         llvm::raw_ostream &os);

} // namespace tgc

#endif // TGC_PIPELINE_PIPELINE_H
