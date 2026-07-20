#include "tiny-gpu-compiler/Pipeline/Pipeline.h"
#include "tiny-gpu-compiler/CodeGen/RegisterAllocator.h"
#include "tiny-gpu-compiler/CodeGen/TinyGPUEmitter.h"
#include "tiny-gpu-compiler/CodeGen/XCore1000Emitter.h"
#include "tiny-gpu-compiler/CodeGen/XCore1000RegAlloc.h"
#include "tiny-gpu-compiler/Dialect/TinyGPU/TinyGPUDialect.h"
#include "tiny-gpu-compiler/Dialect/TinyGPU/TinyGPUOps.h"
#include "tiny-gpu-compiler/Dialect/XCore1000/XCore1000Dialect.h"
#include "tiny-gpu-compiler/Dialect/XCore1000/XCore1000Ops.h"
#include "tiny-gpu-compiler/Frontend/Lexer.h"
#include "tiny-gpu-compiler/Frontend/MLIRGen.h"
#include "tiny-gpu-compiler/Frontend/Parser.h"
#include "tiny-gpu-compiler/Passes/Passes.h"
#include "tiny-gpu-compiler/Passes/XCore1000Lowering.h"
#include "tiny-gpu-compiler/Passes/XCore1000ToLLVM.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

using namespace mlir;

namespace tgc {

/// Capture the current module IR as a string.
static std::string captureIR(ModuleOp module) {
  std::string ir;
  llvm::raw_string_ostream stream(ir);
  module->print(stream);
  return ir;
}

/// Analyze the compiled instructions for divergence and coalescing patterns.
static void analyzeInstructions(tgc::CompilationTrace &trace) {
  auto &analysis = trace.analysis;
  analysis.totalInstructions = trace.instructions.size();

  for (auto &inst : trace.instructions) {
    uint16_t binary = inst.binary;
    uint16_t opcode = (binary >> 12) & 0xF;

    switch (opcode) {
    case 0b0001: // BRnzp
      analysis.branchInstructions++;
      {
        AnalysisResult::DivergenceInfo div;
        div.instructionAddr = inst.address;
        div.type = "branch";
        div.divergentThreads = 0; // static estimate
        div.totalThreads = 0;
        analysis.divergence.push_back(div);
      }
      break;
    case 0b0111: // LDR
    case 0b1000: // STR
    case 0b1010: // SLDR
    case 0b1011: // SSTR
      analysis.memoryInstructions++;
      {
        AnalysisResult::CoalescingInfo coal;
        coal.instructionAddr = inst.address;
        // Heuristic: if the assembly mentions threadIdx-derived register,
        // it's likely coalesced
        if (inst.assembly.find("R0") != std::string::npos ||
            inst.assembly.find("%threadIdx") != std::string::npos) {
          coal.accessPattern = "coalesced";
          coal.transactionsNeeded = 1;
          coal.description = "Sequential thread access (1 transaction)";
        } else {
          coal.accessPattern = "likely_coalesced";
          coal.transactionsNeeded = 1;
          coal.description = "Register-based access";
        }
        if (opcode == 0b1010 || opcode == 0b1011) {
          coal.description += " [shared memory - low latency]";
        }
        analysis.coalescing.push_back(coal);
      }
      break;
    case 0b1100: // BAR
      analysis.barrierCount++;
      break;
    default:
      if (opcode != 0b1111 && opcode != 0b1001 && opcode != 0b0000)
        analysis.computeInstructions++;
      break;
    }
  }
}

/// Run the TinyGPU backend pipeline (existing 16-bit ISA)
static CompilationTrace compileTinyGPU(const std::string &source,
                                       const CompilerOptions &opts,
                                       llvm::raw_ostream &os) {
  CompilationTrace trace;
  trace.sourceCode = source;

  MLIRContext context;
  context.getOrLoadDialect<tinygpu::TinyGPUDialect>();

  // Stage 1: Frontend — Parse .tgc source to AST
  Lexer lexer(source);
  Parser parser(lexer);
  auto program = parser.parseProgram();

  // Stage 2: MLIRGen — AST to MLIR (TinyGPU dialect)
  auto module = mlirGen(context, *program);
  if (!module) {
    llvm::errs() << "MLIRGen failed\n";
    return trace;
  }

  trace.irStages.push_back(
      {"Frontend \xe2\x86\x92 TinyGPU Dialect", captureIR(*module)});

  if (opts.format == OutputFormat::MLIR) {
    module->print(os);
    return trace;
  }

  // Stage 2.5: Optimization Passes
  OptimizationStats optStats;
  for (auto &op : module->getBody()->getOperations()) {
    if (isa<tinygpu::FuncOp>(&op)) {
      optStats = runAllOptimizations(&op);
    }
  }

  trace.irStages.push_back(
      {"Optimization Passes", captureIR(*module)});
  trace.analysis.optimizationSummary = optStats.summary();

  // Stage 3: Register Allocation
  for (auto &op : module->getBody()->getOperations()) {
    if (isa<tinygpu::FuncOp>(&op)) {
      if (failed(allocateRegisters(&op))) {
        llvm::errs() << "Register allocation failed\n";
        return trace;
      }
    }
  }

  // Count registers used
  int maxReg = 0;
  for (auto &op : module->getBody()->getOperations()) {
    if (isa<tinygpu::FuncOp>(&op)) {
      op.walk([&](Operation *child) {
        if (auto rdAttr = child->getAttrOfType<IntegerAttr>("rd")) {
          int reg = rdAttr.getInt();
          if (reg < 13 && reg > maxReg)
            maxReg = reg;
        }
      });
    }
  }
  trace.analysis.registersUsed = maxReg + 1;

  trace.irStages.push_back({"Register Allocation", captureIR(*module)});

  // Stage 4: Binary Emission
  std::vector<Instruction> allInstructions;
  for (auto &op : module->getBody()->getOperations()) {
    if (isa<tinygpu::FuncOp>(&op)) {
      auto instructions = emitBinary(&op);
      allInstructions.insert(allInstructions.end(), instructions.begin(),
                             instructions.end());
    }
  }

  trace.instructions = allInstructions;

  // Run analysis on compiled output
  analyzeInstructions(trace);

  switch (opts.format) {
  case OutputFormat::Assembly:
    emitAssembly(allInstructions, os);
    break;
  case OutputFormat::Hex:
    emitHex(allInstructions, os);
    break;
  case OutputFormat::Binary:
    emitRawBinary(allInstructions, os);
    break;
  case OutputFormat::JsonTrace:
    emitJsonTrace(trace, os);
    break;
  default:
    break;
  }

  return trace;
}

/// Run the xcore1000 backend pipeline (MetaX 32-bit ISA)
static CompilationTrace compileXCore1000(const std::string &source,
                                         const CompilerOptions &opts,
                                         llvm::raw_ostream &os) {
  CompilationTrace trace;
  trace.sourceCode = source;

  MLIRContext context;
  context.getOrLoadDialect<tinygpu::TinyGPUDialect>();
  context.getOrLoadDialect<xcore::XCore1000Dialect>();
  context.getOrLoadDialect<LLVM::LLVMDialect>();

  // Stage 1: Frontend — Parse .tgc source to AST
  Lexer lexer(source);
  Parser parser(lexer);
  auto program = parser.parseProgram();

  // Stage 2: MLIRGen — AST to MLIR (TinyGPU dialect)
  auto module = mlirGen(context, *program);
  if (!module) {
    llvm::errs() << "MLIRGen failed\n";
    return trace;
  }

  trace.irStages.push_back(
      {"Frontend \xe2\x86\x92 TinyGPU Dialect", captureIR(*module)});

  if (opts.format == OutputFormat::MLIR) {
    module->print(os);
    return trace;
  }

  // Stage 2.5: Optimization Passes (on TinyGPU dialect)
  OptimizationStats optStats;
  for (auto &op : module->getBody()->getOperations()) {
    if (isa<tinygpu::FuncOp>(&op)) {
      optStats = runAllOptimizations(&op);
    }
  }

  trace.irStages.push_back(
      {"Optimization Passes", captureIR(*module)});
  trace.analysis.optimizationSummary = optStats.summary();

  // Stage 3: Lower TinyGPU → XCore1000 dialect
  if (!lowerTinyGPUToXCore1000(*module)) {
    llvm::errs() << "TinyGPU → XCore1000 lowering failed\n";
    return trace;
  }

  trace.irStages.push_back(
      {"Lowering: TinyGPU \xe2\x86\x92 XCore1000", captureIR(*module)});

  // Stage 3.5: xcore1000 Register Allocation (VGPR/SGPR dual-class)
  for (auto &op : module->getBody()->getOperations()) {
    if (isa<xcore::FuncOp>(&op)) {
      if (!allocateXCore1000Registers(&op)) {
        llvm::errs() << "xcore1000 register allocation failed\n";
        return trace;
      }
    }
  }

  // Report register pressure
  for (auto &op : module->getBody()->getOperations()) {
    if (isa<xcore::FuncOp>(&op)) {
      auto [maxVgpr, maxSgpr] = getXCore1000RegPressure(&op);
      trace.analysis.registersUsed = maxVgpr + 1;
    }
  }

  trace.irStages.push_back({"xcore1000 Register Allocation", captureIR(*module)});

  // Stage 4: Output
  if (opts.format == OutputFormat::Bitcode) {
    // Lower XCore1000 → LLVM dialect, then emit LLVM IR bitcode
    if (!lowerXCore1000ToLLVM(*module)) {
      llvm::errs() << "XCore1000 → LLVM lowering failed\n";
      return trace;
    }
    trace.irStages.push_back(
        {"XCore1000 \xe2\x86\x92 LLVM Dialect", captureIR(*module)});
    if (!emitLLVMBitcode(*module, os)) {
      llvm::errs() << "LLVM bitcode emission failed\n";
      return trace;
    }
  } else if (opts.format == OutputFormat::CUDA) {
    // Transpile .tgc DSL to CUDA source for mxcc compilation
    std::string cuda = source;

    // "kernel name(...)" → "__global__ void name(...)"
    {
      size_t pos = 0;
      while ((pos = cuda.find("kernel ", pos)) != std::string::npos) {
        cuda.replace(pos, 7, "__global__ void ");
        pos += 16;
      }
    }

    // "global int*" → "int*" (remove global qualifier)
    {
      size_t pos = 0;
      while ((pos = cuda.find("global ", pos)) != std::string::npos) {
        cuda.erase(pos, 7);
      }
    }

    // "shared int" → "__shared__ int"
    {
      size_t pos = 0;
      while ((pos = cuda.find("shared ", pos)) != std::string::npos) {
        cuda.replace(pos, 7, "__shared__ ");
        pos += 11;
      }
    }

    // "threadIdx" → "threadIdx.x"
    {
      size_t pos = 0;
      while ((pos = cuda.find("threadIdx", pos)) != std::string::npos) {
        // Only replace if not already threadIdx.x
        if (pos + 9 >= cuda.size() || cuda[pos + 9] != '.') {
          cuda.replace(pos, 9, "threadIdx.x");
          pos += 11;
        } else {
          pos += 10;
        }
      }
    }

    // "blockIdx" → "blockIdx.x"
    {
      size_t pos = 0;
      while ((pos = cuda.find("blockIdx", pos)) != std::string::npos) {
        if (pos + 8 >= cuda.size() || cuda[pos + 8] != '.') {
          cuda.replace(pos, 8, "blockIdx.x");
          pos += 10;
        } else {
          pos += 9;
        }
      }
    }

    // "blockDim" → "blockDim.x"
    {
      size_t pos = 0;
      while ((pos = cuda.find("blockDim", pos)) != std::string::npos) {
        if (pos + 8 >= cuda.size() || cuda[pos + 8] != '.') {
          cuda.replace(pos, 8, "blockDim.x");
          pos += 10;
        } else {
          pos += 9;
        }
      }
    }

    os << "// Auto-generated CUDA source from tgc --target xcore1000\n";
    os << "// Compile:  mxcc output.cu -o kernel --maca-path=/opt/maca\n";
    os << "// Run:      MACA_VISIBLE_DEVICES=1 ./kernel\n\n";
    os << cuda;
  } else if (opts.format == OutputFormat::Assembly ||
      opts.format == OutputFormat::Hex) {
    emitXCore1000FullAssembly(*module, os);
  } else if (opts.format == OutputFormat::JsonTrace) {
    // For JSON trace, emit xcore1000 assembly as the "binary" output
    std::string asmStr;
    llvm::raw_string_ostream asmOs(asmStr);
    emitXCore1000FullAssembly(*module, asmOs);
    // Wrap in a minimal JSON trace
    os << "{\n";
    os << "  \"target\": \"xcore1000\",\n";
    os << "  \"source\": \"";
    for (char c : source) {
      if (c == '"') os << "\\\"";
      else if (c == '\\') os << "\\\\";
      else if (c == '\n') os << "\\n";
      else if (c == '\t') os << "\\t";
      else os << c;
    }
    os << "\",\n";
    os << "  \"assembly\": \"";
    for (char c : asmStr) {
      if (c == '"') os << "\\\"";
      else if (c == '\\') os << "\\\\";
      else if (c == '\n') os << "\\n";
      else if (c == '\t') os << "\\t";
      else os << c;
    }
    os << "\"\n";
    os << "}\n";
  }

  return trace;
}

CompilationTrace compile(const std::string &source,
                         const CompilerOptions &opts,
                         llvm::raw_ostream &os) {
  switch (opts.target) {
  case Target::XCore1000:
    return compileXCore1000(source, opts, os);
  case Target::TinyGPU:
  default:
    return compileTinyGPU(source, opts, os);
  }
}

} // namespace tgc
