#include "tiny-gpu-compiler/Passes/XCore1000ToLLVM.h"
#include "tiny-gpu-compiler/Dialect/XCore1000/XCore1000Ops.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>

using namespace mlir;

namespace tgc {

/// Convert an XCore1000 module to LLVM dialect.
/// This generates LLVM dialect MLIR that can be translated to LLVM IR bitcode
/// and compiled by mxcc to run on the MetaX xcore1000 GPU.
bool lowerXCore1000ToLLVM(ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  OpBuilder builder(ctx);

  // Set module attributes for xcore1000 target
  module->setAttr("llvm.data_layout",
      builder.getStringAttr(
          "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64"
          "-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32"
          "-v48:64-v96:128-v192:256-v256:256-v512:512"
          "-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"));
  module->setAttr("llvm.target_triple",
      builder.getStringAttr("mxc-metax-macahca"));

  // Collect XCore1000 FuncOps
  std::vector<xcore::FuncOp> oldFuncs;
  for (auto &op : module->getBody()->getOperations()) {
    if (auto funcOp = dyn_cast<xcore::FuncOp>(&op))
      oldFuncs.push_back(funcOp);
  }

  // Map from XCore1000 values to LLVM dialect values
  llvm::DenseMap<Value, Value> vmap;

  for (auto oldFunc : oldFuncs) {
    // Create LLVM function with kernel attributes
    builder.setInsertionPoint(oldFunc);

    auto i32Ty = builder.getI32Type();
    auto i64Ty = builder.getI64Type();
    auto floatTy = builder.getF32Type();
    auto voidTy = builder.getNoneType();

    // Kernel function: void @name()
    auto funcType = LLVM::LLVMFunctionType::get(voidTy, {}, false);
    auto llvmFunc = builder.create<LLVM::LLVMFuncOp>(
        oldFunc.getLoc(), oldFunc.getName(), funcType);

    // Set kernel attributes
    llvmFunc->setAttr("passthrough", builder.getArrayAttr({
        builder.getStringAttr("metaxgpu_kernel"),
        builder.getStringAttr("noinline"),
        builder.getStringAttr("nounwind"),
    }));

    // Create entry block
    Block *entryBlock = llvmFunc.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Walk XCore1000 ops and convert to LLVM dialect
    for (Block &block : oldFunc.getBody()) {
      for (Operation &op : block) {
        // --- Thread ID ---
        if (auto tidOp = dyn_cast<xcore::ThreadIdOp>(&op)) {
          // Call @llvm.mxc.thread.id.x() intrinsic
          auto intrinsicTy = LLVM::LLVMFunctionType::get(i32Ty, {});
          auto intrinsic = LLVM::lookupOrCreateFn(
              module, "llvm.mxc.thread.id.x", {}, i32Ty);
          auto call = builder.create<LLVM::CallOp>(
              op.getLoc(), i32Ty, SymbolRefAttr::get(intrinsic));
          vmap[tidOp.getResult()] = call.getResult();
          continue;
        }

        // --- Block ID ---
        if (auto bidOp = dyn_cast<xcore::BlockIdOp>(&op)) {
          auto intrinsic = LLVM::lookupOrCreateFn(
              module, "llvm.mxc.block.id.x", {}, i32Ty);
          auto call = builder.create<LLVM::CallOp>(
              op.getLoc(), i32Ty, SymbolRefAttr::get(intrinsic));
          vmap[bidOp.getResult()] = call.getResult();
          continue;
        }

        // --- Block Dim ---
        if (auto bdimOp = dyn_cast<xcore::BlockDimOp>(&op)) {
          auto intrinsic = LLVM::lookupOrCreateFn(
              module, "llvm.mxc.block.dim.x", {}, i32Ty);
          auto call = builder.create<LLVM::CallOp>(
              op.getLoc(), i32Ty, SymbolRefAttr::get(intrinsic));
          vmap[bdimOp.getResult()] = call.getResult();
          continue;
        }

        // --- Constants ---
        if (auto constOp = dyn_cast<xcore::ConstIOp>(&op)) {
          auto val = builder.create<LLVM::ConstantOp>(
              op.getLoc(), i32Ty, constOp.getValue());
          vmap[constOp.getResult()] = val.getResult();
          continue;
        }

        // --- Integer Arithmetic ---
        if (auto addOp = dyn_cast<xcore::AddIOp>(&op)) {
          auto lhs = vmap.lookup(addOp.getLhs());
          auto rhs = vmap.lookup(addOp.getRhs());
          if (!lhs || !rhs) continue;
          auto result = builder.create<LLVM::AddOp>(op.getLoc(), lhs, rhs);
          vmap[addOp.getResult()] = result.getResult();
          continue;
        }
        if (auto subOp = dyn_cast<xcore::SubIOp>(&op)) {
          auto lhs = vmap.lookup(subOp.getLhs());
          auto rhs = vmap.lookup(subOp.getRhs());
          if (!lhs || !rhs) continue;
          auto result = builder.create<LLVM::SubOp>(op.getLoc(), lhs, rhs);
          vmap[subOp.getResult()] = result.getResult();
          continue;
        }
        if (auto mulOp = dyn_cast<xcore::MulIOp>(&op)) {
          auto lhs = vmap.lookup(mulOp.getLhs());
          auto rhs = vmap.lookup(mulOp.getRhs());
          if (!lhs || !rhs) continue;
          auto result = builder.create<LLVM::MulOp>(op.getLoc(), lhs, rhs);
          vmap[mulOp.getResult()] = result.getResult();
          continue;
        }

        // --- Float Arithmetic ---
        if (auto addOp = dyn_cast<xcore::AddFOp>(&op)) {
          auto lhs = vmap.lookup(addOp.getLhs());
          auto rhs = vmap.lookup(addOp.getRhs());
          if (!lhs || !rhs) continue;
          auto result = builder.create<LLVM::FAddOp>(op.getLoc(), lhs, rhs);
          vmap[addOp.getResult()] = result.getResult();
          continue;
        }
        if (auto subOp = dyn_cast<xcore::SubFOp>(&op)) {
          auto lhs = vmap.lookup(subOp.getLhs());
          auto rhs = vmap.lookup(subOp.getRhs());
          if (!lhs || !rhs) continue;
          auto result = builder.create<LLVM::FSubOp>(op.getLoc(), lhs, rhs);
          vmap[subOp.getResult()] = result.getResult();
          continue;
        }
        if (auto mulOp = dyn_cast<xcore::MulFOp>(&op)) {
          auto lhs = vmap.lookup(mulOp.getLhs());
          auto rhs = vmap.lookup(mulOp.getRhs());
          if (!lhs || !rhs) continue;
          auto result = builder.create<LLVM::FMulOp>(op.getLoc(), lhs, rhs);
          vmap[mulOp.getResult()] = result.getResult();
          continue;
        }

        // --- Comparison ---
        if (auto cmpOp = dyn_cast<xcore::CmpIOp>(&op)) {
          auto lhs = vmap.lookup(cmpOp.getLhs());
          auto rhs = vmap.lookup(cmpOp.getRhs());
          if (!lhs || !rhs) continue;
          // Use ICMP_SLT as default comparison
          auto result = builder.create<LLVM::ICmpOp>(
              op.getLoc(), LLVM::ICmpPredicate::slt, lhs, rhs);
          vmap[cmpOp.getResult()] = result.getResult();
          continue;
        }

        // --- Memory: Global Load ---
        if (auto loadOp = dyn_cast<xcore::LoadGlobalOp>(&op)) {
          auto addr = vmap.lookup(loadOp.getAddr());
          if (!addr) continue;
          // Cast i64 address to pointer in address space 1 (global)
          auto ptrTy = LLVM::LLVMPointerType::get(ctx, 1);
          auto ptr = builder.create<LLVM::IntToPtrOp>(op.getLoc(), ptrTy, addr);
          auto load = builder.create<LLVM::LoadOp>(
              op.getLoc(), i32Ty, ptr, /*alignment=*/4);
          vmap[loadOp.getResult()] = load.getResult();
          continue;
        }

        // --- Memory: Global Store ---
        if (auto storeOp = dyn_cast<xcore::StoreGlobalOp>(&op)) {
          auto addr = vmap.lookup(storeOp.getAddr());
          auto val = vmap.lookup(storeOp.getValue());
          if (!addr || !val) continue;
          auto ptrTy = LLVM::LLVMPointerType::get(ctx, 1);
          auto ptr = builder.create<LLVM::IntToPtrOp>(op.getLoc(), ptrTy, addr);
          builder.create<LLVM::StoreOp>(op.getLoc(), val, ptr, /*alignment=*/4);
          continue;
        }

        // --- Memory: Shared Load ---
        if (auto loadOp = dyn_cast<xcore::LoadSharedOp>(&op)) {
          auto addr = vmap.lookup(loadOp.getAddr());
          if (!addr) continue;
          auto ptrTy = LLVM::LLVMPointerType::get(ctx, 3);
          auto ptr = builder.create<LLVM::IntToPtrOp>(op.getLoc(), ptrTy, addr);
          auto load = builder.create<LLVM::LoadOp>(
              op.getLoc(), i32Ty, ptr, /*alignment=*/4);
          vmap[loadOp.getResult()] = load.getResult();
          continue;
        }

        // --- Memory: Shared Store ---
        if (auto storeOp = dyn_cast<xcore::StoreSharedOp>(&op)) {
          auto addr = vmap.lookup(storeOp.getAddr());
          auto val = vmap.lookup(storeOp.getValue());
          if (!addr || !val) continue;
          auto ptrTy = LLVM::LLVMPointerType::get(ctx, 3);
          auto ptr = builder.create<LLVM::IntToPtrOp>(op.getLoc(), ptrTy, addr);
          builder.create<LLVM::StoreOp>(op.getLoc(), val, ptr, /*alignment=*/4);
          continue;
        }

        // --- Barrier ---
        if (isa<xcore::BarrierOp>(&op)) {
          // Call @llvm.mxc.barrier() intrinsic
          auto intrinsic = LLVM::lookupOrCreateFn(
              module, "llvm.mxc.barrier", {}, voidTy);
          builder.create<LLVM::CallOp>(
              op.getLoc(), std::nullopt, SymbolRefAttr::get(intrinsic));
          continue;
        }

        // --- Return ---
        if (isa<xcore::ReturnOp>(&op)) {
          builder.create<LLVM::ReturnOp>(op.getLoc(), ValueRange{});
          continue;
        }

        // Unknown op — skip
        llvm::errs() << "Warning: unhandled XCore1000 op in LLVM lowering: "
                     << op.getName() << "\n";
      }
    }

    // Ensure function is terminated
    Block *body = &llvmFunc.getBody().front();
    if (body->empty() || !body->back().hasTrait<OpTrait::IsTerminator>()) {
      builder.setInsertionPointToEnd(body);
      builder.create<LLVM::ReturnOp>(oldFunc.getLoc(), ValueRange{});
    }
  }

  // Remove old XCore1000 functions
  for (auto oldFunc : oldFuncs)
    oldFunc.erase();

  return true;
}

bool emitLLVMBitcode(ModuleOp module, llvm::raw_ostream &os) {
  // Write the MLIR LLVM dialect to a temp file
  std::string tmpMlir = "/tmp/tgc_llvm_dialect.mlir";
  {
    std::error_code ec;
    llvm::raw_fd_ostream tmpOs(tmpMlir, ec);
    if (ec) {
      llvm::errs() << "Failed to open temp file: " << ec.message() << "\n";
      return false;
    }
    module->print(tmpOs);
  }

  // Use mlir-translate to convert to LLVM IR bitcode
  std::string cmd = "/opt/maca-3.7.0/mxgpu_llvm/bin/mlir-translate "
                    "--mlir-to-llvmir " + tmpMlir + " 2>/dev/null";

  FILE *pipe = popen(cmd.c_str(), "r");
  if (!pipe) {
    llvm::errs() << "Failed to run mlir-translate\n";
    return false;
  }

  char buffer[4096];
  while (fgets(buffer, sizeof(buffer), pipe)) {
    os << buffer;
  }

  int ret = pclose(pipe);
  std::remove(tmpMlir.c_str());

  return ret == 0;
}

} // namespace tgc
