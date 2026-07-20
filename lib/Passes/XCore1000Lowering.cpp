#include "tiny-gpu-compiler/Passes/XCore1000Lowering.h"
#include "tiny-gpu-compiler/Dialect/TinyGPU/TinyGPUOps.h"
#include "tiny-gpu-compiler/Dialect/XCore1000/XCore1000Ops.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace tgc {

/// Map from old TinyGPU values to new XCore1000 values.
using ValueMap = llvm::DenseMap<Value, Value>;

/// Lower a single TinyGPU FuncOp to XCore1000 FuncOp.
/// Creates a new xcore.func, walks the body, and converts each operation.
static xcore::FuncOp lowerFuncOp(tinygpu::FuncOp oldFunc, OpBuilder &builder) {
  // Create new function with same name but i32 types
  auto loc = oldFunc.getLoc();
  auto newFuncType = builder.getFunctionType({}, {});
  auto newFunc = builder.create<xcore::FuncOp>(
      loc, oldFunc.getName(), newFuncType);

  // Create entry block
  Block *entryBlock = newFunc.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Walk old function and convert ops
  ValueMap vmap;

  // Helper: get or create i32 equivalent of an i8 value
  auto getI32 = [&](Value v) -> Value {
    auto it = vmap.find(v);
    if (it != vmap.end())
      return it->second;
    // If not in map, it's an argument or external value — shouldn't happen
    return nullptr;
  };

  // Helper: map an old value to a new value
  auto mapValue = [&](Value oldVal, Value newVal) {
    vmap[oldVal] = newVal;
  };

  for (Block &block : oldFunc.getBody()) {
    // For now, we only handle the entry block (single-BB kernels)
    if (&block != &oldFunc.getBody().front())
      continue;

    for (Operation &op : block) {
      // --- Thread/Block ID ---
      if (auto tidOp = dyn_cast<tinygpu::ThreadIdOp>(&op)) {
        auto newOp = builder.create<xcore::ThreadIdOp>(op.getLoc(),
            builder.getI32Type());
        mapValue(tidOp.getResult(), newOp.getResult());
        continue;
      }
      if (auto bidOp = dyn_cast<tinygpu::BlockIdOp>(&op)) {
        auto newOp = builder.create<xcore::BlockIdOp>(op.getLoc(),
            builder.getI32Type());
        mapValue(bidOp.getResult(), newOp.getResult());
        continue;
      }
      if (auto bdimOp = dyn_cast<tinygpu::BlockDimOp>(&op)) {
        auto newOp = builder.create<xcore::BlockDimOp>(op.getLoc(),
            builder.getI32Type());
        mapValue(bdimOp.getResult(), newOp.getResult());
        continue;
      }

      // --- Constants ---
      if (auto constOp = dyn_cast<tinygpu::ConstOp>(&op)) {
        uint8_t val = constOp.getValue().getZExtValue();
        auto newOp = builder.create<xcore::ConstIOp>(op.getLoc(),
            builder.getI32IntegerAttr(val));
        mapValue(constOp.getResult(), newOp.getResult());
        continue;
      }

      // --- Arithmetic ---
      if (auto addOp = dyn_cast<tinygpu::AddOp>(&op)) {
        auto lhs = getI32(addOp.getLhs());
        auto rhs = getI32(addOp.getRhs());
        if (!lhs || !rhs) continue;
        auto newOp = builder.create<xcore::AddIOp>(op.getLoc(), lhs, rhs);
        mapValue(addOp.getResult(), newOp.getResult());
        continue;
      }
      if (auto subOp = dyn_cast<tinygpu::SubOp>(&op)) {
        auto lhs = getI32(subOp.getLhs());
        auto rhs = getI32(subOp.getRhs());
        if (!lhs || !rhs) continue;
        auto newOp = builder.create<xcore::SubIOp>(op.getLoc(), lhs, rhs);
        mapValue(subOp.getResult(), newOp.getResult());
        continue;
      }
      if (auto mulOp = dyn_cast<tinygpu::MulOp>(&op)) {
        auto lhs = getI32(mulOp.getLhs());
        auto rhs = getI32(mulOp.getRhs());
        if (!lhs || !rhs) continue;
        auto newOp = builder.create<xcore::MulIOp>(op.getLoc(), lhs, rhs);
        mapValue(mulOp.getResult(), newOp.getResult());
        continue;
      }
      if (auto divOp = dyn_cast<tinygpu::DivOp>(&op)) {
        auto lhs = getI32(divOp.getLhs());
        auto rhs = getI32(divOp.getRhs());
        if (!lhs || !rhs) continue;
        auto newOp = builder.create<xcore::DivIOp>(op.getLoc(), lhs, rhs);
        mapValue(divOp.getResult(), newOp.getResult());
        continue;
      }

      // --- Comparison ---
      if (auto cmpOp = dyn_cast<tinygpu::CmpOp>(&op)) {
        auto lhs = getI32(cmpOp.getLhs());
        auto rhs = getI32(cmpOp.getRhs());
        if (!lhs || !rhs) continue;
        auto newOp = builder.create<xcore::CmpIOp>(op.getLoc(), lhs, rhs);
        mapValue(cmpOp.getResult(), newOp.getResult());
        continue;
      }

      // --- Memory: Global ---
      if (auto loadOp = dyn_cast<tinygpu::LoadOp>(&op)) {
        auto addr = getI32(loadOp.getAddr());
        if (!addr) continue;
        // Zero-extend i32 address to i64 for global memory
        auto addr64 = builder.create<xcore::ConstIOp>(op.getLoc(),
            builder.getI32IntegerAttr(0));
        // Use addr_calc to form a 64-bit address (simplified)
        auto newOp = builder.create<xcore::LoadGlobalOp>(op.getLoc(),
            builder.getI64Type(), addr64.getResult());
        mapValue(loadOp.getResult(), newOp.getResult());
        continue;
      }
      if (auto storeOp = dyn_cast<tinygpu::StoreOp>(&op)) {
        auto addr = getI32(storeOp.getAddr());
        auto val = getI32(storeOp.getValue());
        if (!addr || !val) continue;
        auto addr64 = builder.create<xcore::ConstIOp>(op.getLoc(),
            builder.getI32IntegerAttr(0));
        builder.create<xcore::StoreGlobalOp>(op.getLoc(),
            addr64.getResult(), val);
        continue;
      }

      // --- Memory: Shared ---
      if (auto loadOp = dyn_cast<tinygpu::SharedLoadOp>(&op)) {
        auto addr = getI32(loadOp.getAddr());
        if (!addr) continue;
        auto newOp = builder.create<xcore::LoadSharedOp>(op.getLoc(),
            builder.getI32Type(), addr);
        mapValue(loadOp.getResult(), newOp.getResult());
        continue;
      }
      if (auto storeOp = dyn_cast<tinygpu::SharedStoreOp>(&op)) {
        auto addr = getI32(storeOp.getAddr());
        auto val = getI32(storeOp.getValue());
        if (!addr || !val) continue;
        builder.create<xcore::StoreSharedOp>(op.getLoc(), addr, val);
        continue;
      }

      // --- Barrier ---
      if (isa<tinygpu::BarrierOp>(&op)) {
        builder.create<xcore::BarrierOp>(op.getLoc());
        continue;
      }

      // --- Branch ---
      if (auto branchOp = dyn_cast<tinygpu::BranchOp>(&op)) {
        // Simplified: create unconditional jump (proper lowering needs CFG)
        builder.create<xcore::ReturnOp>(op.getLoc());
        continue;
      }
      if (auto jumpOp = dyn_cast<tinygpu::JumpOp>(&op)) {
        // Skip — will be handled when we process the target block
        continue;
      }

      // --- Return ---
      if (isa<tinygpu::ReturnOp>(&op)) {
        builder.create<xcore::ReturnOp>(op.getLoc());
        continue;
      }

      // Unknown op — skip with warning
      llvm::errs() << "Warning: unhandled TinyGPU op in lowering: "
                   << op.getName() << "\n";
    }
  }

  // Ensure function is terminated
  Block *body = &newFunc.getBody().front();
  if (body->empty() || !body->back().hasTrait<OpTrait::IsTerminator>()) {
    builder.setInsertionPointToEnd(body);
    builder.create<xcore::ReturnOp>(loc);
  }

  return newFunc;
}

bool lowerTinyGPUToXCore1000(ModuleOp module) {
  OpBuilder builder(module.getContext());

  // Collect all TinyGPU FuncOps
  std::vector<tinygpu::FuncOp> oldFuncs;
  for (auto &op : module->getBody()->getOperations()) {
    if (auto funcOp = dyn_cast<tinygpu::FuncOp>(&op))
      oldFuncs.push_back(funcOp);
  }

  if (oldFuncs.empty()) {
    llvm::errs() << "No TinyGPU functions found to lower\n";
    return false;
  }

  // Lower each function
  for (auto oldFunc : oldFuncs) {
    builder.setInsertionPoint(oldFunc);
    auto newFunc = lowerFuncOp(oldFunc, builder);
    if (!newFunc) {
      llvm::errs() << "Failed to lower function: " << oldFunc.getName() << "\n";
      return false;
    }

    // Verify the new function
    if (failed(verify(newFunc))) {
      llvm::errs() << "Verification failed for lowered function: "
                   << oldFunc.getName() << "\n";
      return false;
    }
  }

  // Remove old TinyGPU functions
  for (auto oldFunc : oldFuncs)
    oldFunc.erase();

  return true;
}

} // namespace tgc
