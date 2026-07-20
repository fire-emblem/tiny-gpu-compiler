#include "tiny-gpu-compiler/Dialect/XCore1000/XCore1000Dialect.h"
#include "tiny-gpu-compiler/Dialect/XCore1000/XCore1000Ops.h"

using namespace mlir;
using namespace mlir::xcore;

#include "tiny-gpu-compiler/Dialect/XCore1000/XCore1000Dialect.cpp.inc"

void XCore1000Dialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tiny-gpu-compiler/Dialect/XCore1000/XCore1000Ops.cpp.inc"
      >();
}
