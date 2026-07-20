add_mlir_dialect_library(XCore1000Dialect
  XCore1000Dialect.cpp
  XCore1000Ops.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/tiny-gpu-compiler/Dialect/XCore1000

  DEPENDS
  XCore1000OpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)
