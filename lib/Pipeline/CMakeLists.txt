add_mlir_library(TGCPipeline
  Pipeline.cpp
  ${PROJECT_SOURCE_DIR}/lib/Passes/Passes.cpp
  ${PROJECT_SOURCE_DIR}/lib/Passes/XCore1000Lowering.cpp
  ${PROJECT_SOURCE_DIR}/lib/Passes/XCore1000ToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/tiny-gpu-compiler/Pipeline
  ${PROJECT_SOURCE_DIR}/include/tiny-gpu-compiler/Passes

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  TinyGPUDialect
  XCore1000Dialect
  TGCFrontend
  TGCCodeGen
)
