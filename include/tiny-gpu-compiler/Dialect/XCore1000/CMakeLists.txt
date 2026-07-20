set(LLVM_TARGET_DEFINITIONS XCore1000Ops.td)

mlir_tablegen(XCore1000Ops.h.inc -gen-op-decls)
mlir_tablegen(XCore1000Ops.cpp.inc -gen-op-defs)
mlir_tablegen(XCore1000Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(XCore1000Dialect.cpp.inc -gen-dialect-defs)

add_public_tablegen_target(XCore1000OpsIncGen)
