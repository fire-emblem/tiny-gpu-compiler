#ifndef XCORE1000_DIALECT_TD
#define XCORE1000_DIALECT_TD

include "mlir/IR/OpBase.td"

def XCore1000_Dialect : Dialect {
  let name = "xcore";
  let summary = "Dialect targeting the MetaX xcore1000 GPU architecture";
  let description = [{
    The XCore dialect provides operations that map to the MetaX xcore1000 GPU
    ISA (MACA platform). The xcore1000 is based on ARCv2.3 64-bit with GPU
    extensions featuring:
    - 32-bit fixed-width instructions
    - VGPR (vector) and SGPR (scalar) register classes
    - 64-thread warps
    - Up to 512 threads per block
    - Shared memory (BSM) with barrier synchronization
    - IEEE754 float32/float64 arithmetic
    - Multi-step division (div_scale + div_fmas + div_fixup)
    - Global, shared, and private memory address spaces
  }];
  let cppNamespace = "::mlir::xcore";
  let useDefaultTypePrinterParser = 1;
}

#endif // XCORE1000_DIALECT_TD
