#ifndef XCORE1000_OPS_TD
#define XCORE1000_OPS_TD

include "tiny-gpu-compiler/Dialect/XCore1000/XCore1000Dialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for all XCore ops
class XCore_Op<string mnemonic, list<Trait> traits = []> :
    Op<XCore1000_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Kernel function
//===----------------------------------------------------------------------===//

def XCore_FuncOp : XCore_Op<"func", [
    IsolatedFromAbove,
    FunctionOpInterface,
    Symbol
]> {
  let summary = "GPU kernel function targeting xcore1000 hardware";
  let description = [{
    Represents a GPU kernel compiled for the MetaX xcore1000 architecture.
    Parameters are passed via the kernarg segment.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);

  let builders = [
    OpBuilder<(ins "StringRef":$name, "FunctionType":$type,
               CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let extraClassDeclaration = [{
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
    Region *getCallableRegion() { return &getBody(); }
  }];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Special register reads (hardware-provided values)
//===----------------------------------------------------------------------===//

def XCore_ThreadIdOp : XCore_Op<"thread_id", [Pure]> {
  let summary = "Read thread index within block (masked to 10 bits)";
  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let builders = [
    OpBuilder<(ins), [{
      $_state.addTypes($_builder.getI32Type());
    }]>
  ];
}

def XCore_BlockIdOp : XCore_Op<"block_id", [Pure]> {
  let summary = "Read block index in grid";
  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let builders = [
    OpBuilder<(ins), [{
      $_state.addTypes($_builder.getI32Type());
    }]>
  ];
}

def XCore_BlockDimOp : XCore_Op<"block_dim", [Pure]> {
  let summary = "Read threads per block (blockDim)";
  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let builders = [
    OpBuilder<(ins), [{
      $_state.addTypes($_builder.getI32Type());
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// Integer arithmetic (32-bit)
//===----------------------------------------------------------------------===//

def XCore_AddIOp : XCore_Op<"addi", [Pure, Commutative]> {
  let summary = "32-bit integer addition";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
      $_state.addOperands({lhs, rhs});
      $_state.addTypes(lhs.getType());
    }]>
  ];
}

def XCore_SubIOp : XCore_Op<"subi", [Pure]> {
  let summary = "32-bit integer subtraction";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
      $_state.addOperands({lhs, rhs});
      $_state.addTypes(lhs.getType());
    }]>
  ];
}

def XCore_MulIOp : XCore_Op<"muli", [Pure, Commutative]> {
  let summary = "32-bit integer multiplication (low 32 bits)";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
      $_state.addOperands({lhs, rhs});
      $_state.addTypes(lhs.getType());
    }]>
  ];
}

def XCore_MulIHiOp : XCore_Op<"mulhi", [Pure, Commutative]> {
  let summary = "32-bit unsigned multiplication (high 32 bits)";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def XCore_MadIOp : XCore_Op<"madi", [Pure]> {
  let summary = "Multiply-add: result = lhs * rhs + addend";
  let arguments = (ins I32:$lhs, I32:$rhs, I32:$addend);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs `,` $rhs `,` $addend attr-dict `:` type($result)";
}

def XCore_DivIOp : XCore_Op<"divi", [Pure]> {
  let summary = "32-bit integer division";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
      $_state.addOperands({lhs, rhs});
      $_state.addTypes(lhs.getType());
    }]>
  ];
}

def XCore_RemIOp : XCore_Op<"remi", [Pure]> {
  let summary = "32-bit integer remainder (modulo)";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Bitwise operations (32-bit)
//===----------------------------------------------------------------------===//

def XCore_AndIOp : XCore_Op<"andi", [Pure, Commutative]> {
  let summary = "32-bit bitwise AND";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def XCore_OrIOp : XCore_Op<"ori", [Pure, Commutative]> {
  let summary = "32-bit bitwise OR";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def XCore_XorIOp : XCore_Op<"xori", [Pure, Commutative]> {
  let summary = "32-bit bitwise XOR";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def XCore_ShlIOp : XCore_Op<"shli", [Pure]> {
  let summary = "32-bit shift left";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def XCore_ShrIOp : XCore_Op<"shri", [Pure]> {
  let summary = "32-bit logical shift right";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def XCore_SarIOp : XCore_Op<"sari", [Pure]> {
  let summary = "32-bit arithmetic shift right (sign-extending)";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def XCore_ClzIOp : XCore_Op<"clzi", [Pure]> {
  let summary = "Count leading zeros";
  let arguments = (ins I32:$operand);
  let results = (outs I32:$result);
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

def XCore_PopcIOp : XCore_Op<"popci", [Pure]> {
  let summary = "Population count (number of set bits)";
  let arguments = (ins I32:$operand);
  let results = (outs I32:$result);
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Float arithmetic (32-bit IEEE754)
//===----------------------------------------------------------------------===//

def XCore_AddFOp : XCore_Op<"addf", [Pure, Commutative]> {
  let summary = "32-bit float addition";
  let arguments = (ins F32:$lhs, F32:$rhs);
  let results = (outs F32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def XCore_SubFOp : XCore_Op<"subf", [Pure]> {
  let summary = "32-bit float subtraction";
  let arguments = (ins F32:$lhs, F32:$rhs);
  let results = (outs F32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def XCore_MulFOp : XCore_Op<"mulf", [Pure, Commutative]> {
  let summary = "32-bit float multiplication";
  let arguments = (ins F32:$lhs, F32:$rhs);
  let results = (outs F32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def XCore_DivFOp : XCore_Op<"divf", [Pure]> {
  let summary = "32-bit float division (multi-step: div_scale + div_fmas + div_fixup)";
  let arguments = (ins F32:$lhs, F32:$rhs);
  let results = (outs F32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def XCore_FmaFOp : XCore_Op<"fmaf", [Pure]> {
  let summary = "Fused multiply-add: result = a * b + c";
  let arguments = (ins F32:$a, F32:$b, F32:$c);
  let results = (outs F32:$result);
  let assemblyFormat = "$a `,` $b `,` $c attr-dict `:` type($result)";
}

def XCore_NegFOp : XCore_Op<"negf", [Pure]> {
  let summary = "Float negate";
  let arguments = (ins F32:$operand);
  let results = (outs F32:$result);
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

def XCore_AbsFOp : XCore_Op<"absf", [Pure]> {
  let summary = "Float absolute value";
  let arguments = (ins F32:$operand);
  let results = (outs F32:$result);
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

def XCore_RcpFOp : XCore_Op<"rcpf", [Pure]> {
  let summary = "Float reciprocal approximation (1/x)";
  let arguments = (ins F32:$operand);
  let results = (outs F32:$result);
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

def XCore_MinFOp : XCore_Op<"minf", [Pure, Commutative]> {
  let summary = "Float minimum";
  let arguments = (ins F32:$lhs, F32:$rhs);
  let results = (outs F32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def XCore_MaxFOp : XCore_Op<"maxf", [Pure, Commutative]> {
  let summary = "Float maximum";
  let arguments = (ins F32:$lhs, F32:$rhs);
  let results = (outs F32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Type conversion
//===----------------------------------------------------------------------===//

def XCore_IntToFloatOp : XCore_Op<"i32tof32", [Pure]> {
  let summary = "Convert int32 to float32";
  let arguments = (ins I32:$operand);
  let results = (outs F32:$result);
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

def XCore_FloatToIntOp : XCore_Op<"f32toi32", [Pure]> {
  let summary = "Convert float32 to int32 (truncating)";
  let arguments = (ins F32:$operand);
  let results = (outs I32:$result);
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

def XCore_BitcastFOp : XCore_Op<"bitcast_f_to_i", [Pure]> {
  let summary = "Bitcast float32 to int32 (no conversion, reinterpret bits)";
  let arguments = (ins F32:$operand);
  let results = (outs I32:$result);
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

def XCore_BitcastIOp : XCore_Op<"bitcast_i_to_f", [Pure]> {
  let summary = "Bitcast int32 to float32 (no conversion, reinterpret bits)";
  let arguments = (ins I32:$operand);
  let results = (outs F32:$result);
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Constant
//===----------------------------------------------------------------------===//

def XCore_ConstIOp : XCore_Op<"consti", [Pure, ConstantLike]> {
  let summary = "Load 32-bit integer immediate";
  let arguments = (ins I32Attr:$value);
  let results = (outs I32:$result);
  let assemblyFormat = "$value attr-dict";
  let hasFolder = 1;
  let builders = [
    OpBuilder<(ins "int32_t":$value), [{
      auto attr = IntegerAttr::get($_builder.getI32Type(), value);
      $_state.addAttribute("value", attr);
      $_state.addTypes($_builder.getI32Type());
    }]>,
    OpBuilder<(ins "IntegerAttr":$value), [{
      $_state.addAttribute("value", value);
      $_state.addTypes(value.getType());
    }]>
  ];
}

def XCore_ConstFOp : XCore_Op<"constf", [Pure, ConstantLike]> {
  let summary = "Load 32-bit float immediate";
  let arguments = (ins F32Attr:$value);
  let results = (outs F32:$result);
  let assemblyFormat = "$value attr-dict";
  let hasFolder = 1;
  let builders = [
    OpBuilder<(ins "float":$value), [{
      auto attr = FloatAttr::get($_builder.getF32Type(), value);
      $_state.addAttribute("value", attr);
      $_state.addTypes($_builder.getF32Type());
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// Comparison and branching
//===----------------------------------------------------------------------===//

def XCore_CmpIOp : XCore_Op<"cmpi", [Pure]> {
  let summary = "Integer comparison, sets scalar flag register";
  let description = [{
    Compares lhs and rhs, producing a boolean result.
    Supports: eq, ne, lt, gt, le, ge (signed and unsigned).
    Maps to xcore1000 cmp_*_i32 / cmp_*_u32 instructions.
  }];
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
      $_state.addOperands({lhs, rhs});
      $_state.addTypes(IntegerType::get(lhs.getContext(), 1));
    }]>
  ];
}

def XCore_CmpFOp : XCore_Op<"cmpf", [Pure]> {
  let summary = "Float comparison";
  let arguments = (ins F32:$lhs, F32:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def XCore_CselOp : XCore_Op<"csel", [Pure]> {
  let summary = "Conditional select: result = cond ? true_val : false_val";
  let arguments = (ins I1:$cond, I32:$true_val, I32:$false_val);
  let results = (outs I32:$result);
  let assemblyFormat = "$cond `,` $true_val `,` $false_val attr-dict `:` type($result)";
}

def XCore_BranchOp : XCore_Op<"branch", [Terminator]> {
  let summary = "Conditional branch based on scalar flag";
  let arguments = (ins I1:$condition);
  let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);
  let assemblyFormat = "$condition `,` $trueDest `,` $falseDest attr-dict";
}

def XCore_JumpOp : XCore_Op<"jump", [Terminator]> {
  let summary = "Unconditional jump to target block";
  let successors = (successor AnySuccessor:$target);
  let assemblyFormat = "$target attr-dict";
}

//===----------------------------------------------------------------------===//
// Memory operations
//===----------------------------------------------------------------------===//

def XCore_LoadGlobalOp : XCore_Op<"load_global"> {
  let summary = "Load 32-bit value from global memory";
  let description = [{
    Loads a value from global device memory at the given address.
    Maps to xcore1000 ldg_b32 instruction.
    The address is a 64-bit pointer passed via kernarg segment.
  }];
  let arguments = (ins I32:$addr);
  let results = (outs I32:$result);
  let assemblyFormat = "$addr attr-dict `:` type($result)";
}

def XCore_StoreGlobalOp : XCore_Op<"store_global"> {
  let summary = "Store 32-bit value to global memory";
  let description = [{
    Stores a value to global device memory.
    Maps to xcore1000 stg_b32 ... devc instruction.
  }];
  let arguments = (ins I32:$addr, I32:$value);
  let assemblyFormat = "$addr `,` $value attr-dict";
}

def XCore_LoadGlobalFOp : XCore_Op<"load_global_f"> {
  let summary = "Load float32 from global memory";
  let arguments = (ins I32:$addr);
  let results = (outs F32:$result);
  let assemblyFormat = "$addr attr-dict `:` type($result)";
}

def XCore_StoreGlobalFOp : XCore_Op<"store_global_f"> {
  let summary = "Store float32 to global memory";
  let arguments = (ins I32:$addr, F32:$value);
  let assemblyFormat = "$addr `,` $value attr-dict";
}

def XCore_LoadSharedOp : XCore_Op<"load_shared"> {
  let summary = "Load 32-bit value from shared memory (BSM)";
  let description = [{
    Loads from block-shared memory.
    Maps to xcore1000 lds_b32 instruction.
  }];
  let arguments = (ins I32:$addr);
  let results = (outs I32:$result);
  let assemblyFormat = "$addr attr-dict `:` type($result)";
}

def XCore_StoreSharedOp : XCore_Op<"store_shared"> {
  let summary = "Store 32-bit value to shared memory (BSM)";
  let description = [{
    Stores to block-shared memory.
    Maps to xcore1000 sts_b32 instruction.
  }];
  let arguments = (ins I32:$addr, I32:$value);
  let assemblyFormat = "$addr `,` $value attr-dict";
}

def XCore_LoadSharedFOp : XCore_Op<"load_shared_f"> {
  let summary = "Load float32 from shared memory";
  let arguments = (ins I32:$addr);
  let results = (outs F32:$result);
  let assemblyFormat = "$addr attr-dict `:` type($result)";
}

def XCore_StoreSharedFOp : XCore_Op<"store_shared_f"> {
  let summary = "Store float32 to shared memory";
  let arguments = (ins I32:$addr, F32:$value);
  let assemblyFormat = "$addr `,` $value attr-dict";
}

//===----------------------------------------------------------------------===//
// Atomic operations
//===----------------------------------------------------------------------===//

def XCore_AtomicAddOp : XCore_Op<"atomic_add"> {
  let summary = "Atomic integer addition on global memory";
  let description = [{
    Atomically adds a value to global memory.
    Maps to xcore1000 atom_global_add_i32 instruction.
  }];
  let arguments = (ins I32:$addr, I32:$value);
  let results = (outs I32:$result);
  let assemblyFormat = "$addr `,` $value attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Synchronization
//===----------------------------------------------------------------------===//

def XCore_BarrierOp : XCore_Op<"barrier"> {
  let summary = "Block-level thread synchronization barrier";
  let description = [{
    Synchronizes all threads within a block.
    Maps to xcore1000 barrier instruction.
    Must be preceded by arrive bsmcnt(0) to ensure shared memory visibility.
  }];
  let assemblyFormat = "attr-dict";
  let builders = [
    OpBuilder<(ins), [{ /* no-op */ }]>
  ];
}

//===----------------------------------------------------------------------===//
// Warp-level operations
//===----------------------------------------------------------------------===//

def XCore_WarpShuffleOp : XCore_Op<"warp_shuffle"> {
  let summary = "Warp shuffle (read value from another lane)";
  let description = [{
    Reads a value from lane src_lane in the same warp.
    Maps to xcore1000 sm_bperm_b32 instruction.
    Uses BSM (block shared memory) for inter-lane communication.
  }];
  let arguments = (ins I32:$value, I32:$src_lane);
  let results = (outs I32:$result);
  let assemblyFormat = "$value `,` $src_lane attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Terminator
//===----------------------------------------------------------------------===//

def XCore_ReturnOp : XCore_Op<"ret", [Terminator]> {
  let summary = "Signal thread completion (endk)";
  let description = [{
    Indicates the current thread has finished execution.
    Maps to xcore1000 endk instruction.
  }];
  let assemblyFormat = "attr-dict";
  let builders = [
    OpBuilder<(ins), [{ /* no-op */ }]>
  ];
}

//===----------------------------------------------------------------------===//
// Address computation helpers
//===----------------------------------------------------------------------===//

def XCore_AddrCalcOp : XCore_Op<"addr_calc", [Pure]> {
  let summary = "Compute global memory address: base + index * stride";
  let description = [{
    Computes a byte address for global memory access.
    Maps to xcore1000 mad_i32 rd, index, stride, base instruction.
    Typically stride=4 for 32-bit values.
  }];
  let arguments = (ins I64:$base, I32:$index, I32:$stride);
  let results = (outs I64:$result);
  let assemblyFormat = "$base `,` $index `,` $stride attr-dict `:` type($result)";
}

#endif // XCORE1000_OPS_TD
