module SimpleChains

# using ChainRules, ChainRulesCore, NNlibCPU
using UnPack,
  VectorizationBase,
  StaticArrayInterface,
  Polyester,
  SLEEFPirates,
  StrideArraysCore,
  Static,
  VectorizedRNG
const ArrayInterface = StaticArrayInterface
using StaticArrayInterface:
  CPUPointer,
  static_size,
  static_strides,
  static_axes,
  StrideIndex,
  contiguous_axis,
  stride_rank,
  static_length,
  static_first,
  static_last,
  static_step,
  indices,
  offsets,
  is_column_major
using SIMDTypes: Bit, NativeTypes
using VectorizationBase: align, relu, stridedpointer, AbstractSIMD, NativeTypesV
using HostCPUFeatures:
  static_sizeof, register_size, register_count, static_sizeof
using CPUSummary: cache_linesize, num_cores
using LayoutPointers:
  bytestrideindex, stridedpointer, zstridedpointer, zero_offsets, val_dense_dims
using Static: One, lt
using CloseOpenIntervals: CloseOpen
using StrideArraysCore: zview, @gc_preserve
import ManualMemory: preserve_buffer
using IfElse: ifelse
import Random
import ChainRulesCore
import ForwardDiff
import LoopVectorization
import StaticArrays
using StaticArrays: SVector, SMatrix
using Random: AbstractRNG

using LoopVectorization: matmul_params, @turbo
# using LoopVectorization: matmul_params
# macro turbo(ex)
#   esc(ex)
# end
# macro turbo(ex0, ex1)
#   esc(ex1)
# end
# macro turbo(ex0, ex1, ex2)
#   esc(ex2)
# end

export SimpleChain,
  TurboDense,
  Dropout,
  Activation,
  Conv,
  MaxPool,
  Flatten,
  AbsoluteLoss,
  SquaredLoss,
  WeightedSquaredLoss,
  LogitCrossEntropyLoss,
  relu,
  static,
  StrideArray,
  valgrad!,
  valgrad,
  NoPenalty,
  L1Penalty,
  L2Penalty,
  FrontLastPenalty

const Integer = Union{StaticInt,Base.Integer}
const MAXSTACK = 16384

include("memory.jl")
include("simple_chain.jl")
include("utils.jl")
include("activation.jl")
include("dense.jl")
include("forwarddiff_matmul.jl")
include("dropout.jl")
include("conv.jl")
include("loss.jl")
include("maxpool.jl")
include("flatten.jl")
include("penalty.jl")
include("chain_rules.jl")
include("optimize.jl")

if VERSION >= v"1.7.0" && hasfield(Method, :recursion_relation)
  dont_limit = Returns(true)
  for f in (
    chain_valgrad!,
    chain_valgrad_pullback!,
    __chain,
    output_size,
    forward_output_size,
    _numparam,
    pullback_layer!,
    contract!,
    _init_params!
  )
    for m in methods(f)
      m.recursion_relation = dont_limit
    end
  end
end

end
