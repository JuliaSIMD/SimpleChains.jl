abstract type AbstractLoss{Y} end

has_loss(sc::SimpleChain) = last(sc.layers) isa AbstractLoss
"""
    add_loss(chn, l::AbstractLoss)

Add the loss function `l` to the simple chain. The loss function
should hold the target you're trying to fit.
"""
function add_loss(sc::SimpleChain, l::AbstractLoss)
  id = chain_input_dims(sc)
  if has_loss(sc)
    SimpleChain(id, (Base.front(sc.layers)..., l))
  else
    SimpleChain(id, (sc.layers..., l))
  end
end
remove_loss(sc::SimpleChain) = has_loss(sc) ? Base.front(sc) : sc
pop_loss(sc::SimpleChain) = last(sc.layers)
function split_loss(sc::SimpleChain)
  layers = sc.layers
  SimpleChain(chain_input_dims(sc), Base.front(layers)), last(layers)
end

target(_) = nothing
target(sc::SimpleChain) = target(last(sc.layers))
preserve_buffer(l::AbstractLoss) = target(l)
StrideArraysCore.object_and_preserve(l::AbstractLoss) = l, target(l)
_iterate_over_losses(::AbstractArray{<:AbstractArray}) = true
_iterate_over_losses(_) = false
iterate_over_losses(sc) = _iterate_over_losses(target(sc))

parameter_free(::AbstractLoss) = true
numparam(::AbstractLoss, _) = static(0), 1
function _layer_output_size_needs_temp(
  ::Val{T},
  sl::AbstractLoss{<:AbstractArray{<:AbstractArray}},
  s
) where {T}
  align(length(first(target(sl))) * static_sizeof(T)), static_sizeof(T)
end
function _layer_output_size_needs_temp_of_equal_len_as_target(
  ::Val{T},
  sl::AbstractLoss,
  s
) where {T}
  align(length(target(sl)) * static_sizeof(T)), static_sizeof(T)
end
function _layer_output_size_no_temp(::Val{T}, sl::AbstractLoss, s) where {T}
  static(0), static_sizeof(T)
end
function forward_layer_output_size(::Val{T}, sl::AbstractLoss, s) where {T}
  _layer_output_size_no_temp(Val{T}(), sl, s)
end

"""
    SquaredLoss(target)

Calculates half of mean squared loss of the target.
"""
struct SquaredLoss{Y} <: AbstractLoss{Y}
  y::Y
end
(::SquaredLoss)(y) = SquaredLoss(y)
SquaredLoss() = SquaredLoss(nothing)
target(sl::SquaredLoss) = getfield(sl, :y)
init_params!(::AbstractLoss, p, _, ::AbstractRNG) = p, 1

Base.getindex(sl::SquaredLoss, r) = SquaredLoss(view_slice_last(target(sl), r))

squared_loss(chn::SimpleChain, y) = add_loss(chn, SquaredLoss(y))

Base.show(io::IO, ::SquaredLoss) = print(io, "SquaredLoss")

@inline loss_multiplier(::AbstractLoss, N, ::Type{T}) where {T} = inv(T(N))
@inline loss_multiplier(::SquaredLoss, N, ::Type{T}) where {T} = T(2) / T(N)

function chain_valgrad!(
  _,
  arg::AbstractArray{T,D},
  layers::Tuple{SquaredLoss},
  p::Ptr,
  pu::Ptr{UInt8}
) where {T,D}
  y = getfield(getfield(layers, 1), :y)
  # invN = T(inv(static_size(arg, D)))
  s = zero(T)
  @turbo for i ∈ eachindex(arg)
    δ = arg[i] - y[i]
    arg[i] = δ
    s += δ * δ
  end
  T(0.5) * s, arg, pu
end
function (sl::SquaredLoss{<:AbstractArray{<:Number}})(
  arg::AbstractArray{T,N},
  p,
  pu
) where {T,N}
  y = getfield(sl, :y)
  s = zero(T)
  @turbo for i ∈ eachindex(arg)
    δ = arg[i] - y[i]
    s += δ * δ
  end
  # NOTE: we're not dividing by static_size(arg,N)
  T(0.5) * s, p, pu
end

"""
WeightedSquaredLoss(target)

Calculates half of mean weighted squared loss of the target.
"""
struct WeightedSquaredLoss{Y, W} <: AbstractLoss{Y}
    y::Y
    weights::W
end
(::WeightedSquaredLoss)(y, w) = WeightedSquaredLoss(y, w)
WeightedSquaredLoss() = WeightedSquaredLoss(nothing)
WeightedSquaredLoss(x::Tuple) = WeightedSquaredLoss(x...)
target(wsl::WeightedSquaredLoss) = getfield(wsl, :y), getfield(wsl, :weights)
function view_slice_last(x::Tuple, r)
    return Tuple(view_slice_last(f, r) for f in x::Tuple)
end

Base.getindex(wsl::WeightedSquaredLoss, r) = WeightedSquaredLoss(view_slice_last(target(wsl), r))

weighted_squared_loss(chn::SimpleChain, y, w) = add_loss(chn, WeightedSquaredLoss(y, w))

Base.show(io::IO, ::WeightedSquaredLoss) = print(io, "WeightedSquaredLoss")

@inline loss_multiplier(::WeightedSquaredLoss, N, ::Type{T}) where {T} = T(2) / T(N)

function chain_valgrad!(
    _,
    arg::AbstractArray{T,D},
    layers::Tuple{WeightedSquaredLoss},
    p::Ptr,
    pu::Ptr{UInt8}
  ) where {T,D}
    y = getfield(getfield(layers, 1), :y)
    w = getfield(getfield(layers, 1), :weights)
    # invN = T(inv(static_size(arg, D)))
    s = zero(T)
    @turbo for i ∈ eachindex(arg)
      δ = arg[i] - y[i]
      δw = δ*w[i]
      arg[i] = δw
      s += δ * δw
    end
    T(0.5) * s, arg, pu
  end
  function (sl::WeightedSquaredLoss{<:AbstractArray{<:Number}})(
    arg::AbstractArray{T,N},
    p,
    pu
  ) where {T,N}
    y = getfield(sl, :y)
    w = getfield(sl, :weights)
    s = zero(T)
    @turbo for i ∈ eachindex(arg)
      δ = arg[i] - y[i]
      s += δ * δ * w[i]
    end
    # NOTE: we're not dividing by static_size(arg,N)
    T(0.5) * s, p, pu
  end

"""
    AbsoluteLoss

Calculates mean absolute loss of the target.
"""
struct AbsoluteLoss{Y} <: AbstractLoss{Y}
  y::Y
end
(::AbsoluteLoss)(y) = AbsoluteLoss(y)
AbsoluteLoss() = AbsoluteLoss(nothing)
target(sl::AbsoluteLoss) = getfield(sl, :y)

absolute_loss(chn::SimpleChain, y) = add_loss(chn, AbsoluteLoss(y))

Base.show(io::IO, ::AbsoluteLoss) = print(io, "AbsoluteLoss")

function Base.getindex(sl::AbsoluteLoss, r)
  AbsoluteLoss(view_slice_last(target(sl), r))
end
function chain_valgrad!(
  __,
  arg::AbstractArray{T},
  layers::Tuple{AbsoluteLoss},
  _::Ptr,
  pu::Ptr{UInt8}
) where {T}
  y = getfield(getfield(layers, 1), :y)
  s = zero(eltype(arg))
  @turbo for i ∈ eachindex(arg)
    δ = arg[i] - y[i]
    pos = δ > zero(T)
    arg[i] = ifelse(pos, one(T), -one(T))
    s += ifelse(pos, δ, -δ)
  end
  return s, arg, pu
end

function (sl::AbsoluteLoss{<:AbstractArray{<:Number}})(arg, p, pu)
  y = getfield(sl, :y)
  s = zero(promote_type(eltype(arg), eltype(y)))
  @turbo for i ∈ eachindex(arg)
    s += abs(arg[i] - y[i])
  end
  s, p, pu
end
function (sl::AbstractLoss{<:AbstractArray{<:AbstractArray}})(arg, p, pu)
  y = getfield(sl, :y)
  s = zero(promote_type(eltype(arg), eltype(first(y))))
  for yᵢ ∈ y
    sᵢ, p, pu = sl(yᵢ)(arg, p, pu)
    @fastmath s += sᵢ
  end
  return s, p, pu
end

"""
    LogitCrossEntropyLoss

Calculates mean logit cross-entropy loss.
"""
struct LogitCrossEntropyLoss{Y<:Union{AbstractVector{UInt32},Nothing}} <:
       AbstractLoss{Y}
  y::Y
end
# function LogitCrossEntropyLoss!(y::AbstractVector{UInt32})
#   m = 0xffffffff
#   @turbo for i = eachindex(y)
#     m = min(i, y[i])
#   end
#   @turbo for i = eachindex(y)
#     y[i] -= m
#   end
#   LogitCrossEntropyLoss(y)
# end
# function LogitCrossEntropyLoss!(y::AbstractVector{UInt32}, x)
#   m = 0xffffffff
#   @turbo for i = eachindex(y)
#     m = min(i, x[i])
#   end
#   @turbo for i = eachindex(y)
#     y[i] = x[i] - m
#   end
#   LogitCrossEntropyLoss(y)
# end
# LogitCrossEntropyLoss(y) = LogitCrossEntropyLoss!(Vector{UInt32}(undef, length(y)), y)

LogitCrossEntropyLoss() = LogitCrossEntropyLoss(nothing)
target(sl::LogitCrossEntropyLoss) = getfield(sl, :y)
(::LogitCrossEntropyLoss)(Y::AbstractVector{UInt32}) = LogitCrossEntropyLoss(Y)

function layer_output_size(
  ::Val{T},
  sl::LogitCrossEntropyLoss,
  s::Tuple
) where {T}
  _layer_output_size_needs_temp_of_equal_len_as_target(Val{T}(), sl, s)
end
function forward_layer_output_size(
  ::Val{T},
  sl::LogitCrossEntropyLoss,
  s
) where {T}
  _layer_output_size_needs_temp_of_equal_len_as_target(Val{T}(), sl, s)
end

function (lcel::LogitCrossEntropyLoss)(
  arg::AbstractArray{T},
  p::Ptr,
  pu
) where {T}
  y = lcel.y
  N = length(y)
  m = PtrArray(Ptr{T}(pu), (N,))
  unnormalized_logsoftmax!(arg, m, arg)
  s = zero(T)
  @turbo for i in eachindex(y)
    s -= arg[y[i], i] - m[i]
  end
  s / N, p, pu
end
function chain_valgrad!(
  __,
  arg::AbstractArray{T},
  layers::Tuple{LogitCrossEntropyLoss},
  _::Ptr,
  pu::Ptr{UInt8}
) where {T}
  y = getfield(getfield(layers, 1), :y)
  N = length(y)
  m = PtrArray(Ptr{T}(pu), (N,))
  logsoftmax!(arg, m, arg)
  s = zero(T)
  @turbo for i in eachindex(y)
    s -= arg[y[i], i]
  end
  @turbo for i in eachindex(arg)
    arg[i] = exp(arg[i])
  end
  @turbo for i in eachindex(y)
    arg[y[i], i] -= one(T)
  end
  return s, arg, pu
end
function Base.getindex(sl::LogitCrossEntropyLoss, r)
  LogitCrossEntropyLoss(view(target(sl), r))
end
function correct_count(Ŷ, Y)
  ntot = 0
  @inbounds for i in eachindex(Y)
    k = -1
    m = typemin(eltype(Ŷ))
    for j in axes(Ŷ, 1)
      cmp = Ŷ[j, i] > m
      k = ifelse(cmp, j, k)
      m = ifelse(cmp, Ŷ[j, i], m)
    end
    ntot += k == Y[i]
  end
  return ntot
end
function correct_count(c::SimpleChain, X, p)
  cnl, loss = split_loss(c)
  Ŷ = cnl(X, p)
  correct_count(Ŷ, target(loss))
end
@inline __loss(_, pu, loss::F, arg, p) where {F} = loss(arg, p, pu)
function correct_count_and_loss(
  c::SimpleChain,
  X::AbstractArray{T},
  p
) where {T}
  cnl, loss = split_loss(c)
  Ŷ = cnl(X, p)
  ec = correct_count(Ŷ, target(loss))
  os = first(layer_output_size(Val(T), loss, static_size(X)))
  GC.@preserve p (ec, with_memory(__loss, c, os, loss, Ŷ, pointer(p))...)
end
function correct_count_and_loss(
  c::SimpleChain,
  X::AbstractArray{T},
  Y,
  p
) where {T}
  correct_count_and_loss(add_loss(c, pop_loss(c)(Y)), X, p)
end
function accuracy_and_loss(c::SimpleChain, X, args...)
  cnt, l = correct_count_and_loss(c, X, args...)
  cnt / static_size(X)[end], l
end

_params(::Tuple{AbstractLoss}, _, __) = ()
