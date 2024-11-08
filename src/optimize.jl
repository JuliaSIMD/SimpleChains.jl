
abstract type AbstractOptimizer end

"""
    ADAM(η = 0.001, β = (0.9, 0.999))

ADAM optimizer.
"""
struct ADAM <: AbstractOptimizer
  η::Float64
  β::Tuple{Float64,Float64}
end

ADAM(η = 0.001) = ADAM(η, (0.9, 0.999))

function update!(o::ADAM, (mt, vt, βp), x, Δ, s = One())
  @unpack η, β = o

  β₁ = β[1]
  β₂ = β[2]
  βp₁ = βp[1]
  βp₂ = βp[2]
  st = eltype(Δ)(s)
  @turbo for i ∈ eachindex(Δ)
    Δᵢ = Δ[i] * st
    mt[i] = β₁ * mt[i] + (1 - β₁) * Δᵢ
    vt[i] = β₂ * vt[i] + (1 - β₂) * Δᵢ^2
    Δxᵢ = mt[i] / ((1 - βp₁) * (sqrt(vt[i] / (1 - βp₂)) + 1.0f-8))
    x[i] -= η * Δxᵢ
  end
  βp[1] = βp₁ * β₁
  βp[2] = βp₂ * β₂
  return
end
@inline function optmemsize(::ADAM, p::AbstractVector{T}) where {T}
  2align(sizeof(T) * length(p)) + align(1)
end
@inline function optmemory(
  opt::ADAM,
  p::AbstractVector{T},
  pu::Ptr{UInt8}
) where {T}
  memoff = align(sizeof(T) * length(p))
  mt = PtrArray(Ptr{T}(pu), (ArrayInterface.static_length(p),))
  pu += memoff
  vt = PtrArray(Ptr{T}(pu), (ArrayInterface.static_length(p),))
  @turbo for i ∈ eachindex(mt)
    mt[i] = 0
    vt[i] = 0
  end
  βp_doesnot_fit_at_end = sizeof(T) * length(p) + 16 > memoff
  pu_p_memoff = pu + memoff # aligned
  pβp = ifelse(βp_doesnot_fit_at_end, pu_p_memoff, pu + sizeof(T) * length(p))
  pu = pu_p_memoff
  βp = PtrArray(reinterpret(Ptr{Float64}, pβp), (static(2),))
  @unpack β = opt
  @inbounds βp[1] = β[1]
  @inbounds βp[2] = β[2]
  pu = ifelse(βp_doesnot_fit_at_end, pu + align(1), pu)
  return (mt, vt, βp), pu
end

function update!(
  g::AbstractVector{T},
  opt,
  Xp::AbstractArray{<:Any,N},
  layers,
  pen,
  sx,
  p,
  pm,
  optbuffer,
  _
) where {T,N}
  GC.@preserve g p chain_valgrad_entry!(
    nothing,
    pointer(g),
    Xp,
    layers,
    pointer(p),
    pm
  )
  apply_penalty!(g, pen, p, sx)
  gmul = loss_multiplier(last(layers), static_size(Xp, static(N)), T)
  update!(opt, optbuffer, p, g, gmul)
end
function chain_valgrad_thread!((g, Xp, layers, p, pm, mpt), start, stop)
  batchsize = static_size(Xp, ndims(Xp))
  if start > stop
    fill!(g, zero(eltype(g)))
    return nothing
  end
  off = start - 1
  nt = static_size(g, static(2))
  goff = static_strides(g, static(2)) * sizeof(eltype(g)) * off
  f = ((off * batchsize) ÷ nt) + 1
  l = (stop * batchsize) ÷ nt
  Xpv = view_slice_last(Xp, f:l)
  loss = last(layers)
  tgt = view_slice_last(target(loss), f:l)
  tgtpb = preserve_buffer(tgt)
  Xpb = preserve_buffer(Xpv)
  newlayers = (Base.front(layers)..., loss(PtrArray(tgt)))
  # newlayers = (Base.front(layers)..., last(layers)[f:l])
  GC.@preserve tgtpb Xpb begin
    chain_valgrad_entry!(
      nothing,
      pointer(g) + goff,
      PtrArray(Xpv),
      newlayers,
      pointer(p),
      pm + mpt * off
    )
  end
  return nothing
end
function update!(
  g::AbstractMatrix{T},
  opt,
  Xp::AbstractArray{<:Any,N},
  layers,
  pen,
  sx,
  p,
  pm,
  optbuffer,
  mpt
) where {T,N}
  nthread = static_size(g, static(2))
  Xpb = preserve_buffer(Xp)
  Xpp = PtrArray(Xp)
  loss = last(layers)
  tgt = target(loss)
  tgtpb = preserve_buffer(tgt)
  newlayers = (Base.front(layers)..., loss(PtrArray(tgt)))
  GC.@preserve Xpb tgtpb begin
    Polyester.batch(
      chain_valgrad_thread!,
      (nthread, nthread),
      g,
      Xpp,
      newlayers,
      p,
      pm,
      mpt
    )
  end
  @turbo for t = 2:nthread, i in axes(g, 1)
    g[i, 1] += g[i, t]
  end
  gpb = preserve_buffer(g)
  GC.@preserve gpb begin
    gv = PtrArray(pointer(g), (length(p),))
    apply_penalty!(gv, pen, p, sx)
    gmul = loss_multiplier(loss, static_size(Xp, static(N)), T)
    update!(opt, optbuffer, p, gv, gmul)
  end
end
# note that pstop - pstart = subrangelen, so it is not a closed-closed i
function shuffle_chain_valgrad_thread!(
  (g, Xp, layers, p, pm, mpt, perm, pstart, pstop),
  start,
  stop
)
  # will work over subrange of pstart+1:pstop
  # it is divided into nthread parts...
  subrangelen = pstop - pstart
  numthread = static_size(g, static(2))
  batchsize, r = divrem(subrangelen, numthread)
  off = start - 1
  goff =
    g isa AbstractVector ? 0 :
    static_strides(g, static(2)) * sizeof(eltype(g)) * off
  pm += mpt * off

  fm1 = off * batchsize + pstart + min(r, off)
  lastdim = batchsize + (start <= r)
  if !((lastdim > 0) & (subrangelen > 0))
    # fill!(g, 0)
    return nothing
  end
  l = fm1 + lastdim

  loss = last(layers)
  tgt = target(loss)
  tgtpb = preserve_buffer(tgt)
  eltgt = eltype(tgt)
  szeltgt = sizeof(eltgt)

  tgtsz = Base.front(static_size(tgt))
  tgttmp = PtrArray(Ptr{eltgt}(pm), (tgtsz..., lastdim))
  ptgttmp = pointer(tgttmp)
  tgtlen = tsprod(tgtsz)
  pm += align(szeltgt * tgtlen * lastdim)
  ptgt = pointer(tgt)
  GC.@preserve tgtpb begin
    for i = fm1:l-1
      @inbounds j = perm[i] # `perm` and `j` are zero-based
      @simd ivdep for k = 0:Int(tgtlen)-1
        # x = tgt[k+1,j+1]
        x = unsafe_load((ptgt + (tgtlen * szeltgt) * j) + k * szeltgt)
        unsafe_store!(ptgttmp + k * szeltgt, x)
      end
      ptgttmp += Int(tgtlen) * szeltgt
    end
  end
  newlayers = (Base.front(layers)..., loss(tgttmp))
  permview = StrideArraysCore.PtrArray0(
    pointer(perm) + (Base.elsize(perm) * fm1),
    (lastdim,)
  )
  chain_valgrad_entry!(
    nothing,
    pointer(g) + goff,
    Xp,
    newlayers,
    permview,
    pointer(p),
    pm
  )
  return nothing
end
function shuffle_update!(
  g::AbstractMatrix{T},
  opt,
  Xp::AbstractArray{<:Any,N},
  layers,
  pen,
  sx,
  p,
  pm,
  optbuffer,
  mpt,
  perm,
  pstart,
  pstop
) where {T,N}
  nthread = static_size(g, static(2))
  #=
  batchsize = pstop - pstart
  if batchsize < nthread
    gpb = preserve_buffer(g)
    GC.@preserve gpb begin
      if batchsize == 1
        gv = PtrArray(pointer(g), (length(p),))
        return shuffle_update!(
          gv,
          opt,
          Xp,
          layers,
          pen,
          sx,
          p,
          pm,
          optbuffer,
          mpt,
          perm,
          pstart,
          pstop,
        )
      else
        gm = PtrArray(stridedpointer(g), (length(p), batchsize), Val{(true, false)}())
        return shuffle_update!(
          gm,
          opt,
          Xp,
          layers,
          pen,
          sx,
          p,
          pm,
          optbuffer,
          mpt,
          perm,
          pstart,
          pstop,
        )
      end
    end
  end
  =#
  Polyester.batch(
    shuffle_chain_valgrad_thread!,
    (nthread, nthread),
    g,
    Xp,
    layers,
    p,
    pm,
    mpt,
    perm,
    pstart,
    pstop
  )

  @turbo for t = 2:nthread, i in axes(g, 1)
    g[i, 1] += g[i, t]
  end
  gpb = preserve_buffer(g)
  GC.@preserve gpb begin
    gv = PtrArray(pointer(g), (length(p),))
    apply_penalty!(gv, pen, p, sx)
    gmul = loss_multiplier(last(layers), static_size(Xp, static(N)), T)
    update!(opt, optbuffer, p, gv, gmul)
  end
  return nothing
end
function shuffle_update!(
  g::AbstractVector{T},
  opt,
  Xp::AbstractArray{<:Any,N},
  layers,
  pen,
  sx,
  p,
  pm,
  optbuffer,
  mpt,
  perm,
  pstart,
  pstop
) where {T,N}
  shuffle_chain_valgrad_thread!(
    (g, Xp, layers, p, pm, mpt, perm, pstart, pstop),
    static(1),
    static(1)
  )
  apply_penalty!(g, pen, p, sx)
  gmul = loss_multiplier(last(layers), static_size(Xp, static(N)), T)
  update!(opt, optbuffer, p, g, gmul)
end

function train_unbatched_core!(
  c::Chain,
  pu::Ptr{UInt8},
  g,
  pX,
  p,
  opt,
  t::AbstractArray,
  mpt
)
  chn = getchain(c)
  @unpack layers = chn
  pen = getpenalty(c)
  fl = Base.front(layers)
  ll = last(layers)
  sx = static_size(pX)
  optbuffer, pm = optmemory(opt, p, pu)
  GC.@preserve p g begin
    for y ∈ t
      layers_y = (fl..., ll(y))
      update!(g, opt, pX, layers_y, pen, sx, p, pm, optbuffer, mpt)
    end
  end
end
function train_unbatched_core!(
  c::Chain,
  pu::Ptr{UInt8},
  g,
  pX,
  p,
  opt,
  iters::Int,
  mpt
)
  chn = getchain(c)
  @unpack layers = chn
  pen = getpenalty(c)
  sx = static_size(pX)
  optbuffer, pm = optmemory(opt, p, pu)
  GC.@preserve p g begin
    for _ ∈ 1:iters
      update!(g, opt, pX, layers, pen, sx, p, pm, optbuffer, mpt)
    end
  end
end
function train_unbatched_core!(
  c::Chain,
  pu::Ptr{UInt8},
  pX,
  p::AbstractVector{T},
  opt,
  it,
  mpt
) where {T}
  numthreads = _numthreads()
  glen = _try_static(numparam(getchain(c)), static_length(p))
  aligned_glen = align(glen)
  g = _alloc_grad(Ptr{T}(pu), glen, numthreads, aligned_glen)
  offset = static_sizeof(T) * aligned_glen * numthreads
  if numthreads == 1
    train_unbatched_core!(c, pu + offset, g[:, begin], pX, p, opt, it, mpt)
  else
    train_unbatched_core!(c, pu + offset, g, pX, p, opt, it, mpt)
  end
end

"""
    train_unbatched!([g::AbstractVecOrMat, ]p, chn, X, opt, iters)

Train without batching inputs.

Arguments:

  - `g` pre-allocated gradient buffer. Can be allocated with `similar(p)` (if you want to run single threaded), or `alloc_threaded_grad(chn, size(X))` (`size(X)` argument is only necessary if the input dimension was not specified when constructing the chain). If a matrix, the number of columns gives how many threads to use. Do not use more threads than batch size would allow. This argument is optional. If excluded, it will run multithreaded (assuming you started Julia with multiple threads).
  - `p` is the parameter vector. It is updated inplace. It should be pre-initialized, e.g. with `init_params`/`init_params!`. This is to allow calling `train_unbatched!` several times to train in increments.
  - `chn` is the `SimpleChain`. It must include a loss (see `SimpleChains.add_loss`) containing the target information (dependent variables) you're trying to fit.
  - `X` the training data input argument (independent variables).
  - `opt` is the optimizer. Currently, only `SimpleChains.ADAM` is supported.
  - `iters`, how many iterations to train for.
"""
function train_unbatched!(
  g,
  p::AbstractVector,
  _chn::Chain,
  X,
  opt::AbstractOptimizer,
  t
)
  if g isa AbstractMatrix && static_size(g, static(2)) == 1
    gpb = preserve_buffer(g)
    gv = PtrArray(pointer(g), (length(p),))
    GC.@preserve gpb train_unbatched!(gv, p, _chn, X, opt, t)
    return p
  end

  chn = getchain(_chn)
  pX = maybe_static_size_arg(chn.inputdim, X)
  optoff = optmemsize(opt, p)
  @unpack layers = chn
  T = Base.promote_eltype(p, X)
  bytes_per_thread, total_bytes = required_bytes(
    Val{T}(),
    layers,
    static_size(pX),
    optoff,
    static(0),
    static_size(g, static(2))
  )
  GC.@preserve X begin
    with_memory(
      train_unbatched_core!,
      _chn,
      total_bytes,
      g,
      pX,
      p,
      opt,
      t,
      bytes_per_thread
    )
  end
  p
end

function train_unbatched!(
  p::AbstractVector,
  _chn::Chain,
  X::AbstractArray,
  opt::AbstractOptimizer,
  t
)
  chn = getchain(_chn)
  pX = maybe_static_size_arg(chn.inputdim, X)
  optoff = optmemsize(opt, p)
  @unpack layers = chn
  glen = _try_static(numparam(chn), static_length(p))
  numthreads = _numthreads()

  T = Base.promote_eltype(p, X)
  bytes_per_thread, total_bytes = required_bytes(
    Val{T}(),
    layers,
    static_size(pX),
    optoff + align(glen) * numthreads * static_sizeof(T),
    static(0),
    numthreads
  )
  GC.@preserve X begin
    with_memory(
      train_unbatched_core!,
      _chn,
      total_bytes,
      pX,
      p,
      opt,
      t,
      bytes_per_thread
    )
  end
  p
end
function train_unbatched!(
  g,
  p::AbstractVector,
  _chn::Chain,
  X,
  opt::AbstractOptimizer
)
  t = target(_chn)
  if _iterate_over_losses(t)
    train_unbatched!(g, p, _chn, X, opt, t)
  else
    train_unbatched!(g, p, _chn, X, opt, 10_000)
  end
end

@generated function turbo_dense_batch_size(
  indputdim::Integer,
  outputdim::Integer,
  Nd::Integer,
  ::StaticInt{W},
  ::StaticInt{RS},
  ::StaticInt{RC},
  ::StaticInt{CLS}
) where {W,RS,RC,CLS}
  Kk = Static.known(indputdim)
  Mk = Static.known(outputdim)
  Nk = Static.known(Nd)
  M = Mk === nothing ? 1024 : Mk
  K = Kk === nothing ? 1024 : Kk
  N = Nk === nothing ? 1024 : Nk
  _, nₖ = matmul_params(RS, RC, CLS; M, K, N, W)
  nₖ = ifelse(nₖ == -1, 16, nₖ)
  StaticInt(nₖ)
end
@inline function batch_size(
  layers::Tuple{L,Vararg},
  argsz::Tuple{I,J},
  ::Val{T}
) where {T,L<:TurboDense,I,J}
  inputdim, N = argsz
  outputdim = first(layers).outputdim
  # id, od = getfield(getfield(layers,1), :dims) # (od × id) * (id x N)
  turbo_dense_batch_size(
    inputdim,
    outputdim,
    N,
    VectorizationBase.pick_vector_width(T),
    register_size(),
    register_count(),
    cache_linesize()
  )
end
@inline function batch_size(
  layers::Tuple{L,Vararg},
  argsz::Tuple,
  ::Val{T}
) where {L,T}
  _, argsz2 = layer_output_size(Val{T}(), getfield(layers, 1), argsz)
  batch_size(Base.tail(layers), argsz2, Val(T))
end
@inline batch_size(::Tuple{}, ::Tuple, ::Val{T}) where {T} = static(18)

@inline view_slice_last(X::AbstractArray{<:Any,1}, r) = view(X, r)
@inline view_slice_last(X::AbstractArray{<:Any,2}, r) = view(X, :, r)
@inline view_slice_last(X::AbstractArray{<:Any,3}, r) = view(X, :, :, r)
@inline view_slice_last(X::AbstractArray{<:Any,4}, r) = view(X, :, :, :, r)
@inline view_slice_last(X::AbstractArray{<:Any,5}, r) = view(X, :, :, :, :, r)
function train_batched_core!(
  _chn::Chain,
  pu::Ptr{UInt8},
  g::AbstractVecOrMat{T},
  p::AbstractVector{T},
  pX,
  opt::AbstractOptimizer,
  iters,
  leaveofflast::Bool,
  mpt,
  N_bs
) where {T}
  chn = getchain(_chn)
  pen = getpenalty(_chn) / N_bs
  @unpack layers = chn
  sx = chain_input_dims(chn, static_size(pX))
  N = last(sx)
  # need to shuffle `N`
  perm_mem = align(sizeof(Int) * N)

  loss = last(layers)
  Y = preserve_buffer(loss)
  newlayers = (Base.front(layers)..., loss(PtrArray(Y)))
  GC.@preserve p g Y begin
    optbuffer, pm = optmemory(opt, p, pu)
    perm = StrideArraysCore.PtrArray0(Ptr{Int}(pm), (N,))
    pm += perm_mem
    d, r = divrem(N, N_bs)
    d += r != 0
    r = ifelse(r != 0, r, N_bs)
    @inbounds for n = 0:N-1
      perm[n] = n
    end
    iter = 0
    while true
      doff = 0
      while true
        doffnext = doff + N_bs
        ifelse(leaveofflast, doffnext, doff) > (N - (!leaveofflast)) && break
        batchstop::Int = min(doffnext, N)
        shuffle_update!(
          g,
          opt,
          pX,
          newlayers,
          pen,
          sx,
          p,
          pm,
          optbuffer,
          mpt,
          perm,
          doff,
          batchstop
        )
        doff = doffnext
      end
      (iter += 1) < iters || break
      randpermzero!(perm)
    end
  end
end
function train_batched_core!(
  c::Chain,
  pu::Ptr{UInt8},
  ::Nothing,
  p::AbstractVector,
  pX,
  opt::AbstractOptimizer,
  iters,
  leaveofflast::Bool,
  mpt,
  N_bs
)
  numthreads = _numthreads()
  glen = _try_static(numparam(getchain(c)), static_length(p))
  aligned_glen = align(glen)
  T = Base.promote_eltype(p, pX)
  g = _alloc_grad(Ptr{T}(pu), glen, numthreads, aligned_glen)
  offset = static_sizeof(T) * aligned_glen * numthreads
  if numthreads == 1
    train_batched_core!(
      c,
      pu + offset,
      g[:, begin],
      p,
      pX,
      opt,
      iters,
      leaveofflast,
      mpt,
      N_bs
    )
  else
    train_batched_core!(
      c,
      pu + offset,
      g,
      p,
      pX,
      opt,
      iters,
      leaveofflast,
      mpt,
      N_bs
    )
  end
end
"""
    train_batched!(g::AbstractVecOrMat, p, chn, X, opt, iters; batchsize = nothing)

Train while batching arguments.

Arguments:

  - `g` pre-allocated gradient buffer. Can be allocated with `similar(p)` (if you want to run single threaded), or `alloc_threaded_grad(chn, size(X))` (`size(X)` argument is only necessary if the input dimension was not specified when constructing the chain). If a matrix, the number of columns gives how many threads to use. Do not use more threads than batch size would allow.
  - `p` is the parameter vector. It is updated inplace. It should be pre-initialized, e.g. with `init_params`/`init_params!`. This is to allow calling `train_unbatched!` several times to train in increments.
  - `chn` is the `SimpleChain`. It must include a loss (see `SimpleChains.add_loss`) containing the target information (dependent variables) you're trying to fit.
  - `X` the training data input argument (independent variables).
  - `opt` is the optimizer. Currently, only `SimpleChains.ADAM` is supported.
  - `iters`, how many iterations to train for.
  - `batchsize` keyword argument: the size of the batches to use. If `batchsize = nothing`, it'll try to do a half-decent job of picking the batch size for you. However, this is not well optimized at the moment.
"""
function train_batched!(
  g::Union{Nothing,AbstractVector,AbstractMatrix},
  p::AbstractVector,
  _chn::Chain,
  X,
  opt::AbstractOptimizer,
  iters;
  batchsize = nothing,
  leaveofflast::Bool = false
)
  if g isa AbstractMatrix && static_size(g, static(2)) == 1
    gpb = preserve_buffer(g)
    gv = PtrArray(pointer(g), (length(p),))
    GC.@preserve gpb train_batched!(gv, p, _chn, X, opt, iters; batchsize)
    return p
  end
  chn = getchain(_chn)
  pX = maybe_static_size_arg(chn.inputdim, X)
  @unpack layers = chn
  optoff = optmemsize(opt, p)
  sx = chain_input_dims(chn, static_size(pX))
  N = last(sx)
  # need to shuffle `N`
  tgt = target(chn)
  nthread = g === nothing ? _numthreads() : static_size(g, static(2))
  N_bs = if batchsize === nothing
    static(8) *
    batch_size(layers, sx, Val(promote_type(eltype(p), eltype(X)))) *
    nthread
  else
    batchsize
  end
  if N_bs >= N
    train_unbatched!(g, p, _chn, X, opt, iters)
    return p
  end
  tgt_batch_len = tsprod(Base.front(static_size(tgt))) * N_bs
  X_batch_len = tsprod(Base.front(sx)) * N_bs
  sxb = (Base.front(sx)..., N_bs)
  shuffle_per_thread =
    align(sizeof(eltype(tgt)) * tgt_batch_len) +
    align(sizeof(eltype(X)) * X_batch_len)
  perm_mem = align(sizeof(Int) * N)
  base_mem = optoff + perm_mem
  T = Base.promote_eltype(p, X)
  if g === nothing
    base_mem +=
      align(_try_static(numparam(chn), static_length(p))) *
      nthread *
      static_sizeof(T)
  end
  mpt, total_bytes =
    required_bytes(Val{T}(), layers, sxb, base_mem, shuffle_per_thread, nthread)
  GC.@preserve X begin
    with_memory(
      train_batched_core!,
      _chn,
      total_bytes,
      g,
      p,
      pX,
      opt,
      iters,
      leaveofflast,
      mpt,
      N_bs
    )
  end
  p
end
function train_batched!(
  p::AbstractVector{T},
  c::Chain,
  X,
  opt::AbstractOptimizer,
  iters;
  batchsize = nothing,
  leaveofflast::Bool = false
) where {T}
  train_batched!(nothing, p, c, X, opt, iters; batchsize, leaveofflast)
end
_isstochastic(::Tuple{}) = false
function _isstochastic(x::Tuple{T,Vararg}) where {T}
  isstochastic(getfield(x, 1)) ? true : _isstochastic(Base.tail(x))
end

isstochastic(chn::Chain) = _isstochastic(getfield(getchain(chn), :layers))

function train!(p, chn::Chain, X, opt::AbstractOptimizer, iters)
  if isstochastic(chn)
    train_unbatched!(p, chn, X, opt, iters)
  else
    train_batched!(p, chn, X, opt, iters)
  end
end
function train!(g, p, chn::Chain, X, opt::AbstractOptimizer, iters)
  if isstochastic(chn)
    train_unbatched!(g, p, chn, X, opt, iters)
  else
    train_batched!(g, p, chn, X, opt, iters)
  end
end

for t ∈ [:train, :train_batched, :train_unbatched]
  t! = Symbol(t, :!)
  @eval function $t(chn::Chain, X, opt, iters; rng::AbstractRNG = local_rng())
    $t!(init_params(chn, nothing, eltype(X); rng), chn, X, opt, iters)
  end
end
