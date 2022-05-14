
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

function update!(o::ADAM, (mt, vt, βp), x, Δ)
  @unpack η, β = o

  β₁ = β[1]
  β₂ = β[2]
  βp₁ = βp[1]
  βp₂ = βp[2]
  @turbo for i ∈ eachindex(Δ)
    mt[i] = β₁ * mt[i] + (1 - β₁) * Δ[i]
    vt[i] = β₂ * vt[i] + (1 - β₂) * Δ[i]^2
    Δᵢ = η * mt[i] / ((1 - βp₁) * (sqrt(vt[i] / (1 - βp₂)) + 1e-8))
    x[i] -= Δᵢ
  end
  βp[1] = βp₁ * β₁
  βp[2] = βp₂ * β₂
  return
end
@inline function optmemsize(::ADAM, p::AbstractVector{T}) where {T}
  2align(sizeof(T) * length(p)) + align(1)
end
@inline function optmemory(opt::ADAM, p::AbstractVector{T}, pu::Ptr{UInt8}) where {T}
  memoff = align(sizeof(T) * length(p))
  mt = PtrArray(reinterpret(Ptr{T}, pu), (ArrayInterface.static_length(p),))
  pu += memoff
  vt = PtrArray(reinterpret(Ptr{T}, pu), (ArrayInterface.static_length(p),))
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


function update!(g::AbstractVector, opt, Xp, layers, pen, sx, p, pm, optbuffer, _)
  chain_valgrad_entry!(pointer(g), Xp, layers, pointer(p), pm)
  apply_penalty!(g, pen, p, sx)
  update!(opt, optbuffer, p, g)
end
function chain_valgrad_thread!((g, Xp, layers, p, pm, mpt), start, stop)
  batchsize = size(Xp, ndims(Xp))
  if start > stop
    fill!(g, zero(eltype(g)))
    return nothing
  end
  off = start - 1
  nt = size(g, static(2))
  goff = stride(g, static(2)) * sizeof(eltype(g)) * off
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
      pointer(g) + goff,
      PtrArray(Xpv),
      newlayers,
      pointer(p),
      pm + mpt * off,
    )
  end
  return nothing
end
function update!(g::AbstractMatrix, opt, Xp, layers, pen, sx, p, pm, optbuffer, mpt)
  nthread = size(g, static(2))
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
      mpt,
    )
  end
  @turbo for t = 2:nthread, i in axes(g, 1)
    g[i, 1] += g[i, t]
  end
  gpb = preserve_buffer(g)
  GC.@preserve gpb begin
    gv = PtrArray(pointer(g), (length(p),))
    apply_penalty!(gv, pen, p, sx)
    update!(opt, optbuffer, p, gv)
  end
end
# note that pstop - pstart = subrangelen, so it is not a closed-closed i
function shuffle_chain_valgrad_thread!(
  (g, Xp, layers, p, pm, mpt, perm, pstart, pstop),
  start,
  stop,
)
  # will work over subrange of pstart+1:pstop
  # it is divided into nthread parts...
  subrangelen = pstop - pstart
  numthread = size(g, static(2))
  batchsize, r = divrem(subrangelen, numthread)
  off = start - 1
  goff = g isa AbstractVector ? 0 : stride(g, static(2)) * sizeof(eltype(g)) * off
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

  tgtsz = Base.front(size(tgt))
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
  permview =
    StrideArraysCore.ptrarray0(pointer(perm) + (Base.elsize(perm) * fm1), (lastdim,))
  chain_valgrad_entry!(pointer(g) + goff, Xp, newlayers, permview, pointer(p), pm)
  return nothing
end
function shuffle_update!(
  g::AbstractMatrix,
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
  nthread = size(g, static(2))
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
    pstop,
  )
  
  @turbo for t = 2:nthread, i in axes(g, 1)
    g[i, 1] += g[i, t]
  end
  gpb = preserve_buffer(g)
  GC.@preserve gpb begin
    gv = PtrArray(pointer(g), (length(p),))
    apply_penalty!(gv, pen, p, sx)
    update!(opt, optbuffer, p, gv)
  end
  return nothing
end
function shuffle_update!(
  g::AbstractVector,
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
  shuffle_chain_valgrad_thread!(
    (g, Xp, layers, p, pm, mpt, perm, pstart, pstop),
    static(1),
    static(1),
  )
  apply_penalty!(g, pen, p, sx)
  update!(opt, optbuffer, p, g)
end

function train_unbatched!(g, p, _chn::Chain, X, opt::AbstractOptimizer, t::AbstractArray)
  if g isa AbstractMatrix && size(g, 2) == 1
    gpb = preserve_buffer(g)
    gv = PtrArray(pointer(g), (length(p),))
    GC.@preserve gpb train_unbatched!(gv, p, _chn, X, opt, t)
    return p
  end
  chn = getchain(_chn)
  pX = maybe_static_size_arg(chn.inputdim, X)
  pen = getpenalty(_chn)
  @unpack layers, memory = chn
  fl = Base.front(layers)
  ll = last(layers)
  optoff = optmemsize(opt, p)
  sx = ArrayInterface.size(pX)
  mpt = resize_memory!(layers, memory, pX, optoff, 0, size(g, static(2)))
  optbuffer, pm = optmemory(opt, p, pointer(memory))
  GC.@preserve p g memory X begin
    for y ∈ t
      layers_y = (fl..., ll(y))
      update!(g, opt, pX, layers_y, pen, sx, p, pm, optbuffer, mpt)
    end
  end
  p
end
"""
    train_unbatched!(g::AbstractVecOrMat, p, chn, X, opt, iters)

Train without batching inputs.

Arguments:
- `g` pre-allocated gradient buffer. Can be allocated with `similar(p)` (if you want to run single threaded), or `alloc_threaded_grad(chn, size(X))` (`size(X)` argument is only necessary if the input dimension was not specified when constructing the chain). If a matrix, the number of columns gives how many threads to use. Do not use more threads than batch size would allow.
- `p` is the parameter vector. It is updated inplace. It should be pre-initialized, e.g. with `init_params`/`init_params!`. This is to allow calling `train_unbatched!` several times to train in increments.
- `chn` is the `SimpleChain`. It must include a loss (see `SimpleChains.add_loss`) containing the target information (dependent variables) you're trying to fit.
- `X` the training data input argument (independent variables).
- `opt` is the optimizer. Currently, only `SimpleChains.ADAM` is supported.
- `iters`, how many iterations to train for.
"""
function train_unbatched!(g, p, _chn::Chain, X, opt::AbstractOptimizer, iters::Int)
  if g isa AbstractMatrix && size(g, 2) == 1
    gpb = preserve_buffer(g)
    gv = PtrArray(pointer(g), (length(p),))
    GC.@preserve gpb train_unbatched!(gv, p, _chn, X, opt, iters)
    return p
  end
  chn = getchain(_chn)
  pX = maybe_static_size_arg(chn.inputdim, X)
  pen = getpenalty(_chn)
  sx = ArrayInterface.size(pX)
  @unpack layers, memory = chn
  optoff = optmemsize(opt, p)
  mpt = resize_memory!(layers, memory, pX, optoff, 0, size(g, static(2)))
  optbuffer, pm = optmemory(opt, p, pointer(memory))
  GC.@preserve p g memory X begin
    for _ ∈ 1:iters
      update!(g, opt, pX, layers, pen, sx, p, pm, optbuffer, mpt)
    end
  end
  p
end
function train_unbatched!(g, p, _chn::Chain, X, opt::AbstractOptimizer)
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
  ::StaticInt{CLS},
) where {W,RS,RC,CLS}
  Kk = Static.known(indputdim)
  Mk = Static.known(outputdim)
  Nk = Static.known(Nd)
  M = Mk === nothing ? 1024 : Mk
  K = Kk === nothing ? 1024 : Kk
  N = Nk === nothing ? 1024 : Nk
  mₖ, nₖ = matmul_params(RS, RC, CLS; M, K, N, W)
  StaticInt(nₖ)
end
@inline function batch_size(
  layers::Tuple{L,Vararg},
  argsz::Tuple{I,J},
  ::Val{T},
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
    cache_linesize(),
  )
end
@inline function batch_size(layers::Tuple{L,Vararg}, argsz::Tuple, ::Val{T}) where {L,T}
  _, argsz2 = layer_output_size(Val{T}(), getfield(layers, 1), argsz)
  batch_size(Base.tail(layers), argsz2, Val(T))
end
@inline batch_size(::Tuple{}, ::Tuple, ::Val{T}) where {T} = static(18)


@inline view_slice_last(X::AbstractArray{<:Any,1}, r) = view(X, r)
@inline view_slice_last(X::AbstractArray{<:Any,2}, r) = view(X, :, r)
@inline view_slice_last(X::AbstractArray{<:Any,3}, r) = view(X, :, :, r)
@inline view_slice_last(X::AbstractArray{<:Any,4}, r) = view(X, :, :, :, r)
@inline view_slice_last(X::AbstractArray{<:Any,5}, r) = view(X, :, :, :, :, r)
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
  g::AbstractVecOrMat,
  p::AbstractVector,
  _chn::Chain,
  X,
  opt::AbstractOptimizer,
  iters;
  batchsize = nothing,
)
  if g isa AbstractMatrix && size(g, 2) == 1
    gpb = preserve_buffer(g)
    gv = PtrArray(pointer(g), (length(p),))
    GC.@preserve gpb train_batched!(gv, p, _chn, X, opt, iters; batchsize)
    return p
  end
  chn = getchain(_chn)
  pX = maybe_static_size_arg(chn.inputdim, X)
  pen = getpenalty(_chn)
  @unpack layers, memory = chn
  optoff = optmemsize(opt, p)
  sx = chain_input_dims(chn, size(pX))
  N = last(sx)
  # need to shuffle `N`
  tgt = target(chn)
  nthread = size(g, static(2))
  N_bs = if batchsize === nothing
    static(8) * batch_size(layers, sx, Val(promote_type(eltype(p), eltype(X)))) * nthread
  else
    batchsize
  end
  if N_bs >= N
    train_unbatched!(g, p, _chn, X, opt, iters)
    return p
  end
  tgt_batch_len = tsprod(Base.front(size(tgt))) * N_bs
  X_batch_len = tsprod(Base.front(sx)) * N_bs
  sxb = (Base.front(sx)..., N_bs)
  shuffle_per_thread =
    align(sizeof(eltype(tgt)) * tgt_batch_len) + align(sizeof(eltype(X)) * X_batch_len)
  perm_mem = align(sizeof(Int) * N)
  mpt = resize_memory!(
    layers,
    memory,
    Val(eltype(pX)),
    sxb,
    optoff + perm_mem,
    shuffle_per_thread,
    nthread,
  )
  loss = last(layers)
  Y = preserve_buffer(loss)
  newlayers = (Base.front(layers)..., loss(PtrArray(Y)))
  GC.@preserve p g memory X Y begin
    optbuffer, pm = optmemory(opt, p, pointer(memory))
    perm = StrideArraysCore.ptrarray0(Ptr{Int}(pm), (N,))
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
        # doffnext > N && break
        # batchstop = doffnext
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
          batchstop,
        )
        doff = doffnext
        doff >= N && break
      end
      (iter += 1) < iters || break
      randpermzero!(perm)
    end
  end
  p
end

_isstochastic(::Tuple{}) = false
function _isstochastic(x::Tuple{T,Vararg}) where {T}
  isstochastic(getfield(x, 1)) ? true : _isstochastic(Base.tail(x))
end

isstochastic(chn::Chain) = _isstochastic(getfield(getchain(chn), :layers))

function train!(g, p, chn::Chain, X, opt::AbstractOptimizer, iters)
  if isstochastic(chn)
    train_unbatched!(g, p, chn, X, opt, iters)
  else
    train_batched!(g, p, chn, X, opt, iters)
  end
end

for t ∈ [:train, :train_batched, :train_unbatched]
  t! = Symbol(t, :!)
  @eval function $t(chn::Chain, X, opt, iters)
    p = init_params(chn)
    g = similar(p)
    $t!(g, p, chn, X, opt, iters)
  end
end
