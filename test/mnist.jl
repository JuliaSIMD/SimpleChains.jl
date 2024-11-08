using Test
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
@testset "LeNet" begin
  using SimpleChains, MLDatasets, JET

  lenet = SimpleChain(
    (static(28), static(28), static(1)),
    SimpleChains.Conv(SimpleChains.relu, (5, 5), 6),
    SimpleChains.MaxPool(2, 2),
    SimpleChains.Conv(SimpleChains.relu, (5, 5), 16),
    SimpleChains.MaxPool(2, 2),
    Flatten(3),
    TurboDense(SimpleChains.relu, 120),
    TurboDense(SimpleChains.relu, 84),
    TurboDense(identity, 10)
  )
  # 3d and 0-indexed
  xtrain3, ytrain0 = MLDatasets.MNIST(:train)[:]
  xtest3, ytest0 = MLDatasets.MNIST(:test)[:]
  xtrain4 = reshape(xtrain3, 28, 28, 1, :)
  xtest4 = reshape(xtest3, 28, 28, 1, :)
  ytrain1 = UInt32.(ytrain0 .+ 1)
  ytest1 = UInt32.(ytest0 .+ 1)
  lenetloss = SimpleChains.add_loss(lenet, LogitCrossEntropyLoss(ytrain1))

  @test SimpleChains.outputdim(lenet, size(xtrain4)) == (10, length(ytrain1))
  @test SimpleChains.outputdim(lenet, size(xtest4)) == (10, length(ytest1))

  # initialize parameters
  @time p = SimpleChains.init_params(lenet; rng = SimpleChains.local_rng())
  @test all(isfinite, p)
  @test_opt SimpleChains.init_params!(p, lenet)

  @testset "Cache Corrupting Results" begin
    g = similar(p)
    subset = 1:200
    x = xtrain4[:, :, :, subset]
    y = ytrain1[subset]
    let lenetloss =
        SimpleChains.add_loss(lenet, SimpleChains.LogitCrossEntropyLoss(y))
      lenentmem = SimpleChains.get_heap_memory(lenetloss, 0)
      lenentmem .= 0x00
      valgrad!(g, lenetloss, x, p)
      @test_opt valgrad!(g, lenetloss, x, p)
      g2 = similar(g)
      lenentmem .= 0xff
      valgrad!(g2, lenetloss, x, p)
      @test g == g2
    end
  end

  # initialize a gradient buffer matrix; number of columns places an upper bound
  # on the number of threads used.
  # G = similar(p, length(p), min(Threads.nthreads(), (Sys.CPU_THREADS ÷ ((Sys.ARCH === :x86_64) + 1))));
  # train
  @time SimpleChains.train_batched!(
    p,
    lenetloss,
    xtrain4,
    SimpleChains.ADAM(3e-4),
    10
  )
  if VERSION >= v"1.10"
    @test_opt SimpleChains.train_batched!(
      p,
      lenetloss,
      xtrain4,
      SimpleChains.ADAM(3e-4),
      10
    )
  end
  @test all(isfinite, p)
  # @test all(isfinite, G)
  # assess training and test loss
  a0, l0 = SimpleChains.accuracy_and_loss(lenetloss, xtrain4, p)
  a1, l1 = SimpleChains.accuracy_and_loss(lenetloss, xtest4, ytest1, p)
  G = SimpleChains.alloc_threaded_grad(lenetloss)
  fill!(G, NaN)
  # train without additional memory allocations
  @time SimpleChains.train_batched!(
    G,
    p,
    lenetloss,
    xtrain4,
    SimpleChains.ADAM(3e-4),
    10
  )
  if VERSION >= v"1.10"
    @test_opt SimpleChains.train_batched!(
      G,
      p,
      lenetloss,
      xtrain4,
      SimpleChains.ADAM(3e-4),
      10
    )
  end
  @test all(isfinite, p)
  @test all(isfinite, G)
  g = Matrix{eltype(G)}(undef, size(G, 1), 1)
  @time SimpleChains.train_batched!(
    g,
    p,
    lenetloss,
    xtrain4,
    SimpleChains.ADAM(3e-4),
    2
  )
  @test all(isfinite, p)
  @test all(isfinite, g)

  # assess training and test loss
  a2, l2 = SimpleChains.accuracy_and_loss(lenetloss, xtrain4, p)
  @test l2 ≈ lenetloss(xtrain4, p)
  a3, l3 = SimpleChains.accuracy_and_loss(lenetloss, xtest4, ytest1, p)
  @test l3 ≈ SimpleChains.add_loss(lenetloss, LogitCrossEntropyLoss(ytest1))(
    xtest4,
    p
  )
  if size(G, 2) <= 4
    @test a0 > 0.94
    @test a2 > 0.96
    @test a1 > 0.94
    @test a3 > 0.96
  else
    @test a0 > 0.93
    @test a2 > 0.95
    @test a1 > 0.93
    @test a3 > 0.95
  end

  @time SimpleChains.init_params!(p, lenet; rng = SimpleChains.local_rng())
  @time SimpleChains.train_unbatched!(
    p,
    lenetloss,
    xtrain4,
    SimpleChains.ADAM(3e-4),
    10
  )
  if VERSION >= v"1.10"
    @test_opt SimpleChains.train_unbatched!(
      p,
      lenetloss,
      xtrain4,
      SimpleChains.ADAM(3e-4),
      10
    )
  end
  a4, l4 = SimpleChains.accuracy_and_loss(lenetloss, xtrain4, p)
  @test l4 ≈ lenetloss(xtrain4, p)
  a5, l5 = SimpleChains.accuracy_and_loss(lenetloss, xtest4, ytest1, p)
  @test l5 ≈ SimpleChains.add_loss(lenetloss, LogitCrossEntropyLoss(ytest1))(
    xtest4,
    p
  )
  # TODO: unbatched training is currently much less effective...
  @test a4 > 0.3
  @test a5 > 0.3
end
