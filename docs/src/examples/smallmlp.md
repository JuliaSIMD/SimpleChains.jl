# Small Multi-Layer Perceptron

Here, we'll fit a simple network:
```julia
using SimpleChains

mlpd = SimpleChain(
  static(4),
  TurboDense(tanh, 32),
  TurboDense(tanh, 16),
  TurboDense(identity, 4)
)
```

Our goal here will be to try and approximate the matrix exponential:
```julia
function f(x)
  N = Base.isqrt(length(x))
  A = reshape(view(x, 1:N*N), (N,N))
  expA = exp(A)
  vec(expA)
end

T = Float32;
X = randn(T, 2*2, 10_000);
Y = reduce(hcat, map(f, eachcol(X)));
Xtest = randn(T, 2*2, 10_000);
Ytest = reduce(hcat, map(f, eachcol(Xtest)));
```

Now, to train our network:
```julia
@time p = SimpleChains.init_params(mlpd);
G = SimpleChains.alloc_threaded_grad(mlpd);

mlpdloss = SimpleChains.add_loss(mlpd, SquaredLoss(Y));
mlpdtest = SimpleChains.add_loss(mlpd, SquaredLoss(Ytest));

# define a function named report to calculate and report the value of loss function with train and test sets.
report = let mtrain = mlpdloss, X=X, Xtest=Xtest, mtest = mlpdtest
  p -> begin
    let train = mtrain(X, p), test = mtest(Xtest, p)
      @info "Loss:" train test
    end
  end
end

report(p)
for _ in 1:3
  @time SimpleChains.train_unbatched!(
    G, p, mlpdloss, X, SimpleChains.ADAM(), 10_000
  );
  report(p)
end
```
I get
```julia
julia> for _ in 1:3
         @time SimpleChains.train_unbatched!(
           G, p, mlpdloss, X, SimpleChains.ADAM(), 10_000
         );
         report(p)
       end
  5.258996 seconds (7.83 M allocations: 539.553 MiB, 4.18% gc time, 69.59% compilation time)
┌ Info: Loss:
│   train = 1243.1248f0
└   test = 483.38852f0
  1.638860 seconds
┌ Info: Loss:
│   train = 96.98259f0
└   test = 210.4579f0
  1.654781 seconds
┌ Info: Loss:
│   train = 44.350838f0
└   test = 164.85913f0

julia> versioninfo()
Julia Version 1.9.0-DEV.1189
Commit 293031b4a5* (2022-08-26 20:24 UTC)
Platform Info:
  OS: Linux (x86_64-redhat-linux)
  CPU: 8 × 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-14.0.5 (ORCJIT, tigerlake)
  Threads: 8 on 8 virtual cores
Environment:
  JULIA_NUM_THREADS = 8
```

