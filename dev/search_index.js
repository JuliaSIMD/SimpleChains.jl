var documenterSearchIndex = {"docs":
[{"location":"examples/mnist/#MNIST-Convolutions","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"","category":"section"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"First, we load the data using MLDatasets.jl:","category":"page"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"using MLDatasets\nxtrain3, ytrain0 = MLDatasets.MNIST.traindata(Float32);\nxtest3, ytest0 = MLDatasets.MNIST.testdata(Float32);\nsize(xtest3)\n# (28, 28, 60000)\nextrema(ytrain0) # digits, 0,...,9\n# (0, 9)","category":"page"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"The covariate data (x) were named 3 as these are three-dimensional arrays, containing the height x width x number of images. The training data are vectors indicating the digit.","category":"page"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"xtrain4 = reshape(xtrain3, 28, 28, 1, :);\nxtest4 = reshape(xtest3, 28, 28, 1, :);\nytrain1 = UInt32.(ytrain0 .+ 1);\nytest1 = UInt32.(ytest0 .+ 1);","category":"page"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"SimpleChains' convolutional layers expect that we have a channels-in dimension, so we shape the images to be four dimensional It also currently defaults to 1-based indexing for its categories, so we shift all categories by 1.","category":"page"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"We now define our model, LeNet5:","category":"page"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"using SimpleChains\n\nlenet = SimpleChain(\n  (static(28), static(28), static(1)),\n  SimpleChains.Conv(SimpleChains.relu, (5, 5), 6),\n  SimpleChains.MaxPool(2, 2),\n  SimpleChains.Conv(SimpleChains.relu, (5, 5), 16),\n  SimpleChains.MaxPool(2, 2),\n  Flatten(3),\n  TurboDense(SimpleChains.relu, 120),\n  TurboDense(SimpleChains.relu, 84),\n  TurboDense(identity, 10),\n)\n\nlenetloss = SimpleChains.add_loss(lenet, LogitCrossEntropyLoss(ytrain1));","category":"page"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"We define the inputs as being statically sized (28,28,1) images. Specifying the input sizes allows these to be checked. Making them static, which we can do either in our simple chain, or by adding static sizing to the images themselves using a package like StrideArrays.jl or HybridArrays.jl. These packages are recomennded for allowing you to mix dynamic and static sizes; the batch size should probably be left dynamic, as you're unlikely to want to specialize code generation on this, given that it is likely to vary, increasing compile times while being unlikely to improve runtimes.","category":"page"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"In SimpleChains, the parameters are not a part of the model, but live as a separate vector that you can pass around to optimizers of your choosing. If you specified the input size, you create a random initial parameter vector corresponding to the model:","category":"page"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"@time p = SimpleChains.init_params(lenet);","category":"page"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"The convolutional layers are initialized with a Glorot (Xavier) unifirom distribution, while the dense layers are initialized with a Glorot (Xaviar) normal distribution. Biases are initialized to zero. Because the number of parameters can be a function of the input size, these must be provided if you didn't specify input dimension. For example:","category":"page"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"@time p = SimpleChains.init_params(lenet, size(xtrain4));","category":"page"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"To allow training to use multiple threads, you can create a gradient matrix, with a number of rows equal to the length of the parameter vector p, and one column per thread. For example:","category":"page"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"estimated_num_cores = (Sys.CPU_THREADS ÷ ((Sys.ARCH === :x86_64) + 1));\nG = similar(p, length(p), min(Threads.nthreads(), estimated_num_cores);","category":"page"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"Here, we're estimating that the number of physical cores is half the number of threads on an x86_64 system, which is true for most – but not all!!! – of them. Otherwise, we're assuming it is equal to the number of threads. This is of course also likely to be wrong, e.g. recent Power CPUs may habe 4 or even 8 threads per core. You may wish to change this, or use Hwloc.jl for an accurate number.","category":"page"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"Now that this is all said and done, we can train for 10 epochs using the ADAM optimizer with a learning rate of 3e-4, and then assess the accuracy and loss of both the training and test data:","category":"page"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"@time SimpleChains.train_batched!(G, p, lenetloss, xtrain4, SimpleChains.ADAM(3e-4), 10);\nSimpleChains.accuracy_and_loss(lenetloss, xtrain4, p)\nSimpleChains.accuracy_and_loss(lenetloss, xtest4, ytest1, p)","category":"page"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"Training for an extra 10 epochs should be fast on most systems. Performance is currently known to be poor on the M1 (PRs welcome, otherwise we'll look into this eventually), but should be  good/great on systems with AVX2/AVX512:","category":"page"},{"location":"examples/mnist/","page":"MNIST - Convolutions","title":"MNIST - Convolutions","text":"@time SimpleChains.train_batched!(G, p, lenetloss, xtrain4, SimpleChains.ADAM(3e-4), 10);\nSimpleChains.accuracy_and_loss(lenetloss, xtrain4, p)\nSimpleChains.accuracy_and_loss(lenetloss, xtest4, ytest1, p)","category":"page"},{"location":"examples/smallmlp/#Small-Multi-Layer-Perceptron","page":"Small Multi-Layer Perceptron","title":"Small Multi-Layer Perceptron","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = SimpleChains","category":"page"},{"location":"#SimpleChains","page":"Home","title":"SimpleChains","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for SimpleChains.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [SimpleChains]","category":"page"},{"location":"#SimpleChains.AbstractPenalty","page":"Home","title":"SimpleChains.AbstractPenalty","text":"AbstractPenalty\n\nThe AbstractPenalty interface requires supporting the following methods:\n\ngetchain(::AbstractPenalty)::SimpleChain returns a SimpleChain if it is carrying one.\napply_penalty(::AbstractPenalty, params)::Number returns the penalty\napply_penalty!(grad, ::AbstractPenalty, params)::Number returns the penalty and updates grad to add the gradient.\n\n\n\n\n\n","category":"type"},{"location":"#SimpleChains.Dropout","page":"Home","title":"SimpleChains.Dropout","text":"Dropout(p) # 0 < p < 1\n\nDropout layer.\n\nWhen evaluated without gradients, it multiplies inputs by (1 - p). When evaluated with gradients, it randomly zeros p proportion of inputs.\n\n\n\n\n\n","category":"type"},{"location":"#SimpleChains.FrontLastPenalty","page":"Home","title":"SimpleChains.FrontLastPenalty","text":"FrontLastPenalty(SimpleChain, frontpen(λ₁...), lastpen(λ₂...))\n\nApplies frontpen to all but the last layer, applying lastpen to the last layer instead. \"Last layer\" here ignores the loss function, i.e. if the last element of the chain is a loss layer, the then lastpen applies to the layer preceding this.\n\n\n\n\n\n","category":"type"},{"location":"#SimpleChains.TurboDense","page":"Home","title":"SimpleChains.TurboDense","text":"TurboDense{B}(outputdim, activation)\n\n\n\n\n\n","category":"type"},{"location":"#Base.front-Tuple{SimpleChain}","page":"Home","title":"Base.front","text":"Base.front(c::SimpleChain)\n\nUseful for popping off a loss layer.\n\n\n\n\n\n","category":"method"},{"location":"#SimpleChains.numparam-Tuple{TurboDense, Tuple}","page":"Home","title":"SimpleChains.numparam","text":"numparam(d::Layer, inputdim::Tuple)\n\nReturns a Tuple{Int,S}. The first element is the number of parameters required by the layer given an argument of size inputdim. The second argument is the size of the object returned by the layer, which can be fed into numparam of the following layer.\n\n\n\n\n\n","category":"method"},{"location":"#SimpleChains.valgrad!-Tuple{Any, SimpleChain, Any, Any}","page":"Home","title":"SimpleChains.valgrad!","text":"Allowed destruction:\n\nvalgrad_layer! Accepts return of previous layer (B) and returns an ouput C. If an internal layer, allowed to destroy B (e.g. dropout layer).\n\npullback! Accepts adjoint of its return (C̄). It is allowed to destroy this. It is also allowed to destroy the previous layer's return B to produce B̄ (the C̄ it receives). Thus, the pullback is not allowed to depend on C, as it may have been destroyed in producing C̄.\n\n\n\n\n\n","category":"method"},{"location":"","page":"Home","title":"Home","text":"Pages = [\"examples/smallmlp.md\", \"examples/mnist.md\"]","category":"page"}]
}
