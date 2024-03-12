### Brief

- Wrapping of popular neural network libraries and providing of unified interface
- Automatic scaling and descaling of data 
- Brute force scanning of neural network configurations
- Graphic evaluation of training history

### Code example

    from blackboxes.box import Black

    X = [[...], [...], ...]  # train input (2D ArrayLike)
    Y = [[...], [...], ...]  # target      (2D ArrayLike)
    x = [[...], [...], ...]  # test input  (2D ArrayLike)
    
    phi = Black()
    y = phi(X=X, Y=Y, x=x, backend='torch', neurons=[6,4], trainer='adam')

### Purpose

The _blackboxes_ Python package serves as a versatile wrapper for various implementations of neural networks, facilitating switching between different backends like _Keras_, _NeuroLab_, _PyTorch_, etc. This flexibility enables users to leverage the specific strengths of each backend, optimizing performance for diverse hardware configurations. By offering this interoperability, _blackboxes_ helps users avoid vendor lock-in and empowers them to harness the potential of the best neural network implementation for a given application.

Additionally, _blackboxes_ specializes in finding optimal hyperparameters for neural networks. It employs brute force scanning to fine-tune model configurations. Moreover, _blackboxes_ exploits the effect of random initialization of neural networks, guaranteeing the discovery of (almost) optimal configurations.

### Motivation

Optimal hyper parameters of neural networks can be difficult to estimate from theoretical considerations. It is mathematically proven that neural networks work effectively for most regression problems of higher complexity.
However, algorithmic instructions for finding the optimal network configuration are often not available. 
Moreover, selecting from multiple optimal network structures contributes to achieving sufficient model performance.

Therefore an automatic configuration of network parameters is being proposed. This covers 

- variations of the number and size of hidden layers
- activation functions of hidden and output layers
- parameters of early stopping of the training or of deacy of weights
- the effect of random initialization of the network weights etc   

### Options for finding the optimal configuration

- Brute force scanning of hyper  parameter space (slow, but transparent) 
- Automatic solutions such as Google’s AutoML (automatic regulariztion, but closed hood with the risk of insufficient model understanding)

Brute force scanning has been employed due to its explicit transparency and robust implementation.

This exhaustive search method relies solely on guessing wide parameter ranges and eliminates the risk of the algorithm getting trapped in local optima.

### Implementation

Class _BruteForce_ in module _bruteforce_ performes nested search loops over selected hyper parameter ranges. 

![loops](https://github.com/dwweiss/blackboxes/blob/main/doc/fig/bruteforce_loops.png)

###### Figure 1: Loops (MSE: mean squared error)

The best configuration is chosen based on the mean squared error, as detailed in the module _metric_.
_BruteForce_ uses different backends such as TensorFlow, PyTorch, NeuroLab etc. 

Functionality specific to each backend is implemented in child classes of the _BruteForce_ class:
- Class _NeuralNlb_: NeuroLab variant
- Class _NeuralTfl_: Tensorflow/Keras variant
- Class _NeuralTch_: PyTorch variant

### Example: Sine curve
_test_blackboxes_box.py_ is an example using synthetic data in 1D space with the backends TensorFlow, PyTorch, and NeuroLab.  

        N = 1000                    # number of training sets
        n = int(np.round(1.4 * N))  # number of test sets
        nse = 5e-2                  # noise

        # X and Y is training data, x is test input and y is prediction
        X = np.linspace(-2. * np.pi, 2. * np.pi, N).reshape(-1, 1)
        dx = 0.25 * (X.max() - X.min())
        x = np.linspace(X.min() - dx, X.max() + dx, n).reshape(-1, 1)
        Y_tru = np.sin(X)
        Y = Y_tru + np.random.uniform(-nse, +nse, size=X.shape)
        y_tru = np.sin(x)

        for backend in [
            'neurolab',
            'tensorflow'
            'torch',
        ]:
            phi = Black()
        
            y = phi(X=X, Y=Y, x=x,
                activation=('leaky', 'elu',) 
                    if backend == 'tensorflow' else 'sigmoid',
                backend=backend,
                epochs=150,
                expected=1e-3,
                learning_rate=0.1,            # tensorflow learning rate
                neurons=[[i]*j for i in range(4, 4+1)       # i: neurons  
                               for j in range(4, 4+1)],       # j: layer
                output='linear',
                patience=10,      # delay of early stopping (tensorflow)
                plot=1,           # 0: none, 1: final only, 2: all plots 
                rr=0.1,                 # bfgs regularization (neurolab)
                show=1,
                tolerated=5e-3,
                trainer='adam' if backend != 'neurolab' else 'bfgs',
                trials=5,   # repetition of every training configuration 
                )

#### Results

The training data and the true values are plotted in Figure 2.

![train_and_true](https://github.com/dwweiss/blackboxes/blob/main/doc/fig/bruteforce_train_and_true1.png)

###### Figure 2: Training data and true values without noise


Figure 3 shows the history of the mean squared error of all trials for the TensorFlow backend. 

![history_all](https://github.com/dwweiss/blackboxes/blob/main/doc/fig/bruteforce_history1_all.png)

###### Figure 3: Mean squared error history of all trials


In Figure 4 the history of the five best trials out of all trials plotted in Figure 3 is shown. 

![history_5best](https://github.com/dwweiss/blackboxes/blob/main/doc/fig/bruteforce_history1_5best.png)

###### Figure 4: Mean squared error history of five best trials


The resulting errorbars are summarized in Figure 5. 

![MSE_history_all](https://github.com/dwweiss/blackboxes/blob/main/doc/fig/bruteforce_errorbars1.png)

###### Figure 5: Errorbars of all trials

It is evident that conducting a single training session is risky, as illustrated by the mean squared error (MSE) of training with _leakyReLU_ in Figure 5. The first trial (#0) fails entirely. Therefore, it is advised to perform a minimum of 3 repetitions.

### Example: UIC airfoil + noise dataset

This real-world example with 6 input, 1 output and 1503 data points is taken from the UIC database:

https://archive.ics.uci.edu/dataset/291/airfoil+self+noise

Each of the 5 hidden layers contains 8 neurons. The trainer is _adam_, the types of activation of hidden layers are: (_elu_, _leakyrelu_, _sigmoid_) and every configuration was repeated 5 times.   

Figure 6 shows the history of the mean squared error of all trials for the TensorFlow backend. 

![history_all](https://github.com/dwweiss/blackboxes/blob/main/doc/fig/bruteforce_history_uic_airfoil.png)

###### Figure 6: Mean squared error history of all trials


The resulting errorbars are summarized in Figure 7. 

![MSE_history_all](https://github.com/dwweiss/blackboxes/blob/main/doc/fig/bruteforce_errorbars_uic_airfoil.png)

###### Figure 7: Errorbars of all trials

A single training session is risky, as indicated by the mean squared error (MSE) of training with the _sigmoid_ activation function in Figure 6. Both the first and second trials (#0 and #1) with _sigmoid_ activation fail. Although the influence of the choice of activation function is minimal, there is an indication that the MSE variation with the _leakyReLU_ activation function is less than the variation with other activation functions. In contrast, the sine curve example has shown that _leakyReLU_ is not a good choice. Therefore, it is recommended to conduct a minimum of 3 repetitions.


### Conclusion

The required number of training repetitions is highly problem-specific in regression analysis of measurements. There are examples where a single training is sufficient, and examples where multiple random initializations of the network weights are definitely needed. Relying solely on a single training of a network configuration on a new dataset can pose a substantial risk of missing an acceptable solution. A preference for a particular optimizer or activation function for minimizing the MSE variation across multiple trials has not been identified. Therefore, brute force scanning of the network parameter space is recommended. The random initialization of weights should be repeated 3-5 times for each network configuration.


### Dependencies
- Module _neuralnlb_ is dependent on package _neurolab_ [[NLB15]](https://github.com/dwweiss/grayboxes/wiki/References#nlb15)
- Module _neuraltch_ is dependent on package _torch_ [[PAS24]](https://github.com/dwweiss/grayboxes/wiki/References#pas24)
- Module _neuraltfl_ is dependent on package _tensorflow_ [[ABA15]](https://github.com/dwweiss/grayboxes/wiki/References#aba15)
