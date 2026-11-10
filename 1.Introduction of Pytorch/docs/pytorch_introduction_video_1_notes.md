# PyTorch for Beginners 

## 1. What Is PyTorch?

**PyTorch** is an open-source, Python-based deep learning framework. It
provides tools for tensor computation, automatic differentiation, GPU
acceleration, and building and training neural networks.

In simple words, PyTorch gives us the building blocks needed to create
and train deep learning models.

### Real-world example

Suppose we want to build an image-classification model that identifies
whether an image contains a cat or a dog.

We need to:

1.  Load images and convert them into numerical data.
2.  Perform mathematical operations on that data.
3.  Build a neural network.
4.  Calculate how wrong the model's predictions are.
5.  Calculate gradients and update the model's weights.
6.  Repeat the process until the model learns.

PyTorch provides tools for these steps.

### Why is it called PyTorch?

-   **Python:** PyTorch provides a Python-first interface.
-   **Torch:** It builds on ideas and technology from the earlier Torch
    framework.

PyTorch combines an accessible Python interface with efficient tensor
operations and deep learning functionality.

------------------------------------------------------------------------

## 2. Why Was PyTorch Created?

Before PyTorch, the Torch framework was already used for scientific
computing and deep learning.

However, Torch used the **Lua** programming language. Many machine
learning researchers and developers preferred Python because of its
broad scientific ecosystem.

PyTorch brought Torch-style tensor computation and deep learning tools
into a Python-friendly framework.

### Why this mattered

-   Developers could use familiar Python syntax.
-   It worked naturally with the wider Python ecosystem.
-   Researchers could experiment with neural network architectures more
    easily.
-   It offered flexible computation graphs and automatic
    differentiation.

**Key idea:** PyTorch made powerful deep learning tools easier to use
within Python.

------------------------------------------------------------------------

## 3. What Is a Computation Graph?

A **computation graph** represents a mathematical calculation as a
sequence of connected operations.

Consider this expression:

``` text
y = (a + b) * c
```

The operations are:

1.  Add `a` and `b`.
2.  Multiply the result by `c`.
3.  Produce `y`.

A graph represents the flow of values and dependencies between these
operations.

### Why do neural networks use computation graphs?

A neural network is a collection of mathematical operations:

-   Multiplication of inputs and weights
-   Addition of biases
-   Activation functions
-   Loss calculation

A computation graph represents these operations and their dependencies.
This is useful for calculating gradients during backpropagation.

------------------------------------------------------------------------

## 4. Static vs Dynamic Computation Graphs

One important distinction in deep learning frameworks is when and how
the computation graph is constructed.

### 4.1 Static computation graph

A static graph is defined before its execution. The computation
structure is established in advance.

**Real-world example --- fixed cake recipe**

Imagine following a recipe that specifies every ingredient and step
before you start baking. You follow the prepared sequence.

This resembles a static graph: the computation structure is prepared
before execution.

### 4.2 Dynamic computation graph

A dynamic graph is constructed as operations are executed. The graph can
reflect the operations performed during the current forward pass.

**Real-world example --- flexible cooking**

Imagine cooking with the ingredients available. You can change the steps
depending on what you have.

This resembles dynamic computation: the operations can depend on the
current input and runtime decisions.

### Comparison

  -----------------------------------------------------------------------
  Static graph                        Dynamic graph
  ----------------------------------- -----------------------------------
  Structure is defined ahead of       Structure is built as operations
  execution                           run

  More fixed execution structure      Flexible execution structure

  Runtime changes to the graph are    Runtime-dependent control flow is
  less direct                         natural

  Can enable graph-level optimization Often easier to experiment and
                                      debug interactively
  -----------------------------------------------------------------------

### Why dynamic graphs are useful

Consider a model that processes text sequences of different lengths. Its
computation may depend on the length or content of each sequence.

Dynamic execution makes it natural to express such input-dependent
behavior using ordinary Python control flow.

**Important clarification:** Static and dynamic graphs are not a simple
ranking where one is always better. Each approach has trade-offs. Modern
frameworks can also offer multiple execution and compilation modes.

------------------------------------------------------------------------

## 5. Core Features of PyTorch

### 5.1 Tensor computation

A **tensor** is a multi-dimensional data structure. It is the
fundamental data structure used by PyTorch.

Examples:

-   Scalar: a single number
-   Vector: a one-dimensional collection
-   Matrix: a two-dimensional collection
-   Higher-dimensional tensor: data with three or more dimensions

Example:

``` python
import torch

# Create a tensor
x = torch.tensor([1, 2, 3])

# Perform an operation
y = x * 2

print(y)
```

Output:

``` text
tensor([2, 4, 6])
```

**How it works:** PyTorch stores numerical values in tensors and
provides operations to calculate with them.

**Why it matters:** Neural networks perform large numbers of tensor
operations.

### 5.2 GPU acceleration

A GPU can perform many numerical operations in parallel. This can make
deep learning workloads much faster than running the same work on a CPU.

Example:

``` python
import torch

# Check whether CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create a tensor on the available device
x = torch.tensor([1, 2, 3], device=device)

print(device)
print(x)
```

**How it works:**

1.  PyTorch checks whether a CUDA-compatible GPU is available.
2.  It selects the GPU when available; otherwise it selects the CPU.
3.  The tensor is created on the selected device.

**Important:** CUDA is primarily associated with NVIDIA GPUs. Other
hardware backends and accelerator options also exist.

### 5.3 Automatic differentiation --- Autograd

Training a neural network requires gradients. Gradients tell us how
changing a parameter affects the loss.

PyTorch provides **autograd**, its automatic differentiation system, to
calculate these gradients.

Example:

``` python
import torch

x = torch.tensor(2.0, requires_grad=True)

y = x ** 2

y.backward()

print(x.grad)
```

Output:

``` text
tensor(4.)
```

**How it works:**

1.  `requires_grad=True` tells PyTorch to track operations involving
    `x`.
2.  `y = x ** 2` calculates the output.
3.  `y.backward()` calculates the gradient of `y` with respect to `x`.
4.  `x.grad` contains the calculated gradient.

For this example:

``` text
y = x²

dy/dx = 2x

When x = 2:
dy/dx = 4
```

**Why it matters:** Autograd provides the gradients needed for
backpropagation and neural network training.

### 5.4 Dynamic computation graphs

PyTorch normally records operations as they execute, creating a graph
for automatic differentiation.

This makes it convenient to write models with conditional logic, loops,
and input-dependent operations.

### 5.5 Distributed training

Distributed training allows a model's training workload to be shared
across multiple GPUs or machines.

**Example:** A very large dataset may take too long to process on one
GPU. Distributed training can divide the workload across several
devices.

### 5.6 Python ecosystem integration

PyTorch works with common Python tools and libraries, including NumPy
and the wider machine learning ecosystem.

This makes it easier to combine data processing, model training,
evaluation, and deployment workflows.

------------------------------------------------------------------------

## 6. PyTorch vs TensorFlow

Both PyTorch and TensorFlow are deep learning frameworks. Both can
build, train, and deploy neural networks.

  -----------------------------------------------------------------------
  Area                    PyTorch                 TensorFlow
  ----------------------- ----------------------- -----------------------
  Main interface          Python-first            Python-first, with
                                                  other language tools

  Programming style       Commonly considered     TensorFlow 2 uses eager
                          intuitive and Pythonic  execution and Keras
                                                  APIs

  Model building          `torch.nn` and related  Keras and TensorFlow
                          tools                   APIs

  Automatic               Autograd                `tf.GradientTape`
  differentiation                                 

  GPU support             Yes                     Yes

  Research and            Widely used             Also widely used
  experimentation                                 

  Deployment              Multiple export and     TensorFlow Serving,
                          serving options         LiteRT and other tools

  Ecosystem               TorchVision,            Keras, TensorBoard,
                          TorchAudio, Hugging     TensorFlow Hub, and
                          Face integrations, and  more
                          more                    
  -----------------------------------------------------------------------

### Which one should you learn?

There is no universal winner.

-   **PyTorch:** A strong choice for learning deep learning,
    experimenting with architectures, and working with many modern
    research and generative AI projects.
-   **TensorFlow:** Also valuable, especially when working with its
    ecosystem and deployment tools.

**Practical recommendation:** Learn one framework deeply before trying
to master both. PyTorch is a sensible choice if your goal is to
understand modern deep learning and work with research-oriented
implementations.

------------------------------------------------------------------------

## 7. Important PyTorch Modules

PyTorch is organized into modules. Each module has a particular
responsibility.

### 7.1 `torch`

The core PyTorch package.

It provides:

-   Tensor creation
-   Mathematical operations
-   Tensor manipulation
-   Device management

Example:

``` python
import torch

x = torch.tensor([1, 2, 3])
print(x)
```

### 7.2 `torch.autograd`

Provides automatic differentiation.

**Main purpose:** Calculate gradients automatically.

Important concepts:

-   `requires_grad`
-   Computational graph
-   `backward()`
-   `.grad`

### 7.3 `torch.nn`

Provides the building blocks for neural networks.

It includes:

-   Layers
-   Activation functions
-   Loss functions
-   Neural network modules

Example:

``` python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

print(model)
```

Here:

-   `nn.Linear(10, 5)` maps 10 input features to 5 outputs.
-   `nn.ReLU()` applies the ReLU activation function.
-   `nn.Linear(5, 1)` produces one output.

### 7.4 `torch.optim`

Provides optimization algorithms used to update model parameters.

Common optimizers:

-   SGD
-   Adam
-   AdamW
-   RMSprop

Example:

``` python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.001)
```

**Meaning:** Adam is the optimizer, `model.parameters()` supplies the
trainable parameters, and `lr` is the learning rate.

### 7.5 `torch.utils.data`

Provides tools for preparing and loading datasets.

Important classes:

-   `Dataset`
-   `DataLoader`

**Dataset:** Defines how individual data samples are accessed.

**DataLoader:** Loads samples in batches and can shuffle the data.

Example:

``` python
from torch.utils.data import TensorDataset, DataLoader
import torch

X = torch.randn(100, 3)
y = torch.randn(100, 1)

dataset = TensorDataset(X, y)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

for batch_X, batch_y in loader:
    print(batch_X.shape, batch_y.shape)
```

### 7.6 `torch.cuda`

Provides interfaces for working with CUDA-compatible NVIDIA GPUs.

It helps check GPU availability and manage CUDA-related operations.

### 7.7 `torch.distributed`

Provides tools for distributed training across multiple devices and
machines.

### 7.8 `torch.jit`

Provides Just-In-Time compilation and TorchScript-related functionality.

It has been used to optimize and package models for execution outside
ordinary Python workflows. Modern PyTorch also provides other
compilation and export tools.

### 7.9 `torch.backends`

Provides access to backend settings and libraries that influence how
certain operations are executed.

### 7.10 `torch.multiprocessing`

Provides multiprocessing support, including functionality designed to
work with PyTorch tensors.

### 7.11 Quantization tools

Quantization represents model weights and computations using
lower-precision numerical formats where supported.

Potential benefits:

-   Smaller model size
-   Lower memory use
-   Faster inference on suitable hardware

**Note:** PyTorch's quantization APIs and recommended workflows have
evolved. Consult current documentation before choosing a specific API.

### 7.12 `torch.onnx`

Supports exporting PyTorch models to ONNX (Open Neural Network
Exchange), an open model format used for interoperability with other
tools and runtimes.

------------------------------------------------------------------------

## 8. PyTorch Ecosystem Libraries

PyTorch is the core framework. Its ecosystem contains additional
libraries that make specific tasks easier.

  -----------------------------------------------------------------------
  Library                             Main purpose
  ----------------------------------- -----------------------------------
  Hugging Face Transformers           Pre-trained transformer models for
                                      NLP, vision, audio, and multimodal
                                      tasks

  TorchVision                         Computer vision datasets,
                                      transformations, and model
                                      utilities

  TorchAudio                          Audio processing and audio-related
                                      model tools

  Fastai                              Higher-level deep learning API
                                      built on PyTorch

  PyTorch Geometric                   Graph neural networks and geometric
                                      deep learning

  TorchMetrics                        Standardized evaluation metrics

  Optuna                              Hyperparameter optimization

  PyTorch Lightning                   Higher-level structure for
                                      organizing PyTorch training code

  Ignite                              Utilities and engines for training
                                      and evaluation

  Skorch                              Scikit-learn-compatible wrapper for
                                      PyTorch models

  PyTorch Forecasting                 Tools for time-series forecasting

  TensorBoard                         Visualization of training metrics
                                      and other experiment information
  -----------------------------------------------------------------------

### 8.1 Hugging Face Transformers

This library provides access to many pre-trained transformer models.

Examples of tasks:

-   Text classification
-   Text generation
-   Translation
-   Question answering
-   Summarization

**Why it matters:** Instead of building every transformer from scratch,
we can use and fine-tune existing models.

### 8.2 TorchVision

Provides computer vision utilities.

Examples:

-   Image datasets
-   Image transformations
-   Pre-trained vision models

### 8.3 TorchAudio

Provides tools for audio processing and machine learning with audio
data.

### 8.4 Fastai

Offers a higher-level API that simplifies many deep learning workflows
while using PyTorch underneath.

### 8.5 PyTorch Geometric

Useful for graph-based data, such as:

-   Social networks
-   Molecule structures
-   Recommendation graphs

### 8.6 TorchMetrics

Provides reusable metrics for evaluating model performance.

### 8.7 Optuna

Helps search for good hyperparameters automatically.

For example, it can help find suitable:

-   Learning rates
-   Batch sizes
-   Number of layers
-   Hidden units
-   Optimizer settings

------------------------------------------------------------------------

## 9. How the Main Components Work Together

A typical PyTorch training workflow connects several modules.

``` text
Dataset
   |
   v
DataLoader
   |
   v
Input Batch
   |
   v
Neural Network (torch.nn)
   |
   v
Predictions
   |
   v
Loss Function
   |
   v
Autograd (calculate gradients)
   |
   v
Optimizer (update parameters)
   |
   v
Next Training Step
```

### Step-by-step explanation

1.  **Dataset:** Stores or provides access to training examples.
2.  **DataLoader:** Groups examples into batches.
3.  **Neural network:** Receives a batch and produces predictions.
4.  **Loss function:** Measures the difference between predictions and
    targets.
5.  **Autograd:** Calculates gradients through backpropagation.
6.  **Optimizer:** Uses gradients to update model parameters.
7.  **Repeat:** Continue for multiple batches and epochs.

This is the basic structure behind training a neural network in PyTorch.

------------------------------------------------------------------------

## 10. Important Terms to Remember

  -----------------------------------------------------------------------
  Term                                Simple meaning
  ----------------------------------- -----------------------------------
  Framework                           A collection of tools and
                                      structures for building
                                      applications

  Tensor                              A multi-dimensional numerical data
                                      structure

  GPU                                 Processor designed for parallel
                                      computations

  CUDA                                NVIDIA's platform for GPU computing

  Computation graph                   A representation of connected
                                      mathematical operations

  Static graph                        A graph whose structure is defined
                                      before execution

  Dynamic graph                       A graph built as operations execute

  Autograd                            PyTorch's automatic differentiation
                                      system

  Gradient                            A value describing how a function
                                      changes with respect to a variable

  Backpropagation                     The process of calculating
                                      gradients through a neural network

  Optimizer                           An algorithm that updates model
                                      parameters

  Distributed training                Training across multiple devices or
                                      machines

  Quantization                        Using lower-precision
                                      representations to reduce resource
                                      requirements

  ONNX                                An open format for exchanging
                                      machine learning models
  -----------------------------------------------------------------------

------------------------------------------------------------------------

##  Key Takeaways

1.  PyTorch is an open-source, Python-based deep learning framework.
2.  It provides tensors and mathematical operations for deep learning.
3.  Dynamic computation graphs make runtime-dependent model logic
    convenient.
4.  Autograd calculates gradients automatically.
5.  GPU acceleration can speed up tensor computations.
6.  `torch.nn` provides neural network building blocks.
7.  `torch.optim` provides optimizers.
8.  `torch.utils.data` provides Dataset and DataLoader utilities.
9.  Hugging Face Transformers is an important part of the PyTorch
    ecosystem.
10. PyTorch and TensorFlow are both capable frameworks; the best choice
    depends on the project.

------------------------------------------------------------------------

##  Official Resources

-   [PyTorch Official Website](https://pytorch.org/)
-   [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
-   [PyTorch Tutorials](https://pytorch.org/tutorials/)
-   [PyTorch Learn the
    Basics](https://pytorch.org/tutorials/beginner/basics/intro.html)
-   [Hugging Face Documentation](https://huggingface.co/docs)
------------------------------------------------------------------------
