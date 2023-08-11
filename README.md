# Layer Simulation 

Layer Simulation program for the MLNoC project. 

## Build 

Clone the repository: 

`$ git clone https://github.com/justinmgarrigus/layer-simulation.git`

Export a new environment variable that points to your local installation of [CUDA's CUTLASS v1.3.2 library](https://github.com/NVIDIA/cutlass/tree/v1.3.2): 

`$ export CUTLASS_INSTALL_DIR=path/to/cutlass`

CMake the project: 

```
$ cd layer-simulation
$ mkdir build
$ cd build
$ cmake ..
$ make
```

## Run

There are multiple ways to run the project: 

### Classification

To run a classification on an image or set of images (where each layer is executed sequentially, generating `.bin` files along the way), place `.jpg` files into the `data` directory and run: 

`$ python3 run.py <model_type> (<layer_index>) (--data_path=<data_path> --zero_type=<zero_type> --epsilon=<epsilon>)`

Parameters: 

model_type: 
* `-alexnet` 
* `-resnet18`
* `-vgg16`
* `-yolov5l`

`layer_index`: The conv operator index in the model to start computation at. If not provided, equals 1. To skip the first _n_ convolutional operations in a network, set this to _n+1_. 

`data_path`: "data/" by default, but can be set to either a directory containing ".jpg" images or a single ".jpg" image. Each of the discovered ".jpg" images will be run through the inference.

`zero_type`: Before tensors are passed to the convolution function, they can be preprocessed by setting values to zero if their absolute value is less than a threshold. This flag determines which tensors to preprocess in this way: 
* `none`: No values in tensors are zeroed.
* `input`: The input tensors to each conv2D operation has values which are zeroed.
* `weight`: The weight tensors to each conv2D operation has values which are zeroed.
* `both`: Both the input and the weight tensors to each conv2D operation has values which are zeroed.

`epsilon`: Must be a valid floating-point number. If `zero_type` is not equal to "`none`", this is the threshold that determines if a tensor value should be set to 0. Before a convolution operation is applied, we loop through every value in the tensor. If the absolute value of this is less than or equal to `epsilon`, then it is replaced with zero.

### Single

To run a single layer on a given model, first generate the `.bin` files as above. Then, run the following command: 

`$ ./single.sh <model_type> <layer_index>` 

In this command, `layer_index` is dependent on the model, and should be chosen in the range `[1, len(model.layers)]`.

## Verification 

After executing a classification or a collection of single's (see the `Run` section above), you can obtain a list of statistics using the command: 

`$ ./end-verify.sh` 

This outputs whether or not the error files (e.g., `output/alexnet/simErrors_cL1_.txt`) are empty, indicating that there were no errors that occured during execution, as well as the time that each layer/model took to run (assuming a sequential execution). 

## Sub-build

Sub-builds are copies of the project, in which each sub-build contains only a single test image. We can fill the base "data/" directory with ".jpg" images, and then create sub-builds with: 

`$ ./sub-run.sh -c`

This creates several new directories underneath the "sub-builds/" directory. Each sub-build contains the same files as the main directory (except for files beginning with a dot like ".git/"), but the "sub-builds/*/data/" directory will contain only a single image.

After sub-builds are created, we can run a command across every sub-build with: 

`$ ./sub-run.sh -e "command"`

For instance, we can test if this works with `./sub-run.sh -e "pwd"` to print each sub-build directory. We can run an inference on alexnet for each sub-build (one inference per image) with `./sub-run.sh -e "python3 run.py alexnet > run.out"`.
