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

`$ python3 run.py <model_type>`

The different model_type's are: 
* `-alexnet` 
* `-resnet18`
* `-vgg16`
* `-yolov5l`

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
