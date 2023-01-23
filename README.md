# Layer Simulation 

Layer Simulation program for the MLNoC project. 

## Build 

Clone the repository: 

`git clone https://github.com/justinmgarrigus/layer-simulation.git`

Export a new environment variable that points to your local installation of [CUDA's CUTLASS v1.3.2 library](https://github.com/NVIDIA/cutlass/tree/v1.3.2): 

`export CUTLASS_INSTALL_DIR=path/to/cutlass`

CMake the project: 

```
cd layer-simulation
mkdir build
cd build
cmake ..
make
```

Run the project: 

`python3 run.py -alexnet`

Model types: 
* -alexnet 
* -resnet18
* -vgg16 
* -yolov5l
