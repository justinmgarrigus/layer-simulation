cmake_minimum_required(VERSION 3.18)

project(gemm LANGUAGES CUDA)

add_executable(${PROJECT_NAME} src/gemm.cu)

set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
    CUDA_STANDARD 11
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
    CUDA_ARCHITECTURES "70"
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)

add_subdirectory(external)
target_link_libraries(${PROJECT_NAME} PRIVATE nvidia::cutlass)
target_include_directories(${PROJECT_NAME} PRIVATE $ENV{CUTLASS_INSTALL_DIR})

# This probably isn't the best method of doing this, but it's easy! 
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/models/yolov5l-cls.pt")
    execute_process(COMMAND wget --no-verbose 
        https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5l-cls.pt 
        -P ${CMAKE_SOURCE_DIR}/models/) 
endif() 
