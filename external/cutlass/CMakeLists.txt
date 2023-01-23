add_library(CUTLASS INTERFACE)
add_library(nvidia::cutlass ALIAS CUTLASS)

target_include_directories(CUTLASS INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools"
)
