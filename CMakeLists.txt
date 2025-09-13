cmake_minimum_required(VERSION 3.28)

project(llama3-server LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
 
find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND)
    message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
    enable_language(CUDA)
    set(HAS_CUDA TRUE)
else()
    message(STATUS "CUDA not found, building without GPU support")
    set(HAS_CUDA FALSE)
endif()

add_library(tannic STATIC IMPORTED)
set_target_properties(tannic PROPERTIES
    IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/external/tannic-nn/external/tannic/build/libtannic.a
)

add_library(tannic-nn STATIC IMPORTED)
set_target_properties(tannic-nn PROPERTIES
    IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/external/tannic-nn/build/libtannic-nn.a
)

target_include_directories(tannic INTERFACE
    ${CMAKE_SOURCE_DIR}/external/tannic-nn/external/tannic/include
)

target_include_directories(tannic-nn INTERFACE
    ${CMAKE_SOURCE_DIR}/external/tannic-nn/include
    ${CMAKE_SOURCE_DIR}/external/tannic-nn/external/tannic/include
)

find_library(OPENBLAS_LIB openblas)
if(OPENBLAS_LIB)
    message(STATUS "Found OpenBLAS: ${OPENBLAS_LIB}")
endif()

add_executable(llama3-server main.cpp)

target_include_directories(llama3-server PRIVATE
    include/
    model/
    ${CMAKE_SOURCE_DIR}/external/tannic-nn/include
    ${CMAKE_SOURCE_DIR}/external/tannic-nn/external/tannic/include
)

target_link_libraries(llama3-server PRIVATE
    tannic-nn
    tannic
    pthread
    dl
)

if(OPENBLAS_LIB)
    target_link_libraries(llama3-server PRIVATE ${OPENBLAS_LIB})
endif()
 
if(HAS_CUDA) 
    target_link_libraries(llama3-server PRIVATE CUDA::cudart)
     
    find_library(CUDA_cudadevrt_LIB cudadevrt PATHS ${CUDAToolkit_LIBRARY_DIR})
    if(CUDA_cudadevrt_LIB)
        target_link_libraries(llama3-server PRIVATE ${CUDA_cudadevrt_LIB})
    endif()
     
    target_link_libraries(llama3-server PRIVATE CUDA::cuda_driver)
     
    set_target_properties(llama3-server PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
     
    target_link_options(llama3-server PRIVATE
        -Wl,--no-as-needed
    )
endif()