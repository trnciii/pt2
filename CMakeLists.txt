cmake_minimum_required(VERSION 3.1)
project(pt2)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# libraries
find_package(STB_IMAGE REQUIRED)
include_directories(${STB_INCLUDE_DIRS})

find_package(CUDA REQUIRED)
if(CUDA_FOUND AND STB_IMAGE_FOUND)
	message("Cuda version: " ${CUDA_VERSION})
	CUDA_ADD_EXECUTABLE(pt2 pt2.cu)
endif()