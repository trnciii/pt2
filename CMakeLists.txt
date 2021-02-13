cmake_minimum_required(VERSION 3.1)
project(pt2)

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
	message("Cuda version: " ${CUDA_VERSION})
	CUDA_ADD_EXECUTABLE(pt2 pt2.cu)
endif()