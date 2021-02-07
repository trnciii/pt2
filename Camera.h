#pragma once

#include <cuda.h>
#include <glm/glm.hpp>
#include "Ray.h"
#include "functions.h"

struct Camera{
	glm::vec3 pos;
	glm::vec3 basis[3];
	float  focal;

	__host__ __device__ void setBasis(glm::vec3 dir, glm::vec3 up){
		basis[0] = glm::normalize(glm::cross(dir, up));
		basis[2] = -glm::normalize(dir);
		basis[1] = cross(basis[2], basis[0]);
	}

	__device__ Ray ray(float x, float y){
		return Ray(pos, glm::normalize(basis[0]*x + basis[1]*y - basis[2]*focal));
	}

	__host__ __device__ void print(){
		printf("position: "); print_vec3(pos); printf("\n");
		printf("cam_x: "); print_vec3(basis[0]); printf("\n");
		printf("cam_y: "); print_vec3(basis[1]); printf("\n");
		printf("cam_z: "); print_vec3(basis[2]); printf("\n");
		printf("length: %f\n", focal);
		printf("\n");
	}
};