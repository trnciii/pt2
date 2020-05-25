#pragma once

#include "vec.h"
#include "Ray.h"

struct Camera{
	float3 pos;
	float3 basis[3];
	float  focal;

	__host__ __device__ void setBasis(float3 dir, float3 up){
		basis[0] = normalize(cross(dir, up));
		basis[2] = -normalize(dir);
		basis[1] = cross(basis[2], basis[0]);
	}

	__device__ Ray ray(float x, float y){
		return Ray(pos, normalize(basis[0]*make_float3(x) + basis[1]*make_float3(y) - basis[2]*make_float3(focal)));
	}

	__host__ __device__ void print(){
		printf("position: "); print_float3(pos);
		printf("cam_x: "); print_float3(basis[0]);
		printf("cam_y: "); print_float3(basis[1]);
		printf("cam_z: "); print_float3(basis[2]);
		printf("length: %f\n", focal);
		printf("\n");
	}
};