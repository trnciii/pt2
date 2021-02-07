#pragma once

#include <cuda.h>
#include <glm/glm.hpp>
#include "functions.h"

enum MtlType{
	null,
	emit,
	lambert,
	GGX_ref_iso
};

class Material{
public:
	MtlType type;
	glm::vec3 col;
	float alpha2;

	__host__ __device__ Material():type(null){}

	__host__ __device__ void setEmission(glm::vec3 e){type=emit; col = e;}
	__host__ __device__ void setLambert(glm::vec3 l){type = lambert; col = l;}
	__host__ __device__ void setGGX_iso(glm::vec3 c, float a2){
		type = GGX_ref_iso;
		col = c;
		alpha2 = a2;
	}

	__host__ __device__ void print(){
		if(type==emit){
			printf("---- type: emit, color: "); print_vec3(col); printf("\n");
		}

		if(type==lambert){
			printf("----type: lambert, color: "); print_vec3(col); printf("\n");
		}
	}
};