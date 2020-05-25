#pragma once

#include "functions.h"
#include "vec.h"

enum MtlType{
	null,
	emit,
	lambert,
	GGX_ref_iso
};

class Material{
public:
	MtlType type;
	float3 col;
	float alpha2;

	__host__ __device__ Material():type(null){}

	__host__ __device__ void setEmission(float3 e){type=emit; col = e;}
	__host__ __device__ void setLambert(float3 l){type = lambert; col = l;}
	__host__ __device__ void setGGX_iso(float3 c, float a2){
		type = GGX_ref_iso;
		col = c;
		alpha2 = a2;
	}

	__host__ __device__ void print(){
		if(type==emit){
			printf("---- type: emit, color: ");
			print_float3(col);
		}

		if(type==lambert){
			printf("----type: lambert, color: ");
			print_float3(col);
		}
	}
};