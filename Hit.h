#pragma once

class Material;

struct Hit{
	float  dist;
	float3 pos;
	float3 n;
	float3 dpdu;
	float3 dpdv;
	float3 tan;
	float2 uv;
	Material *mtl;
	bool backface = false;
};