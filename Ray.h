#pragma once

struct Ray{
	float3 o;
	float3 d;

	__device__ Ray(float3 _o, float3 _d):o(_o), d(_d){}
};