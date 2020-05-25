#pragma once

struct Hit;
struct Ray;

struct Object{
	__device__ virtual bool intersect(Hit *hit, Ray &ray)=0;
};