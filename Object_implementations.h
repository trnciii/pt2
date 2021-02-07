#pragma once

#include <cuda.h>
#include <glm/glm.hpp>

#include "Ray.h"
#include "Hit.h"

class Material;

struct Sphere{
	glm::vec3 center;
	float  r;
	Material *mtl;

	__host__ __device__ Sphere(glm::vec3 _p, float _r, Material *_m):center(_p), r(_r), mtl(_m){}

	__device__ float dist(Ray &ray){
		glm::vec3 OC = ray.o-center;
		float b = dot(ray.d, OC);
		float b2 = b*b;
		float c = dot(OC, OC) - r*r;
		if(c<b2){
			float t = b+sqrt(b2-c);
			if(0<t) t = b-sqrt(b2-c);
			if(t<0)return -t;
		}
		return -1;
	}
	__device__ bool intersect(Hit *hit, Ray &ray){
		float t = dist(ray);
		if(0<t && t<hit->dist){
			hit->dist = t;
			hit->n = (ray.o + ray.d*t - center)/r;
			
			int sg =(hit->n.z < 0) ?-1 :1;
			double a = -1.0/(sg+hit->n.z);
			hit->dpdu = glm::vec3(
				1.0 + sg * hit->n.x*hit->n.x * a,
				sg * hit->n.x * hit->n.y * a,
				-sg*hit->n.x
			);

			hit->mtl = mtl;
			return true;
		}
		return false;
	}

	__host__ __device__ void print(){
		printf("--Sphere\n");
		printf("center: "); print_vec3(center); printf("\n");
		printf("radius: %f\n", r);
		printf("material %p: \n", mtl);	// mtl->print();
		printf("\n");
	}
};


struct Plane{
	glm::vec3 pos;
	glm::vec3 b1;
	glm::vec3 b2;
	Material *mtl;

	glm::vec3 normal;

	__host__ __device__ Plane(glm::vec3 _p, glm::vec3 _b1, glm::vec3 _b2, Material *_m)
	:pos(_p), b1(_b1), b2(_b2), mtl(_m){
		normal = glm::normalize(glm::cross(b1,b2));
	}

	__device__ float dist(Ray &ray, float2 *uv){
		glm::vec3 o = ray.o - pos;
		glm::vec3 q = glm::cross(b1, o);
		glm::vec3 p = glm::cross(b2, ray.d);
		float det = glm::dot(b1,p);

		uv->x = glm::dot(o,p)/det;
		uv->y = glm::dot(ray.d,q)/det;
		return (-1<uv->x && uv->x<1 && -1<uv->y && uv->y<1)? glm::dot(q,b2)/det : -1;
	}

	__device__ bool intersect(Hit *hit, Ray &ray){
		float2 uv;
		float t = dist(ray, &uv);
		if(0<t && t<hit->dist){
			hit->dist = t;
			hit->n = normal;
			hit->dpdu = b1;
			hit->dpdv = b2;
			hit->uv = glm::vec2(uv.x*0.5+0.5, uv.y*0.5+0.5);
			hit->mtl = mtl;
			return true;
		}return false;
	}
};