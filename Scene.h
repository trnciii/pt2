#pragma once

#include <cuda.h>
#include <glm/glm.hpp>

#include "Camera.h"
// #include "Object.h"
#include "Object_implementations.h"
#include "Hit.h"
#include "Ray.h"
#include "Material.h"

struct Scene{
	Material background;
	Camera camera;

	Sphere *spheres = nullptr;
	int nSpheres = 0;

	Plane *planes = nullptr;
	int nPlanes = 0;

	__device__ Hit nearest(Ray &ray){
		Hit hit;
			hit.dist = 1e10;
			hit.mtl = &background;

		for(int i=0; i<nSpheres; i++){
			spheres[i].intersect(&hit, ray);
		}

		for(int i=0; i<nPlanes; i++){
			planes[i].intersect(&hit, ray);
		}

		hit.pos = ray.o + hit.dist*ray.d;
		hit.tan = glm::normalize(hit.dpdu);
		if(glm::dot(ray.d,hit.n)>0){
			hit.n = -hit.n;
			hit.backface = true;
		}

		return hit;
	}

	__host__ __device__ void print(){
		printf("background: "); background.print();
		printf("\n");
		
		printf("camera\n");
		camera.print();

		printf("%d spheres\n", nSpheres);
		for(int i=0; i<nSpheres; i++)
			spheres[i].print();
	}

	__host__ void createScene(){
		background.setEmission(glm::vec3(0.1));

		camera.pos = glm::vec3(0,-4,-0);
		camera.setBasis(glm::vec3(0,1,0), glm::vec3(0,0,1));
		camera.focal = 1;

		Material *emit;
		cudaMallocManaged(&emit, sizeof(Material));
		emit->setEmission(glm::vec3(4));

		Material *left;
		cudaMallocManaged(&left, sizeof(Material));
		left->setGGX_iso(glm::vec3(0.9, 0.1, 0.1), 0.1);

		Material *right;
		cudaMallocManaged(&right, sizeof(Material));
		right->setLambert(glm::vec3(0.1, 0.1, 0.9));

		Material *floor;
		cudaMallocManaged(&floor, sizeof(Material));
		floor->setGGX_iso(glm::vec3(0.1), 0.1);

		nSpheres = 3;
		if(nSpheres>0){
			cudaMallocManaged(&spheres, nSpheres*sizeof(Sphere));
			spheres[0] = Sphere(glm::vec3(-1, 1,-1  ), 1.5, left);
			spheres[1] = Sphere(glm::vec3( 1, 0,-1  ), 1.5, right);
			spheres[2] = Sphere(glm::vec3( 0,-1, 2.2), 0.8, emit);
		}

		nPlanes = 1;
		if(nPlanes>0){
			cudaMallocManaged(&planes, nPlanes*sizeof(Plane));
			planes[0] = Plane(glm::vec3(0,0,-3), glm::vec3(4,0,0), glm::vec3(0,4,0), floor);
		}

		printf("scene created\n");
	}

	__host__ void destroyScene(){
		cudaFree(spheres[0].mtl);
		cudaFree(spheres[1].mtl);
		// cudaFree(spheres[2].mtl);

		cudaFree(spheres);
		cudaFree(planes);
		spheres = nullptr;
		nSpheres = 0;
	}
};