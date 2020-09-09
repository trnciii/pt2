#pragma once

#include "vec.h"
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

		hit.pos = ray.o + make_float3(hit.dist)*ray.d;
		hit.tan = normalize(hit.dpdu);
		if(dot(ray.d,hit.n)>0){
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
		background.setEmission(make_float3(0.1));

		camera.pos = make_float3(0,-4,-0);
		camera.setBasis(make_float3(0,1,0), make_float3(0,0,1));
		camera.focal = 1;

		Material *emit;
		cudaMallocManaged(&emit, sizeof(Material));
		emit->setEmission(make_float3(4));

		Material *left;
		cudaMallocManaged(&left, sizeof(Material));
		left->setLambert(make_float3(0.9, 0.1, 0.1));
		left->setNonPhotorealistic();

		Material *right;
		cudaMallocManaged(&right, sizeof(Material));
		right->setLambert(make_float3(0.1, 0.1, 0.9));

		Material *floor;
		cudaMallocManaged(&floor, sizeof(Material));
		floor->setGGX_iso(make_float3(0.1), 0.1);

		nSpheres = 3;
		if(nSpheres>0){
			cudaMallocManaged(&spheres, nSpheres*sizeof(Sphere));
			spheres[0] = Sphere(make_float3(-0, 1,-1  ), 1.5, left);
			spheres[1] = Sphere(make_float3( 1, 0,-1  ), 1.5, right);
			spheres[2] = Sphere(make_float3( 0,-1, 2.2), 0.8, emit);
		}

		nPlanes = 1;
		if(nPlanes>0){
			cudaMallocManaged(&planes, nPlanes*sizeof(Plane));
			planes[0] = Plane(make_float3(0,0,-3), make_float3(4,0,0), make_float3(0,4,0), floor);
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