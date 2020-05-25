// iterative path tracer written in cuda

#define _USE_MATH_DEFINES
#include <iostream>
#include <curand_kernel.h>

#include "functions.h"
#include "vec.h"
#include "image.h"
#include "Camera.h"
#include "Scene.h"
#include "PathState.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
		file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

__global__ void initRandom(curandState* state, const int w, const int h){
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	const int j = blockIdx.y*blockDim.y + threadIdx.y;
	if(i<w && j<h){
		const int idx = j*w+i;
		// curand_init(9999, idx, 0, state+idx);
		curand_init(idx, 0, 0, state+idx);
	}
}

__device__ void throuput(	float3 *th, Ray *ray,
							Hit &hit, curandState *randState){
	if(hit.mtl->type == lambert){
		float u1 = curand_uniform(randState);
		float u2 = curand_uniform(randState) * 2*M_PI;

		float3 bitan = normalize(cross(hit.n, hit.tan));
		float r = sqrt(u1);
		
		ray->o = hit.pos + (1e-6*hit.n);
		ray->d = (hit.n * (sqrt(1-u1)))
			+ (hit.tan * (r*cos(u2)))
			+ (bitan * (r*sin(u2)));
		*th *= hit.mtl->col;
	}

	if(hit.mtl->type == GGX_ref_iso){
		float u1 = curand_uniform(randState);
		float u2 = curand_uniform(randState) * 2*M_PI;

		float3 bitan = normalize(cross(hit.n, hit.tan));
		float r2 = hit.mtl->alpha2*u1/(1+u1*(hit.mtl->alpha2-1));
		float r = sqrt(r2);

		ray->o = hit.pos + (1e-6*hit.n);
		ray->d = (sqrt(1-r2))*hit.n
			+ (cos(u2)*r)*hit.tan
			+ (sin(u2)*r)*bitan;
		*th *= hit.mtl->col;
	}
}

__device__ float GGX_iso_D(float hn, float a2){
	if(hn>0){
		float cos2 = hn*hn;
		float t = a2*cos2 + 1-cos2;
		return (t>1e-6)? a2/(M_PI*t*t) : 0;
	}else return 0;
}

__device__ float smith_mask(float3 x, float3 n, float a2){
	float xn2 = dot(x,n);
	xn2 *= xn2;
	return 2/(1+sqrt(1+a2*(1-xn2)/xn2));
}

__global__ void render(	float3* const pResult, 
						const uint32_t w, const uint32_t h,
						Scene* const scene,
						curandState* randState){
	const uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
	const uint32_t j = blockIdx.y*blockDim.y + threadIdx.y;
	
	if((i >= w) || (j >= h)) return;
	
	const float imgDimNorm = 1.0/h;
	const uint32_t idx = j*w+i;
	curandState *rand_local = randState+idx;
	uint32_t spp = 4000;
	bool NEE = !true;

	for(uint32_t n=0; n<spp; n++){
		float x =  2*(i+curand_uniform(rand_local))*imgDimNorm - 1;
		float y = -2*(j+curand_uniform(rand_local))*imgDimNorm + 1;

		PathState ps(2,rand_local,true);
		Ray ray = scene->camera.ray(x,y);
		float3 th = make_float3(1);

		while(ps.contIntegration(&th)){
			Hit hit = scene->nearest(ray);

			if(hit.mtl->type == emit){
				ps.terminate = true;
				pResult[idx] += th * hit.mtl->col;
			}
			else if(NEE){
				//currently sampleing facing facet only from sphere light
				float3 lightPos = scene->spheres[2].center;
				float3 lightEmit = scene->spheres[2].mtl->col;
				float lightR = scene->spheres[2].r;

				float3 PL = lightPos-hit.pos;
				if(dot(PL,hit.n)>0){
					Ray shadowRay(hit.pos + 1e-6*hit.n, normalize(PL));
					Hit shadowTrace = scene->nearest(shadowRay);
					float dist = abs(PL)-lightR;

					if(shadowTrace.dist < dist+1e-6){
						float r;

						if(hit.mtl->type == lambert){
							r = M_1_PI;
						}

						if(hit.mtl->type == GGX_ref_iso){
							float3 wi = shadowRay.d;
							float3 wo = -ray.d;

							float3 m = normalize(wi+wo);
							float mn = dot(m, hit.n);
							if(0<mn){
								float D = GGX_iso_D(dot(m, hit.n), hit.mtl->alpha2);
								float gi = smith_mask(wi, hit.n, hit.mtl->alpha2);
								float go = smith_mask(wo, hit.n, hit.mtl->alpha2);
								float w = 0.25*fabs(gi*go*dot(wo, m)/(dot(wo, hit.n)*mn));
								r = w*D;
							}
							else r = 0;
						}

						float G = dot(hit.n,shadowRay.d)/(dist*dist);
						pResult[idx] += th*hit.mtl->col*r*lightEmit*G;
					}
				}
			}
			// end NEE

			throuput(&th, &ray, hit, rand_local);
		}
	}
	pResult[idx] *= make_float3(1.0/spp);

}

int main(){
	printBreak();

	const int w = 512;
	const int h = 512;
	float3* result;
	cudaMallocManaged(&result, w*h*sizeof(float3));

	curandState *randState;
	cudaMalloc(&randState, w*h*sizeof(curandState));

	Scene* scene;
	cudaMallocManaged(&scene, sizeof(Scene));
	scene->createScene();
	printBreak();


	const int tx = 16;
	const int ty = 16;

	const dim3 blocks(w/tx+1, h/ty+1);
	const dim3 threads(tx,ty);

	initRandom<<<blocks, threads>>>(randState, w, h);
	cudaDeviceSynchronize();

	render<<<blocks, threads>>>(result, w, h, scene, randState);
	checkCudaErrors(cudaGetLastError());
	cudaDeviceSynchronize();

	writeJPG(result, w, h);


	cudaFree(scene->spheres);

	cudaFree(scene);
	cudaFree(result);

	cudaDeviceReset();
	printBreak();
	return 0;
}