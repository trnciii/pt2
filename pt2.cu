// iterative path tracer written in cuda

#define _USE_MATH_DEFINES
#include <iostream>
#include <curand_kernel.h>

#include <cuda.h>
#include <glm/glm.hpp>

#include "functions.h"
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
		curand_init(0, idx, 0, state+idx);
	}
}

__device__ float GGX_iso_D(float hn, float a2){
	if(hn>0){
		float cos2 = hn*hn;
		float t = a2*cos2 + 1-cos2;
		return (t>1e-6)? a2/(M_PI*t*t) : 0;
	}else return 0;
}

__device__ float smith_mask(glm::vec3 x, glm::vec3 n, float a2){
	float xn2 = dot(x,n);
	xn2 *= xn2;
	return 2/(1+sqrt(1+a2*(1-xn2)/xn2));
}

__device__ glm::vec3 sample_GGX_NDF(float u1, float u2, float a2){
	u2 *= 2*M_PI;
	float r2 = a2*u1/(1+u1*(a2-1));
	float r = sqrt(r2);
	float z = sqrt(1-r2);

	return glm::vec3(r*cos(u2), r*sin(u2), z);
}

__device__ void throuput(	glm::vec3 *th, float *pdf, Ray *ray,
							Hit &hit, curandState *randState){

	if(hit.mtl->type == lambert){
		float u1 = curand_uniform(randState);
		float u2 = curand_uniform(randState) * 2*M_PI;

		glm::vec3 bitan = glm::normalize(glm::cross(hit.n, hit.tan));
		float r = sqrt(u1);
		
		ray->o = hit.pos + ((float)1e-6*hit.n);
		ray->d = (hit.n * (float)(sqrt(1-u1)))
			+ (hit.tan * (float)(r*cos(u2)))
			+ (bitan * (float)(r*sin(u2)));
		
		*th *= hit.mtl->col;
	}

	if(hit.mtl->type == GGX_ref_iso){
		glm::vec3 wi = -ray->d;
		glm::vec3 bitan = glm::normalize(glm::cross(hit.n, hit.tan));
		
		float u1 = curand_uniform(randState);
		float u2 = curand_uniform(randState);
		glm::vec3 m_tan = sample_GGX_NDF(u1, u2, hit.mtl->alpha2);
		glm::vec3 m = m_tan.x*hit.tan + m_tan.y*bitan + m_tan.z*hit.n;
		
		glm::vec3 wo = ray->d - 2*glm::dot(ray->d, m)*m;

		float mn = glm::dot(m, hit.n);
		float gi = smith_mask(wi, hit.n, hit.mtl->alpha2);
		float go = smith_mask(wo, hit.n, hit.mtl->alpha2);
		float w = fabs(gi*go*glm::dot(wi, m)/(glm::dot(wi, hit.n)*m_tan.z));

		ray->o = hit.pos + ((float)1e-6*hit.n);
		ray->d = wo;
		*th *= w*hit.mtl->col;
	}

	return;
}

__global__ void render(	glm::vec3* const pResult, 
						const uint32_t w, const uint32_t h, const uint32_t spp,
						Scene* const scene,
						curandState* randState){
	const uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
	const uint32_t j = blockIdx.y*blockDim.y + threadIdx.y;
	
	if((i >= w) || (j >= h)) return;
	
	const float imgDimNorm = 1.0/h;
	const uint32_t idx = j*w+i;
	curandState *rand_local = randState+idx;
	bool NEE = !true;

	for(uint32_t n=0; n<spp; n++){
		float x =  2*(i+curand_uniform(rand_local))*imgDimNorm - 1;
		float y = -2*(j+curand_uniform(rand_local))*imgDimNorm + 1;

		PathState ps(2,rand_local,true);
		Ray ray = scene->camera.ray(x,y);
		glm::vec3 th = glm::vec3(1);
		float pdf = 1;

		while(ps.contIntegration(&th)){
			Hit hit = scene->nearest(ray);


			if(hit.mtl->type == emit){
				ps.terminate = true;

				glm::vec3 cont = th * hit.mtl->col;
				pResult[idx] += cont;
			}

			else if(NEE){
				float u1 = 2*curand_uniform(rand_local) - 1;
				float u2 = 2*M_PI*curand_uniform(rand_local);
				float r = sqrt(1-u1*u1);

				float lightR = scene->spheres[2].r;
				glm::vec3 lightNormal = glm::vec3(r*cos(u2), r*sin(u2), u1);
				glm::vec3 lightPos = scene->spheres[2].center + lightR*lightNormal;
				glm::vec3 lightEmit = scene->spheres[2].mtl->col;

				glm::vec3 PL = lightPos-hit.pos;
				if(dot(PL,hit.n)>0 && dot(lightNormal, PL)<0){
					Ray shadowRay(hit.pos + (float)1e-6*hit.n, glm::normalize(PL));
					Hit shadowTrace = scene->nearest(shadowRay);
					float dist = glm::length(PL);

					if(shadowTrace.dist < dist+1e-6){
						float r;

						if(hit.mtl->type == lambert){
							r = M_1_PI;
						}

						if(hit.mtl->type == GGX_ref_iso){
							glm::vec3 wi = shadowRay.d;
							glm::vec3 wo = -ray.d;

							glm::vec3 m = glm::normalize(wi+wo);
							float mn = glm::dot(m, hit.n);
							if(0<mn){
								float D = GGX_iso_D(glm::dot(m, hit.n), hit.mtl->alpha2);
								float gi = smith_mask(wi, hit.n, hit.mtl->alpha2);
								float go = smith_mask(wo, hit.n, hit.mtl->alpha2);
								float w = 0.25*fabs(gi*go*glm::dot(wo, m)/(glm::dot(wo, hit.n)*mn));
								r = w*D;
							}
							else r = 0;
						}

						float G = -glm::dot(hit.n,shadowRay.d)*glm::dot(shadowRay.d,lightNormal)/(dist*dist);
						glm::vec3 cont = th*hit.mtl->col*lightEmit*G*r;
						pResult[idx] += cont;
					}
				}
			}
			// end NEE

			throuput(&th, &pdf, &ray, hit, rand_local);
		}
	}
	pResult[idx] = pResult[idx]/glm::vec3(spp);

}

int main(){
	printBreak();

	const uint32_t w = 512;
	const uint32_t h = 512;
	const uint32_t spp = 100;
	glm::vec3* result;
	cudaMallocManaged(&result, w*h*sizeof(glm::vec3));

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
	checkCudaErrors(cudaGetLastError());
	cudaDeviceSynchronize();

	render<<<blocks, threads>>>(result, w, h, spp, scene, randState);
	checkCudaErrors(cudaGetLastError());
	cudaDeviceSynchronize();

	writeimage(result, w, h);


	cudaFree(scene->spheres);

	cudaFree(scene);
	cudaFree(result);

	cudaDeviceReset();
	printBreak();
	return 0;
}