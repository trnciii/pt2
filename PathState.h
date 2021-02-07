#pragma once

struct PathState{
	bool terminate;
	uint32_t curDepth;
	uint32_t maxDepth;
	curandState *randState;
	const bool enableRussianRoulette;

	__device__ PathState(uint32_t depth, curandState* rand, const bool RR)
	:terminate(false), curDepth(0), maxDepth(depth), randState(rand), enableRussianRoulette(RR){}

	__device__ bool contIntegration(glm::vec3 *th){
		if(terminate)return false;

		if(++curDepth > maxDepth){
			if(!enableRussianRoulette)return false;

			float p = fmaxf(th->x, fmaxf(th->y,th->z));
			if(curand_uniform(randState)<p)
				*th *= 1/p;
			else return false;
		}
		return true;
	}
};