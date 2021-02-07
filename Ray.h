#pragma once

#include <cuda.h>
#include <glm/glm.hpp>

struct Ray{
	glm::vec3 o;
	glm::vec3 d;

	__device__ Ray(glm::vec3 _o, glm::vec3 _d):o(_o), d(_d){}
};