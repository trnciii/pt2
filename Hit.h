#pragma once

#include <cuda.h>
#include <glm/glm.hpp>

class Material;

struct Hit{
	float  dist;
	glm::vec3 pos;
	glm::vec3 n;
	glm::vec3 dpdu;
	glm::vec3 dpdv;
	glm::vec3 tan;
	glm::vec2 uv;
	Material *mtl;
	bool backface = false;
};