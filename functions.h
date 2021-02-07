#pragma once

#include <cuda.h>
#include <glm/glm.hpp>

__host__ __device__ void printBreak(){printf("----------  ----------  ----------  ----------\n");}
__host__ __device__ void print_vec3(glm::vec3& a){printf("%f %f %f\n", a.x, a.y, a.z);}