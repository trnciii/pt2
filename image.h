#pragma once

#include "functions.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./stbi/stb_image_write.h"

__host__ unsigned char tonemap(double c){
	int c_out = 255*pow(c,(1/2.2)) +0.5;
	if(255 < c_out)c_out = 255;
	if(c_out < 0)c_out = 0;
	return c_out&0xff;
}

__host__ int writeJPG(float3 *color, int w, int h){
	unsigned char *tone = new unsigned char[3*w*h];
	for(int i=0; i<w*h; i++){
		tone[3*i  ] = tonemap(color[i].x);
		tone[3*i+1] = tonemap(color[i].y);
		tone[3*i+2] = tonemap(color[i].z);
	}

	int result = stbi_write_jpg("result.jpg", w, h, 3, tone, 3*w);
	std::cout <<"image saved as result.jpg" <<std::endl;
	delete[] tone;
	return result;
}