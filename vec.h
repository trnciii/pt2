#pragma once

// float3

__host__ __device__ inline float3 make_float3(float a){return make_float3(a,a,a);}

__host__ __device__ inline float3 operator +(float3 a, float3 b){return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);}
__host__ __device__ inline float3 operator -(float3 a, float3 b){return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);}
__host__ __device__ inline float3 operator -(float3 a){return make_float3(-a.x, -a.y, -a.z);}
__host__ __device__ inline float3 operator *(float3 a, float3 b){return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);}
__host__ __device__ inline float3 operator *(float3 a, float t){return make_float3(a.x*t, a.y*t, a.z*t);}
__host__ __device__ inline float3 operator *(float t, float3 a){return make_float3(a.x*t, a.y*t, a.z*t);}
__host__ __device__ inline float3 operator /(float3 a, float3 b){return make_float3(a.x/b.x, a.y/b.y, a.z/b.z);}
__host__ __device__ inline float3 operator /(float3 a, float t){t=1/t; return make_float3(a.x*t, a.y*t, a.z*t);}

__host__ __device__ inline void operator +=(float3 &a, float3 &b){a.x+=b.x; a.y+=b.y; a.z+=b.z;}
__host__ __device__ inline void operator -=(float3 &a, float3 &b){a.x-=b.x; a.y-=b.y; a.z-=b.z;}
__host__ __device__ inline void operator *=(float3 &a, float3 &b){a.x*=b.x; a.y*=b.y; a.z*=b.z;}
__host__ __device__ inline void operator /=(float3 &a, float3 &b){a.x/=b.x; a.y/=b.y; a.z/=b.z;}

__host__ __device__ inline float dot(float3 &a, float3 &b){return a.x*b.x + a.y*b.y + a.z*b.z;}
__host__ __device__ inline float abs(float3 a){return sqrt(dot(a,a));}
__host__ __device__ inline float3 cross(float3 a, float3 b){return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);}
__host__ __device__ inline float3 normalize(float3 a){float len = abs(a); return make_float3(a.x/len, a.y/len, a.z/len);}

__host__ __device__ inline void print_float3(float3 &v){printf("%f, %f, %f\n", v.x, v.y, v.z);}



// float2

__host__ __device__ inline float2 make_float2(float a){return make_float2(a,a);}

__host__ __device__ inline float2 operator +(float2 a, float2 b){return make_float2(a.x+b.x, a.y+b.y);}
__host__ __device__ inline float2 operator -(float2 a, float2 b){return make_float2(a.x-b.x, a.y-b.y);}
__host__ __device__ inline float2 operator -(float2 a){return make_float2(-a.x, -a.y);}
__host__ __device__ inline float2 operator *(float2 a, float2 b){return make_float2(a.x*b.x, a.y*b.y);}
__host__ __device__ inline float2 operator /(float2 a, float2 b){return make_float2(a.x/b.x, a.y/b.y);}

__host__ __device__ inline void operator +=(float2 &a, float2 &b){a.x+=b.x; a.y+=b.y;}
__host__ __device__ inline void operator -=(float2 &a, float2 &b){a.x-=b.x; a.y-=b.y;}
__host__ __device__ inline void operator *=(float2 &a, float2 &b){a.x*=b.x; a.y*=b.y;}
__host__ __device__ inline void operator /=(float2 &a, float2 &b){a.x/=b.x; a.y/=b.y;}

__host__ __device__ inline float dot(float2 a, float2 b){return a.x*b.x + a.y*b.y;}
__host__ __device__ inline float abs(float2 a){return sqrt(dot(a,a));}
__host__ __device__ inline float2 normalize(float2 a){float len=abs(a); return make_float2(a.x/len, a.y/len);}