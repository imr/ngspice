#ifndef NG_SIMD_VECTOR_H
#define NG_SIMD_VECTOR_H

#include "ngspice/SIMD/simddef.h"

#ifndef NSIMD
#error NSIMD must be defined
#endif

#if NSIMD==4
typedef double Vec4d __attribute__ ((vector_size (sizeof(double)*4), aligned (sizeof(double)*4)));
typedef int64_t Vec4m __attribute__ ((vector_size (sizeof(double)*4), aligned (sizeof(double)*4)));
typedef int64_t Vec4i __attribute__ ((vector_size (sizeof(double)*4), aligned (sizeof(double)*4)));

#define VecNd Vec4d
#define VecNm Vec4m
#define VecNi Vec4i

#endif

#if NSIMD==8
typedef double Vec8d __attribute__ ((vector_size (sizeof(double)*8), aligned (sizeof(double)*8)));
typedef int64_t Vec8m __attribute__ ((vector_size (sizeof(double)*8), aligned (sizeof(double)*8)));
typedef int64_t Vec8i __attribute__ ((vector_size (sizeof(double)*8), aligned (sizeof(double)*8)));

#define VecNd Vec8d
#define VecNm Vec8m
#define VecNi Vec8i

#endif

#if NSIMD==2
typedef double Vec2d __attribute__ ((vector_size (sizeof(double)*2), aligned (sizeof(double)*2)));
typedef int64_t Vec2m __attribute__ ((vector_size (sizeof(double)*2), aligned (sizeof(double)*2)));
typedef int64_t Vec2i __attribute__ ((vector_size (sizeof(double)*2), aligned (sizeof(double)*2)));

#define VecNd Vec2d
#define VecNm Vec2m
#define VecNi Vec2i

#endif

#endif
