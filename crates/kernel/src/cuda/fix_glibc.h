// Fix glibc 2.42+ / CUDA header conflict.
// Must be included BEFORE any CUDA header.
// Prevents glibc from declaring rsqrt/rsqrtf (which conflict with CUDA's version).
#pragma once

// Pre-declare rsqrt/rsqrtf matching glibc's signature so CUDA doesn't conflict
#ifdef __cplusplus
extern "C" {
#endif

// Shadow glibc's declarations by providing them before CUDA headers load
double rsqrt(double __x) __attribute__((weak)) __attribute__((__nothrow__));
float rsqrtf(float __x) __attribute__((weak)) __attribute__((__nothrow__));

#ifdef __cplusplus
}
#endif

// Prevent glibc from re-declaring these
#define __GLIBC_USE_IEC_60559_FUNCS_EXT_C23 0
