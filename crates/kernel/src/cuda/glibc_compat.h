// Workaround for glibc 2.41 / CUDA header conflict.
// glibc 2.41 declares rsqrt/rsqrtf/cospi/sinpi/cospif/sinpif
// with noexcept(true), but CUDA's math_functions.h declares them
// without noexcept. This causes a compilation error.
//
// Fix: pre-declare them before CUDA headers are included,
// matching CUDA's exception spec (no noexcept).
#pragma once

// Suppress glibc's conflicting math declarations
#define __GLIBC_USE_IEC_60559_FUNCS_EXT 0
#define __GLIBC_USE_ISOC2X 0
