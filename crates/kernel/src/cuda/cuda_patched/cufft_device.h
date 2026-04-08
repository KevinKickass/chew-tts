 /* Copyright 2005-2025 NVIDIA Corporation.  All rights reserved.
  *
  * NOTICE TO LICENSEE:
  *
  * The source code and/or documentation ("Licensed Deliverables") are
  * subject to NVIDIA intellectual property rights under U.S. and
  * international Copyright laws.
  *
  * The Licensed Deliverables contained herein are PROPRIETARY and
  * CONFIDENTIAL to NVIDIA and are being provided under the terms and
  * conditions of a form of NVIDIA software license agreement by and
  * between NVIDIA and Licensee ("License Agreement") or electronically
  * accepted by Licensee.  Notwithstanding any terms or conditions to
  * the contrary in the License Agreement, reproduction or disclosure
  * of the Licensed Deliverables to any third party without the express
  * written consent of NVIDIA is prohibited.
  *
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
  * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  THEY ARE
  * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
  * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
  * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
  * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
  * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
  * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
  * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
  * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
  * OF THESE LICENSED DELIVERABLES.
  *
  * U.S. Government End Users.  These Licensed Deliverables are a
  * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
  * 1995), consisting of "commercial computer software" and "commercial
  * computer software documentation" as such terms are used in 48
  * C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government
  * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
  * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
  * U.S. Government End Users acquire the Licensed Deliverables with
  * only those rights set forth herein.
  *
  * Any use of the Licensed Deliverables in individual and commercial
  * software must include, in the user documentation and internal
  * comments to the code, the above Disclaimer and U.S. Government End
  * Users Notice.
  */

/*!
* \file cufft_device.h
* \brief Public header file for the NVIDIA CUDA FFT library (CUFFT)
*/

#ifndef _CUFFT_DEVICE_H_
#define _CUFFT_DEVICE_H_

#ifndef CUFFT_ENABLE_EXPERIMENTAL_API
#  error "cuFFT device API is experimental and subject to change. Define CUFFT_ENABLE_EXPERIMENTAL_API to acknowledge this notice."
#endif

#include "cufft.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CUFFT_DEVICE_VER_MAJOR 0
#define CUFFT_DEVICE_VER_MINOR 2
#define CUFFT_DEVICE_VER_PATCH 0

#define CUFFT_DEVICE_VERSION 200

typedef enum cufftDescriptionDirection_t
{
  CUFFT_DESC_FORWARD = 0,
  CUFFT_DESC_INVERSE = 1,
} cufftDescriptionDirection;

typedef enum cufftDescriptionType_t
{
  CUFFT_DESC_C2C = 0,
  CUFFT_DESC_C2R = 1,
  CUFFT_DESC_R2C = 2
} cufftDescriptionType;

typedef enum cufftDescriptionPrecision_t
{
  CUFFT_DESC_SINGLE = 0,
  CUFFT_DESC_DOUBLE = 1,
  CUFFT_DESC_HALF = 2
} cufftDescriptionPrecision;

typedef enum cufftDescriptionTrait_t {
  CUFFT_DESC_TRAIT_SIZE = 0,
  CUFFT_DESC_TRAIT_DIRECTION = 1,
  CUFFT_DESC_TRAIT_PRECISION = 2,
  CUFFT_DESC_TRAIT_TYPE = 3,
  CUFFT_DESC_TRAIT_SM = 4,
  CUFFT_DESC_TRAIT_ELEMENTS_PER_THREAD = 5
} cufftDescriptionTrait;

typedef enum cufftDeviceFunctionTrait_t {
  CUFFT_DEVICE_FUNC_TRAIT_ELEMENTS_PER_THREAD = 0,
  CUFFT_DEVICE_FUNC_TRAIT_STORAGE_SIZE = 1,
  CUFFT_DEVICE_FUNC_TRAIT_SHARED_MEMORY_PER_FFT = 2,
  CUFFT_DEVICE_FUNC_TRAIT_THREADS_PER_FFT = 3,
  CUFFT_DEVICE_FUNC_TRAIT_SUGGESTED_FFTS_PER_BLOCK = 4,
} cufftDeviceFunctionTrait;

typedef enum cufftDeviceCodeContainer_t {
  CUFFT_DEVICE_LTOIR = 0,
  CUFFT_DEVICE_FATBIN = 1
} cufftDeviceCodeContainer;

typedef int cufftDescriptionHandle;
typedef int cufftDeviceHandle;
typedef int cufftDeviceFunctionHandle;

cufftResult CUFFTAPI cufftDeviceGetVersion(int *version);

cufftResult CUFFTAPI cufftDescriptionCreate(cufftDescriptionHandle* desc_handle);
cufftResult CUFFTAPI cufftDescriptionSetTraitInt64(cufftDescriptionHandle desc_handle, cufftDescriptionTrait trait, const long long int value);
cufftResult CUFFTAPI cufftDescriptionGetTraitInt64(cufftDescriptionHandle desc_handle, cufftDescriptionTrait trait, long long int* value);

cufftResult CUFFTAPI cufftDeviceCreate(cufftDeviceHandle* handle, size_t desc_count, cufftDescriptionHandle* desc_handles);
cufftResult CUFFTAPI cufftDeviceIsSupported(cufftDeviceHandle handle, cufftDescriptionHandle desc_handle, bool* is_supported);

cufftResult CUFFTAPI cufftDeviceGetNumDeviceFunctions(cufftDeviceHandle handle, cufftDescriptionHandle desc_handle, size_t* func_count);
cufftResult CUFFTAPI cufftDeviceGetDeviceFunctions(cufftDeviceHandle handle, cufftDescriptionHandle desc_handle, size_t func_count, cufftDeviceFunctionHandle* func_handles);
cufftResult CUFFTAPI cufftDeviceGetDeviceFunctionTraitInt64(cufftDeviceHandle handle, cufftDeviceFunctionHandle func_handle, cufftDeviceFunctionTrait trait, long long int* value);

cufftResult CUFFTAPI cufftDeviceGetDatabaseStrSize(cufftDeviceHandle handle, size_t* db_str_size, cufftDeviceFunctionHandle func_handle = -1);
cufftResult CUFFTAPI cufftDeviceGetDatabaseStr(cufftDeviceHandle handle, size_t db_str_size, char* db_str, cufftDeviceFunctionHandle func_handle = -1);

cufftResult CUFFTAPI cufftDeviceGetNumLTOIRs(cufftDeviceHandle handle, size_t* code_count, cufftDeviceFunctionHandle func_handle = -1);
cufftResult CUFFTAPI cufftDeviceGetLTOIRSizes(cufftDeviceHandle handle, size_t code_count, size_t* code_sizes, cufftDeviceFunctionHandle func_handle = -1);
cufftResult CUFFTAPI cufftDeviceGetLTOIRs(cufftDeviceHandle handle, size_t code_count, char** code_ptrs, cufftDeviceCodeContainer* code_containers, cufftDeviceFunctionHandle func_handle = -1);

cufftResult CUFFTAPI cufftDeviceDestroy(cufftDeviceHandle handle);

#ifdef __cplusplus
}
#endif

#endif /* _CUFFT_DEVICE_H_ */
