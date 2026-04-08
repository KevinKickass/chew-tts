/*
 * SPDX-FileCopyrightText: Copyright (c) 2024  NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */
#ifndef _CUOBJCLIENT_H_
#define _CUOBJCLIENT_H_

#define OBJ_RDMA_V1 "CUOBJ"

/**
 * @brief Maximum memory registration size limit for RDMA operations (4GiB)
 */
#define CUOBJ_MAX_MEMORY_REG_SIZE (4ULL * 1024 * 1024 * 1024)

#include <unistd.h>
#include <stdio.h>
#include <mutex>

#include "cufile.h"
#include "cuobjtelem.h"

/**
 * @file cuobjclient.h
 * @brief  cuObject client C++ APIs
 *
 * This file contains all the C++ APIs to perform GPUDirect Storage supported client IO operations for objects
 */


/**
 * Telemetry object
 */
extern std::shared_ptr<cuObjTelem> clientTelem;

/**
 * @brief cuObject error numbers.
 *
 * @note  These errors will be expanded in future
 *
 */

typedef enum cuObjErr_enum {
	CU_OBJ_SUCCESS =0, /**< Operation successfully completed */
	CU_OBJ_FAIL =1,    /**< Operation failed */
} cuObjErr_t;	

/**
 * @brief cuObject RDMA descriptor protocol version
 *
 */
typedef enum cuObjProto_enum {
	CUOBJ_PROTO_RDMA_DC_V1=1001, /**< RDMA support version 1 */
	CUOBJ_PROTO_MAX
} cuObjProto_t;


/**
 * @brief cuObject Operation type
 *
 */
typedef enum cuObjOpType_enum {
	CUOBJ_GET = 0, /**< GET operation */
	CUOBJ_PUT = 1, /**< PUT operation */
	CUOBJ_INVALID=9999
} cuObjOpType_t;

/**
 * @brief cuObject Operation callbacks
 * This struct specifies the callback interfaces used by cuObjClient class object during IO operations.
 * @note The callbacks can be called from a different thread than the caller thread. user must lock any shared resources
 * that can be used concurrently across multiple callers.
 */
typedef struct CUObjIOOps {
/**
 * @brief cuObject GET callback
 * @param handle cookie to the user context provided in the cuObjGet call. cuObjClient::getCtx(handle) should be called for getting the user context
 * @param ptr pointer to the start of the memory chunk
 * @param size size of the memory chunk being read.
 * @param offset starting object offset for this memory chunk.
 * @param cufileRDMAInfo_t Pointer to a RDMA memory descriptor string
 *
 * @return size of the data read on success or negative -1, the data read is obtained from control path
 *
 * @note offset will be set to zero for cases where the MaxReqCallbackSize is equal to or greater the cuObjectGet call size
 * @note size will be set to total requested size n cuObjectGet for cases where the MaxReqCallbackSize is equal to or greater the cuObjectGet call size
 *
 *
 * @see cuObjClient::cuObjGet
 */

      ssize_t (*get) (const void *handle, char *ptr, size_t size, loff_t offset, const cufileRDMAInfo_t*);
/**
 * @brief cuObject PUT callback
 * @param handle to the user context provided in the cuObjPut call. cuObjClient::getCtx(handle) should be called for getting the user context
 * @param ptr pointer to the start of the memory chunk
 * @param size size of the memory chunk being written
 * @param offset starting object offset for this memory chunk.
 * @param cufileRDMAInfo_t Pointer to a RDMA memory descriptor string
 *
 * @return size of the data written on success or negative -1, the data written is obtained from control path
 *
 * @note offset will be set to zero for cases where the MaxReqCallbackSize is equal to or greater the cuObjectPut call size
 * @note size will be set to total requested size n cuObjectGet for cases where the MaxReqCallbackSize is equal to or greater the cuObjectPut call size
 *
 *
 * @see cuObjClient::cuObjPut
 */

      ssize_t (*put) (const void *handle, const char *ptr, size_t size, loff_t offset, const cufileRDMAInfo_t*);
}CUObjOps_t;

/**
 * @brief cuObject memory type
 *
 */
typedef enum cuObjMemoryType_enum {
	CUOBJ_MEMORY_SYSTEM = 0,
	CUOBJ_MEMORY_CUDA_MANAGED = 1,
	CUOBJ_MEMORY_CUDA_DEVICE = 2,
	CUOBJ_MEMORY_UNKNOWN = 3,
	CUOBJ_MEMORY_INVALID = 4
} cuObjMemoryType_t;

/**
 * @brief cuObjClient class.
 *
 * cuObjClient Provides client side APIs to prepare PUT/GET operations for out-of-band RDMA IO operations.
 * The user of this object is expected to implement a set of callback interfaces specified in CUObjIOOps.
 * Once the client object is created, user is expected to optionally register the memory for RDMA and
 * perform PUT and GET operations on this client object.
 * The cuObjClient will validate the memory and prepare a memory region for RDMA transer.
 * The cuObjClient will call one or more callback operations with relevant RDMA information.
 * The user is expected to relay the RDMA information and other data to cuObjServer using standard control path.
 * After the IO completion, the callback should return the total data read or written in this context.
 * The client object may perform additional operations before the cuObjectPut/cuObjectGet operation finishes.
 */

class cuObjClient {
	public:

                /**
                * @brief constructor for cuObjClient class.
                * @param ops callback reference to CUObjIOOps
                * @param proto RDMA descriptor protocol used for this client. defaults to CUOBJ_PROTO_RDMA_DC_V1
                */
		cuObjClient(CUObjOps_t& ops, cuObjProto_t proto=CUOBJ_PROTO_RDMA_DC_V1);
		~cuObjClient();

                /**
                * @brief Acquire a RDMA memory descriptor for the user memory
                * @param ptr start address of user memory
                * @param size size of memory that needs pinning starting from the start address of user memory
                * @return CU_OBJ_SUCCESS on success, CU_OBJ_FAIL on failure
                * @note Memory sizes greater than or equal to CUOBJ_MAX_MEMORY_SIZE (4GiB) are not supported
                */
		cuObjErr_t cuMemObjGetDescriptor(void *ptr, size_t size);
                /**
                * @brief Get a RDMA memory descriptor for the user memory
                * @param ptr start address of user memory
                * @return max size of the callback for this memory pointer
                * @note: The size can be smaller than allocated memory if the memory is not registered or
                * the underlying RDMA subsystem does not allow for pinning/transfer of the entire memory in a single callback
                */
		ssize_t cuMemObjGetMaxRequestCallbackSize(void *ptr);

                /**
                * @brief release the RDMA memory descriptor for the user memory
                * @param ptr start address of user memory used during cuMemObjGetDescriptor
                * @return error status if the memory cannot be unregistered
                */

		cuObjErr_t cuMemObjPutDescriptor(void *ptr);

                /**
                * @brief Get RDMA descriptor string for a registered memory buffer with size and offset
                * @param ptr start address of user memory that was registered with cuMemObjGetDescriptor
                * @param size size of the memory region for which to generate the descriptor string
                * @param buffer_offset offset from the base address to start the descriptor region
                * @param operation operation type (CUOBJ_GET, CUOBJ_PUT)
                * @param desc_str_out pointer to store the allocated descriptor string (caller must free using cuMemObjPutRDMAToken)
                * @return CU_OBJ_SUCCESS on success, CU_OBJ_FAIL on failure
                * @note This function calls cuFileRDMADescStrGet which allocates memory for the descriptor string. The caller must call cuMemObjPutRDMAToken to free it.
                * @note The buffer must be registered with cuMemObjGetDescriptor and be RDMA capable for this function to succeed.
                * @note The size and buffer_offset parameters modify the descriptor string's address and size fields accordingly.
                * @note The buffer_offset + size must not exceed the originally registered buffer size.
                */
		cuObjErr_t cuMemObjGetRDMAToken(void *ptr, size_t size, size_t buffer_offset, cuObjOpType_t operation, char **desc_str_out);

                /**
                * @brief Free RDMA descriptor string allocated by cuMemObjGetRDMAToken
                * @param desc_str descriptor string to free (allocated by the underlying cuFileRDMADescStrGet)
                * @return CU_OBJ_SUCCESS on success, CU_OBJ_FAIL on failure
                * @note This function calls cuFileRDMADescStrPut to free memory allocated by the underlying cuFile API.
                */
		cuObjErr_t cuMemObjPutRDMAToken(char *desc_str);

                /**
                * @brief Get the user context provided in the cuObjGet and cuObjPut from handle in the callback
                * @param handle pointer to the handle from the callback
                * @return void pointer to the user context.
                */

		static void* getCtx(const void *handle);

                /**
                * @brief API to perform GET operation using cuObject
                * @param ctx pointer to a user control context, used in GET callback
                * @param ptr pointer to a user memory
                * @param size size of the GET operation
                * @param offset currently set to 0. reserved for future use
                * @param buf_offset currently set to 0. reserved for future use
                * @return data returned by the cuObjServer or a negative error code.
                */

		ssize_t cuObjGet(void *ctx, void *ptr, size_t size, loff_t offset=0, loff_t buf_offset=0);
                /**
                * @brief API to perform PUT operation using cuObject
                * @param ctx pointer to a user control context, used in PUT callback
                * @param ptr pointer to a user memory
                * @param size size of the PUT operation
                * @param offset currently set to 0. reserved for future use
                * @param buf_offset currently set to 0. reserved for future use
                * @return data returned by the cuObjServer or a negative error code.
                */

		ssize_t cuObjPut(void *ctx, void *ptr, size_t size, loff_t offset=0, loff_t buf_offset=0);
                /**
                * @brief check if the client is connected
                */
		bool isConnected(void);
                /**
                * @brief setup telemetry output stream. must call shutdownTelemetry() before closing the os
                * @warning the os must be valid until all the cuObjClient objects are destroyed
                */
                static void setupTelemetry(bool use_OTEL, std::ostream *os);

                /**
                * @brief shutdown telemetry. reset the telemetry to default output stream
                * @note the telemetry will be closed when all the cuObjClient objects are destroyed
                */
                static void shutdownTelemetry();

                /**
                * @brief setup telemetry stream logging level
                */
                static void setTelemFlags(unsigned flags);

                 /**
                  * @brief Get the memory type of a given pointer
                  * @param ptr pointer to the memory
                  * @return memory type
                  */
                 static cuObjMemoryType_t getMemoryType(const void* ptr);

        private:
		bool cuObjRegisterKey();
        	CUfileHandle_t _cufh;
		CUfileFSOps _objectFsOps;
		CUObjOps_t _userOps;  // Store user-provided ops
		bool _connected;
		cuObjProto_t _proto;
		static std::mutex _telemMutex;
		static std::ostream *_os;
		static int _telemRefCnt;
		static unsigned _debugFlags;
		static bool _useOTEL;

		// Client wrapper functions that cufile will call
		static ssize_t cuObjClientRead(const void *handle, char *ptr, size_t size, loff_t offset, const cufileRDMAInfo_t* rdmaInfo);
		static ssize_t cuObjClientWrite(const void *handle, const char *ptr, size_t size, loff_t offset, const cufileRDMAInfo_t* rdmaInfo);
};

#endif
