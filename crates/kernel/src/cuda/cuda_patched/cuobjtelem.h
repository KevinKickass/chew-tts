/* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

 NVIDIA CORPORATION and its licensors retain all intellectual property
 and proprietary rights in and to this software, related documentation
 and any modifications thereto.  Any use, reproduction, disclosure or
 distribution of this software and related documentation without an express
 license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef CUOBJTELEM_H
#define CUOBJTELEM_H

#define PUBLIC_VISIBLE  __attribute__((visibility("default")))

/****************************************/
/* Telemetry for logging and statistics */
/****************************************/

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <variant>

// Logging "path" flags:
#define CUOBJ_LOG_PATH_INFO	0x0001
#define CUOBJ_LOG_PATH_DEBUG	0x0002
#define CUOBJ_LOG_PATH_ERROR	0x0004

#ifdef CUOBJ_USE_OPENTELEMETRY
#include "opentelemetry/sdk/version/version.h"

#include "opentelemetry/trace/provider.h"
#include "opentelemetry/logs/provider.h"
#include "opentelemetry/metrics/provider.h"

namespace trace_api = opentelemetry::trace;
namespace logs_api = opentelemetry::logs;
namespace metrics_api = opentelemetry::metrics;

#endif /* CUOBJ_USE_OPENTELEMETRY */

class cuObjTelem {
public:
        cuObjTelem(unsigned flags = CUOBJ_LOG_PATH_ERROR) : log_flags(flags) {};
	virtual ~cuObjTelem() {};

        virtual void logInfo(const char *fmt, ...) = 0;
        virtual void logDebug(const char *fmt, ...) = 0;
        virtual void logError(const char *fmt, ...) = 0;
        virtual void incPutCounter(int v) = 0;
        virtual void incGetCounter(int v) = 0;
        unsigned getFlags(void) { return log_flags; };
        cuObjTelem&  setFlags(unsigned flags)
            { log_flags = flags; return *this; };
private:
        unsigned log_flags;
};

#ifdef CUOBJ_USE_OPENTELEMETRY
class cuObjTelem_OTEL : public cuObjTelem {
public:
	cuObjTelem_OTEL(void);
	~cuObjTelem_OTEL() override {};

        trace_api::Scope getSpan(std::string name);
        void logInfo(const char *fmt, ...) override;
        void logDebug(const char *fmt, ...) override;
        void logError(const char *fmt, ...) override;
        void incPutCounter(int v) override;
        void incGetCounter(int v) override;

private:
        opentelemetry::nostd::shared_ptr<logs_api::Logger> logger;
        opentelemetry::nostd::shared_ptr<metrics_api::Counter<uint64_t>> put_counter;
        opentelemetry::nostd::shared_ptr<metrics_api::Counter<uint64_t>> get_counter;
        opentelemetry::nostd::shared_ptr<trace_api::Tracer> tracer;
};
#endif

class cuObjSpan {
public:
	cuObjSpan(std::string str, std::ostream &sout);
	~cuObjSpan();

private:
        std::ostream &os;      // Stream to output span
        std::string str;
        std::chrono::time_point<std::chrono::system_clock> start;
};

class cuObjTelem_ostream : public cuObjTelem {
public:
	cuObjTelem_ostream(std::ostream &sout = std::cout);
	~cuObjTelem_ostream() override;

        std::unique_ptr<cuObjSpan> getSpan(std::string name);
        void logInfo(const char *fmt, ...) override;
        void logDebug(const char *fmt, ...) override;
        void logError(const char *fmt, ...) override;
        void incPutCounter(int v) override;
        void incGetCounter(int v) override;

private:
        std::ostream &os;      // Stream to output telemetry
        struct {
            uint64_t put;
            uint64_t get;
        } counters;
};

#ifdef CUOBJ_USE_OPENTELEMETRY
std::variant<std::unique_ptr<cuObjSpan>, trace_api::Scope> getSpan(std::shared_ptr<cuObjTelem> &t, std::string name);
#else
std::unique_ptr<cuObjSpan> getSpan(std::shared_ptr<cuObjTelem> &t, std::string name);
#endif

#endif
