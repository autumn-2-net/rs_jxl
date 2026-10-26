#ifndef RS_JXL_H
#define RS_JXL_H
#include <stddef.h>
#include <stdint.h>

#define RS_JXL_ABI_VERSION 1u
#if defined(_WIN32) && defined(RS_JXL_IMPORT)
#define RS_JXL_API __declspec(dllimport)
#else
#define RS_JXL_API
#endif
#ifdef __cplusplus
extern "C" {
#endif

/* Native-endian C layout. JSON is UTF-8; pixel words are LITTLE-ENDIAN. */
typedef struct RsJxlRequest {
    uint32_t abi_version;
    size_t struct_size;
    const uint8_t *json;
    size_t json_len;
    const uint8_t *data;
    size_t data_len;
} RsJxlRequest;
typedef struct RsJxlResult RsJxlResult;

RS_JXL_API uint32_t rs_jxl_abi_version(void);
RS_JXL_API size_t rs_jxl_request_size(void);
/* Nonnull request must expose at least the version/size prefix. If those match,
   the entire struct and each nonempty buffer must be readable, immutable and
   allocated for its advertised length throughout the call. Null buffers are
   allowed only for length zero. Inputs are not retained. Arbitrary dangling
   pointers cannot be validated. Panics are caught except abort/OOM failures. */
RS_JXL_API RsJxlResult *rs_jxl_execute(const RsJxlRequest *request);
/* Status: 0 success, 1 request/codec failure, 2 panic, -1 null result. */
RS_JXL_API int32_t rs_jxl_result_status(const RsJxlResult *result);
/* Buffers are immutable borrows until free; JSON is not NUL-terminated.
   length may be NULL, otherwise must be writable and not alias result storage.
   Empty buffers/null results return NULL and set *length=0. Results must be
   live and may not be freed concurrently with getters or readers. */
RS_JXL_API const uint8_t *rs_jxl_result_json(const RsJxlResult *result, size_t *length);
RS_JXL_API const uint8_t *rs_jxl_result_data(const RsJxlResult *result, size_t *length);
/* NULL is a no-op. A nonnull result must be freed exactly once. */
RS_JXL_API void rs_jxl_result_free(RsJxlResult *result);

#ifdef __cplusplus
}
#endif
#endif
