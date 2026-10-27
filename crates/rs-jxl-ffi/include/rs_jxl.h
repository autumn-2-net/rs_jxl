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
typedef struct RsJxlContext RsJxlContext;

RS_JXL_API uint32_t rs_jxl_abi_version(void);
RS_JXL_API size_t rs_jxl_request_size(void);
/* Nonnull request must expose at least the version/size prefix. If those match,
   the entire struct and each nonempty buffer must be readable, immutable and
   allocated for its advertised length throughout the call. Null buffers are
   allowed only for length zero. Inputs are not retained. Arbitrary dangling
   pointers cannot be validated. Panics are caught except abort/OOM failures. */
RS_JXL_API RsJxlResult *rs_jxl_execute(const RsJxlRequest *request);
/* Optional additive v1 symbols; callers may capability-detect their presence.
   Constructor JSON is {"kind":"encoder"|"decoder","threads":N}; threads
   defaults to 1 and must be 1..256. No operation/options fields or data payload.
   request follows rs_jxl_execute's storage contract. out_context must be writable
   for one pointer (unaligned is allowed), not overlapping request/input storage.
   It is set to NULL before validation and stays NULL on failure. A NULL output
   pointer produces an error. Free an old context before reusing its output slot.
   Always free the returned result; status 0 transfers a new context to caller. */
RS_JXL_API RsJxlResult *rs_jxl_context_create(const RsJxlRequest *request,
                                           RsJxlContext **out_context);
/* context must be NULL or live. All execute/free access to one context must be
   exclusive; caller supplies synchronization. Independent contexts may run in
   parallel. Request buffers follow rs_jxl_execute and must not alias context.
   Omitted options.threads inherits context threads; explicit mismatch is an error.
   All other options have per-request defaults; no decoder defaults are retained.
   Encoder supports encode/encode_multiframe/transcode_jpeg and stateless
   analyze_multiframe. Decoder supports decode/decode_to_png/reconstruct_jpeg and
   stateless info/read_image (no parser/worker reuse for those two operations).
   A kind mismatch is an error. No input buffers are retained. Ordinary errors
   permit reuse; status 2 poisons the context, which must then only be freed.
   Results own their buffers independently of context lifetime. */
RS_JXL_API RsJxlResult *rs_jxl_context_execute(RsJxlContext *context,
                                            const RsJxlRequest *request);
/* NULL is a no-op; otherwise free exactly once, with no concurrent use/free. */
RS_JXL_API void rs_jxl_context_free(RsJxlContext *context);
/* Status: 0 success, 1 request/codec failure, 2 panic, -1 null result. */
RS_JXL_API int32_t rs_jxl_result_status(const RsJxlResult *result);
/* Errors preserve the v1 "error" string and add "error_kind": invalid_input,
   resource_limit, encoding, decoding, io, internal, or unsupported. Optional
   "error_stage" identifies a boundary without changing the error message.
   Buffers are immutable borrows until free; JSON is not NUL-terminated.
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
