#ifndef ERROR_CODE_C_H
#define ERROR_CODE_C_H
#ifdef __cplusplus
extern "C" {
#endif
typedef enum {
    ERROR_CODE_NO_ERROR           = 0,
    ERROR_CODE_OUT_OF_MEMORY      = 1,
    ERROR_CODE_NOT_SUPPORT        = 2,
    ERROR_CODE_COMPUTE_SIZE_ERROR = 3,
    ERROR_CODE_NO_EXECUTION       = 4,
    ERROR_CODE_INVALID_VALUE      = 5,
    // User error
    ERROR_CODE_INPUT_DATA_ERROR = 10,
    ERROR_CODE_CALL_BACK_STOP   = 11,
    // Op Resize Error
    ERROR_CODE_TENSOR_NOT_SUPPORT = 20,
    ERROR_CODE_TENSOR_NEED_DIVIDE = 21,
} ErrorCode;
#ifdef __cplusplus
}
#endif
#endif // ERROR_CODE_C_H
