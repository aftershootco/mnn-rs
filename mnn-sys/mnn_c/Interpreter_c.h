#ifndef INTERPRETER_C_H
#define INTERPRETER_C_H
#include "backend_c.h"
#include "schedule_c.h"
#include "error_code_c.h"
#include "tensor_c.h"
#include "utils.h"
#include <MNN/HalideRuntime.h>
#include <MNN/MNNForwardType.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct Interpreter Interpreter;
typedef struct Session Session;
typedef struct Backend Backend;

/** acquire runtime status by Runtime::getCurrentStatus with following keys,
 */
enum RuntimeStatus {
  /**
   * get status whether this runtime support 16-bits float point arithmetic
   */
  STATUS_SUPPORT_FP16,
  /**
   * get status whether this runtime support dot-product arithmetic
   */
  STATUS_SUPPORT_DOT_PRODUCT,
  /**
   * get status whether this runtime support power-low (means low priority for
   * opencl)
   */
  STATUS_SUPPORT_POWER_LOW,
  /**
   * emum total number
   */
  STATUS_COUNT
};

// typedef struct {
//   char **saveTensors;
//   size_t saveTensorsSize;
//   MNNForwardType type;
//   union {
//     int numThread;
//     int mode;
//   };
//   struct {
//     char **inputs;
//     size_t inputsSize;
//     char **outputs;
//     size_t outputsSize;
//     int mode;
//   } path;
//   MNNForwardType backupType;
//   MNNBackendConfig *backendConfig;
// } ScheduleConfig;

typedef struct {
  const char *name;
  const char *type;
  float flops;
} OperatorInfo;
typedef int (*TensorCallBack)(const Tensor **tensors, size_t tensorCount,
                              const char *opName);
typedef int (*TensorCallBackWithInfo)(const Tensor **tensors,
                                      size_t tensorCount,
                                      const OperatorInfo *opInfo);
#if 0
typedef struct {
  std::map<MNNForwardType, std::shared_ptr<Runtime>> *runtimeMap;
  std::shared_ptr<Runtime> *defaultRuntime;
} RuntimeInfo;
#endif

void modelPrintIO(const char *model);

/**
 * @brief get mnn version info.
 * @return mnn version string.
 */
const char *getVersion();
/**
 * @brief create net from file.
 * @param file  given file.
 * @return created net if success, NULL otherwise.
 */
Interpreter *Interpreter_createFromFile(const char *file);
/**
 * @brief create net from buffer.
 * @param buffer    given data buffer.
 * @param size      size of data buffer.
 * @return created net if success, NULL otherwise.
 */
Interpreter *Interpreter_createFromBuffer(const void *buffer, size_t size);
void Interpreter_destroy(Interpreter *interpreter);
typedef enum {
  /** About CallBack, Default Session_Debug*/
  /** runSessionWithCallBack is allowed and can get internal op info*/
  Session_Debug = 0,
  /** runSessionWithCallBack is not valid and can't get any info of op in
     session*/
  Session_Release = 1,

  /** About input tenosr, Default Session_Input_Inside*/
  /** The input tensor is alloced by session, input data after session resized*/
  Session_Input_Inside = 2,
  /** The input tensor is alloced by user, set input data before session
     resize*/
  Session_Input_User = 3,

  /** The output tensor depends on session, and can't be separate used*/
  Session_Output_Inside = 4,
  /** The output tensor can be separated from session*/
  Session_Output_User = 5,

  /** Try Resize Session when create Session or not, default direct: */
  Session_Resize_Direct = 6,
  Session_Resize_Defer = 7,

  /** Determine the Execution's forward type is determine by user or auto
     determine */
  Session_Backend_Fix =
      8, // Use the backend user set, when not support use default backend
  Session_Backend_Auto = 9, // Auto Determine the Op type by MNN

  /** Determine static memory whether recyle in resizeSession or just cache the
     memory */
  Session_Memory_Collect =
      10, // Recycle static memory when session resize in case memory explosion
  Session_Memory_Cache = 11, // Cache the static memory for next forward usage

  /** Determine whether use codegen function */
  Session_Codegen_Disable =
      12, // Disable codegen in case extra build codegen cost
  Session_Codegen_Enable = 13, // Enable codegen

  /** Dynamic Reisze Optimization */
  Session_Resize_Check = 14, // Open Trace for resize
  Session_Resize_Fix = 15,   // Apply Resize Optimization
} SessionMode;
void Interpreter_setSessionMode(Interpreter *interpreter, SessionMode mode);
void Interpreter_setCacheFile(Interpreter *interpreter, const char *cacheFile,
                              size_t keySize);
void Interpreter_setExternalFile(Interpreter *interpreter, const char *file,
                                 size_t flag);
ErrorCode Interpreter_updateCacheFile(Interpreter *interpreter,
                                      Session *session, int flag);
void Interpreter_setSessionHint(Interpreter *interpreter, int mode, int value);
// RuntimeInfo *Interpreter_createRuntime(const ScheduleConfig *configs,
//                                        size_t configSize);
Session *Interpreter_createSession(Interpreter *interpreter,
                                   const MNNScheduleConfig *config);
// Session *Interpreter_createSessionWithRuntime(Interpreter *interpreter,
//                                               const ScheduleConfig *config,
//                                               const RuntimeInfo *runtime);
Session *Interpreter_createMultiPathSession(Interpreter *interpreter,
                                            const MNNScheduleConfig *configs,
                                            size_t configSize);
// Session *Interpreter_createMultiPathSessionWithRuntime(
//     Interpreter *interpreter, const ScheduleConfig *configs, size_t
//     configSize, const RuntimeInfo *runtime);
int Interpreter_releaseSession(Interpreter *interpreter, Session *session);
void Interpreter_resizeSession(Interpreter *interpreter, Session *session);
void Interpreter_resizeSessionWithFlag(Interpreter *interpreter,
                                       Session *session, int needRelloc);
void Interpreter_releaseModel(Interpreter *interpreter);
// std::pair<const void *, size_t>
// Interpreter_getModelBuffer(const Interpreter *interpreter);
const char *Interpreter_getModelVersion(const Interpreter *interpreter);
ErrorCode Interpreter_updateSessionToModel(Interpreter *interpreter,
                                           Session *session);
ErrorCode Interpreter_runSession(const Interpreter *interpreter,
                                 Session *session);
ErrorCode Interpreter_runSessionWithCallBack(const Interpreter *interpreter,
                                             const Session *session,
                                             TensorCallBack before,
                                             TensorCallBack end, int sync);
ErrorCode Interpreter_runSessionWithCallBackInfo(const Interpreter *interpreter,
                                                 const Session *session,
                                                 TensorCallBackWithInfo before,
                                                 TensorCallBackWithInfo end,
                                                 int sync);
Tensor *Interpreter_getSessionInput(Interpreter *interpreter,
                                    const Session *session, const char *name);
Tensor *Interpreter_getSessionOutput(Interpreter *interpreter,
                                     const Session *session, const char *name);
int Interpreter_getSessionInfo(Interpreter *interpreter, const Session *session,
                               int code, void *ptr);
TensorInfoArray const * Interpreter_getSessionOutputAll(const Interpreter *interpreter,
                                                const Session *session);

TensorInfoArray const * Interpreter_getSessionInputAll(const Interpreter *interpreter,
                                               const Session *session);
void Interpreter_resizeTensor(Interpreter *interpreter, Tensor *tensor,
                              const int *dims, size_t dimsSize);
void Interpreter_resizeTensorByNCHW(Interpreter *interpreter, Tensor *tensor,
                                    int batch, int channel, int height,
                                    int width);
const Backend *Interpreter_getBackend(const Interpreter *interpreter,
                                      const Session *session,
                                      const Tensor *tensor);
const char *Interpreter_bizCode(const Interpreter *interpreter);
const char *Interpreter_uuid(const Interpreter *interpreter);
#ifdef __cplusplus
}
#endif
#endif // INTERPRETER_C_H
