#ifndef BACKEND_C_H
#define BACKEND_C_H
#include <MNN/MNNForwardType.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { Memory_Normal = 0, Memory_High, Memory_Low } MemoryMode;
typedef enum { Power_Normal = 0, Power_High, Power_Low } PowerMode;
typedef enum {
  Precision_Normal = 0,
  Precision_High,
  Precision_Low,
  Precision_Low_BF16
} PrecisionMode;
typedef struct MNNBackendConfig MNNBackendConfig;
// struct BackendConfig {
//   MemoryMode memory;       // = Memory_Normal;
//   PowerMode power;         // = Power_Normal;
//   PrecisionMode precision; // = Precision_Normal;
//   /** user defined context */
//   union {
//     void *sharedContext; // = nullptr;
//     size_t flags;        // Valid for CPU Backend
//   };
// };

MNNBackendConfig *mnnbc_create();
MNNBackendConfig *mnnbc_clone(const MNNBackendConfig *config);
void mnnbc_destroy(MNNBackendConfig *config);
void mnnbc_set_memory_mode(MNNBackendConfig *config, MemoryMode memory_mode);
void mnnbc_set_power_mode(MNNBackendConfig *config, PowerMode power_mode);
void mnnbc_set_precision_mode(MNNBackendConfig *config,
                              PrecisionMode precision_mode);
void mnnbc_set_shared_context(MNNBackendConfig *config, void *shared_context);
void mnnbc_set_flags(MNNBackendConfig *config, size_t flags);
void mnnbc_reset(MNNBackendConfig *config);

#ifdef __cplusplus
}
#endif
#endif // BACKEND_C_H
