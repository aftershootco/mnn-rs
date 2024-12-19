#ifndef SCHEDULE_C_H
#define SCHEDULE_C_H
#include "backend_c.h"
#include <MNN/MNNForwardType.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MNNScheduleConfig MNNScheduleConfig;

MNNScheduleConfig *mnnsc_create();
MNNScheduleConfig *mnnsc_clone(const MNNScheduleConfig *from);
void mnnsc_destroy(MNNScheduleConfig *config);
void mnnsc_set_save_tensors(MNNScheduleConfig *config,
                            const char *const *saveTensors,
                            size_t saveTensorsSize);
void mnnsc_set_type(MNNScheduleConfig *config, MNNForwardType type);
void mnnsc_set_num_threads(MNNScheduleConfig *config, int numThread);
void mnnsc_set_mode(MNNScheduleConfig *config, int mode);
void mnnsc_set_backup_type(MNNScheduleConfig *config,
                           MNNForwardType backupType);
void mnnsc_set_backend_config(MNNScheduleConfig *config,
                              MNNBackendConfig *backendConfig);
MNNForwardType mnnsc_get_type(MNNScheduleConfig *config);
MNNForwardType mnnsc_get_backup_type(MNNScheduleConfig *config);

#ifdef __cplusplus
}
#endif
#endif // SCHEDULE_C_H
