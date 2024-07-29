#include "Schedule_c.h"
#include <MNN/Interpreter.hpp>
#include <MNN/MNNForwardType.h>

MNNScheduleConfig *mnnsc_create() {
  auto mnnsc = new MNN::ScheduleConfig();
  mnnsc->saveTensors = std::vector<std::string>();
  return reinterpret_cast<MNNScheduleConfig *>(mnnsc);
}

void mnnsc_destroy(MNNScheduleConfig *config) {
  auto mnn_config = reinterpret_cast<MNN::ScheduleConfig *>(config);
  delete mnn_config;
}

void mnnsc_set_save_tensors(MNNScheduleConfig *config,
                            const char *const *saveTensors,
                            size_t saveTensorsSize) {
  auto mnn_config = reinterpret_cast<MNN::ScheduleConfig *>(config);
  auto mnn_saveTensors =
      std::vector<std::string>(saveTensors, saveTensors + saveTensorsSize);
  mnn_config->saveTensors = std::move(mnn_saveTensors);
}

void mnnsc_set_type(MNNScheduleConfig *config, MNNForwardType type) {
  auto mnn_config = reinterpret_cast<MNN::ScheduleConfig *>(config);
  mnn_config->type = type;
}

void mnnsc_set_num_threads(MNNScheduleConfig *config, int numThread) {
  auto mnn_config = reinterpret_cast<MNN::ScheduleConfig *>(config);
  mnn_config->numThread = numThread;
}

void mnnsc_set_mode(MNNScheduleConfig *config, int mode) {
  auto mnn_config = reinterpret_cast<MNN::ScheduleConfig *>(config);
  mnn_config->mode = mode;
}

void mnnsc_set_backup_type(MNNScheduleConfig *config,
                           MNNForwardType backupType) {
  auto mnn_config = reinterpret_cast<MNN::ScheduleConfig *>(config);
  mnn_config->backupType = backupType;
}
void mnnsc_set_backend_config(MNNScheduleConfig *config,
                              MNNBackendConfig *backendConfig) {
  auto mnn_config = reinterpret_cast<MNN::ScheduleConfig *>(config);
  mnn_config->backendConfig =
      reinterpret_cast<MNN::BackendConfig *>(backendConfig);
}
