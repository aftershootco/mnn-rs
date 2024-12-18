#include "backend_c.h"
#include <MNN/MNNForwardType.h>

MNNBackendConfig *mnnbc_create() {
  return reinterpret_cast<MNNBackendConfig *>(new MNN::BackendConfig());
}

MNNBackendConfig *mnnbc_clone(const MNNBackendConfig *config) {
  return reinterpret_cast<MNNBackendConfig *>(new MNN::BackendConfig(
      *reinterpret_cast<const MNN::BackendConfig *>(config)));
}

void mnnbc_destroy(MNNBackendConfig *config) {
  delete reinterpret_cast<MNN::BackendConfig *>(config);
}

void mnnbc_set_memory_mode(MNNBackendConfig *config, MemoryMode memory_mode) {
  reinterpret_cast<MNN::BackendConfig *>(config)->memory =
      static_cast<MNN::BackendConfig::MemoryMode>(memory_mode);
}
void mnnbc_set_power_mode(MNNBackendConfig *config, PowerMode power_mode) {
  reinterpret_cast<MNN::BackendConfig *>(config)->power =
      static_cast<MNN::BackendConfig::PowerMode>(power_mode);
}
void mnnbc_set_precision_mode(MNNBackendConfig *config,
                              PrecisionMode precision_mode) {
  reinterpret_cast<MNN::BackendConfig *>(config)->precision =
      static_cast<MNN::BackendConfig::PrecisionMode>(precision_mode);
}
void mnnbc_set_shared_context(MNNBackendConfig *config, void *shared_context) {
  reinterpret_cast<MNN::BackendConfig *>(config)->sharedContext =
      shared_context;
}
void mnnbc_set_flags(MNNBackendConfig *config, size_t flags) {
  reinterpret_cast<MNN::BackendConfig *>(config)->flags = flags;
}
void mnnbc_reset(MNNBackendConfig *config) {
  reinterpret_cast<MNN::BackendConfig *>(config)->memory =
      MNN::BackendConfig::Memory_Normal;
  reinterpret_cast<MNN::BackendConfig *>(config)->power =
      MNN::BackendConfig::Power_Normal;
  reinterpret_cast<MNN::BackendConfig *>(config)->precision =
      MNN::BackendConfig::Precision_Normal;
  reinterpret_cast<MNN::BackendConfig *>(config)->sharedContext = nullptr;
}

MemoryMode mnnbc_get_memory_mode(MNNBackendConfig *config) {
  return static_cast<MemoryMode>(
      reinterpret_cast<MNN::BackendConfig *>(config)->memory);
}
PowerMode mnnbc_get_power_mode(MNNBackendConfig *config) {
  return static_cast<PowerMode>(
      reinterpret_cast<MNN::BackendConfig *>(config)->power);
}
PrecisionMode mnnbc_get_precision_mode(MNNBackendConfig *config) {
  return static_cast<PrecisionMode>(
      reinterpret_cast<MNN::BackendConfig *>(config)->precision);
}
