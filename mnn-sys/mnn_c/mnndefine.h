#ifndef MNNDEFINE_H
#define MNNDEFINE_H
#include <cstddef>

enum class Level {
  Info = 0,
  Error = 1,
};

extern "C" {
void mnn_ffi_emit(const char *file, size_t line, Level level,
                  const char *message);
}

#define MNN_PRINT(format, ...)                                                 \
  {                                                                            \
    char logtmp[4096];                                                         \
    snprintf(logtmp, 4096, format, ##__VA_ARGS__);                             \
    mnn_ffi_emit(__FILE__, __LINE__, Level::Info, logtmp);                     \
  }

#define MNN_ERROR(format, ...)                                                 \
  {                                                                            \
    char logtmp[4096];                                                         \
    snprintf(logtmp, 4096, format, ##__VA_ARGS__);                             \
    mnn_ffi_emit(__FILE__, __LINE__, Level::Error, logtmp);                    \
  }

#endif
