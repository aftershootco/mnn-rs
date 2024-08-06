#include "interpreter_c.h"
#include "MNN/Interpreter.hpp"
#include <MNN/MNNForwardType.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
extern "C" {

void modelPrintIO(const char *model) {
  auto net = MNN::Interpreter::createFromFile(model);
  MNN::ScheduleConfig config;
  config.numThread = 4;
  config.type = MNN_FORWARD_METAL;
  MNN::Session *session = net->createSession(config);
  auto inputs = net->getSessionInputAll(session);
  for (auto input : inputs) {
    std::cout << "Input: " << input.first << " ";
    input.second->printShape();
  }
  auto outputs = net->getSessionOutputAll(session);
  for (auto output : outputs) {
    std::cout << "Output: " << output.first << " ";
    output.second->printShape();
  }
}

const char *getVersion() { return MNN::getVersion(); }
Interpreter *Interpreter_createFromFile(const char *file) {
  return reinterpret_cast<Interpreter *>(
      MNN::Interpreter::createFromFile(file));
}
Interpreter *Interpreter_createFromBuffer(const void *buffer, size_t size) {
  return reinterpret_cast<Interpreter *>(
      MNN::Interpreter::createFromBuffer(buffer, size));
}
void Interpreter_destroy(Interpreter *interpreter) {
  auto mnn_interpreter = reinterpret_cast<MNN::Interpreter *>(interpreter);
  MNN::Interpreter::destroy(mnn_interpreter);
}
void Interpreter_setSessionMode(Interpreter *interpreter, SessionMode mode) {
  auto mnn_interpreter = reinterpret_cast<MNN::Interpreter *>(interpreter);
  mnn_interpreter->setSessionMode(
      static_cast<MNN::Interpreter::SessionMode>(mode));
}
void Interpreter_setCacheFile(Interpreter *interpreter, const char *cacheFile,
                              size_t keySize) {
  auto mnn_interpreter = reinterpret_cast<MNN::Interpreter *>(interpreter);
  mnn_interpreter->setCacheFile(cacheFile, keySize);
}
void Interpreter_setExternalFile(Interpreter *interpreter, const char *file,
                                 size_t flag) {
  auto mnn_interpreter = reinterpret_cast<MNN::Interpreter *>(interpreter);
  mnn_interpreter->setExternalFile(file, flag);
}
ErrorCode Interpreter_updateCacheFile(Interpreter *interpreter,
                                      Session *session, int flag) {
  auto mnn_interpreter = reinterpret_cast<MNN::Interpreter *>(interpreter);
  auto mnn_session = reinterpret_cast<MNN::Session *>(session);
  return static_cast<ErrorCode>(
      mnn_interpreter->updateCacheFile(mnn_session, flag));
}
void Interpreter_setSessionHint(Interpreter *interpreter, int mode, int value) {
  auto mnn_interpreter = reinterpret_cast<MNN::Interpreter *>(interpreter);
  mnn_interpreter->setSessionHint(static_cast<MNN::Interpreter::HintMode>(mode),
                                  value);
}
// RuntimeInfo* Interpreter_createRuntime(const ScheduleConfig* configs, size_t
// configSize) {
//     std::vector<MNN::ScheduleConfig> cppConfigs(configSize);
//     for (size_t i = 0; i < configSize; ++i) {
//         cppConfigs[i].saveTensors.assign(configs[i].saveTensors,
//         configs[i].saveTensors + configs[i].saveTensorsSize);
//         cppConfigs[i].type = configs[i].type;
//         cppConfigs[i].numThread = configs[i].numThread;
//         cppConfigs[i].path.inputs.assign(configs[i].path.inputs,
//         configs[i].path.inputs + configs[i].path.inputsSize);
//         cppConfigs[i].path.outputs.assign(configs[i].path.outputs,
//         configs[i].path.outputs + configs[i].path.outputsSize);
//         cppConfigs[i].path.mode =
//         static_cast<MNN::ScheduleConfig::Path::Mode>(configs[i].path.mode);
//         cppConfigs[i].backupType = configs[i].backupType;
//         cppConfigs[i].backendConfig = configs[i].backendConfig;
//     }
//     auto runtimeInfo = MNN::Interpreter::createRuntime(cppConfigs);
//     return new RuntimeInfo{new std::map<MNNForwardType,
//     std::shared_ptr<MNN::Runtime>>(runtimeInfo.first), new
//     std::shared_ptr<MNN::Runtime>(runtimeInfo.second)};
// }
Session *Interpreter_createSession(Interpreter *interpreter,
                                   const MNNScheduleConfig *config) {
  auto mnn_interpreter = reinterpret_cast<MNN::Interpreter *>(interpreter);
  auto mnn_schedule_config =
      reinterpret_cast<const MNN::ScheduleConfig *>(config);

  return reinterpret_cast<Session *>(
      mnn_interpreter->createSession(*mnn_schedule_config));
  ;
}
// Session* Interpreter_createSessionWithRuntime(Interpreter* interpreter, const
// ScheduleConfig* config, const RuntimeInfo* runtime) {
//     MNN::ScheduleConfig cppConfig;
//     cppConfig.saveTensors.assign(config->saveTensors, config->saveTensors +
//     config->saveTensorsSize); cppConfig.type = config->type;
//     cppConfig.numThread = config->numThread;
//     cppConfig.path.inputs.assign(config->path.inputs, config->path.inputs +
//     config->path.inputsSize);
//     cppConfig.path.outputs.assign(config->path.outputs, config->path.outputs
//     + config->path.outputsSize); cppConfig.path.mode =
//     static_cast<MNN::ScheduleConfig::Path::Mode>(config->path.mode);
//     cppConfig.backupType = config->backupType;
//     cppConfig.backendConfig = config->backendConfig;
//     return interpreter->createSession(cppConfig, *runtime);
// }
Session *Interpreter_createMultiPathSession(Interpreter *interpreter,
                                            const MNNScheduleConfig *configs,
                                            size_t configSize) {

  auto mnn_configs = reinterpret_cast<const MNN::ScheduleConfig *>(configs);
  // @todo: check if this is correct
  std::vector<MNN::ScheduleConfig> cppConfigs(mnn_configs,
                                              mnn_configs + configSize);
  auto mnn_interpreter = reinterpret_cast<MNN::Interpreter *>(interpreter);
  return reinterpret_cast<Session *>(
      mnn_interpreter->createMultiPathSession(cppConfigs));
}

// Session* Interpreter_createMultiPathSessionWithRuntime(Interpreter*
// interpreter, const ScheduleConfig* configs, size_t configSize, const
// RuntimeInfo* runtime) {
// }
int Interpreter_releaseSession(Interpreter *interpreter, Session *session) {
  auto mnn_interpreter = reinterpret_cast<MNN::Interpreter *>(interpreter);
  auto mnn_session = reinterpret_cast<MNN::Session *>(session);
  return mnn_interpreter->releaseSession(mnn_session);
}
void Interpreter_resizeSession(Interpreter *interpreter, Session *session) {
  auto mnn_interpreter = reinterpret_cast<MNN::Interpreter *>(interpreter);
  auto mnn_session = reinterpret_cast<MNN::Session *>(session);
  mnn_interpreter->resizeSession(mnn_session);
}
void Interpreter_resizeSessionWithFlag(Interpreter *interpreter,
                                       Session *session, int needRelloc) {
  auto mnn_interpreter = reinterpret_cast<MNN::Interpreter *>(interpreter);
  auto mnn_session = reinterpret_cast<MNN::Session *>(session);
  mnn_interpreter->resizeSession(mnn_session, needRelloc);
}
void Interpreter_releaseModel(Interpreter *interpreter) {
  auto mnn_interpreter = reinterpret_cast<MNN::Interpreter *>(interpreter);
  mnn_interpreter->releaseModel();
}
// std::pair<const void*, size_t> Interpreter_getModelBuffer(const Interpreter*
// interpreter) {
//     auto mnn_interpreter = reinterpret_cast<MNN::Interpreter
//     const*>(interpreter); return mnn_interpreter->getModelBuffer();
// }
const char *Interpreter_getModelVersion(const Interpreter *interpreter) {
  auto mnn_interpreter =
      reinterpret_cast<MNN::Interpreter const *>(interpreter);
  return mnn_interpreter->getModelVersion();
}
ErrorCode Interpreter_updateSessionToModel(Interpreter *interpreter,
                                           Session *session) {
  auto mnn_interpreter = reinterpret_cast<MNN::Interpreter *>(interpreter);
  auto mnn_session = reinterpret_cast<MNN::Session *>(session);
  return static_cast<ErrorCode>(
      mnn_interpreter->updateSessionToModel(mnn_session));
}
ErrorCode Interpreter_runSession(const Interpreter *interpreter,
                                 Session *session) {
  auto mnn_interpreter =
      reinterpret_cast<MNN::Interpreter const *>(interpreter);
  auto mnn_session = reinterpret_cast<MNN::Session *>(session);
  return static_cast<ErrorCode>(mnn_interpreter->runSession(mnn_session));
}
// ErrorCode Interpreter_runSessionWithCallBack(const Interpreter* interpreter,
// const Session* session, TensorCallBack before, TensorCallBack end, int sync)
// {
//     auto beforeCpp = [before](const std::vector<MNN::Tensor*>& tensors, const
//     std::string& opName) {
//         std::vector<const Tensor*> cTensors(tensors.begin(), tensors.end());
//         return before(cTensors.data(), cTensors.size(), opName.c_str());
//     };
//     auto endCpp = [end](const std::vector<MNN::Tensor*>& tensors, const
//     std::string& opName) {
//         std::vector<const Tensor*> cTensors(tensors.begin(), tensors.end());
//         return end(cTensors.data(), cTensors.size(), opName.c_str());
//     };
//     return interpreter->runSessionWithCallBack(session, beforeCpp, endCpp,
//     sync);
// }
// ErrorCode Interpreter_runSessionWithCallBackInfo(const Interpreter*
// interpreter, const Session* session, TensorCallBackWithInfo before,
// TensorCallBackWithInfo end, int sync) {
//     auto beforeCpp = [before](const std::vector<MNN::Tensor*>& tensors, const
//     MNN::OperatorInfo* opInfo) {
//         std::vector<const Tensor*> cTensors(tensors.begin(), tensors.end());
//         OperatorInfo cOpInfo{opInfo->name().c_str(), opInfo->type().c_str(),
//         opInfo->flops()}; return before(cTensors.data(), cTensors.size(),
//         &cOpInfo);
//     };
//     auto endCpp = [end](const std::vector<MNN::Tensor*>& tensors, const
//     MNN::OperatorInfo* opInfo) {
//         std::vector<const Tensor*> cTensors(tensors.begin(), tensors.end());
//         OperatorInfo cOpInfo{opInfo->name().c_str(), opInfo->type().c_str(),
//         opInfo->flops()}; return end(cTensors.data(), cTensors.size(),
//         &cOpInfo);
//     };
//     return interpreter->runSessionWithCallBackInfo(session, beforeCpp,
//     endCpp, sync);
// }
Tensor *Interpreter_getSessionInput(Interpreter *interpreter,
                                    const Session *session, const char *name) {
  auto mnn_interpreter = reinterpret_cast<MNN::Interpreter *>(interpreter);
  auto mnn_session = reinterpret_cast<MNN::Session const *>(session);
  return reinterpret_cast<Tensor *>(
      mnn_interpreter->getSessionInput(mnn_session, name));
}
Tensor *Interpreter_getSessionOutput(Interpreter *interpreter,
                                     const Session *session, const char *name) {
  auto mnn_interpreter = reinterpret_cast<MNN::Interpreter *>(interpreter);
  auto mnn_session = reinterpret_cast<MNN::Session const *>(session);
  return reinterpret_cast<Tensor *>(
      mnn_interpreter->getSessionOutput(mnn_session, name));
}
int Interpreter_getSessionInfo(Interpreter *interpreter, const Session *session,
                               int code, void *ptr) {
  auto mnn_interpreter = reinterpret_cast<MNN::Interpreter *>(interpreter);
  auto mnn_session = reinterpret_cast<const MNN::Session *>(session);
  return mnn_interpreter->getSessionInfo(
      mnn_session, static_cast<MNN::Interpreter::SessionInfoCode>(code), ptr);
}
TensorInfoArray const *
Interpreter_getSessionOutputAll(const Interpreter *interpreter,
                                const Session *session) {
  auto mnn_interpreter =
      reinterpret_cast<MNN::Interpreter const *>(interpreter);
  auto mnn_session = reinterpret_cast<MNN::Session const *>(session);
  auto outputMap = mnn_interpreter->getSessionOutputAll(mnn_session);
  auto out = createTensorInfoArray(outputMap.size());
  size_t index = 0;
  for (const auto &entry : outputMap) {
    out->tensors[index].name =
        createCString(entry.first.c_str(), entry.first.size());
    out->tensors[index].tensor = static_cast<void *>(entry.second);
    ++index;
  }
  return out;
}
TensorInfoArray const *
Interpreter_getSessionInputAll(const Interpreter *interpreter,
                               const Session *session) {
  auto mnn_interpreter =
      reinterpret_cast<MNN::Interpreter const *>(interpreter);
  auto mnn_session = reinterpret_cast<MNN::Session const *>(session);
  auto inputMap = mnn_interpreter->getSessionInputAll(mnn_session);
  auto in = createTensorInfoArray(inputMap.size());
  size_t index = 0;
  for (const auto &entry : inputMap) {
    in->tensors[index].name =
        createCString(entry.first.c_str(), entry.first.size());
    in->tensors[index].tensor = static_cast<void *>(entry.second);
    ++index;
  }
  return in;
}
void Interpreter_resizeTensor(Interpreter *interpreter, Tensor *tensor,
                              const int *dims, size_t dimsSize) {
  std::vector<int> cppDims(dims, dims + dimsSize);
  auto mnn_interpreter = reinterpret_cast<MNN::Interpreter *>(interpreter);
  auto mnn_tensor = reinterpret_cast<MNN::Tensor *>(tensor);
  mnn_interpreter->resizeTensor(mnn_tensor, cppDims);
}
void Interpreter_resizeTensorByNCHW(Interpreter *interpreter, Tensor *tensor,
                                    int batch, int channel, int height,
                                    int width) {
  auto mnn_interpreter = reinterpret_cast<MNN::Interpreter *>(interpreter);
  auto mnn_tensor = reinterpret_cast<MNN::Tensor *>(tensor);
  mnn_interpreter->resizeTensor(mnn_tensor, batch, channel, height, width);
}
const Backend *Interpreter_getBackend(const Interpreter *interpreter,
                                      const Session *session,
                                      const Tensor *tensor) {
  auto mnn_interpreter =
      reinterpret_cast<MNN::Interpreter const *>(interpreter);
  auto mnn_session = reinterpret_cast<MNN::Session const *>(session);
  auto mnn_tensor = reinterpret_cast<MNN::Tensor const *>(tensor);
  return reinterpret_cast<const Backend *>(
      mnn_interpreter->getBackend(mnn_session, mnn_tensor));
}
const char *Interpreter_bizCode(const Interpreter *interpreter) {
  auto mnn_interpreter =
      reinterpret_cast<MNN::Interpreter const *>(interpreter);
  return mnn_interpreter->bizCode();
}
const char *Interpreter_uuid(const Interpreter *interpreter) {
  auto mnn_interpreter =
      reinterpret_cast<MNN::Interpreter const *>(interpreter);
  return mnn_interpreter->uuid();
}
void Session_destroy(Session *session) {
  auto mnn_session = reinterpret_cast<MNN::Session *>(session);
  delete mnn_session;
}
} // extern "C"
