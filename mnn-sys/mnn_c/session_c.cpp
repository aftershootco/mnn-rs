#include <MNN/Interpreter.hpp>
#include "session_c.h"

void Session_destroy(Session *session) {
  auto mnn_session = reinterpret_cast<MNN::Session *>(session);
  // delete mnn_session;
}

int Session_hasAsyncWork(Session *session) {
    auto mnn_session = reinterpret_cast<MNN::Session *>(session);
    // return mnn_session->hasAsyncWork();
    return true;
}
