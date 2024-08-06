#include "llm_c.h"
#include <sstream>

LLM LLM_create(const char *config_path) {
  LLM llm;
  llm.llm = MNN::Transformer::Llm::createLLM(config_path);
  return llm;
}

void LLM_destroy(LLM *self) { delete self->llm; }

void LLM_chat(LLM self) { self.llm->chat(); }
void LLM_reset(LLM self) { self.llm->reset(); }
void LLM_trace(LLM self, int start) { self.llm->trace(start); }
void LLM_load(LLM self) { self.llm->load(); }

LLMString LLM_generate(LLM self, const int *ids, size_t size) {
  auto s = std::ostringstream();
  LLMString str;
  str.cppstr = self.llm->generate(std::vector<int>(ids, ids + size) , stdout );
  return str;
}
