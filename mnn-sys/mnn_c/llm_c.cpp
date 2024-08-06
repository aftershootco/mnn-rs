#include "llm_c.h"
#include <iostream>
#include <sstream>

const char *LLMString_as_str(LLMString const *str) { return str->cppstr->c_str(); }

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

LLMString LLM_generate(LLM self, const int *ids, size_t size, void *out,
                       const char *end_with) {
  LLMString str;
  auto out_stream = reinterpret_cast<std::ostream *>(out);
  auto generated = self.llm->generate(std::vector<int>(ids, ids + size),
                                      out_stream, end_with);
  str.cppstr = new std::string(generated);
  return str;
}

void *create_ostream(int out) {
  std::ostream *ret;
  switch (out) {
  case 0:
    ret = nullptr;
  case 1:
    ret = &std::cout;
  case 2:
    ret = &std::cerr;
  default:
    ret = &std::cerr;
  }
  return reinterpret_cast<void *>(ret);
}
