#ifndef LLM_C_H
#define LLM_C_H
#include "utils.h"

#ifdef __cplusplus
#include <MNN/llm/llm.hpp>
extern "C" {
#endif

void *create_ostream(int out);

typedef struct {
#ifdef __cplusplus
  std::string *cppstr;
#else
  void *cppstr;
#endif
} LLMString;
const char *LLMString_as_str(LLMString *);

typedef struct {
#ifdef __cplusplus
  MNN::Transformer::Llm *llm;
#else
  void *llm;
#endif
} LLM;
LLM LLM_create(const char *config_path);
void LLM_destroy(LLM *self);

void LLM_chat(LLM self);
void LLM_reset(LLM self);
void LLM_trace(LLM self, int start);
void LLM_load(LLM self);
// LLMString LLM_generate(LLM self, const int *ids, size_t size);
LLMString LLM_generate(LLM self, const int *ids, size_t size, void *out,
                       const char *end_with);
LLMString LLM_response(LLM self, const char *user_content);

typedef struct {
#ifdef __cplusplus
  MNN::Transformer::Embedding *embedding;
#else
  void *embedding;
#endif
} Embedding;
Embedding Embedding_create(const char *config_path);
void embedding_destroy(Embedding *embedding);
void embedding_load(Embedding *embedding);
float embedding_dist(Embedding *embedding, const char *txt1, const char *txt2);

#ifdef __cplusplus
}
#endif

#endif // LLM_C_H
