#ifndef SESSION_C_H
#define SESSION_C_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Session Session;
void Session_destroy(Session *session);
int Session_hasAsyncWork(Session *session);

#ifdef __cplusplus
}
#endif

#endif // SESSION_C_H
