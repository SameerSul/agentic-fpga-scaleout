#ifndef SDBOOT_STRING_H
#define SDBOOT_STRING_H
#include <stddef.h>
void *memset(void *d, int c, size_t n);
void *memcpy(void *d, const void *s, size_t n);
int strcmp(const char *a, const char *b);
#endif
