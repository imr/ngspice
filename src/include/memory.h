#ifndef _MEMORY_H
#define _MEMORY_H

extern void *tmalloc(size_t num);
extern void *trealloc(void *str, size_t num);
extern void txfree(void *ptr);

#endif
