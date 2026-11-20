/* Link into the test executable only, with --wrap=malloc,--wrap=calloc. */
#include <stddef.h>
static int fail_at, seen;
void ng_test_alloc_fail(int n) { fail_at=n;seen=0; }
void *__real_malloc(size_t);
void *__real_calloc(size_t,size_t);
void *__wrap_malloc(size_t n) {
    if(fail_at && ++seen==fail_at)return NULL;
    return __real_malloc(n);
}
void *__wrap_calloc(size_t n,size_t size) {
    if(fail_at && ++seen==fail_at)return NULL;
    return __real_calloc(n,size);
}
