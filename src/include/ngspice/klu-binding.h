#ifndef _KLU_BINDING_H
#define _KLU_BINDING_H

#define CREATE_KLU_BINDING_TABLE(ptr, binding, a, b)                     \
    if ((here->a != 0) && (here->b != 0)) {                              \
        i = here->ptr ;                                                  \
        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ; \
        here->binding = matched ;                                        \
        here->ptr = matched->CSC ;                                       \
    }

#define CREATE_KLU_BINDING_TABLE_DYNAMIC(ptr, binding, a, b)             \
    if ((here->a != 0) && (here->b != 0)) {                              \
        i = here->ptr ;                                                  \
        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ; \
        here->binding = matched ;                                        \
        here->ptr = matched->CSC_LinearDynamic ;                         \
    }

#define CREATE_KLU_BINDING_TABLE_STATIC(ptr, binding, a, b)              \
    if ((here->a != 0) && (here->b != 0)) {                              \
        i = here->ptr ;                                                  \
        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ; \
        here->binding = matched ;                                        \
        here->ptr = matched->CSC_LinearStatic ;                          \
    }

#define CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(ptr, binding, a, b)         \
    if ((here->a != 0) && (here->b != 0))                                \
        here->ptr = here->binding->CSC_Complex ;

#define CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_DYNAMIC(ptr, binding, a, b) \
    if ((here->a != 0) && (here->b != 0))                                \
        here->ptr = here->binding->CSC_Complex_LinearDynamic ;

#define CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_STATIC(ptr, binding, a, b)  \
    if ((here->a != 0) && (here->b != 0))                                \
        here->ptr = here->binding->CSC_Complex_LinearStatic ;

#define CONVERT_KLU_BINDING_TABLE_TO_REAL(ptr, binding, a, b)            \
    if ((here->a != 0) && (here->b != 0))                                \
        here->ptr = here->binding->CSC ;

#define CONVERT_KLU_BINDING_TABLE_TO_REAL_DYNAMIC(ptr, binding, a, b)    \
    if ((here->a != 0) && (here->b != 0))                                \
        here->ptr = here->binding->CSC_LinearDynamic ;

#define CONVERT_KLU_BINDING_TABLE_TO_REAL_STATIC(ptr, binding, a, b)     \
    if ((here->a != 0) && (here->b != 0))                                \
        here->ptr = here->binding->CSC_LinearStatic ;

#endif
