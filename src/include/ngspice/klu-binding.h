#ifndef _KLU_BINDING_H
#define _KLU_BINDING_H

#define CREATE_KLU_BINDING_TABLE(ptr, binding, a, b)                    \
    if ((here->a != 0) && (here->b != 0)) {                             \
        i = here->ptr ;                                                 \
        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ; \
        here->binding = matched ;                                       \
        here->ptr = matched->CSC ;                                      \
    }

#define CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(ptr, binding, a, b)        \
    if ((here->a != 0) && (here->b != 0))                               \
        here->ptr = here->binding->CSC_Complex ;

#define CONVERT_KLU_BINDING_TABLE_TO_REAL(ptr, binding, a, b)   \
    if ((here->a != 0) && (here->b != 0))                       \
        here->ptr = here->binding->CSC ;

#endif
