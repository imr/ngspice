#ifndef _KLU_BINDING_H
#define _KLU_BINDING_H

#define CREATE_KLU_BINDING_TABLE(ptr, binding, a, b)                             \
    if ((here->a != 0) && (here->b != 0)) {                                      \
        i = here->ptr ;                                                          \
        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ; \
        here->binding = matched ;                                                \
        here->ptr = matched->CSC ;                                               \
    }

#define CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(ptr, binding, a, b)                 \
    if ((here->a != 0) && (here->b != 0))                                        \
        here->ptr = here->binding->CSC_Complex ;

#define CONVERT_KLU_BINDING_TABLE_TO_REAL(ptr, binding, a, b)                    \
    if ((here->a != 0) && (here->b != 0))                                        \
        here->ptr = here->binding->CSC ;

#ifdef XSPICE
#define CREATE_KLU_BINDING_TABLE_XSPICE_OUTPUTS(ptr, binding, a, b)              \
    if ((smp_data_out->a != 0) && (smp_data_out->b != 0)) {                      \
        i = smp_data_out->ptr ;                                                  \
        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ; \
        smp_data_out->binding = matched ;                                        \
        smp_data_out->ptr = matched->CSC ;                                       \
    }

#define CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_XSPICE_OUTPUTS(ptr, binding, a, b)  \
    if ((smp_data_out->a != 0) && (smp_data_out->b != 0))                        \
        smp_data_out->ptr = smp_data_out->binding->CSC_Complex ;

#define CONVERT_KLU_BINDING_TABLE_TO_REAL_XSPICE_OUTPUTS(ptr, binding, a, b)     \
    if ((smp_data_out->a != 0) && (smp_data_out->b != 0))                        \
        smp_data_out->ptr = smp_data_out->binding->CSC ;

#define CREATE_KLU_BINDING_TABLE_XSPICE_INPUTS_E(ptr, binding, a, b)             \
    if ((smp_data_out->a != 0) && (smp_data_cntl->b != 0)) {                     \
        i = smp_data_out->input[k].port[l].ptr ;                                 \
        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ; \
        smp_data_out->input[k].port[l].e.binding = matched ;                     \
        smp_data_out->input[k].port[l].ptr = matched->CSC ;                      \
    }

#define CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_XSPICE_INPUTS_E(ptr, binding, a, b) \
    if ((smp_data_out->a != 0) && (smp_data_cntl->b != 0))                       \
        smp_data_out->input[k].port[l].ptr = smp_data_out->input[k].port[l].e.binding->CSC_Complex ;

#define CONVERT_KLU_BINDING_TABLE_TO_REAL_XSPICE_INPUTS_E(ptr, binding, a, b)    \
    if ((smp_data_out->a != 0) && (smp_data_cntl->b != 0))                       \
        smp_data_out->input[k].port[l].ptr = smp_data_out->input[k].port[l].e.binding->CSC ;

#define CREATE_KLU_BINDING_TABLE_XSPICE_INPUTS_F(ptr, binding, a, b)             \
    if ((smp_data_out->a != 0) && (smp_data_cntl->b != 0)) {                     \
        i = smp_data_out->input[k].port[l].ptr ;                                 \
        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ; \
        smp_data_out->input[k].port[l].f.binding = matched ;                     \
        smp_data_out->input[k].port[l].ptr = matched->CSC ;                      \
    }

#define CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_XSPICE_INPUTS_F(ptr, binding, a, b) \
    if ((smp_data_out->a != 0) && (smp_data_cntl->b != 0))                       \
        smp_data_out->input[k].port[l].ptr = smp_data_out->input[k].port[l].f.binding->CSC_Complex ;

#define CONVERT_KLU_BINDING_TABLE_TO_REAL_XSPICE_INPUTS_F(ptr, binding, a, b)    \
    if ((smp_data_out->a != 0) && (smp_data_cntl->b != 0))                       \
        smp_data_out->input[k].port[l].ptr = smp_data_out->input[k].port[l].f.binding->CSC ;

#define CREATE_KLU_BINDING_TABLE_XSPICE_INPUTS_G(ptr, binding, a, b)             \
    if ((smp_data_out->a != 0) && (smp_data_cntl->b != 0)) {                     \
        i = smp_data_out->input[k].port[l].ptr ;                                 \
        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ; \
        smp_data_out->input[k].port[l].g.binding = matched ;                     \
        smp_data_out->input[k].port[l].ptr = matched->CSC ;                      \
    }

#define CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_XSPICE_INPUTS_G(ptr, binding, a, b) \
    if ((smp_data_out->a != 0) && (smp_data_cntl->b != 0))                       \
        smp_data_out->input[k].port[l].ptr = smp_data_out->input[k].port[l].g.binding->CSC_Complex ;

#define CONVERT_KLU_BINDING_TABLE_TO_REAL_XSPICE_INPUTS_G(ptr, binding, a, b)    \
    if ((smp_data_out->a != 0) && (smp_data_cntl->b != 0))                       \
        smp_data_out->input[k].port[l].ptr = smp_data_out->input[k].port[l].g.binding->CSC ;

#define CREATE_KLU_BINDING_TABLE_XSPICE_INPUTS_H(ptr, binding, a, b)             \
    if ((smp_data_out->a != 0) && (smp_data_cntl->b != 0)) {                     \
        i = smp_data_out->input[k].port[l].ptr ;                                 \
        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ; \
        smp_data_out->input[k].port[l].h.binding = matched ;                     \
        smp_data_out->input[k].port[l].ptr = matched->CSC ;                      \
    }

#define CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_XSPICE_INPUTS_H(ptr, binding, a, b) \
    if ((smp_data_out->a != 0) && (smp_data_cntl->b != 0))                       \
        smp_data_out->input[k].port[l].ptr = smp_data_out->input[k].port[l].h.binding->CSC_Complex ;

#define CONVERT_KLU_BINDING_TABLE_TO_REAL_XSPICE_INPUTS_H(ptr, binding, a, b)    \
    if ((smp_data_out->a != 0) && (smp_data_cntl->b != 0))                       \
        smp_data_out->input[k].port[l].ptr = smp_data_out->input[k].port[l].h.binding->CSC ;
#endif

#endif
