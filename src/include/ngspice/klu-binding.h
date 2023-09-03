#ifndef _KLU_BINDING_H
#define _KLU_BINDING_H

#include "ngspice/klu.h"

#define CREATE_KLU_BINDING_TABLE(ptr, binding, a, b)                             \
    if ((here->a > 0) && (here->b > 0)) {                                        \
        i.COO = here->ptr ;                                                      \
        i.CSC = NULL ;                                                           \
        i.CSC_Complex = NULL ;                                                   \
        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof (BindElement), BindCompare) ; \
        if (matched == NULL) {                                                   \
            printf ("Ptr %p not found in BindStruct Table\n", here->ptr) ;       \
        } \
        here->binding = matched ;                                                \
        here->ptr = matched->CSC ;                                               \
    }

#define CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(ptr, binding, a, b)                 \
    if ((here->a > 0) && (here->b > 0))                                          \
        here->ptr = here->binding->CSC_Complex ;

#define CONVERT_KLU_BINDING_TABLE_TO_REAL(ptr, binding, a, b)                    \
    if ((here->a > 0) && (here->b > 0))                                          \
        here->ptr = here->binding->CSC ;

#ifdef XSPICE
#define CREATE_KLU_BINDING_TABLE_XSPICE_OUTPUTS(ptr, binding, a, b)              \
    if ((smp_data_out->a > 0) && (smp_data_out->b > 0)) {                        \
        i.COO = smp_data_out->ptr ;                                              \
        i.CSC = NULL ;                                                           \
        i.CSC_Complex = NULL ;                                                   \
        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof (BindElement), BindCompare) ; \
        smp_data_out->binding = matched ;                                        \
        smp_data_out->ptr = matched->CSC ;                                       \
    }

#define CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_XSPICE_OUTPUTS(ptr, binding, a, b)  \
    if ((smp_data_out->a > 0) && (smp_data_out->b > 0))                          \
        smp_data_out->ptr = smp_data_out->binding->CSC_Complex ;

#define CONVERT_KLU_BINDING_TABLE_TO_REAL_XSPICE_OUTPUTS(ptr, binding, a, b)     \
    if ((smp_data_out->a > 0) && (smp_data_out->b > 0))                          \
        smp_data_out->ptr = smp_data_out->binding->CSC ;

#define CREATE_KLU_BINDING_TABLE_XSPICE_INPUTS_E(ptr, binding, a, b)             \
    if ((smp_data_out->a > 0) && (smp_data_cntl->b > 0)) {                       \
        i.COO = smp_data_out->input[k].port[l].ptr ;                             \
        i.CSC = NULL ;                                                           \
        i.CSC_Complex = NULL ;                                                   \
        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof (BindElement), BindCompare) ; \
        smp_data_out->input[k].port[l].e.binding = matched ;                     \
        smp_data_out->input[k].port[l].ptr = matched->CSC ;                      \
    }

#define CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_XSPICE_INPUTS_E(ptr, binding, a, b) \
    if ((smp_data_out->a > 0) && (smp_data_cntl->b > 0))                         \
        smp_data_out->input[k].port[l].ptr = smp_data_out->input[k].port[l].e.binding->CSC_Complex ;

#define CONVERT_KLU_BINDING_TABLE_TO_REAL_XSPICE_INPUTS_E(ptr, binding, a, b)    \
    if ((smp_data_out->a > 0) && (smp_data_cntl->b > 0))                         \
        smp_data_out->input[k].port[l].ptr = smp_data_out->input[k].port[l].e.binding->CSC ;

#define CREATE_KLU_BINDING_TABLE_XSPICE_INPUTS_F(ptr, binding, a, b)             \
    if ((smp_data_out->a > 0) && (smp_data_cntl->b > 0)) {                       \
        i.COO = smp_data_out->input[k].port[l].ptr ;                             \
        i.CSC = NULL ;                                                           \
        i.CSC_Complex = NULL ;                                                   \
        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof (BindElement), BindCompare) ; \
        smp_data_out->input[k].port[l].f.binding = matched ;                     \
        smp_data_out->input[k].port[l].ptr = matched->CSC ;                      \
    }

#define CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_XSPICE_INPUTS_F(ptr, binding, a, b) \
    if ((smp_data_out->a > 0) && (smp_data_cntl->b > 0))                         \
        smp_data_out->input[k].port[l].ptr = smp_data_out->input[k].port[l].f.binding->CSC_Complex ;

#define CONVERT_KLU_BINDING_TABLE_TO_REAL_XSPICE_INPUTS_F(ptr, binding, a, b)    \
    if ((smp_data_out->a > 0) && (smp_data_cntl->b > 0))                         \
        smp_data_out->input[k].port[l].ptr = smp_data_out->input[k].port[l].f.binding->CSC ;

#define CREATE_KLU_BINDING_TABLE_XSPICE_INPUTS_G(ptr, binding, a, b)             \
    if ((smp_data_out->a > 0) && (smp_data_cntl->b > 0)) {                       \
        i.COO = smp_data_out->input[k].port[l].ptr ;                             \
        i.CSC = NULL ;                                                           \
        i.CSC_Complex = NULL ;                                                   \
        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof (BindElement), BindCompare) ; \
        smp_data_out->input[k].port[l].g.binding = matched ;                     \
        smp_data_out->input[k].port[l].ptr = matched->CSC ;                      \
    }

#define CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_XSPICE_INPUTS_G(ptr, binding, a, b) \
    if ((smp_data_out->a > 0) && (smp_data_cntl->b > 0))                         \
        smp_data_out->input[k].port[l].ptr = smp_data_out->input[k].port[l].g.binding->CSC_Complex ;

#define CONVERT_KLU_BINDING_TABLE_TO_REAL_XSPICE_INPUTS_G(ptr, binding, a, b)    \
    if ((smp_data_out->a > 0) && (smp_data_cntl->b > 0))                         \
        smp_data_out->input[k].port[l].ptr = smp_data_out->input[k].port[l].g.binding->CSC ;

#define CREATE_KLU_BINDING_TABLE_XSPICE_INPUTS_H(ptr, binding, a, b)             \
    if ((smp_data_out->a > 0) && (smp_data_cntl->b > 0)) {                       \
        i.COO = smp_data_out->input[k].port[l].ptr ;                             \
        i.CSC = NULL ;                                                           \
        i.CSC_Complex = NULL ;                                                   \
        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof (BindElement), BindCompare) ; \
        smp_data_out->input[k].port[l].h.binding = matched ;                     \
        smp_data_out->input[k].port[l].ptr = matched->CSC ;                      \
    }

#define CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_XSPICE_INPUTS_H(ptr, binding, a, b) \
    if ((smp_data_out->a > 0) && (smp_data_cntl->b > 0))                         \
        smp_data_out->input[k].port[l].ptr = smp_data_out->input[k].port[l].h.binding->CSC_Complex ;

#define CONVERT_KLU_BINDING_TABLE_TO_REAL_XSPICE_INPUTS_H(ptr, binding, a, b)    \
    if ((smp_data_out->a > 0) && (smp_data_cntl->b > 0))                         \
        smp_data_out->input[k].port[l].ptr = smp_data_out->input[k].port[l].h.binding->CSC ;
#endif

#ifdef CIDER
#define CREATE_KLU_BINDING_TABLE_CIDER(ptr, binding, a, b)                                \
    if ((a > 0) && (b > 0)) {                                                             \
        if (pNode->binding != NULL) {                                                     \
            if (pNode->binding->CSC_Complex != NULL) {                                    \
                qsort (BindStructCSC, nz, sizeof (BindElementKLUforCIDER), BindKluCompareCSCKLUforCIDER) ; \
                i.COO = NULL ;                                                            \
                i.CSC_Complex = pNode->binding->CSC_Complex ;                             \
                matched = (BindElementKLUforCIDER *) bsearch (&i, BindStructCSC, nz, sizeof (BindElementKLUforCIDER), BindKluCompareCSCKLUforCIDER) ; \
                if (matched == NULL) {                                                    \
                    i.COO = pNode->ptr ;                                                  \
                    i.CSC_Complex = NULL ;                                                \
                    matched = (BindElementKLUforCIDER *) bsearch (&i, BindStruct, nz, sizeof (BindElementKLUforCIDER), BindCompareKLUforCIDER) ; \
                    if (matched != NULL) {                                                \
                        pNode->binding = matched ;                                        \
                        pNode->ptr = matched->CSC_Complex ;                               \
                    }                                                                     \
                }                                                                         \
            } else {                                                                      \
                i.COO = pNode->ptr ;                                                      \
                i.CSC_Complex = NULL ;                                                    \
                matched = (BindElementKLUforCIDER *) bsearch (&i, BindStruct, nz, sizeof (BindElementKLUforCIDER), BindCompareKLUforCIDER) ; \
                pNode->binding = matched ;                                                \
                pNode->ptr = matched->CSC_Complex ;                                       \
            }                                                                             \
        } else {                                                                          \
            i.COO = pNode->ptr ;                                                          \
            i.CSC_Complex = NULL ;                                                        \
            matched = (BindElementKLUforCIDER *) bsearch (&i, BindStruct, nz, sizeof (BindElementKLUforCIDER), BindCompareKLUforCIDER) ; \
            pNode->binding = matched ;                                                    \
            pNode->ptr = matched->CSC_Complex ;                                           \
        }                                                                                 \
    }
#endif

#endif

