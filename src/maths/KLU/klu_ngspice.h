/*
 * klu_ngspice.h
 *
 * Private header for the ngspice additions to KLU (klu_utils.c,
 * klu_multiply.c).  Like the KLU sources these files are compiled twice,
 * without and with -DCOMPLEX.
 *
 * Only the public KLU structures (klu_common, klu_symbolic, klu_numeric) are
 * used, never the internal SuiteSparse headers, so the code works unchanged
 * with the bundled SuiteSparse and with any installed SuiteSparse version.
 */

#ifndef ngspice_KLU_NGSPICE_H
#define ngspice_KLU_NGSPICE_H

#include <stdio.h>
#include "ngspice/config.h"
#include "ngspice/memory.h"
#include "ngspice/klu.h"

#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif

#define ASSERT(a)

/* ngspice uses the int32_t ("int") version of KLU only */
#define Int int

/* layout of a complex entry in KLU: two doubles, real part first */
typedef struct {
    double Real ;
    double Imag ;
} ngspice_klu_complex ;

#define KLU_common   klu_common
#define KLU_symbolic klu_symbolic
#define KLU_numeric  klu_numeric

#ifdef COMPLEX

#define Entry ngspice_klu_complex
#define REAL(c) ((c).Real)
#define IMAG(c) ((c).Imag)

#define KLU_extract_Udiag          klu_z_extract_Udiag
#define KLU_print                  klu_z_print
#define KLU_constant_multiply      klu_z_constant_multiply
#define KLU_matrix_vector_multiply klu_z_matrix_vector_multiply
#define KLU_convert_matrix_in_CSR  klu_z_convert_matrix_in_CSR

#else

#define Entry double
#define REAL(c) (c)
#define IMAG(c) (0.)

#define KLU_extract_Udiag          klu_extract_Udiag
#define KLU_print                  klu_print
#define KLU_constant_multiply      klu_constant_multiply
#define KLU_matrix_vector_multiply klu_matrix_vector_multiply
#define KLU_convert_matrix_in_CSR  klu_convert_matrix_in_CSR

#endif

#endif /* ngspice_KLU_NGSPICE_H */
