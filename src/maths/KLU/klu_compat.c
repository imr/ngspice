/*
 * klu_compat.c
 *
 * Only compiled when ngspice is linked against an external SuiteSparse KLU
 * (NGSPICE_SYSTEM_KLU).
 *
 * The bundled KLU was modified by Francesco Lannutti to handle a matrix of
 * dimension 0: klu_analyze() sets Common->status = KLU_EMPTY_MATRIX and all
 * following factor/solve/extract calls return immediately, leaving that status
 * untouched.  ngspice (klusmp.c, CIDER) relies on this.  SuiteSparse instead
 * reports KLU_INVALID.  The wrappers below restore the bundled behaviour;
 * klu_ext.h maps the klu_* names onto them for all other translation units.
 */

#include "ngspice/config.h"

#define NGSPICE_KLU_COMPAT_IMPL
#include "ngspice/klu.h"

#define EMPTY_STATUS(Common) ((Common) != NULL && (Common)->status == KLU_EMPTY_MATRIX)

klu_symbolic *
ngspice_klu_analyze (int n, int Ap [ ], int Ai [ ], klu_common *Common)
{
    if (Common != NULL && n == 0) {
        Common->status = KLU_EMPTY_MATRIX ;
        return NULL ;
    }
    return klu_analyze (n, Ap, Ai, Common) ;
}

klu_numeric *
ngspice_klu_factor (int Ap [ ], int Ai [ ], double Ax [ ], klu_symbolic *Symbolic, klu_common *Common)
{
    if (EMPTY_STATUS (Common))
        return NULL ;
    return klu_factor (Ap, Ai, Ax, Symbolic, Common) ;
}

klu_numeric *
ngspice_klu_z_factor (int Ap [ ], int Ai [ ], double Ax [ ], klu_symbolic *Symbolic, klu_common *Common)
{
    if (EMPTY_STATUS (Common))
        return NULL ;
    return klu_z_factor (Ap, Ai, Ax, Symbolic, Common) ;
}

int
ngspice_klu_refactor (int Ap [ ], int Ai [ ], double Ax [ ], klu_symbolic *Symbolic, klu_numeric *Numeric, klu_common *Common)
{
    if (EMPTY_STATUS (Common))
        return 0 ;
    return klu_refactor (Ap, Ai, Ax, Symbolic, Numeric, Common) ;
}

int
ngspice_klu_z_refactor (int Ap [ ], int Ai [ ], double Ax [ ], klu_symbolic *Symbolic, klu_numeric *Numeric, klu_common *Common)
{
    if (EMPTY_STATUS (Common))
        return 0 ;
    return klu_z_refactor (Ap, Ai, Ax, Symbolic, Numeric, Common) ;
}

int
ngspice_klu_solve (klu_symbolic *Symbolic, klu_numeric *Numeric, int ldim, int nrhs, double B [ ], klu_common *Common)
{
    if (EMPTY_STATUS (Common))
        return 0 ;
    return klu_solve (Symbolic, Numeric, ldim, nrhs, B, Common) ;
}

int
ngspice_klu_z_solve (klu_symbolic *Symbolic, klu_numeric *Numeric, int ldim, int nrhs, double B [ ], klu_common *Common)
{
    if (EMPTY_STATUS (Common))
        return 0 ;
    return klu_z_solve (Symbolic, Numeric, ldim, nrhs, B, Common) ;
}

int
ngspice_klu_tsolve (klu_symbolic *Symbolic, klu_numeric *Numeric, int ldim, int nrhs, double B [ ], klu_common *Common)
{
    if (EMPTY_STATUS (Common))
        return 0 ;
    return klu_tsolve (Symbolic, Numeric, ldim, nrhs, B, Common) ;
}

int
ngspice_klu_z_tsolve (klu_symbolic *Symbolic, klu_numeric *Numeric, int ldim, int nrhs, double B [ ], int conj_solve, klu_common *Common)
{
    if (EMPTY_STATUS (Common))
        return 0 ;
    return klu_z_tsolve (Symbolic, Numeric, ldim, nrhs, B, conj_solve, Common) ;
}

int
ngspice_klu_extract (klu_numeric *Numeric, klu_symbolic *Symbolic,
    int *Lp, int *Li, double *Lx, int *Up, int *Ui, double *Ux,
    int *Fp, int *Fi, double *Fx, int *P, int *Q, double *Rs, int *R, klu_common *Common)
{
    if (EMPTY_STATUS (Common))
        return 0 ;
    return klu_extract (Numeric, Symbolic, Lp, Li, Lx, Up, Ui, Ux, Fp, Fi, Fx, P, Q, Rs, R, Common) ;
}

int
ngspice_klu_z_extract (klu_numeric *Numeric, klu_symbolic *Symbolic,
    int *Lp, int *Li, double *Lx, double *Lz, int *Up, int *Ui, double *Ux, double *Uz,
    int *Fp, int *Fi, double *Fx, double *Fz, int *P, int *Q, double *Rs, int *R, klu_common *Common)
{
    if (EMPTY_STATUS (Common))
        return 0 ;
    return klu_z_extract (Numeric, Symbolic, Lp, Li, Lx, Lz, Up, Ui, Ux, Uz, Fp, Fi, Fx, Fz, P, Q, Rs, R, Common) ;
}
