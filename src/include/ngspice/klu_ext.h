/*
 * ngspice/klu_ext.h
 *
 * ngspice-specific extensions to the KLU sparse solver.  These are not part
 * of SuiteSparse and are used identically with the bundled KLU (src/maths/KLU)
 * and with an external SuiteSparse KLU (NGSPICE_SYSTEM_KLU).
 *
 * Include "ngspice/klu.h", not this file directly.
 */

#ifndef ngspice_KLU_EXT_H
#define ngspice_KLU_EXT_H

#ifdef __cplusplus
extern "C" {
#endif

/* Common->status: matrix of dimension 0 (Francesco Lannutti).
 * Set by klu_analyze() when n == 0; klu_factor(), klu_refactor(),
 * klu_solve(), klu_tsolve() and klu_extract() then return immediately. */
#define KLU_EMPTY_MATRIX (2)

/* Francesco - Extract only Udiag */
int klu_extract_Udiag     /* returns TRUE if successful, FALSE otherwise */
(
    /* inputs: */
    klu_numeric *Numeric,
    klu_symbolic *Symbolic,

    /* outputs, all of which must be allocated on input */

    /* U */
    double *Ux,     /* size nnz(U) */

    int *P,
    int *Q,
    double *Rs,

    klu_common *Common
) ;

/* Francesco - Extract only Udiag */
int klu_z_extract_Udiag     /* returns TRUE if successful, FALSE otherwise */
(
    /* inputs: */
    klu_numeric *Numeric,
    klu_symbolic *Symbolic,

    /* outputs, all of which must be allocated on input */

    /* U */
    double *Ux,     /* size nnz(U) */
    double *Uz,     /* size nnz(U) for the complex case, ignored if real */

    int *P,
    int *Q,
    double *Rs,

    klu_common *Common
) ;

int klu_print
(
    int *Ap,
    int *Ai,
    double *Ax,
    int n,
    int *IntToExtRowMap,
    int *IntToExtColMap
) ;

int klu_z_print
(
    int *Ap,
    int *Ai,
    double *Ax,
    int n,
    int *IntToExtRowMap,
    int *IntToExtColMap
) ;


int klu_constant_multiply
(
    int *Ap,
    double *Ax,
    int n,
    klu_common *Common,
    double constant
) ;

int klu_z_constant_multiply
(
    int *Ap,
    double *Ax,
    int n,
    klu_common *Common,
    double constant
) ;


int klu_matrix_vector_multiply
(
    int *Ap,    /* CSR */
    int *Ai,    /* CSR */
    double *Ax, /* CSR */
    double *RHS,
    double *Solution,
    int *IntToExtRowMap,
    int *IntToExtColMap,
    int n,
    klu_common *Common
) ;

int klu_z_matrix_vector_multiply
(
    int *Ap,    /* CSR */
    int *Ai,    /* CSR */
    double *Ax, /* CSR */
    double *RHS,
    double *Solution,
    double *iRHS,
    double *iSolution,
    int *IntToExtRowMap,
    int *IntToExtColMap,
    int n,
    klu_common *Common
) ;


int klu_convert_matrix_in_CSR
(
    int *Ap_CSC,    /* CSC */
    int *Ai_CSC,    /* CSC */
    double *Ax_CSC, /* CSC */
    int *Ap_CSR,    /* CSR */
    int *Ai_CSR,    /* CSR */
    double *Ax_CSR, /* CSR */
    int n,
    int nz,
    klu_common *Common
) ;

int klu_z_convert_matrix_in_CSR
(
    int *Ap_CSC,    /* CSC */
    int *Ai_CSC,    /* CSC */
    double *Ax_CSC, /* CSC */
    int *Ap_CSR,    /* CSR */
    int *Ai_CSR,    /* CSR */
    double *Ax_CSR, /* CSR */
    int n,
    int nz,
    klu_common *Common
) ;

typedef struct sBindElement {
    double *COO ;
    double *CSC ;
    double *CSC_Complex ;
} BindElement ;

#ifdef CIDER
typedef struct sBindElementKLUforCIDER {
    double *COO ;
    double *CSC_Complex ;
} BindElementKLUforCIDER ;
#endif

typedef struct sKluLinkedListCOO {
    unsigned int row ;
    unsigned int col ;
    double *pointer ;
    struct sKluLinkedListCOO *next ;
} KluLinkedListCOO ;

int BindCompare (const void *a, const void *b) ;

#ifdef CIDER
int BindCompareKLUforCIDER (const void *a, const void *b) ;
int BindKluCompareCSCKLUforCIDER (const void *a, const void *b) ;
#endif

typedef struct sKLUmatrix {
    klu_common *KLUmatrixCommon ;                   /* KLU common object */
    klu_symbolic *KLUmatrixSymbolic ;               /* KLU symbolic object */
    klu_numeric *KLUmatrixNumeric ;                 /* KLU numeric object */
    int *KLUmatrixAp ;                              /* KLU column pointer */
    int *KLUmatrixAi ;                              /* KLU row pointer */
    double *KLUmatrixAx ;                           /* KLU Real Elements */
    double *KLUmatrixAxComplex ;                    /* KLU Complex Elements */
    unsigned int KLUmatrixIsComplex:1 ;             /* KLU Matrix Is Complex Flag */
    #define KLUmatrixReal 0                         /* KLU Matrix Real definition */
    #define KLUMatrixComplex 1                      /* KLU Matrix Complex definition */
    double *KLUmatrixIntermediate ;                 /* KLU RHS Intermediate for Solve Real Step */
    double *KLUmatrixIntermediateComplex ;          /* KLU iRHS Intermediate for Solve Complex Step */
    unsigned int KLUmatrixN ;                       /* KLU N */
    unsigned int KLUmatrixNrhs ;                    /* KLU N for RHS - needed by Node Collapsing */
    unsigned int KLUmatrixNZ ;                      /* KLU nz */
    BindElement *KLUmatrixBindStructCOO ;           /* KLU COO Binding Structure */
    KluLinkedListCOO *KLUmatrixLinkedListCOO ;      /* KLU COO in Linked List Format for Initial Parsing */
//    unsigned int *KLUmatrixNodeCollapsingOldToNew ; /* KLU Node Collapsing Mapping from New Node to Old Node */
    unsigned int *KLUmatrixNodeCollapsingNewToOld ; /* KLU Node Collapsing Mapping from New Node to Old Node */
    unsigned int KLUmatrixLinkedListNZ ;            /* KLU nz for the Initial Parsing */
    double *KLUmatrixTrashCOO ;                     /* KLU COO Trash Pointer for Ground Node not Stored in the Matrix */
    double **KLUmatrixDiag ;                        /* KLU pointer to diagonal element to perform Gmin */
    unsigned int KLUloadDiagGmin:1 ;                /* KLU flag to load Diag Gmin */

#ifdef CIDER
    int *KLUmatrixColCOOforCIDER ;             /* KLU Col Index for COO storage (for CIDER) */
    int *KLUmatrixRowCOOforCIDER ;             /* KLU Row Index for COO storage (for CIDER) */
    double *KLUmatrixValueComplexCOOforCIDER ; /* KLU Complex Elements for COO storage (for CIDER) */
    BindElementKLUforCIDER *KLUmatrixBindStructForCIDER ; /* KLU COO Binding Structure (for CIDER) */
#endif

} KLUmatrix ;

#ifdef NGSPICE_SYSTEM_KLU
/*
 * SuiteSparse KLU reports n == 0 as KLU_INVALID and has no notion of
 * KLU_EMPTY_MATRIX.  Route the affected calls through thin wrappers
 * (src/maths/KLU/klu_compat.c) that reproduce the behaviour of the
 * bundled KLU, so that the callers need not distinguish both cases.
 */
klu_symbolic *ngspice_klu_analyze (int n, int Ap [ ], int Ai [ ], klu_common *Common) ;
klu_numeric *ngspice_klu_factor (int Ap [ ], int Ai [ ], double Ax [ ], klu_symbolic *Symbolic, klu_common *Common) ;
klu_numeric *ngspice_klu_z_factor (int Ap [ ], int Ai [ ], double Ax [ ], klu_symbolic *Symbolic, klu_common *Common) ;
int ngspice_klu_refactor (int Ap [ ], int Ai [ ], double Ax [ ], klu_symbolic *Symbolic, klu_numeric *Numeric, klu_common *Common) ;
int ngspice_klu_z_refactor (int Ap [ ], int Ai [ ], double Ax [ ], klu_symbolic *Symbolic, klu_numeric *Numeric, klu_common *Common) ;
int ngspice_klu_solve (klu_symbolic *Symbolic, klu_numeric *Numeric, int ldim, int nrhs, double B [ ], klu_common *Common) ;
int ngspice_klu_z_solve (klu_symbolic *Symbolic, klu_numeric *Numeric, int ldim, int nrhs, double B [ ], klu_common *Common) ;
int ngspice_klu_tsolve (klu_symbolic *Symbolic, klu_numeric *Numeric, int ldim, int nrhs, double B [ ], klu_common *Common) ;
int ngspice_klu_z_tsolve (klu_symbolic *Symbolic, klu_numeric *Numeric, int ldim, int nrhs, double B [ ], int conj_solve, klu_common *Common) ;
int ngspice_klu_extract (klu_numeric *Numeric, klu_symbolic *Symbolic,
    int *Lp, int *Li, double *Lx, int *Up, int *Ui, double *Ux,
    int *Fp, int *Fi, double *Fx, int *P, int *Q, double *Rs, int *R, klu_common *Common) ;
int ngspice_klu_z_extract (klu_numeric *Numeric, klu_symbolic *Symbolic,
    int *Lp, int *Li, double *Lx, double *Lz, int *Up, int *Ui, double *Ux, double *Uz,
    int *Fp, int *Fi, double *Fx, double *Fz, int *P, int *Q, double *Rs, int *R, klu_common *Common) ;

#ifndef NGSPICE_KLU_COMPAT_IMPL
#define klu_analyze     ngspice_klu_analyze
#define klu_factor      ngspice_klu_factor
#define klu_z_factor    ngspice_klu_z_factor
#define klu_refactor    ngspice_klu_refactor
#define klu_z_refactor  ngspice_klu_z_refactor
#define klu_solve       ngspice_klu_solve
#define klu_z_solve     ngspice_klu_z_solve
#define klu_tsolve      ngspice_klu_tsolve
#define klu_z_tsolve    ngspice_klu_z_tsolve
#define klu_extract     ngspice_klu_extract
#define klu_z_extract   ngspice_klu_z_extract
#endif
#endif /* NGSPICE_SYSTEM_KLU */

#ifdef __cplusplus
}
#endif

#endif /* ngspice_KLU_EXT_H */
