//------------------------------------------------------------------------------
// KLU/Source/klu_ext.h: include file for KLU
//------------------------------------------------------------------------------

/* Include file for user programs that call klu_* routines */

#ifndef _KLU_EXT_H
#define _KLU_EXT_H

#define KLU_EMPTY_MATRIX (2)        /* Modified by Francesco Lannutti - Case when the matrix is empty */

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

/* Francesco - Utilities */
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

#endif

