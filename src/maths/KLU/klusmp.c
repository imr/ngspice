/* KLU to SMP Interface
 * Francesco Lannutti
 * July 2020
*/

#include "ngspice/config.h"
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include "ngspice/spmatrix.h"
#include "../sparse/spdefs.h"
#include "ngspice/smpdefs.h"

#if defined (_MSC_VER)
extern double scalbn(double, int);
#define logb _logb
extern double logb(double);
#endif

static void LoadGmin_CSC (double **diag, unsigned int n, double Gmin) ;
static void LoadGmin (SMPmatrix *eMatrix, double Gmin) ;

typedef struct sElement {
    unsigned int row ;
    unsigned int col ;
    double *pointer ;
    unsigned int group ;
} Element ;

static int
CompareRow (const void *a, const void *b)
{
    Element *A = (Element *) a ;
    Element *B = (Element *) b ;

    return
        (A->row > B->row) ?  1 :
        (A->row < B->row) ? -1 :
        0 ;
}

static int
CompareColumn (const void *a, const void *b)
{
    Element *A = (Element *) a ;
    Element *B = (Element *) b ;

    return
        (A->col > B->col) ?  1 :
        (A->col < B->col) ? -1 :
        0 ;
}

static void
Compress (unsigned int *Ai, unsigned int *Bp, unsigned int n, unsigned int nz)
{
    unsigned int i, j ;

    for (i = 0 ; i <= Ai [0] ; i++)
        Bp [i] = 0 ;

    j = Ai [0] + 1 ;
    for (i = 1 ; i < nz ; i++)
    {
        if (Ai [i] == Ai [i - 1] + 1)
        {
            Bp [j] = i ;
            j++ ;
        }
        else if (Ai [i] > Ai [i - 1] + 1)
        {
            for ( ; j <= Ai [i] ; j++)
                Bp [j] = i ;
        }
    }

    for ( ; j <= n ; j++)
        Bp [j] = i ;
}

int
BindCompare (const void *a, const void *b)
{
    BindElement *A = (BindElement *) a ;
    BindElement *B = (BindElement *) b ;

    return
        (A->COO > B->COO) ?  1 :
        (A->COO < B->COO) ? -1 :
        0 ;
}

#ifdef CIDER
int
BindCompareKLUforCIDER (const void *a, const void *b)
{
    BindElementKLUforCIDER *A = (BindElementKLUforCIDER *) a ;
    BindElementKLUforCIDER *B = (BindElementKLUforCIDER *) b ;

    return
        (A->COO > B->COO) ?  1 :
        (A->COO < B->COO) ? -1 :
        0 ;
}

int
BindKluCompareCSCKLUforCIDER (const void *a, const void *b)
{
    BindElementKLUforCIDER *A = (BindElementKLUforCIDER *) a ;
    BindElementKLUforCIDER *B = (BindElementKLUforCIDER *) b ;

    return
        (A->CSC_Complex > B->CSC_Complex) ?  1 :
        (A->CSC_Complex < B->CSC_Complex) ? -1 :
        0 ;
}
#endif

void SMPconvertCOOtoCSC (SMPmatrix *Matrix)
{
    Element *MatrixCOO ;
    KluLinkedListCOO *current, *temp ;
    unsigned int *Ap_COO, current_group, i, j ;

    /* Allocate the compressed COO elements */
    MatrixCOO = (Element *) malloc (Matrix->SMPkluMatrix->KLUmatrixLinkedListNZ * sizeof (Element)) ;

    /* Populate the compressed COO elements and COO value of Binding Table */
    /* Delete the Linked List in the meantime */
    i = 0 ;
    temp = Matrix->SMPkluMatrix->KLUmatrixLinkedListCOO ;
    while (temp != NULL) {
        MatrixCOO [i].row = temp->row ;
        MatrixCOO [i].col = temp->col ;
        MatrixCOO [i].pointer = temp->pointer ;
        MatrixCOO [i].group = 0 ;
        current = temp ;
        temp = temp->next ;
        free (current->pointer) ;
        free (current) ;
        current = NULL ;
        i++ ;
    }

    /* Order the MatrixCOO along the columns */
    qsort (MatrixCOO, Matrix->SMPkluMatrix->KLUmatrixLinkedListNZ, sizeof (Element), CompareColumn) ;

    /* Order the MatrixCOO along the rows */
    i = 0 ;
    while (i < Matrix->SMPkluMatrix->KLUmatrixLinkedListNZ)
    {
        /* Look for the next column */
        for (j = i + 1 ; j < Matrix->SMPkluMatrix->KLUmatrixLinkedListNZ ; j++)
        {
            if (MatrixCOO [j].col != MatrixCOO [i].col)
            {
                break ;
            }
        }

        qsort (MatrixCOO + i, j - i, sizeof (Element), CompareRow) ;

        i = j ;
    }

    /* Assign labels to avoid duplicates */
    for (i = 0, j = 1 ; i < Matrix->SMPkluMatrix->KLUmatrixLinkedListNZ - 1 ; i++, j++) {
        if ((MatrixCOO [i].col == MatrixCOO [j].col) && (MatrixCOO [i].row == MatrixCOO [j].row)) {
            // If col and row are the same
            MatrixCOO [j].group = MatrixCOO [i].group ;
        } else if ((MatrixCOO [i].col != MatrixCOO [j].col) || (MatrixCOO [i].row != MatrixCOO [j].row)) {
            // If or col either row are different, it isn't a duplicate, so assign the next label and store it in 'nz'
            MatrixCOO [j].group = MatrixCOO [i].group + 1 ;
        } else {
            printf ("Error: Strange behavior during label assignment\n") ;
        }
    }

    /* Assign N and NZ */
    Matrix->SMPkluMatrix->KLUmatrixN = (unsigned int)MatrixCOO [Matrix->SMPkluMatrix->KLUmatrixLinkedListNZ - 1].col + 1 ;
    Matrix->SMPkluMatrix->KLUmatrixNZ = MatrixCOO [Matrix->SMPkluMatrix->KLUmatrixLinkedListNZ - 1].group + 1 ;

    /* Allocate Diag Gmin CSC Vector */
    Matrix->SMPkluMatrix->KLUmatrixDiag = (double **) malloc (Matrix->SMPkluMatrix->KLUmatrixN * sizeof (double *)) ;
    for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixN ; i++) {
        Matrix->SMPkluMatrix->KLUmatrixDiag [i] = NULL ;
    }

    /* Allocate the temporary COO Column Index */
    Ap_COO = (unsigned int *) malloc (Matrix->SMPkluMatrix->KLUmatrixNZ * sizeof (unsigned int)) ;

    /* Allocate the needed KLU data structures */
    Matrix->SMPkluMatrix->KLUmatrixAp = (int *) malloc ((Matrix->SMPkluMatrix->KLUmatrixN + 1) * sizeof (int)) ;
    Matrix->SMPkluMatrix->KLUmatrixAi = (int *) malloc (Matrix->SMPkluMatrix->KLUmatrixNZ * sizeof (int)) ;
    Matrix->SMPkluMatrix->KLUmatrixBindStructCOO = (BindElement *) malloc (Matrix->SMPkluMatrix->KLUmatrixLinkedListNZ * sizeof (BindElement)) ;
    Matrix->SMPkluMatrix->KLUmatrixAx = (double *) malloc (Matrix->SMPkluMatrix->KLUmatrixNZ * sizeof (double)) ;
    Matrix->SMPkluMatrix->KLUmatrixAxComplex = (double *) malloc (2 * Matrix->SMPkluMatrix->KLUmatrixNZ * sizeof (double)) ;
    Matrix->SMPkluMatrix->KLUmatrixIntermediate = (double *) malloc (Matrix->SMPkluMatrix->KLUmatrixN * sizeof (double)) ;
    Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex = (double *) malloc (2 * Matrix->SMPkluMatrix->KLUmatrixN * sizeof (double)) ;

    /* Copy back the Matrix in partial CSC */
    for (i = 0, current_group = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixLinkedListNZ ; i++)
    {
        if (MatrixCOO [i].group > current_group) {
            current_group = MatrixCOO [i].group ;
        }

        Ap_COO [current_group] = MatrixCOO [i].col ;
        Matrix->SMPkluMatrix->KLUmatrixAi [current_group] = (int)MatrixCOO [i].row ;
        Matrix->SMPkluMatrix->KLUmatrixBindStructCOO [i].COO = MatrixCOO [i].pointer ;
        Matrix->SMPkluMatrix->KLUmatrixBindStructCOO [i].CSC = &(Matrix->SMPkluMatrix->KLUmatrixAx [current_group]) ;
        Matrix->SMPkluMatrix->KLUmatrixBindStructCOO [i].CSC_Complex = &(Matrix->SMPkluMatrix->KLUmatrixAxComplex [2 * current_group]) ;
        if (MatrixCOO [i].col == MatrixCOO [i].row) {
            Matrix->SMPkluMatrix->KLUmatrixDiag [MatrixCOO [i].col] = Matrix->SMPkluMatrix->KLUmatrixBindStructCOO [i].CSC ;
        }
    }

    /* Compress the COO Column Index to CSC Column Index */
    Compress (Ap_COO, (unsigned int *)Matrix->SMPkluMatrix->KLUmatrixAp, Matrix->SMPkluMatrix->KLUmatrixN, Matrix->SMPkluMatrix->KLUmatrixNZ) ;

    /* Free the temporary stuff */
    free (Ap_COO) ;
    free (MatrixCOO) ;

    /* Sort the Binding Table */
    qsort (Matrix->SMPkluMatrix->KLUmatrixBindStructCOO, Matrix->SMPkluMatrix->KLUmatrixLinkedListNZ, sizeof (BindElement), BindCompare) ;

    /* Set the Matrix as Real */
    Matrix->SMPkluMatrix->KLUmatrixIsComplex = KLUmatrixReal ;

    return ;
}

#ifdef CIDER
typedef struct sElementKLUforCIDER {
    unsigned int row ;
    unsigned int col ;
    double *pointer ;
} ElementKLUforCIDER ;

void SMPconvertCOOtoCSCKLUforCIDER (SMPmatrix *Matrix)
{
    ElementKLUforCIDER *MatrixCOO ;
    unsigned int *Ap_COO, i, j, nz ;

    /* Count the non-zero elements and store it */
    nz = 0 ;
    for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixN * Matrix->SMPkluMatrix->KLUmatrixN ; i++) {
        if ((Matrix->SMPkluMatrix->KLUmatrixRowCOOforCIDER [i] != -1) && (Matrix->SMPkluMatrix->KLUmatrixColCOOforCIDER [i] != -1)) {
            nz++ ;
        }
    }
    Matrix->SMPkluMatrix->KLUmatrixNZ = nz ;

    /* Allocate the compressed COO elements */
    MatrixCOO = (ElementKLUforCIDER *) malloc (nz * sizeof (ElementKLUforCIDER)) ;

    /* Allocate the temporary COO Column Index */
    Ap_COO = (unsigned int *) malloc (nz * sizeof (unsigned int)) ;

    /* Allocate the needed KLU data structures */
    Matrix->SMPkluMatrix->KLUmatrixAp = (int *) malloc ((Matrix->SMPkluMatrix->KLUmatrixN + 1) * sizeof (int)) ;
    Matrix->SMPkluMatrix->KLUmatrixAi = (int *) malloc (nz * sizeof (int)) ;
    Matrix->SMPkluMatrix->KLUmatrixBindStructForCIDER = (BindElementKLUforCIDER *) malloc (nz * sizeof (BindElementKLUforCIDER)) ;
    Matrix->SMPkluMatrix->KLUmatrixAxComplex = (double *) malloc (2 * nz * sizeof (double)) ;
    Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex = (double *) malloc (2 * Matrix->SMPkluMatrix->KLUmatrixN * sizeof (double)) ;

    /* Populate the compressed COO elements and COO value of Binding Table */
    j = 0 ;
    for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixN * Matrix->SMPkluMatrix->KLUmatrixN ; i++) {
        if ((Matrix->SMPkluMatrix->KLUmatrixRowCOOforCIDER [i] != -1) && (Matrix->SMPkluMatrix->KLUmatrixColCOOforCIDER [i] != -1)) {
            MatrixCOO [j].row = (unsigned int)Matrix->SMPkluMatrix->KLUmatrixRowCOOforCIDER [i] ;
            MatrixCOO [j].col = (unsigned int)Matrix->SMPkluMatrix->KLUmatrixColCOOforCIDER [i] ;
            MatrixCOO [j].pointer = &(Matrix->SMPkluMatrix->KLUmatrixValueComplexCOOforCIDER [2 * i]) ;
            j++ ;
        }
    }

    /* Order the MatrixCOO along the columns */
    qsort (MatrixCOO, nz, sizeof (ElementKLUforCIDER), CompareColumn) ;

    /* Order the MatrixCOO along the rows */
    i = 0 ;
    while (i < nz)
    {
        /* Look for the next column */
        for (j = i + 1 ; j < nz ; j++)
        {
            if (MatrixCOO [j].col != MatrixCOO [i].col)
            {
                break ;
            }
        }

        qsort (MatrixCOO + i, j - i, sizeof (ElementKLUforCIDER), CompareRow) ;

        i = j ;
    }

    /* Copy back the Matrix in partial CSC */
    for (i = 0 ; i < nz ; i++)
    {
        Ap_COO [i] = MatrixCOO [i].col ;
        Matrix->SMPkluMatrix->KLUmatrixAi [i] = (int)MatrixCOO [i].row ;
        Matrix->SMPkluMatrix->KLUmatrixBindStructForCIDER [i].COO = MatrixCOO [i].pointer ;
        Matrix->SMPkluMatrix->KLUmatrixBindStructForCIDER [i].CSC_Complex = &(Matrix->SMPkluMatrix->KLUmatrixAxComplex [2 * i]) ;
    }

    /* Compress the COO Column Index to CSC Column Index */
    Compress (Ap_COO, (unsigned int *)Matrix->SMPkluMatrix->KLUmatrixAp, Matrix->SMPkluMatrix->KLUmatrixN, nz) ;

    /* Free the temporary stuff */
    free (Ap_COO) ;
    free (MatrixCOO) ;

    /* Sort the Binding Table */
    qsort (Matrix->SMPkluMatrix->KLUmatrixBindStructForCIDER, nz, sizeof (BindElementKLUforCIDER), BindCompareKLUforCIDER) ;

    return ;
}
#endif

/*
 * SMPmakeElt()
 */
double *
SMPmakeElt (SMPmatrix *Matrix, int Row, int Col)
{
    KluLinkedListCOO *temp ;

    if (Matrix->CKTkluMODE) {
        if ((Row > 0) && (Col > 0)) {
            Row = Row - 1 ;
            Col = Col - 1 ;
            temp = (KluLinkedListCOO *) malloc (sizeof (KluLinkedListCOO)) ;
            temp->row = (unsigned int)Row ;
            temp->col = (unsigned int)Col ;
            temp->pointer = (double *) malloc (sizeof (double)) ;
            temp->next = Matrix->SMPkluMatrix->KLUmatrixLinkedListCOO ;
            Matrix->SMPkluMatrix->KLUmatrixLinkedListCOO = temp ;
            Matrix->SMPkluMatrix->KLUmatrixLinkedListNZ++ ;
            return temp->pointer ;
        } else {
            return Matrix->SMPkluMatrix->KLUmatrixTrashCOO ;
        }
    } else {
        return spGetElement (Matrix->SPmatrix, Row, Col) ;
    }
}

#ifdef CIDER
double *
SMPmakeEltKLUforCIDER (SMPmatrix *Matrix, int Row, int Col)
{
    if (Matrix->CKTkluMODE) {
        if ((Row > 0) && (Col > 0)) {
            Row = Row - 1 ;
            Col = Col - 1 ;
            Matrix->SMPkluMatrix->KLUmatrixRowCOOforCIDER [Row * (int)Matrix->SMPkluMatrix->KLUmatrixN + Col] = Row ;
            Matrix->SMPkluMatrix->KLUmatrixColCOOforCIDER [Row * (int)Matrix->SMPkluMatrix->KLUmatrixN + Col] = Col ;
            return &(Matrix->SMPkluMatrix->KLUmatrixValueComplexCOOforCIDER [2 * (Row * (int)Matrix->SMPkluMatrix->KLUmatrixN + Col)]) ;
        } else {
            return Matrix->SMPkluMatrix->KLUmatrixTrashCOO ;
        }
    } else {
        return spGetElement (Matrix->SPmatrix, Row, Col) ;
    }
}
#endif

/*
 * SMPcClear()
 */

void
SMPcClear (SMPmatrix *Matrix)
{
    unsigned int i ;

    if (Matrix->CKTkluMODE)
    {
        for (i = 0 ; i < 2 * Matrix->SMPkluMatrix->KLUmatrixNZ ; i++) {
            Matrix->SMPkluMatrix->KLUmatrixAxComplex [i] = 0 ;
        }
    } else {
        spClear (Matrix->SPmatrix) ;
    }
}

/*
 * SMPclear()
 */

void
SMPclear (SMPmatrix *Matrix)
{
    unsigned int i ;

    if (Matrix->CKTkluMODE)
    {
        for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixNZ ; i++) {
            Matrix->SMPkluMatrix->KLUmatrixAx [i] = 0 ;
        }
    } else {
        spClear (Matrix->SPmatrix) ;
    }
}

#ifdef CIDER
void
SMPclearKLUforCIDER (SMPmatrix *Matrix)
{
    unsigned int i ;

    for (i = 0 ; i < 2 * Matrix->SMPkluMatrix->KLUmatrixNZ ; i++) {
        Matrix->SMPkluMatrix->KLUmatrixAxComplex [i] = 0 ;
    }
}
#endif

#define NG_IGNORE(x)  (void)x

/*
 * SMPcLUfac()
 */
/*ARGSUSED*/

int
SMPcLUfac (SMPmatrix *Matrix, double PivTol)
{
    int ret ;

    NG_IGNORE (PivTol) ;

    if (Matrix->CKTkluMODE)
    {
        ret = klu_z_refactor (Matrix->SMPkluMatrix->KLUmatrixAp, Matrix->SMPkluMatrix->KLUmatrixAi, Matrix->SMPkluMatrix->KLUmatrixAxComplex,
                              Matrix->SMPkluMatrix->KLUmatrixSymbolic, Matrix->SMPkluMatrix->KLUmatrixNumeric, Matrix->SMPkluMatrix->KLUmatrixCommon) ;

        if (ret == 0)
        {
            if (Matrix->SMPkluMatrix->KLUmatrixCommon->status == KLU_SINGULAR) {
                fprintf (stderr, "Warning (ReFactor): KLU Matrix is SINGULAR\n") ;
                return E_SINGULAR ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixCommon == NULL) {
                fprintf (stderr, "Error (ReFactor): KLUcommon object is NULL. A problem occurred\n") ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixCommon->status == KLU_EMPTY_MATRIX)
            {
                fprintf (stderr, "Error (ReFactor): KLU Matrix is empty\n") ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixNumeric == NULL) {
                fprintf (stderr, "Error (ReFactor): KLUnumeric object is NULL. A problem occurred\n") ;
            }
            return 1 ;
        } else {
            return 0 ;
        }
    } else {
        spSetComplex (Matrix->SPmatrix) ;
        return spFactor (Matrix->SPmatrix) ;
    }
}

/*
 * SMPluFac()
 */
/*ARGSUSED*/

int
SMPluFac (SMPmatrix *Matrix, double PivTol, double Gmin)
{
    int ret ;

    NG_IGNORE (PivTol) ;

    if (Matrix->CKTkluMODE)
    {
        LoadGmin_CSC (Matrix->SMPkluMatrix->KLUmatrixDiag, Matrix->SMPkluMatrix->KLUmatrixN, Gmin) ;

        ret = klu_refactor (Matrix->SMPkluMatrix->KLUmatrixAp, Matrix->SMPkluMatrix->KLUmatrixAi, Matrix->SMPkluMatrix->KLUmatrixAx,
                            Matrix->SMPkluMatrix->KLUmatrixSymbolic, Matrix->SMPkluMatrix->KLUmatrixNumeric, Matrix->SMPkluMatrix->KLUmatrixCommon) ;

        if (ret == 0)
        {
            if (Matrix->SMPkluMatrix->KLUmatrixCommon->status == KLU_SINGULAR) {
                fprintf (stderr, "Warning (ReFactor): KLU Matrix is SINGULAR\n") ;
                return E_SINGULAR ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixCommon == NULL) {
                fprintf (stderr, "Error (ReFactor): KLUcommon object is NULL. A problem occurred\n") ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixCommon->status == KLU_EMPTY_MATRIX)
            {
                fprintf (stderr, "Error (ReFactor): KLU Matrix is empty\n") ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixNumeric == NULL) {
                fprintf (stderr, "Error (ReFactor): KLUnumeric object is NULL. A problem occurred\n") ;
            }
            return 1 ;
        } else {
            return 0 ;
        }
    } else {
        spSetReal (Matrix->SPmatrix) ;
        LoadGmin (Matrix, Gmin) ;
        return spFactor (Matrix->SPmatrix) ;
    }
}

#ifdef CIDER
int
SMPluFacKLUforCIDER (SMPmatrix *Matrix)
{
    unsigned int i ;
    double *KLUmatrixAx ;
    int ret ;

    if (Matrix->CKTkluMODE)
    {
        if (Matrix->SMPkluMatrix->KLUmatrixIsComplex) {
            ret = klu_z_refactor (Matrix->SMPkluMatrix->KLUmatrixAp, Matrix->SMPkluMatrix->KLUmatrixAi, Matrix->SMPkluMatrix->KLUmatrixAxComplex,
                                  Matrix->SMPkluMatrix->KLUmatrixSymbolic, Matrix->SMPkluMatrix->KLUmatrixNumeric, Matrix->SMPkluMatrix->KLUmatrixCommon) ;
        } else {
            /* Allocate the Real Matrix */
            KLUmatrixAx = (double *) malloc (Matrix->SMPkluMatrix->KLUmatrixNZ * sizeof(double)) ;

            /* Copy the Complex Matrix into the Real Matrix */
            for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixNZ ; i++) {
                KLUmatrixAx [i] = Matrix->SMPkluMatrix->KLUmatrixAxComplex [2 * i] ;
            }

            /* Re-Factor the Real Matrix */
            ret = klu_refactor (Matrix->SMPkluMatrix->KLUmatrixAp, Matrix->SMPkluMatrix->KLUmatrixAi, KLUmatrixAx,
                                Matrix->SMPkluMatrix->KLUmatrixSymbolic, Matrix->SMPkluMatrix->KLUmatrixNumeric, Matrix->SMPkluMatrix->KLUmatrixCommon) ;

            /* Free the Real Matrix Storage */
            free (KLUmatrixAx) ;
        }

        if (ret == 0)
        {
            if (Matrix->SMPkluMatrix->KLUmatrixCommon->status == KLU_SINGULAR) {
                fprintf (stderr, "Warning (ReFactor): KLU Matrix is SINGULAR\n") ;
                return E_SINGULAR ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixCommon == NULL) {
                fprintf (stderr, "Error (ReFactor): KLUcommon object is NULL. A problem occurred\n") ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixCommon->status == KLU_EMPTY_MATRIX)
            {
                fprintf (stderr, "Error (ReFactor): KLU Matrix is empty\n") ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixNumeric == NULL) {
                fprintf (stderr, "Error (ReFactor): KLUnumeric object is NULL. A problem occurred\n") ;
            }
            return 1 ;
        } else {
            return 0 ;
        }
    } else {
        return spFactor (Matrix->SPmatrix) ;
    }
}
#endif

/*
 * SMPcReorder()
 */

int
SMPcReorder (SMPmatrix *Matrix, double PivTol, double PivRel, int *NumSwaps)
{
    if (Matrix->CKTkluMODE)
    {
        Matrix->SMPkluMatrix->KLUmatrixCommon->tol = PivRel ;

        if (Matrix->SMPkluMatrix->KLUmatrixNumeric != NULL) {
            klu_free_numeric (&(Matrix->SMPkluMatrix->KLUmatrixNumeric), Matrix->SMPkluMatrix->KLUmatrixCommon) ;
        }
        Matrix->SMPkluMatrix->KLUmatrixNumeric = klu_z_factor (Matrix->SMPkluMatrix->KLUmatrixAp, Matrix->SMPkluMatrix->KLUmatrixAi,
                                                               Matrix->SMPkluMatrix->KLUmatrixAxComplex, Matrix->SMPkluMatrix->KLUmatrixSymbolic,
                                                               Matrix->SMPkluMatrix->KLUmatrixCommon) ;

        if (Matrix->SMPkluMatrix->KLUmatrixNumeric == NULL)
        {
            fprintf (stderr, "Error (Factor): KLUnumeric object is NULL. A problem occurred\n") ;
            if (Matrix->SMPkluMatrix->KLUmatrixCommon->status == KLU_SINGULAR) {
                fprintf (stderr, "Warning (Factor): KLU Matrix is SINGULAR\n") ;
                return 0 ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixCommon == NULL) {
                fprintf (stderr, "Error (Factor): KLUcommon object is NULL. A problem occurred\n") ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixCommon->status == KLU_EMPTY_MATRIX) {
                fprintf (stderr, "Error (Factor): KLU Matrix is empty\n") ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixSymbolic == NULL) {
                fprintf (stderr, "Error (Factor): KLUsymbolic object is NULL. A problem occurred\n") ;
            }
            return 1 ;
        } else {
            return 0 ;
        }
    } else {
        *NumSwaps = 1 ;
        spSetComplex (Matrix->SPmatrix) ;
        return spOrderAndFactor (Matrix->SPmatrix, NULL, (spREAL)PivRel, (spREAL)PivTol, YES) ;
    }
}

/*
 * SMPreorder()
 */

int
SMPreorder (SMPmatrix *Matrix, double PivTol, double PivRel, double Gmin)
{
    if (Matrix->CKTkluMODE)
    {
        LoadGmin_CSC (Matrix->SMPkluMatrix->KLUmatrixDiag, Matrix->SMPkluMatrix->KLUmatrixN, Gmin) ;
        Matrix->SMPkluMatrix->KLUmatrixCommon->tol = PivRel ;

        if (Matrix->SMPkluMatrix->KLUmatrixNumeric != NULL) {
            klu_free_numeric (&(Matrix->SMPkluMatrix->KLUmatrixNumeric), Matrix->SMPkluMatrix->KLUmatrixCommon) ;
        }

        Matrix->SMPkluMatrix->KLUmatrixNumeric = klu_factor (Matrix->SMPkluMatrix->KLUmatrixAp, Matrix->SMPkluMatrix->KLUmatrixAi,
                                                             Matrix->SMPkluMatrix->KLUmatrixAx, Matrix->SMPkluMatrix->KLUmatrixSymbolic,
                                                             Matrix->SMPkluMatrix->KLUmatrixCommon) ;

        if (Matrix->SMPkluMatrix->KLUmatrixNumeric == NULL)
        {
            fprintf (stderr, "Error (Factor): KLUnumeric object is NULL. A problem occurred\n") ;
            if (Matrix->SMPkluMatrix->KLUmatrixCommon->status == KLU_SINGULAR) {
                fprintf (stderr, "Warning (Factor): KLU Matrix is SINGULAR\n") ;
                return 0 ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixCommon == NULL) {
                fprintf (stderr, "Error (Factor): KLUcommon object is NULL. A problem occurred\n") ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixCommon->status == KLU_EMPTY_MATRIX) {
                fprintf (stderr, "Error (Factor): KLU Matrix is empty\n") ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixSymbolic == NULL) {
                fprintf (stderr, "Error (Factor): KLUsymbolic object is NULL. A problem occurred\n") ;
            }
            return 1 ;
        } else {
            return 0 ;
        }
    } else {
        spSetReal (Matrix->SPmatrix) ;
        LoadGmin (Matrix, Gmin) ;
        return spOrderAndFactor (Matrix->SPmatrix, NULL, (spREAL)PivRel, (spREAL)PivTol, YES) ;
    }
}

#ifdef CIDER
int
SMPreorderKLUforCIDER (SMPmatrix *Matrix)
{
    unsigned int i ;
    double *KLUmatrixAx ;

    if (Matrix->CKTkluMODE)
    {
        if (Matrix->SMPkluMatrix->KLUmatrixNumeric != NULL) {
            klu_free_numeric (&(Matrix->SMPkluMatrix->KLUmatrixNumeric), Matrix->SMPkluMatrix->KLUmatrixCommon) ;
        }
        if (Matrix->SMPkluMatrix->KLUmatrixIsComplex) {
            Matrix->SMPkluMatrix->KLUmatrixNumeric = klu_z_factor (Matrix->SMPkluMatrix->KLUmatrixAp, Matrix->SMPkluMatrix->KLUmatrixAi,
                                                                   Matrix->SMPkluMatrix->KLUmatrixAxComplex, Matrix->SMPkluMatrix->KLUmatrixSymbolic,
                                                                   Matrix->SMPkluMatrix->KLUmatrixCommon) ;
        } else {
            /* Allocate the Real Matrix */
            KLUmatrixAx = (double *) malloc (Matrix->SMPkluMatrix->KLUmatrixNZ * sizeof(double)) ;

            /* Copy the Complex Matrix into the Real Matrix */
            for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixNZ ; i++) {
                KLUmatrixAx [i] = Matrix->SMPkluMatrix->KLUmatrixAxComplex [2 * i] ;
            }

            /* Factor the Real Matrix */
            Matrix->SMPkluMatrix->KLUmatrixNumeric = klu_factor (Matrix->SMPkluMatrix->KLUmatrixAp, Matrix->SMPkluMatrix->KLUmatrixAi,
                                                                 KLUmatrixAx, Matrix->SMPkluMatrix->KLUmatrixSymbolic,
                                                                 Matrix->SMPkluMatrix->KLUmatrixCommon) ;

            /* Free the Real Matrix Storage */
            free (KLUmatrixAx) ;
        }

        if (Matrix->SMPkluMatrix->KLUmatrixNumeric == NULL)
        {
            fprintf (stderr, "Error (Factor): KLUnumeric object is NULL. A problem occurred\n") ;
            if (Matrix->SMPkluMatrix->KLUmatrixCommon->status == KLU_SINGULAR) {
                fprintf (stderr, "Warning (Factor): KLU Matrix is SINGULAR\n") ;
                return 0 ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixCommon == NULL) {
                fprintf (stderr, "Error (Factor): KLUcommon object is NULL. A problem occurred\n") ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixCommon->status == KLU_EMPTY_MATRIX) {
                fprintf (stderr, "Error (Factor): KLU Matrix is empty\n") ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixSymbolic == NULL) {
                fprintf (stderr, "Error (Factor): KLUsymbolic object is NULL. A problem occurred\n") ;
            }
            return 1 ;
        } else {
            return 0 ;
        }
    } else {
        return spFactor (Matrix->SPmatrix) ;
    }
}
#endif

/*
 * SMPcaSolve()
 */
void
SMPcaSolve (SMPmatrix *Matrix, double RHS[], double iRHS[], double Spare[], double iSpare[])
{
    int ret ;
    unsigned int i ;

    NG_IGNORE (iSpare) ;
    NG_IGNORE (Spare) ;

    if (Matrix->CKTkluMODE)
    {
        for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixN ; i++)
        {
            Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex [2 * i] = RHS [i + 1] ;
            Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex [2 * i + 1] = iRHS [i + 1] ;
        }

        ret = klu_z_solve (Matrix->SMPkluMatrix->KLUmatrixSymbolic, Matrix->SMPkluMatrix->KLUmatrixNumeric, (int)Matrix->SMPkluMatrix->KLUmatrixN, 1,
                           Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex, Matrix->SMPkluMatrix->KLUmatrixCommon) ;

        for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixN ; i++)
        {
            RHS [i + 1] = Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex [2 * i] ;
            iRHS [i + 1] = Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex [2 * i + 1] ;
        }
    } else {
        spSolveTransposed (Matrix->SPmatrix, RHS, RHS, iRHS, iRHS) ;
    }
}

/*
 * SMPcSolve()
 */

void
SMPcSolve (SMPmatrix *Matrix, double RHS[], double iRHS[], double Spare[], double iSpare[])
{
    int ret ;
    unsigned int i ;

    NG_IGNORE (iSpare) ;
    NG_IGNORE (Spare) ;

    if (Matrix->CKTkluMODE)
    {
        for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixN ; i++)
        {
            Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex [2 * i] = RHS [i + 1] ;
            Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex [2 * i + 1] = iRHS [i + 1] ;
        }

        ret = klu_z_solve (Matrix->SMPkluMatrix->KLUmatrixSymbolic, Matrix->SMPkluMatrix->KLUmatrixNumeric, (int)Matrix->SMPkluMatrix->KLUmatrixN, 1,
                           Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex, Matrix->SMPkluMatrix->KLUmatrixCommon) ;

        for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixN ; i++)
        {
            RHS [i + 1] = Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex [2 * i] ;
            iRHS [i + 1] = Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex [2 * i + 1] ;
        }
    } else {
        spSolve (Matrix->SPmatrix, RHS, RHS, iRHS, iRHS) ;
    }
}

/*
 * SMPsolve()
 */

void
SMPsolve (SMPmatrix *Matrix, double RHS[], double Spare[])
{
    int ret ;
    unsigned int i ;

    NG_IGNORE (Spare) ;

    if (Matrix->CKTkluMODE)
    {
        for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixN ; i++) {
            Matrix->SMPkluMatrix->KLUmatrixIntermediate [i] = RHS [i + 1] ;
        }

        ret = klu_solve (Matrix->SMPkluMatrix->KLUmatrixSymbolic, Matrix->SMPkluMatrix->KLUmatrixNumeric, (int)Matrix->SMPkluMatrix->KLUmatrixN, 1,
                         Matrix->SMPkluMatrix->KLUmatrixIntermediate, Matrix->SMPkluMatrix->KLUmatrixCommon) ;

        if (ret == 0)
        {
            if (Matrix->SMPkluMatrix->KLUmatrixCommon->status == KLU_SINGULAR) {
                fprintf (stderr, "Warning (Solve): KLU Matrix is SINGULAR\n") ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixCommon == NULL) {
                fprintf (stderr, "Error (Solve): KLUcommon object is NULL. A problem occurred\n") ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixCommon->status == KLU_EMPTY_MATRIX)
            {
                fprintf (stderr, "Error (Solve): KLU Matrix is empty\n") ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixNumeric == NULL) {
                fprintf (stderr, "Error (Solve): KLUnumeric object is NULL. A problem occurred\n") ;
            }
            if (Matrix->SMPkluMatrix->KLUmatrixSymbolic == NULL) {
                fprintf (stderr, "Error (Solve): KLUsymbolic object is NULL. A problem occurred\n") ;
            }
        }

        for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixN ; i++) {
            RHS [i + 1] = Matrix->SMPkluMatrix->KLUmatrixIntermediate [i] ;
        }
    } else {
        spSolve (Matrix->SPmatrix, RHS, RHS, NULL, NULL) ;
    }
}

#ifdef CIDER
void
SMPsolveKLUforCIDER (SMPmatrix *Matrix, double RHS[], double RHSsolution[], double iRHS[], double iRHSsolution[])
{
    int ret ;
    unsigned int i ;
    double *KLUmatrixIntermediate ;

    if (Matrix->CKTkluMODE)
    {
        if (Matrix->SMPkluMatrix->KLUmatrixIsComplex) {
            for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixN ; i++)
            {
                Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex [2 * i] = RHS [i + 1] ;
                Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex [2 * i + 1] = iRHS [i + 1] ;
            }

            ret = klu_z_solve (Matrix->SMPkluMatrix->KLUmatrixSymbolic, Matrix->SMPkluMatrix->KLUmatrixNumeric, (int)Matrix->SMPkluMatrix->KLUmatrixN, 1,
                               Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex, Matrix->SMPkluMatrix->KLUmatrixCommon) ;

            for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixN ; i++)
            {
                RHSsolution [i + 1] = Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex [2 * i] ;
                iRHSsolution [i + 1] = Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex [2 * i + 1] ;
            }
        } else {
            /* Allocate the Intermediate Vector */
            KLUmatrixIntermediate = (double *) malloc (Matrix->SMPkluMatrix->KLUmatrixN * sizeof(double)) ;

            for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixN ; i++) {
                KLUmatrixIntermediate [i] = RHS [i + 1] ;
            }

            ret = klu_solve (Matrix->SMPkluMatrix->KLUmatrixSymbolic, Matrix->SMPkluMatrix->KLUmatrixNumeric, (int)Matrix->SMPkluMatrix->KLUmatrixN, 1,
                             KLUmatrixIntermediate, Matrix->SMPkluMatrix->KLUmatrixCommon) ;

            for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixN ; i++) {
                RHSsolution [i + 1] = KLUmatrixIntermediate [i] ;
            }

            /* Free the Intermediate Vector */
            free (KLUmatrixIntermediate) ;
        }

    } else {

        spSolve (Matrix->SPmatrix, RHS, RHSsolution, iRHS, iRHSsolution) ;
    }
}
#endif

/*
 * SMPmatSize()
 */
int
SMPmatSize (SMPmatrix *Matrix)
{
    if (Matrix->CKTkluMODE) {
        return (int)Matrix->SMPkluMatrix->KLUmatrixN ;
    } else {
        return spGetSize (Matrix->SPmatrix, 1) ;
    }
}

/*
 * SMPnewMatrix()
 */
int
SMPnewMatrix (SMPmatrix *Matrix, int size)
{
    int Error ;

    if (Matrix->CKTkluMODE) {
        /* Allocate the KLU Matrix Data Structure */
        Matrix->SMPkluMatrix = (KLUmatrix *) malloc (sizeof (KLUmatrix)) ;
        Matrix->SMPkluMatrix->KLUmatrixLinkedListNZ = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixLinkedListCOO = NULL ;

        /* Initialize the KLU Matrix Internal Pointers */
        Matrix->SMPkluMatrix->KLUmatrixCommon = (klu_common *) malloc (sizeof (klu_common)) ; ;
        Matrix->SMPkluMatrix->KLUmatrixSymbolic = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixNumeric = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixAp = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixAi = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixAx = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixAxComplex = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixIsComplex = KLUmatrixReal ;
        Matrix->SMPkluMatrix->KLUmatrixIntermediate = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixNZ = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixBindStructCOO = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixDiag = NULL ;

        /* Initialize the KLU Common Data Structure */
        klu_defaults (Matrix->SMPkluMatrix->KLUmatrixCommon) ;
        Matrix->SMPkluMatrix->KLUmatrixCommon->memgrow = Matrix->CKTkluMemGrowFactor ;

        /* Allocate KLU data structures */
        Matrix->SMPkluMatrix->KLUmatrixN = (unsigned int)size ;
        Matrix->SMPkluMatrix->KLUmatrixTrashCOO = (double *) malloc (sizeof (double)) ;

        return spOKAY ;
    } else {
        Matrix->SPmatrix = spCreate (size, 1, &Error) ;
        return Error ;
    }
}

#ifdef CIDER
int
SMPnewMatrixKLUforCIDER (SMPmatrix *Matrix, int size, unsigned int KLUmatrixIsComplex)
{
    int Error ;
    unsigned int i ;

    if (Matrix->CKTkluMODE) {
        /* Allocate the KLU Matrix Data Structure */
        Matrix->SMPkluMatrix = (KLUmatrix *) malloc (sizeof (KLUmatrix)) ;

        /* Initialize the KLU Matrix Internal Pointers */
        Matrix->SMPkluMatrix->KLUmatrixCommon = (klu_common *) malloc (sizeof (klu_common)) ; ;
        Matrix->SMPkluMatrix->KLUmatrixSymbolic = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixNumeric = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixAp = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixAi = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixAxComplex = NULL ;
        if (KLUmatrixIsComplex) {
            Matrix->SMPkluMatrix->KLUmatrixIsComplex = KLUMatrixComplex ;
        } else {
            Matrix->SMPkluMatrix->KLUmatrixIsComplex = KLUmatrixReal ;
        }
        Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixNZ = 0 ;
        Matrix->SMPkluMatrix->KLUmatrixBindStructForCIDER = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixValueComplexCOOforCIDER = NULL ;

        /* Initialize the KLU Common Data Structure */
        klu_defaults (Matrix->SMPkluMatrix->KLUmatrixCommon) ;

        /* Allocate KLU data structures */
        Matrix->SMPkluMatrix->KLUmatrixN = (unsigned int)size ;
        Matrix->SMPkluMatrix->KLUmatrixColCOOforCIDER = (int *) malloc (Matrix->SMPkluMatrix->KLUmatrixN * Matrix->SMPkluMatrix->KLUmatrixN * sizeof(int)) ;
        Matrix->SMPkluMatrix->KLUmatrixRowCOOforCIDER = (int *) malloc (Matrix->SMPkluMatrix->KLUmatrixN * Matrix->SMPkluMatrix->KLUmatrixN * sizeof(int)) ;
        Matrix->SMPkluMatrix->KLUmatrixTrashCOO = (double *) malloc (sizeof(double)) ;
        Matrix->SMPkluMatrix->KLUmatrixValueComplexCOOforCIDER = (double *) malloc (2 * Matrix->SMPkluMatrix->KLUmatrixN * Matrix->SMPkluMatrix->KLUmatrixN * sizeof(double)) ;

        /* Pre-set the values of Row and Col */
        for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixN * Matrix->SMPkluMatrix->KLUmatrixN ; i++) {
            Matrix->SMPkluMatrix->KLUmatrixRowCOOforCIDER [i] = -1 ;
            Matrix->SMPkluMatrix->KLUmatrixColCOOforCIDER [i] = -1 ;
        }

        return spOKAY ;
    } else {
        Matrix->SPmatrix = spCreate (size, (int)KLUmatrixIsComplex, &Error) ;
        return Error ;
    }
}
#endif

/*
 * SMPdestroy()
 */

void
SMPdestroy (SMPmatrix *Matrix)
{
    if (Matrix->CKTkluMODE)
    {
        klu_free_numeric (&(Matrix->SMPkluMatrix->KLUmatrixNumeric), Matrix->SMPkluMatrix->KLUmatrixCommon) ;
        klu_free_symbolic (&(Matrix->SMPkluMatrix->KLUmatrixSymbolic), Matrix->SMPkluMatrix->KLUmatrixCommon) ;
        free (Matrix->SMPkluMatrix->KLUmatrixAp) ;
        free (Matrix->SMPkluMatrix->KLUmatrixAi) ;
        free (Matrix->SMPkluMatrix->KLUmatrixAx) ;
        free (Matrix->SMPkluMatrix->KLUmatrixAxComplex) ;
        free (Matrix->SMPkluMatrix->KLUmatrixIntermediate) ;
        free (Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex) ;
        free (Matrix->SMPkluMatrix->KLUmatrixBindStructCOO) ;
        free (Matrix->SMPkluMatrix->KLUmatrixTrashCOO) ;
        Matrix->SMPkluMatrix->KLUmatrixAp = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixAi = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixAx = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixAxComplex = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixIntermediate = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixBindStructCOO = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixTrashCOO = NULL ;
    } else {
        spDestroy (Matrix->SPmatrix) ;
    }
}

#ifdef CIDER
void
SMPdestroyKLUforCIDER (SMPmatrix *Matrix)
{
    if (Matrix->CKTkluMODE)
    {
        klu_free_numeric (&(Matrix->SMPkluMatrix->KLUmatrixNumeric), Matrix->SMPkluMatrix->KLUmatrixCommon) ;
        klu_free_symbolic (&(Matrix->SMPkluMatrix->KLUmatrixSymbolic), Matrix->SMPkluMatrix->KLUmatrixCommon) ;
        free (Matrix->SMPkluMatrix->KLUmatrixAp) ;
        free (Matrix->SMPkluMatrix->KLUmatrixAi) ;
        free (Matrix->SMPkluMatrix->KLUmatrixAxComplex) ;
        free (Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex) ;
        free (Matrix->SMPkluMatrix->KLUmatrixBindStructForCIDER) ;
        free (Matrix->SMPkluMatrix->KLUmatrixColCOOforCIDER) ;
        free (Matrix->SMPkluMatrix->KLUmatrixRowCOOforCIDER) ;
        free (Matrix->SMPkluMatrix->KLUmatrixValueComplexCOOforCIDER) ;
        free (Matrix->SMPkluMatrix->KLUmatrixTrashCOO) ;
        Matrix->SMPkluMatrix->KLUmatrixAp = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixAi = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixAxComplex = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixIntermediateComplex = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixBindStructForCIDER = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixColCOOforCIDER = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixRowCOOforCIDER = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixValueComplexCOOforCIDER = NULL ;
        Matrix->SMPkluMatrix->KLUmatrixTrashCOO = NULL ;
    } else {
        spDestroy (Matrix->SPmatrix) ;
    }
}
#endif

/*
 * SMPpreOrder()
 */

int
SMPpreOrder (SMPmatrix *Matrix)
{
    if (Matrix->CKTkluMODE)
    {
        Matrix->SMPkluMatrix->KLUmatrixSymbolic = klu_analyze ((int)Matrix->SMPkluMatrix->KLUmatrixN, Matrix->SMPkluMatrix->KLUmatrixAp,
                                                               Matrix->SMPkluMatrix->KLUmatrixAi, Matrix->SMPkluMatrix->KLUmatrixCommon) ;

        if (Matrix->SMPkluMatrix->KLUmatrixSymbolic == NULL)
        {
            fprintf (stderr, "Error (PreOrder): KLUsymbolic object is NULL. A problem occurred\n") ;
            if (Matrix->SMPkluMatrix->KLUmatrixCommon->status == KLU_EMPTY_MATRIX)
            {
                return 0 ;
            }
            return 1 ;
        } else {
            return 0 ;
        }
    } else {
        spMNA_Preorder (Matrix->SPmatrix) ;
        return spError (Matrix->SPmatrix) ;
    }
}

/*
 * SMPprintRHS()
 */

void
SMPprintRHS (SMPmatrix *Matrix, char *Filename, RealVector RHS, RealVector iRHS)
{
    if (!Matrix->CKTkluMODE)
        spFileVector (Matrix->SPmatrix, Filename, RHS, iRHS) ;
}

/*
 * SMPprint()
 */

void
SMPprint (SMPmatrix *Matrix, char *Filename)
{
    if (Matrix->CKTkluMODE)
    {
        if (Matrix->SMPkluMatrix->KLUmatrixIsComplex)
        {
            klu_z_print (Matrix->SMPkluMatrix->KLUmatrixAp, Matrix->SMPkluMatrix->KLUmatrixAi, Matrix->SMPkluMatrix->KLUmatrixAxComplex,
                         (int)Matrix->SMPkluMatrix->KLUmatrixN, NULL, NULL) ;
        } else {
            klu_print (Matrix->SMPkluMatrix->KLUmatrixAp, Matrix->SMPkluMatrix->KLUmatrixAi, Matrix->SMPkluMatrix->KLUmatrixAx,
                       (int)Matrix->SMPkluMatrix->KLUmatrixN, NULL, NULL) ;
        }
    } else {
        if (Filename)
            spFileMatrix (Matrix->SPmatrix, Filename, "Circuit Matrix", 0, 1, 1) ;
        else
            spPrint (Matrix->SPmatrix, 0, 1, 1) ;
    }
}

#ifdef CIDER
void
SMPprintKLUforCIDER (SMPmatrix *Matrix, char *Filename)
{
    unsigned int i ;
    double *KLUmatrixAx ;

    if (Matrix->CKTkluMODE)
    {
        if (Matrix->SMPkluMatrix->KLUmatrixIsComplex)
        {
            klu_z_print (Matrix->SMPkluMatrix->KLUmatrixAp, Matrix->SMPkluMatrix->KLUmatrixAi, Matrix->SMPkluMatrix->KLUmatrixAxComplex,
                         (int)Matrix->SMPkluMatrix->KLUmatrixN, NULL, NULL) ;
        } else {
            /* Allocate the Real Matrix */
            KLUmatrixAx = (double *) malloc (Matrix->SMPkluMatrix->KLUmatrixNZ * sizeof(double)) ;

            /* Copy the Complex Matrix into the Real Matrix */
            for (i = 0 ; i < Matrix->SMPkluMatrix->KLUmatrixNZ ; i++) {
                KLUmatrixAx [i] = Matrix->SMPkluMatrix->KLUmatrixAxComplex [2 * i] ;
            }

            /* Print the Real Matrix */
            klu_print (Matrix->SMPkluMatrix->KLUmatrixAp, Matrix->SMPkluMatrix->KLUmatrixAi, KLUmatrixAx, (int)Matrix->SMPkluMatrix->KLUmatrixN, NULL, NULL) ;

            /* Free the Real Matrix Storage */
            free (KLUmatrixAx) ;
        }
    } else {
        if (Filename)
            spFileMatrix (Matrix->SPmatrix, Filename, "Circuit Matrix", 0, 1, 1) ;
        else
            spPrint (Matrix->SPmatrix, 0, 1, 1) ;
    }
}
#endif

/*
 * SMPgetError()
 */
void
SMPgetError (SMPmatrix *Matrix, int *Col, int *Row)
{
    if (Matrix->CKTkluMODE)
    {
        *Row = Matrix->SMPkluMatrix->KLUmatrixCommon->singular_col + 1 ;
        *Col = Matrix->SMPkluMatrix->KLUmatrixCommon->singular_col + 1 ;
    } else {
        spWhereSingular (Matrix->SPmatrix, Row, Col) ;
    }
}

void
spDeterminant_KLU (SMPmatrix *Matrix, int *pExponent, RealNumber *pDeterminant, RealNumber *piDeterminant)
{
    int I, Size ;
    RealNumber Norm, nr, ni ;
    ComplexNumber Pivot, cDeterminant, Udiag ;

    int *P, *Q ;
    double *Rs, *Ux, *Uz ;
    unsigned int nSwap, nSwapP, nSwapQ ;

#define  NORM(a)     (nr = ABS((a).Real), ni = ABS((a).Imag), MAX (nr,ni))

    *pExponent = 0 ;

    if (Matrix->SMPkluMatrix->KLUmatrixCommon->status == KLU_SINGULAR)
    {
	*pDeterminant = 0.0 ;
        if (Matrix->SMPkluMatrix->KLUmatrixIsComplex == KLUMatrixComplex)
        {
            *piDeterminant = 0.0 ;
        }
        return ;
    }

    Size = (int)Matrix->SMPkluMatrix->KLUmatrixN ;
    I = 0 ;

    P = (int *) malloc ((size_t)Matrix->SMPkluMatrix->KLUmatrixN * sizeof (int)) ;
    Q = (int *) malloc ((size_t)Matrix->SMPkluMatrix->KLUmatrixN * sizeof (int)) ;

    Ux = (double *) malloc ((size_t)Matrix->SMPkluMatrix->KLUmatrixN * sizeof (double)) ;

    Rs = (double *) malloc ((size_t)Matrix->SMPkluMatrix->KLUmatrixN * sizeof (double)) ;

    if (Matrix->SMPkluMatrix->KLUmatrixIsComplex == KLUMatrixComplex)        /* Complex Case. */
    {
	cDeterminant.Real = 1.0 ;
        cDeterminant.Imag = 0.0 ;

        Uz = (double *) malloc ((size_t)Matrix->SMPkluMatrix->KLUmatrixN * sizeof (double)) ;
/*
        int *Lp, *Li, *Up, *Ui, *Fp, *Fi, *P, *Q ;
        double *Lx, *Lz, *Ux, *Uz, *Fx, *Fz, *Rs ;
        Lp = (int *) malloc (((size_t)Matrix->CKTkluN + 1) * sizeof (int)) ;
        Li = (int *) malloc ((size_t)Matrix->CKTkluNumeric->lnz * sizeof (int)) ;
        Lx = (double *) malloc ((size_t)Matrix->CKTkluNumeric->lnz * sizeof (double)) ;
        Lz = (double *) malloc ((size_t)Matrix->CKTkluNumeric->lnz * sizeof (double)) ;
        Up = (int *) malloc (((size_t)Matrix->CKTkluN + 1) * sizeof (int)) ;
        Ui = (int *) malloc ((size_t)Matrix->CKTkluNumeric->unz * sizeof (int)) ;
        Ux = (double *) malloc ((size_t)Matrix->CKTkluNumeric->unz * sizeof (double)) ;
        Uz = (double *) malloc ((size_t)Matrix->CKTkluNumeric->unz * sizeof (double)) ;
        Fp = (int *) malloc (((size_t)Matrix->CKTkluN + 1) * sizeof (int)) ;
        Fi = (int *) malloc ((size_t)Matrix->CKTkluNumeric->Offp [Matrix->CKTkluN] * sizeof (int)) ;
        Fx = (double *) malloc ((size_t)Matrix->CKTkluNumeric->Offp [Matrix->CKTkluN] * sizeof (double)) ;
        Fz = (double *) malloc ((size_t)Matrix->CKTkluNumeric->Offp [Matrix->CKTkluN] * sizeof (double)) ;
        klu_z_extract (Matrix->CKTkluNumeric, Matrix->CKTkluSymbolic,
                       Lp, Li, Lx, Lz,
                       Up, Ui, Ux, Uz,
                       Fp, Fi, Fx, Fz,
                       P, Q, Rs, NULL,
                       Matrix->CKTkluCommon) ;
*/
        klu_z_extract_Udiag (Matrix->SMPkluMatrix->KLUmatrixNumeric, Matrix->SMPkluMatrix->KLUmatrixSymbolic, Ux, Uz, P, Q, Rs, Matrix->SMPkluMatrix->KLUmatrixCommon) ;
/*
        for (I = 0 ; I < Matrix->CKTkluNumeric->lnz ; I++)
        {
            printf ("L - Value: %-.9g\t%-.9g\n", Lx [I], Lz [I]) ;
        }
        for (I = 0 ; I < Matrix->CKTkluNumeric->unz ; I++)
        {
            printf ("U - Value: %-.9g\t%-.9g\n", Ux [I], Uz [I]) ;
        }
        for (I = 0 ; I < Matrix->CKTkluNumeric->Offp [Matrix->CKTkluN] ; I++)
        {
            printf ("F - Value: %-.9g\t%-.9g\n", Fx [I], Fz [I]) ;
        }

        for (I = 0 ; I < Matrix->CKTkluN ; I++)
        {
            printf ("U - Value: %-.9g\t%-.9g\n", Ux [I], Uz [I]) ;
        }
*/
        nSwapP = 0 ;
        for (I = 0 ; I < (int)Matrix->SMPkluMatrix->KLUmatrixN ; I++)
        {
            if (P [I] != I)
            {
                nSwapP++ ;
            }
        }
        nSwapP /= 2 ;

        nSwapQ = 0 ;
        for (I = 0 ; I < (int)Matrix->SMPkluMatrix->KLUmatrixN ; I++)
        {
            if (Q [I] != I)
            {
                nSwapQ++ ;
            }
        }
        nSwapQ /= 2 ;

        nSwap = nSwapP + nSwapQ ;
/*
        free (Lp) ;
        free (Li) ;
        free (Lx) ;
        free (Lz) ;
        free (Up) ;
        free (Ui) ;
        free (Fp) ;
        free (Fi) ;
        free (Fx) ;
        free (Fz) ;
*/
        I = 0 ;
        while (I < Size)
        {
            Udiag.Real = 1 / (Ux [I] * Rs [I]) ;
            Udiag.Imag = Uz [I] * Rs [I] ;

//            printf ("Udiag.Real: %-.9g\tUdiag.Imag %-.9g\n", Udiag.Real, Udiag.Imag) ;

            CMPLX_RECIPROCAL (Pivot, Udiag) ;
            CMPLX_MULT_ASSIGN (cDeterminant, Pivot) ;

//            printf ("cDeterminant.Real: %-.9g\tcDeterminant.Imag %-.9g\n", cDeterminant.Real, cDeterminant.Imag) ;

	    /* Scale Determinant. */
            Norm = NORM (cDeterminant) ;
            if (Norm != 0.0)
            {
		while (Norm >= 1.0e12)
                {
		    cDeterminant.Real *= 1.0e-12 ;
                    cDeterminant.Imag *= 1.0e-12 ;
                    *pExponent += 12 ;
                    Norm = NORM (cDeterminant) ;
                }
                while (Norm < 1.0e-12)
                {
		    cDeterminant.Real *= 1.0e12 ;
                    cDeterminant.Imag *= 1.0e12 ;
                    *pExponent -= 12 ;
                    Norm = NORM (cDeterminant) ;
                }
            }
            I++ ;
        }

	/* Scale Determinant again, this time to be between 1.0 <= x < 10.0. */
        Norm = NORM (cDeterminant) ;
        if (Norm != 0.0)
        {
	    while (Norm >= 10.0)
            {
		cDeterminant.Real *= 0.1 ;
                cDeterminant.Imag *= 0.1 ;
                (*pExponent)++ ;
                Norm = NORM (cDeterminant) ;
            }
            while (Norm < 1.0)
            {
		cDeterminant.Real *= 10.0 ;
                cDeterminant.Imag *= 10.0 ;
                (*pExponent)-- ;
                Norm = NORM (cDeterminant) ;
            }
        }
        if (nSwap % 2 != 0)
        {
            CMPLX_NEGATE (cDeterminant) ;
        }

        *pDeterminant = cDeterminant.Real ;
        *piDeterminant = cDeterminant.Imag ;

        free (Uz) ;
    }
    else
    {
	/* Real Case. */
        *pDeterminant = 1.0 ;

        klu_extract_Udiag (Matrix->SMPkluMatrix->KLUmatrixNumeric, Matrix->SMPkluMatrix->KLUmatrixSymbolic, Ux, P, Q, Rs, Matrix->SMPkluMatrix->KLUmatrixCommon) ;

        nSwapP = 0 ;
        for (I = 0 ; I < (int)Matrix->SMPkluMatrix->KLUmatrixN ; I++)
        {
            if (P [I] != I)
            {
                nSwapP++ ;
            }
        }
        nSwapP /= 2 ;

        nSwapQ = 0 ;
        for (I = 0 ; I < (int)Matrix->SMPkluMatrix->KLUmatrixN ; I++)
        {
            if (Q [I] != I)
            {
                nSwapQ++ ;
            }
        }
        nSwapQ /= 2 ;

        nSwap = nSwapP + nSwapQ ;

        while (I < Size)
        {
            *pDeterminant /= (Ux [I] * Rs [I]) ;

	    /* Scale Determinant. */
            if (*pDeterminant != 0.0)
            {
		while (ABS(*pDeterminant) >= 1.0e12)
                {
		    *pDeterminant *= 1.0e-12 ;
                    *pExponent += 12 ;
                }
                while (ABS(*pDeterminant) < 1.0e-12)
                {
		    *pDeterminant *= 1.0e12 ;
                    *pExponent -= 12 ;
                }
            }
            I++ ;
        }

	/* Scale Determinant again, this time to be between 1.0 <= x <
           10.0. */
        if (*pDeterminant != 0.0)
        {
	    while (ABS(*pDeterminant) >= 10.0)
            {
		*pDeterminant *= 0.1 ;
                (*pExponent)++ ;
            }
            while (ABS(*pDeterminant) < 1.0)
            {
		*pDeterminant *= 10.0 ;
                (*pExponent)-- ;
            }
        }
        if (nSwap % 2 != 0)
        {
            *pDeterminant = -*pDeterminant ;
        }
    }

    free (P) ;
    free (Q) ;
    free (Ux) ;
    free (Rs) ;
}

/*
 * SMPcProdDiag()
 *    note: obsolete for Spice3d2 and later
 */
int
SMPcProdDiag (SMPmatrix *Matrix, SPcomplex *pMantissa, int *pExponent)
{
    if (Matrix->CKTkluMODE)
    {
        spDeterminant_KLU (Matrix, pExponent, &(pMantissa->real), &(pMantissa->imag)) ;
    } else {
        spDeterminant (Matrix->SPmatrix, pExponent, &(pMantissa->real), &(pMantissa->imag)) ;
    }
    return spError (Matrix->SPmatrix) ;
}

/*
 * SMPcDProd()
 */
int
SMPcDProd (SMPmatrix *Matrix, SPcomplex *pMantissa, int *pExponent)
{
    double	re, im, x, y, z;
    int		p;

    if (Matrix->CKTkluMODE)
    {
        spDeterminant_KLU (Matrix, &p, &re, &im) ;
    } else {
        spDeterminant (Matrix->SPmatrix, &p, &re, &im) ;
    }

#ifndef M_LN2
#define M_LN2   0.69314718055994530942
#endif
#ifndef M_LN10
#define M_LN10  2.30258509299404568402
#endif

#ifdef debug_print
    printf ("Determinant 10: (%20g,%20g)^%d\n", re, im, p) ;
#endif

    /* Convert base 10 numbers to base 2 numbers, for comparison */
    y = p * M_LN10 / M_LN2;
    x = (int) y;
    y -= x;

    /* ASSERT
     *	x = integral part of exponent, y = fraction part of exponent
     */

    /* Fold in the fractional part */
#ifdef debug_print
    printf (" ** base10 -> base2 int =  %g, frac = %20g\n", x, y) ;
#endif
    z = pow (2.0, y) ;
    re *= z ;
    im *= z ;
#ifdef debug_print
    printf (" ** multiplier = %20g\n", z) ;
#endif

    /* Re-normalize (re or im may be > 2.0 or both < 1.0 */
    if (re != 0.0)
    {
	y = logb (re) ;
	if (im != 0.0)
	    z = logb (im) ;
	else
	    z = 0 ;
    } else if (im != 0.0) {
	z = logb (im) ;
	y = 0 ;
    } else {
	/* Singular */
	/*printf("10 -> singular\n");*/
	y = 0 ;
	z = 0 ;
    }

#ifdef debug_print
    printf (" ** renormalize changes = %g,%g\n", y, z) ;
#endif
    if (y < z)
	y = z ;

    *pExponent = (int)(x + y) ;
    x = scalbn (re, (int) -y) ;
    z = scalbn (im, (int) -y) ;
#ifdef debug_print
    printf (" ** values are: re %g, im %g, y %g, re' %g, im' %g\n", re, im, y, x, z) ;
#endif
    pMantissa->real = scalbn (re, (int) -y) ;
    pMantissa->imag = scalbn (im, (int) -y) ;

#ifdef debug_print
    printf ("Determinant 10->2: (%20g,%20g)^%d\n", pMantissa->real, pMantissa->imag, *pExponent) ;
#endif

    if (Matrix->CKTkluMODE)
    {
        return 0 ;
    } else {
        return spError (Matrix->SPmatrix) ;
    }
}



/*
 *  The following routines need internal knowledge of the Sparse data
 *  structures.
 */

/*
 *  LOAD GMIN
 *
 *  This routine adds Gmin to each diagonal element.  Because Gmin is
 *  added to the current diagonal, which may bear little relation to
 *  what the outside world thinks is a diagonal, and because the
 *  elements that are diagonals may change after calling spOrderAndFactor,
 *  use of this routine is not recommended.  It is included here simply
 *  for compatibility with Spice3.
 */


static void
LoadGmin_CSC (double **diag, unsigned int n, double Gmin)
{
    unsigned int i ;

    if (Gmin != 0.0) {
        for (i = 0 ; i < n ; i++) {
            if (diag [i] != NULL) {
                // Not all the elements on the diagonal are present, when the circuit is parsed
                *(diag [i]) += Gmin ;
            }
        }
    }
}

static void
LoadGmin (SMPmatrix *eMatrix, double Gmin)
{
    MatrixPtr Matrix = eMatrix->SPmatrix ;
    int I ;
    ArrayOfElementPtrs Diag ;
    ElementPtr diag ;

    /* Begin `LoadGmin'. */
    assert (IS_SPARSE (Matrix)) ;

    if (Gmin != 0.0) {
	Diag = Matrix->Diag ;
	for (I = Matrix->Size ; I > 0 ; I--)
        {
	    if ((diag = Diag [I]) != NULL)
		diag->Real += Gmin ;
	}
    }
    return ;
}




/*
 *  FIND ELEMENT
 *
 *  This routine finds an element in the matrix by row and column number.
 *  If the element exists, a pointer to it is returned.  If not, then NULL
 *  is returned unless the CreateIfMissing flag is TRUE, in which case a
 *  pointer to the new element is returned.
 */

SMPelement *
SMPfindElt (SMPmatrix *eMatrix, int Row, int Col, int CreateIfMissing)
{
    MatrixPtr Matrix = eMatrix->SPmatrix ;

    if (eMatrix->CKTkluMODE)
    {
        int i ;

        Row = Row - 1 ;
        Col = Col - 1 ;
        if ((Row < 0) || (Col < 0)) {
            printf ("Information: Cannot find an element with row '%d' and column '%d' in the KLU matrix\n", Row, Col) ;
            return NULL ;
        }
        for (i = eMatrix->SMPkluMatrix->KLUmatrixAp [Col] ; i < eMatrix->SMPkluMatrix->KLUmatrixAp [Col + 1] ; i++) {
            if (eMatrix->SMPkluMatrix->KLUmatrixAi [i] == Row) {
                if (eMatrix->SMPkluMatrix->KLUmatrixIsComplex == KLUmatrixReal) {
                    return (SMPelement *) &(eMatrix->SMPkluMatrix->KLUmatrixAx [i]) ;
                } else if (eMatrix->SMPkluMatrix->KLUmatrixIsComplex == KLUMatrixComplex) {
                    return (SMPelement *) &(eMatrix->SMPkluMatrix->KLUmatrixAxComplex [2 * i]) ;
                } else {
                    return NULL ;
                }
            }
        }
        return NULL ;
    } else {
        ElementPtr Element ;

        /* Begin `SMPfindElt'. */
        assert (IS_SPARSE (Matrix)) ;
        Row = Matrix->ExtToIntRowMap [Row] ;
        Col = Matrix->ExtToIntColMap [Col] ;
        Element = Matrix->FirstInCol [Col] ;
        Element = spcFindElementInCol (Matrix, &Element, Row, Col, CreateIfMissing) ;
        return (SMPelement *)Element ;
    }
}

/* XXX The following should probably be implemented in spUtils */

/*
 * SMPcZeroCol()
 */
int
SMPcZeroCol (SMPmatrix *eMatrix, int Col)
{
    MatrixPtr Matrix = eMatrix->SPmatrix ;
    ElementPtr	Element ;

    if (eMatrix->CKTkluMODE)
    {
        int i ;
        for (i = eMatrix->SMPkluMatrix->KLUmatrixAp [Col - 1] ; i < eMatrix->SMPkluMatrix->KLUmatrixAp [Col] ; i++)
        {
            eMatrix->SMPkluMatrix->KLUmatrixAxComplex [2 * i] = 0 ;
            eMatrix->SMPkluMatrix->KLUmatrixAxComplex [2 * i + 1] = 0 ;
        }
        return 0 ;
    } else {
        Col = Matrix->ExtToIntColMap [Col] ;
        for (Element = Matrix->FirstInCol [Col] ; Element != NULL ; Element = Element->NextInCol)
        {
            Element->Real = 0.0 ;
            Element->Imag = 0.0 ;
        }
        return spError (Matrix) ;
    }
}

/*
 * SMPcAddCol()
 */
int
SMPcAddCol (SMPmatrix *eMatrix, int Accum_Col, int Addend_Col)
{
    MatrixPtr Matrix = eMatrix->SPmatrix ;
    ElementPtr	Accum, Addend, *Prev ;

    Accum_Col = Matrix->ExtToIntColMap [Accum_Col] ;
    Addend_Col = Matrix->ExtToIntColMap [Addend_Col] ;

    Addend = Matrix->FirstInCol [Addend_Col] ;
    Prev = &Matrix->FirstInCol [Accum_Col] ;
    Accum = *Prev;

    while (Addend != NULL)
    {
	while (Accum && Accum->Row < Addend->Row)
        {
	    Prev = &Accum->NextInCol ;
	    Accum = *Prev ;
	}
	if (!Accum || Accum->Row > Addend->Row)
        {
	    Accum = spcCreateElement (Matrix, Addend->Row, Accum_Col, Prev, 0) ;
	}
	Accum->Real += Addend->Real ;
	Accum->Imag += Addend->Imag ;
	Addend = Addend->NextInCol ;
    }

    return spError (Matrix) ;
}

/*
 * SMPzeroRow()
 */
int
SMPzeroRow (SMPmatrix *eMatrix, int Row)
{
    MatrixPtr Matrix = eMatrix->SPmatrix ;
    ElementPtr	Element ;

    Row = Matrix->ExtToIntColMap [Row] ;

    if (Matrix->RowsLinked == NO)
	spcLinkRows (Matrix) ;

    if (Matrix->PreviousMatrixWasComplex || Matrix->Complex)
    {
	for (Element = Matrix->FirstInRow[Row] ; Element != NULL; Element = Element->NextInRow)
	{
	    Element->Real = 0.0 ;
	    Element->Imag = 0.0 ;
	}
    } else {
	for (Element = Matrix->FirstInRow [Row] ; Element != NULL ; Element = Element->NextInRow)
	{
	    Element->Real = 0.0 ;
	}
    }

    return spError (Matrix) ;
}

/*
 * SMPconstMult()
 */
void
SMPconstMult (SMPmatrix *Matrix, double constant)
{
    if (Matrix->CKTkluMODE)
    {
        if (Matrix->SMPkluMatrix->KLUmatrixIsComplex)
        {
            klu_z_constant_multiply (Matrix->SMPkluMatrix->KLUmatrixAp, Matrix->SMPkluMatrix->KLUmatrixAxComplex,
                                     (int)Matrix->SMPkluMatrix->KLUmatrixN, Matrix->SMPkluMatrix->KLUmatrixCommon, constant) ;
        } else {
            klu_constant_multiply (Matrix->SMPkluMatrix->KLUmatrixAp, Matrix->SMPkluMatrix->KLUmatrixAx,
                                   (int)Matrix->SMPkluMatrix->KLUmatrixN, Matrix->SMPkluMatrix->KLUmatrixCommon, constant) ;
        }
    } else {
        spConstMult (Matrix->SPmatrix, constant) ;
    }
}

/*
 * SMPmultiply()
 */
void
SMPmultiply (SMPmatrix *Matrix, double *RHS, double *Solution, double *iRHS, double *iSolution)
{
    if (Matrix->CKTkluMODE)
    {
        int *Ap_CSR, *Ai_CSR ;
        double *Ax_CSR ;

        Ap_CSR = (int *) malloc ((size_t)(Matrix->SMPkluMatrix->KLUmatrixN + 1) * sizeof (int)) ;
        Ai_CSR = (int *) malloc ((size_t)Matrix->SMPkluMatrix->KLUmatrixNZ * sizeof (int)) ;

        if (Matrix->SMPkluMatrix->KLUmatrixIsComplex)
        {
            Ax_CSR = (double *) malloc ((size_t)(2 * Matrix->SMPkluMatrix->KLUmatrixNZ) * sizeof (double)) ;
            klu_z_convert_matrix_in_CSR (Matrix->SMPkluMatrix->KLUmatrixAp, Matrix->SMPkluMatrix->KLUmatrixAi, Matrix->SMPkluMatrix->KLUmatrixAxComplex, Ap_CSR,
                                         Ai_CSR, Ax_CSR, (int)Matrix->SMPkluMatrix->KLUmatrixN, (int)Matrix->SMPkluMatrix->KLUmatrixNZ, Matrix->SMPkluMatrix->KLUmatrixCommon) ;
            klu_z_matrix_vector_multiply (Ap_CSR, Ai_CSR, Ax_CSR, RHS, Solution, iRHS, iSolution, NULL, NULL,
                                          (int)Matrix->SMPkluMatrix->KLUmatrixN, Matrix->SMPkluMatrix->KLUmatrixCommon) ;
        } else {
            Ax_CSR = (double *) malloc ((size_t)Matrix->SMPkluMatrix->KLUmatrixNZ * sizeof (double)) ;
            klu_convert_matrix_in_CSR (Matrix->SMPkluMatrix->KLUmatrixAp, Matrix->SMPkluMatrix->KLUmatrixAi, Matrix->SMPkluMatrix->KLUmatrixAx, Ap_CSR, Ai_CSR,
                                       Ax_CSR, (int)Matrix->SMPkluMatrix->KLUmatrixN, (int)Matrix->SMPkluMatrix->KLUmatrixNZ, Matrix->SMPkluMatrix->KLUmatrixCommon) ;
            klu_matrix_vector_multiply (Ap_CSR, Ai_CSR, Ax_CSR, RHS, Solution, NULL, NULL,
                                        (int)Matrix->SMPkluMatrix->KLUmatrixN, Matrix->SMPkluMatrix->KLUmatrixCommon) ;
            iSolution = iRHS ;
        }

        free (Ap_CSR) ;
        free (Ai_CSR) ;
        free (Ax_CSR) ;
    } else {
        spMultiply (Matrix->SPmatrix, RHS, Solution, iRHS, iSolution) ;
    }
}

