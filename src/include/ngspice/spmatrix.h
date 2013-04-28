/*
 *  EXPORTS for sparse matrix routines with SPICE3.
 *
 *  Author:                     Advising professor:
 *      Kenneth S. Kundert          Alberto Sangiovanni-Vincentelli
 *      UC Berkeley
 *
 *  This file contains definitions that are useful to the calling
 *  program.  In particular, this file contains error keyword
 *  definitions, some macro functions that are used to quickly enter
 *  data into the matrix and the type definition of a data structure
 *  that acts as a template for entering admittances into the matrix.
 *  Also included is the type definitions for the various functions
 *  available to the user.
 *
 *  This file is a modified version of spMatrix.h that is used when
 *  interfacing to Spice3.
 */


/*
 *  Revision and copyright information.
 *
 *  Copyright (c) 1985,86,87,88,89,90
 *  by Kenneth S. Kundert and the University of California.
 *
 *  Permission to use, copy, modify, and distribute this software and
 *  its documentation for any purpose and without fee is hereby granted,
 *  provided that the copyright notices appear in all copies and
 *  supporting documentation and that the authors and the University of
 *  California are properly credited.  The authors and the University of
 *  California make no representations as to the suitability of this
 *  software for any purpose.  It is provided `as is', without express
 *  or implied warranty.
 */




#ifndef  spOKAY





/*
 *  ERROR KEYWORDS
 *
 *  The actual numbers used in the error codes are not sacred, they can be
 *  changed under the condition that the codes for the nonfatal errors are
 *  less than the code for spFATAL and similarly the codes for the fatal
 *  errors are greater than that for spFATAL.
 *
 *  >>> Error descriptions:
 *  spOKAY
 *      No error has occurred.
 *  spSMALL_PIVOT
 *      When reordering the matrix, no element was found which satisfies the
 *      threshold criteria.  The largest element in the matrix was chosen
 *      as pivot.  Non-fatal.
 *  spZERO_DIAG
 *      Fatal error.  A zero was encountered on the diagonal the matrix.  This
 *      does not necessarily imply that the matrix is singular.  When this
 *      error occurs, the matrix should be reconstructed and factored using
 *      spOrderAndFactor().
 *  spSINGULAR
 *      Fatal error.  Matrix is singular, so no unique solution exists.
 *  spNO_MEMORY
 *      Fatal error.  Indicates that not enough memory is available to handle
 *      the matrix.
 *  spPANIC
 *      Fatal error indicating that the routines are not prepared to
 *      handle the matrix that has been requested.  This may occur when
 *      the matrix is specified to be real and the routines are not
 *      compiled for real matrices, or when the matrix is specified to
 *      be complex and the routines are not compiled to handle complex
 *      matrices.
 *  spFATAL
 *      Not an error flag, but rather the dividing line between fatal errors
 *      and warnings.
 */

#include "ngspice/sperror.h"  /* Spice error definitions. */

/* Begin error macros. */
#define  spOKAY                 OK
#define  spSMALL_PIVOT          OK
#define  spZERO_DIAG            E_SINGULAR
#define  spSINGULAR             E_SINGULAR
#define  spNO_MEMORY            E_NOMEM
#define  spPANIC                E_BADMATRIX

#define  spFATAL                E_BADMATRIX






/*
 *  KEYWORD DEFINITIONS
 *
 *  Here we define what precision arithmetic Sparse will use.  Double
 *  precision is suggested as being most appropriate for circuit
 *  simulation and for C.  However, it is possible to change spREAL
 *  to a float for single precision arithmetic.  Note that in C, single
 *  precision arithmetic is often slower than double precision.  Sparse
 *  internally refers to spREALs as RealNumbers.
 *
 *  Some C compilers, notably the old VMS compiler, do not handle the keyword
 *  "void" correctly.  If this is true for your compiler, remove the
 *  comment delimiters from the redefinition of void to int below.
 */

#define  spREAL double
/* #define  void    int   */



/*
 *  PARTITION TYPES
 *
 *  When factoring a previously ordered matrix using spFactor(), Sparse
 *  operates on a row-at-a-time basis.  For speed, on each step, the row
 *  being updated is copied into a full vector and the operations are
 *  performed on that vector.  This can be done one of two ways, either
 *  using direct addressing or indirect addressing.  Direct addressing
 *  is fastest when the matrix is relatively dense and indirect addressing
 *  is quite sparse.  The user can select which partitioning mode is used.
 *  The following keywords are passed to spPartition() and indicate that
 *  Sparse should use only direct addressing, only indirect addressing, or
 *  that it should choose the best mode on a row-by-row basis.  The time
 *  required to choose a partition is of the same order of the cost to factor
 *  the matrix.
 *
 *  If you plan to factor a large number of matrices with the same structure,
 *  it is best to let Sparse choose the partition.  Otherwise, you should
 *  choose the partition based on the predicted density of the matrix.
 */

/* Begin partition keywords. */

#define spDEFAULT_PARTITION     0
#define spDIRECT_PARTITION      1
#define spINDIRECT_PARTITION    2
#define spAUTO_PARTITION        3





/*
 *  MACRO FUNCTION DEFINITIONS
 *
 *  >>> Macro descriptions:
 *  spADD_REAL_ELEMENT
 *      Macro function that adds data to a real element in the matrix by a
 *      pointer.
 *  spADD_IMAG_ELEMENT
 *      Macro function that adds data to a imaginary element in the matrix by
 *      a pointer.
 *  spADD_COMPLEX_ELEMENT
 *      Macro function that adds data to a complex element in the matrix by a
 *      pointer.
 *  spADD_REAL_QUAD
 *      Macro function that adds data to each of the four real matrix elements
 *      specified by the given template.
 *  spADD_IMAG_QUAD
 *      Macro function that adds data to each of the four imaginary matrix
 *      elements specified by the given template.
 *  spADD_COMPLEX_QUAD
 *      Macro function that adds data to each of the four complex matrix
 *      elements specified by the given template.
 */

/* Begin Macros. */
#define  spADD_REAL_ELEMENT(element,real)       *(element) += real

#define  spADD_IMAG_ELEMENT(element,imag)       *(element+1) += imag

#define  spADD_COMPLEX_ELEMENT(element,real,imag)       \
{   *(element) += real;                                 \
    *(element+1) += imag;                               \
}

#define  spADD_REAL_QUAD(template,real)         \
{   *((template).Element1) += real;             \
    *((template).Element2) += real;             \
    *((template).Element3Negated) -= real;      \
    *((template).Element4Negated) -= real;      \
}

#define  spADD_IMAG_QUAD(template,imag)         \
{   *((template).Element1+1) += imag;           \
    *((template).Element2+1) += imag;           \
    *((template).Element3Negated+1) -= imag;    \
    *((template).Element4Negated+1) -= imag;    \
}

#define  spADD_COMPLEX_QUAD(template,real,imag) \
{   *((template).Element1) += real;             \
    *((template).Element2) += real;             \
    *((template).Element3Negated) -= real;      \
    *((template).Element4Negated) -= real;      \
    *((template).Element1+1) += imag;           \
    *((template).Element2+1) += imag;           \
    *((template).Element3Negated+1) -= imag;    \
    *((template).Element4Negated+1) -= imag;    \
}






/*
 *   TYPE DEFINITION FOR COMPONENT TEMPLATE
 *
 *   This data structure is used to hold pointers to four related elements in
 *   matrix.  It is used in conjunction with the routines
 *       spGetAdmittance
 *       spGetQuad
 *       spGetOnes
 *   These routines stuff the structure which is later used by the spADD_QUAD
 *   macro functions above.  It is also possible for the user to collect four
 *   pointers returned by spGetElement and stuff them into the template.
 *   The spADD_QUAD routines stuff data into the matrix in locations specified
 *   by Element1 and Element2 without changing the data.  The data is negated
 *   before being placed in Element3 and Element4.
 */

/* Begin `spTemplate'. */
struct  spTemplate
{   spREAL    *Element1       ;
    spREAL    *Element2       ;
    spREAL    *Element3Negated;
    spREAL    *Element4Negated;
};


typedef struct MatrixFrame *MatrixPtr;


/*
 *   FUNCTION TYPE DEFINITIONS
 *
 *   The type of every user accessible function is declared here.
 */

/* Begin function declarations. */

extern  void     spClear( MatrixPtr );
extern  spREAL   spCondition( MatrixPtr, spREAL, int* );
extern  MatrixPtr spCreate( int, int, int* );
extern  void     spDeleteRowAndCol( MatrixPtr, int, int );
extern  void     spDestroy( MatrixPtr);
extern  int      spElementCount( MatrixPtr );
extern  int      spError( MatrixPtr );
extern  int      spFactor( MatrixPtr );
extern  int      spFileMatrix( MatrixPtr, char *, char *, int, int, int );
extern  int      spFileStats( MatrixPtr, char *, char * );
extern  int      spFillinCount( MatrixPtr );
extern  int      spGetAdmittance( MatrixPtr, int, int, struct spTemplate* );
extern  spREAL  *spFindElement(MatrixPtr Matrix, int Row, int Col );
extern  spREAL  *spGetElement(MatrixPtr, int, int );
extern  void    *spGetInitInfo( spREAL* );
extern  int      spGetOnes( MatrixPtr, int, int, int, struct spTemplate* );
extern  int      spGetQuad( MatrixPtr, int, int, int, int, struct spTemplate* );
extern  int      spGetSize( MatrixPtr, int );
extern  int      spInitialize(MatrixPtr, int (*pInit)(spREAL*, void *InitInfo, int, int Col));
extern  void     spInstallInitInfo( spREAL*, void * );
extern  spREAL   spLargestElement( MatrixPtr );
extern  void     spMNA_Preorder( MatrixPtr );
extern  spREAL   spNorm( MatrixPtr );
extern  int      spOrderAndFactor(MatrixPtr, spREAL*, spREAL, spREAL, int );
extern  int      spOriginalCount( MatrixPtr);
extern  void     spPartition( MatrixPtr, int );
extern  void     spPrint(MatrixPtr, int, int, int );
extern  spREAL   spPseudoCondition( MatrixPtr );
extern  spREAL   spRoundoff( MatrixPtr, spREAL );
extern  void     spScale( MatrixPtr, spREAL*, spREAL* );
extern  void     spSetComplex( MatrixPtr );
extern  void     spSetReal( MatrixPtr );
extern  void     spStripFills( MatrixPtr );
extern  void     spWhereSingular(MatrixPtr, int*, int* );
extern  void     spConstMult(MatrixPtr, double);

/* Functions with argument lists that are dependent on options. */

extern  void     spDeterminant ( MatrixPtr, int*, spREAL*, spREAL* );
extern  int      spFileVector( MatrixPtr, char * , spREAL*, spREAL*);
extern  void     spMultiply( MatrixPtr, spREAL*, spREAL*, spREAL*, spREAL* );
extern  void     spMultTransposed(MatrixPtr,spREAL*,spREAL*,spREAL*,spREAL*);
extern  void     spSolve( MatrixPtr, spREAL*, spREAL*, spREAL*, spREAL* );
extern  void     spSolveTransposed(MatrixPtr,spREAL*,spREAL*,spREAL*,spREAL*);

#endif  /* spOKAY */
