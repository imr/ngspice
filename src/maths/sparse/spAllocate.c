/*
 *  MATRIX ALLOCATION MODULE
 *
 *  Author:                     Advising professor:
 *      Kenneth S. Kundert          Alberto Sangiovanni-Vincentelli
 *      UC Berkeley
 */
/*!\file
 *  This file contains functions for allocating and freeing matrices, configuring them, and for
 *  accessing global information about the matrix (size, error status, etc.).
 *
 *  Objects that begin with the \a spc prefix are considered private
 *  and should not be used.
 *
 * \author
 *  Kenneth S. Kundert <kundert@users.sourceforge.net>
 */
/*  >>> User accessible functions contained in this file:
 *  spCreate
 *  spDestroy
 *  spErrorState
 *  spWhereSingular
 *  spGetSize
 *  spSetReal
 *  spSetComplex
 *  spFillinCount
 *  spElementCount
 *
 *  >>> Other functions contained in this file:
 *  spcGetElement
 *  InitializeElementBlocks
 *  spcGetFillin
 *  RecordAllocation
 *  AllocateBlockOfAllocationList
 *  EnlargeMatrix
 *  ExpandTranslationArrays
 */


/*
 *  Revision and copyright information.
 *
 *  Copyright (c) 1985-2003 by Kenneth S. Kundert
 */



/*
 *  IMPORTS
 *
 *  >>> Import descriptions:
 *  spConfig.h
 *      Macros that customize the sparse matrix routines.
 *  spMatrix.h
 *      Macros and declarations to be imported by the user.
 *  spDefs.h
 *      Matrix type and macro definitions for the sparse matrix routines.
 */

#define spINSIDE_SPARSE
#include <stdio.h>
#include "spConfig.h"
#include "ngspice/spmatrix.h"
#include "spDefs.h"





/*
 * Global strings
 */

char spcMatrixIsNotValid[] = "Matrix passed to Sparse is not valid";
char spcErrorsMustBeCleared[] = "Error not cleared";
char spcMatrixMustBeFactored[] = "Matrix must be factored";
char spcMatrixMustNotBeFactored[] = "Matrix must not be factored";




/*
 *  Function declarations
 */

//static spError ReserveElements( MatrixPtr, int );
static void InitializeElementBlocks( MatrixPtr, int, int );
static void RecordAllocation( MatrixPtr, void* );
static void AllocateBlockOfAllocationList( MatrixPtr );



/*!
 *  Allocates and initializes the data structures associated with a matrix.
 *
 *  \return
 *  A pointer to the matrix is returned cast into \a spMatrix (typically a
 *  pointer to a void).  This pointer is then passed and used by the other
 *  matrix routines to refer to a particular matrix.  If an error occurs,
 *  the \a NULL pointer is returned.
 *
 *  \param Size
 *      Size of matrix or estimate of size of matrix if matrix is \a EXPANDABLE.
 *  \param Complex
 *      Type of matrix.  If \a Complex is 0 then the matrix is real, otherwise
 *      the matrix will be complex.  Note that if the routines are not set up
 *      to handle the type of matrix requested, then an \a spPANIC error will occur.
 *      Further note that if a matrix will be both real and complex, it must
 *      be specified here as being complex.
 *  \param pError
 *      Returns error flag, needed because function \a spErrorState() will
 *      not work correctly if \a spCreate() returns \a NULL. Possible errors
 *      include \a  spNO_MEMORY and \a spPANIC.
 */
/*  >>> Local variables:
 *  AllocatedSize  (int)
 *      The size of the matrix being allocated.
 *  Matrix  (MatrixPtr)
 *      A pointer to the matrix frame being created.
 */

MatrixPtr
spCreate(
    int  Size,
    int  Complex,
    spError *pError
)
{
register  unsigned  SizePlusOne;
register  MatrixPtr  Matrix;
register  int  I;
int  AllocatedSize;

/* Begin `spCreate'. */
/* Clear error flag. */
    *pError = spOKAY;

/* Test for valid size. */
    vASSERT( (Size >= 0) AND (Size != 0 OR EXPANDABLE), "Invalid size" );

/* Test for valid type. */
#if NOT spCOMPLEX
    ASSERT( NOT Complex );
#endif
#if NOT REAL
    ASSERT( Complex );
#endif

/* Create Matrix. */
    AllocatedSize = MAX( Size, MINIMUM_ALLOCATED_SIZE );
    SizePlusOne = (unsigned)(AllocatedSize + 1);

    if ((Matrix = SP_MALLOC(struct MatrixFrame, 1)) == NULL)
    {   *pError = spNO_MEMORY;
        return NULL;
    }

/* Initialize matrix */
    Matrix->ID = SPARSE_ID;
    Matrix->Complex = Complex;
    Matrix->PreviousMatrixWasComplex = Complex;
    Matrix->Factored = NO;
    Matrix->Elements = 0;
    Matrix->Error = *pError;
    Matrix->Fillins = 0;
    Matrix->Reordered = NO;
    Matrix->NeedsOrdering = YES;
    Matrix->NumberOfInterchangesIsOdd = NO;
    Matrix->Partitioned = NO;
    Matrix->RowsLinked = NO;
    Matrix->InternalVectorsAllocated = NO;
    Matrix->SingularCol = 0;
    Matrix->SingularRow = 0;
    Matrix->Size = Size;
    Matrix->AllocatedSize = AllocatedSize;
    Matrix->ExtSize = Size;
    Matrix->AllocatedExtSize = AllocatedSize;
    Matrix->CurrentSize = 0;
    Matrix->ExtToIntColMap = NULL;
    Matrix->ExtToIntRowMap = NULL;
    Matrix->IntToExtColMap = NULL;
    Matrix->IntToExtRowMap = NULL;
    Matrix->MarkowitzRow = NULL;
    Matrix->MarkowitzCol = NULL;
    Matrix->MarkowitzProd = NULL;
    Matrix->DoCmplxDirect = NULL;
    Matrix->DoRealDirect = NULL;
    Matrix->Intermediate = NULL;
    Matrix->RelThreshold = DEFAULT_THRESHOLD;
    Matrix->AbsThreshold = 0.0;

    Matrix->TopOfAllocationList = NULL;
    Matrix->RecordsRemaining = 0;
    Matrix->ElementsRemaining = 0;
    Matrix->FillinsRemaining = 0;

    RecordAllocation( Matrix, (void *)Matrix );
    if (Matrix->Error == spNO_MEMORY) goto MemoryError;

/* Take out the trash. */
    Matrix->TrashCan.Real = 0.0;
#if spCOMPLEX
    Matrix->TrashCan.Imag = 0.0;
#endif
    Matrix->TrashCan.Row = 0;
    Matrix->TrashCan.Col = 0;
    Matrix->TrashCan.NextInRow = NULL;
    Matrix->TrashCan.NextInCol = NULL;
#if INITIALIZE
    Matrix->TrashCan.pInitInfo = NULL;
#endif

/* Allocate space in memory for Diag pointer vector. */
    SP_CALLOC( Matrix->Diag, ElementPtr, SizePlusOne);
    if (Matrix->Diag == NULL)
        goto MemoryError;

/* Allocate space in memory for FirstInCol pointer vector. */
    SP_CALLOC( Matrix->FirstInCol, ElementPtr, SizePlusOne);
    if (Matrix->FirstInCol == NULL)
        goto MemoryError;

/* Allocate space in memory for FirstInRow pointer vector. */
    SP_CALLOC( Matrix->FirstInRow, ElementPtr, SizePlusOne);
    if (Matrix->FirstInRow == NULL)
        goto MemoryError;

/* Allocate space in memory for IntToExtColMap vector. */
    if (( Matrix->IntToExtColMap = SP_MALLOC(int, SizePlusOne)) == NULL)
        goto MemoryError;

/* Allocate space in memory for IntToExtRowMap vector. */
    if (( Matrix->IntToExtRowMap = SP_MALLOC(int, SizePlusOne)) == NULL)
        goto MemoryError;

/* Initialize MapIntToExt vectors. */
    for (I = 1; I <= AllocatedSize; I++)
    {   Matrix->IntToExtRowMap[I] = I;
        Matrix->IntToExtColMap[I] = I;
    }

#if TRANSLATE
/* Allocate space in memory for ExtToIntColMap vector. */
    if (( Matrix->ExtToIntColMap = SP_MALLOC(int, SizePlusOne)) == NULL)
        goto MemoryError;

/* Allocate space in memory for ExtToIntRowMap vector. */
    if (( Matrix->ExtToIntRowMap = SP_MALLOC(int, SizePlusOne)) == NULL)
        goto MemoryError;

/* Initialize MapExtToInt vectors. */
    for (I = 1; I <= AllocatedSize; I++)
    {   Matrix->ExtToIntColMap[I] = -1;
        Matrix->ExtToIntRowMap[I] = -1;
    }
    Matrix->ExtToIntColMap[0] = 0;
    Matrix->ExtToIntRowMap[0] = 0;
#endif

/* Allocate space for fill-ins and initial set of elements. */
    InitializeElementBlocks( Matrix, SPACE_FOR_ELEMENTS*AllocatedSize,
                                     SPACE_FOR_FILL_INS*AllocatedSize );
    if (Matrix->Error == spNO_MEMORY)
        goto MemoryError;

    return Matrix;

MemoryError:

/* Deallocate matrix and return no pointer to matrix if there is not enough
   memory. */
    *pError = spNO_MEMORY;
    spDestroy( Matrix );
    return NULL;
}









/*
 *  ELEMENT ALLOCATION
 *
 *  This routine allocates space for matrix elements. It requests large blocks
 *  of storage from the system and doles out individual elements as required.
 *  This technique, as opposed to allocating elements individually, tends to
 *  speed the allocation process.
 *
 *  >>> Returned:
 *  A pointer to an element.
 *
 *  >>> Arguments:
 *  Matrix  <input>  (MatrixPtr)
 *      Pointer to matrix.
 *
 *  >>> Local variables:
 *  pElement  (ElementPtr)
 *      A pointer to the first element in the group of elements being allocated.
 *
 *  >>> Possible errors:
 *  spNO_MEMORY
 */

ElementPtr
spcGetElement( MatrixPtr Matrix )
{
ElementPtr  pElement;

/* Begin `spcGetElement'. */

/* Allocate block of MatrixElements if necessary. */
    if (Matrix->ElementsRemaining == 0)
    {   pElement = SP_MALLOC(struct MatrixElement, ELEMENTS_PER_ALLOCATION);
        RecordAllocation( Matrix, (void *)pElement );
        if (Matrix->Error == spNO_MEMORY) return NULL;
        Matrix->ElementsRemaining = ELEMENTS_PER_ALLOCATION;
        Matrix->NextAvailElement = pElement;
    }

/* Update Element counter and return pointer to Element. */
    Matrix->ElementsRemaining--;
    return Matrix->NextAvailElement++;
}








/*
 *  ELEMENT ALLOCATION INITIALIZATION
 *
 *  This routine allocates space for matrix fill-ins and an initial set of
 *  elements.  Besides being faster than allocating space for elements one
 *  at a time, it tends to keep the fill-ins physically close to the other
 *  matrix elements in the computer memory.  This keeps virtual memory paging
 *  to a minimum.
 *
 *  >>> Arguments:
 *  Matrix  <input>    (MatrixPtr)
 *      Pointer to the matrix.
 *  InitialNumberOfElements  <input> (int)
 *      This number is used as the size of the block of memory, in
 *      MatrixElements, reserved for elements. If more than this number of
 *      elements are generated, then more space is allocated later.
 *  NumberOfFillinsExpected  <input> (int)
 *      This number is used as the size of the block of memory, in
 *      MatrixElements, reserved for fill-ins. If more than this number of
 *      fill-ins are generated, then more space is allocated, but they may
 *      not be physically close in computer's memory.
 *
 *  >>> Local variables:
 *  pElement  (ElementPtr)
 *      A pointer to the first element in the group of elements being allocated.
 *
 *  >>> Possible errors:
 *  spNO_MEMORY
 */

static void
InitializeElementBlocks(
    MatrixPtr Matrix,
    int InitialNumberOfElements,
    int NumberOfFillinsExpected
)
{
ElementPtr  pElement;

/* Begin `InitializeElementBlocks'. */

/* Allocate block of MatrixElements for elements. */
    pElement = SP_MALLOC(struct MatrixElement, InitialNumberOfElements);
    RecordAllocation( Matrix, (void *)pElement );
    if (Matrix->Error == spNO_MEMORY) return;
    Matrix->ElementsRemaining = InitialNumberOfElements;
    Matrix->NextAvailElement = pElement;

/* Allocate block of MatrixElements for fill-ins. */
    pElement = SP_MALLOC(struct MatrixElement, NumberOfFillinsExpected);
    RecordAllocation( Matrix, (void *)pElement );
    if (Matrix->Error == spNO_MEMORY) return;
    Matrix->FillinsRemaining = NumberOfFillinsExpected;
    Matrix->NextAvailFillin = pElement;

/* Allocate a fill-in list structure. */
    Matrix->FirstFillinListNode = SP_MALLOC(struct FillinListNodeStruct,1);
    RecordAllocation( Matrix, (void *)Matrix->FirstFillinListNode );
    if (Matrix->Error == spNO_MEMORY) return;
    Matrix->LastFillinListNode = Matrix->FirstFillinListNode;

    Matrix->FirstFillinListNode->pFillinList = pElement;
    Matrix->FirstFillinListNode->NumberOfFillinsInList =NumberOfFillinsExpected;
    Matrix->FirstFillinListNode->Next = NULL;

    return;
}










/*
 *  FILL-IN ALLOCATION
 *
 *  This routine allocates space for matrix fill-ins. It requests large blocks
 *  of storage from the system and doles out individual elements as required.
 *  This technique, as opposed to allocating elements individually, tends to
 *  speed the allocation process.
 *
 *  >>> Returned:
 *  A pointer to the fill-in.
 *
 *  >>> Arguments:
 *  Matrix  <input>  (MatrixPtr)
 *      Pointer to matrix.
 *
 *  >>> Possible errors:
 *  spNO_MEMORY
 */

ElementPtr
spcGetFillin( MatrixPtr Matrix )
{
#if STRIP OR LINT
struct FillinListNodeStruct *pListNode;
ElementPtr  pFillins;
#endif

/* Begin `spcGetFillin'. */

#if NOT STRIP OR LINT
    if (Matrix->FillinsRemaining == 0)
        return spcGetElement( Matrix );
#endif
#if STRIP OR LINT

    if (Matrix->FillinsRemaining == 0)
    {   pListNode = Matrix->LastFillinListNode;

/* First see if there are any stripped fill-ins left. */
        if (pListNode->Next != NULL)
        {   Matrix->LastFillinListNode = pListNode = pListNode->Next;
            Matrix->FillinsRemaining = pListNode->NumberOfFillinsInList;
            Matrix->NextAvailFillin = pListNode->pFillinList;
        }
        else
        {
/* Allocate block of fill-ins. */
            pFillins = SP_MALLOC(struct MatrixElement, ELEMENTS_PER_ALLOCATION);
            RecordAllocation( Matrix, (void *)pFillins );
            if (Matrix->Error == spNO_MEMORY) return NULL;
            Matrix->FillinsRemaining = ELEMENTS_PER_ALLOCATION;
            Matrix->NextAvailFillin = pFillins;

/* Allocate a fill-in list structure. */
            pListNode->Next = SP_MALLOC(struct FillinListNodeStruct,1);
            RecordAllocation( Matrix, (void *)pListNode->Next );
            if (Matrix->Error == spNO_MEMORY) return NULL;
            Matrix->LastFillinListNode = pListNode = pListNode->Next;

            pListNode->pFillinList = pFillins;
            pListNode->NumberOfFillinsInList = ELEMENTS_PER_ALLOCATION;
            pListNode->Next = NULL;
        }
    }
#endif

/* Update Fill-in counter and return pointer to Fill-in. */
    Matrix->FillinsRemaining--;
    return Matrix->NextAvailFillin++;
}









/*
 *  RECORD A MEMORY ALLOCATION
 *
 *  This routine is used to record all memory allocations so that the memory
 *  can be freed later.
 *
 *  >>> Arguments:
 *  Matrix  <input>    (MatrixPtr)
 *      Pointer to the matrix.
 *  AllocatedPtr  <input>  (void *)
 *      The pointer returned by malloc or calloc.  These pointers are saved in
 *      a list so that they can be easily freed.
 *
 *  >>> Possible errors:
 *  spNO_MEMORY
 */

static void
RecordAllocation(
    MatrixPtr Matrix,
    void *AllocatedPtr
)
{
/* Begin `RecordAllocation'. */
/*
 * If Allocated pointer is NULL, assume that malloc returned a NULL pointer,
 * which indicates a spNO_MEMORY error.
 */
    if (AllocatedPtr == NULL)
    {   Matrix->Error = spNO_MEMORY;
        return;
    }

/* Allocate block of MatrixElements if necessary. */
    if (Matrix->RecordsRemaining == 0)
    {   AllocateBlockOfAllocationList( Matrix );
        if (Matrix->Error == spNO_MEMORY)
        {   SP_FREE(AllocatedPtr);
            return;
        }
    }

/* Add Allocated pointer to Allocation List. */
    (++Matrix->TopOfAllocationList)->AllocatedPtr = AllocatedPtr;
    Matrix->RecordsRemaining--;
    return;

}








/*
 *  ADD A BLOCK OF SLOTS TO ALLOCATION LIST
 *
 *  This routine increases the size of the allocation list.
 *
 *  >>> Arguments:
 *  Matrix  <input>    (MatrixPtr)
 *      Pointer to the matrix.
 *
 *  >>> Local variables:
 *  ListPtr  (AllocationListPtr)
 *      Pointer to the list that contains the pointers to segments of memory
 *      that were allocated by the operating system for the current matrix.
 *
 *  >>> Possible errors:
 *  spNO_MEMORY
 */

static void
AllocateBlockOfAllocationList( MatrixPtr Matrix )
{
register  int  I;
register  AllocationListPtr  ListPtr;

/* Begin `AllocateBlockOfAllocationList'. */
/* Allocate block of records for allocation list. */
    ListPtr = SP_MALLOC(struct AllocationRecord, (ELEMENTS_PER_ALLOCATION+1));
    if (ListPtr == NULL)
    {   Matrix->Error = spNO_MEMORY;
        return;
    }

/* String entries of allocation list into singly linked list.  List is linked
   such that any record points to the one before it. */

    ListPtr->NextRecord = Matrix->TopOfAllocationList;
    Matrix->TopOfAllocationList = ListPtr;
    ListPtr += ELEMENTS_PER_ALLOCATION;
    for (I = ELEMENTS_PER_ALLOCATION; I > 0; I--)
    {    ListPtr->NextRecord = ListPtr - 1;
         ListPtr--;
    }

/* Record allocation of space for allocation list on allocation list. */
    Matrix->TopOfAllocationList->AllocatedPtr = (void *)ListPtr;
    Matrix->RecordsRemaining = ELEMENTS_PER_ALLOCATION;

    return;
}








/*!
 *  Destroys a matrix and frees all memory associated with it.
 *
 *  \param Matrix
 *      Pointer to the matrix frame which is to be destroyed.
 */
/*  >>> Local variables:
 *  ListPtr  (AllocationListPtr)
 *      Pointer into the linked list of pointers to allocated data structures.
 *      Points to pointer to structure to be freed.
 *  NextListPtr  (AllocationListPtr)
 *      Pointer into the linked list of pointers to allocated data structures.
 *      Points to the next pointer to structure to be freed.  This is needed
 *      because the data structure to be freed could include the current node
 *      in the allocation list.
 */

void
spDestroy( MatrixPtr Matrix )
{
register  AllocationListPtr  ListPtr, NextListPtr;

/* Begin `spDestroy'. */
    ASSERT_IS_SPARSE( Matrix );

/* Deallocate the vectors that are located in the matrix frame. */
    SP_FREE( Matrix->IntToExtColMap );
    SP_FREE( Matrix->IntToExtRowMap );
    SP_FREE( Matrix->ExtToIntColMap );
    SP_FREE( Matrix->ExtToIntRowMap );
    SP_FREE( Matrix->Diag );
    SP_FREE( Matrix->FirstInRow );
    SP_FREE( Matrix->FirstInCol );
    SP_FREE( Matrix->MarkowitzRow );
    SP_FREE( Matrix->MarkowitzCol );
    SP_FREE( Matrix->MarkowitzProd );
    SP_FREE( Matrix->DoCmplxDirect );
    SP_FREE( Matrix->DoRealDirect );
    SP_FREE( Matrix->Intermediate );

/* Sequentially step through the list of allocated pointers freeing pointers
 * along the way. */
    ListPtr = Matrix->TopOfAllocationList;
    while (ListPtr != NULL)
    {   NextListPtr = ListPtr->NextRecord;
        free( ListPtr->AllocatedPtr );
        ListPtr = NextListPtr;
    }
    return;
}







/*!
 *  This function returns the error status of the given matrix.
 *
 *  \return
 *      The error status of the given matrix.
 *
 *  \param Matrix
 *      The pointer to the matrix for which the error status is desired.
 */

spError
spErrorState( MatrixPtr Matrix )
{
/* Begin `spErrorState'. */

    if (Matrix != NULL)
    {   ASSERT_IS_SPARSE( Matrix );
        return (Matrix)->Error;
    }
    else return spNO_MEMORY;   /* This error may actually be spPANIC,
                                * no way to tell. */
}









/*!
 *  This function returns the row and column number where the matrix was
 *  detected as singular (if pivoting was allowed on the last factorization)
 *  or where a zero was detected on the diagonal (if pivoting was not
 *  allowed on the last factorization). Pivoting is performed only in
 *  spOrderAndFactor().
 *
 *  \param Matrix
 *      The matrix for which the error status is desired.
 *  \param pRow
 *      The row number.
 *  \param pCol
 *      The column number.
 */

void
spWhereSingular(
    MatrixPtr Matrix,
    int *pRow,
    int *pCol
)
{
/* Begin `spWhereSingular'. */
    ASSERT_IS_SPARSE( Matrix );

    if (Matrix->Error == spSINGULAR OR Matrix->Error == spZERO_DIAG)
    {   *pRow = Matrix->SingularRow;
        *pCol = Matrix->SingularCol;
    }
    else *pRow = *pCol = 0;
    return;
}






/*!
 *  Returns the size of the matrix.  Either the internal or external size of
 *  the matrix is returned.
 *
 *  \param Matrix
 *      Pointer to matrix.
 *  \param External
 *      If \a External is set true, the external size , i.e., the value of the
 *      largest external row or column number encountered is returned.
 *      Otherwise the true size of the matrix is returned.  These two sizes
 *      may differ if the \a TRANSLATE option is set true.
 */

int
spGetSize(
    MatrixPtr Matrix,
    int External
)
{
/* Begin `spGetSize'. */
    ASSERT_IS_SPARSE( Matrix );

#if TRANSLATE
    if (External)
        return Matrix->ExtSize;
    else
        return Matrix->Size;
#else
    return Matrix->Size;
#endif
}








/*!
 *  Forces matrix to be real.
 *
 *  \param Matrix
 *      Pointer to matrix.
 */

void
spSetReal( MatrixPtr Matrix )
{
/* Begin `spSetReal'. */

    ASSERT_IS_SPARSE( Matrix );
    vASSERT( REAL, "Sparse not compiled to handle real matrices" );
    Matrix->Complex = NO;
    return;
}


/*!
 *  Forces matrix to be complex.
 *
 *  \param Matrix
 *      Pointer to matrix.
 */

void
spSetComplex( MatrixPtr Matrix )
{
/* Begin `spSetComplex'. */

    ASSERT_IS_SPARSE( Matrix );
    vASSERT( spCOMPLEX, "Sparse not compiled to handle complex matrices" );
    Matrix->Complex = YES;
    return;
}









/*!
 *  This function returns the number of fill-ins that currently exists in a matrix.
 *
 *  \param Matrix
 *      Pointer to matrix.
 */

int
spFillinCount( MatrixPtr Matrix )
{
/* Begin `spFillinCount'. */

    ASSERT_IS_SPARSE( Matrix );
    return Matrix->Fillins;
}


/*!
 *  This function returns the total number of elements (including fill-ins) that currently exists in a matrix.
 *
 *  \param Matrix
 *      Pointer to matrix.
 */

/* FIXME: Seems no different size entries available anymore */

int
spElementCount( MatrixPtr Matrix )
{
/* Begin `spElementCount'. */

    ASSERT_IS_SPARSE( Matrix );
    return Matrix->Elements;
}

int
spOriginalCount( MatrixPtr Matrix )
{
    /* Begin `spOriginalCount'. */

    ASSERT_IS_SPARSE( Matrix );
    return Matrix->Elements;
}

