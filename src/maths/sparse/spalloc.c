/*
 *  MATRIX ALLOCATION MODULE
 *
 *  Author:                     Advising professor:
 *      Kenneth S. Kundert          Alberto Sangiovanni-Vincentelli
 *      UC Berkeley
 *
 *  This file contains the allocation and deallocation routines for the
 *  sparse matrix routines.
 *
 *  >>> User accessible functions contained in this file:
 *  spCreate
 *  spDestroy
 *  spError
 *  spWhereSingular
 *  spGetSize
 *  spSetReal
 *  spSetComplex
 *  spFillinCount
 *  spElementCount
 *  spOriginalCount
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
#include <assert.h>
#include <stdlib.h>

#define spINSIDE_SPARSE

#include "spconfig.h"
#include "ngspice/spmatrix.h"
#include "spdefs.h"

/*
 *  Function declarations
 */

static void InitializeElementBlocks( MatrixPtr, int, int );
static void RecordAllocation( MatrixPtr, void *);
static void AllocateBlockOfAllocationList( MatrixPtr );




/*
 *  MATRIX ALLOCATION
 *
 *  Allocates and initializes the data structures associated with a matrix.
 *
 *  >>> Returned:
 *  A pointer to the matrix is returned cast into the form of a pointer to
 *  a character.  This pointer is then passed and used by the other matrix
 *  routines to refer to a particular matrix.  If an error occurs, the NULL
 *  pointer is returned.
 *
 *  >>> Arguments:
 *  Size  <input>  (int)
 *      Size of matrix or estimate of size of matrix if matrix is EXPANDABLE.
 *  Complex  <input>  (int)
 *      Type of matrix.  If Complex is 0 then the matrix is real, otherwise
 *      the matrix will be complex.  Note that if the routines are not set up
 *      to handle the type of matrix requested, then a spPANIC error will occur.
 *      Further note that if a matrix will be both real and complex, it must
 *      be specified here as being complex.
 *  pError  <output>  (int *)
 *      Returns error flag, needed because function spError() will not work
 *      correctly if spCreate() returns NULL.
 *
 *  >>> Local variables:
 *  AllocatedSize  (int)
 *      The size of the matrix being allocated.
 *  Matrix  (MatrixPtr)
 *      A pointer to the matrix frame being created.
 *
 *  >>> Possible errors:
 *  spNO_MEMORY
 *  spPANIC
 *  Error is cleared in this routine.
 */

MatrixPtr
spCreate(int Size, int Complex, int *pError)
{
    unsigned  SizePlusOne;
    MatrixPtr  Matrix;
    int  I;
    int  AllocatedSize;

    /* Begin `spCreate'. */
    /* Clear error flag. */
    *pError = spOKAY;

    /* Test for valid size. */
#if EXPANDABLE
    if (Size < 0) {
	*pError = spPANIC;
        return NULL;
    }
#else
    if (Size <= 0) {
	*pError = spPANIC;
        return NULL;
    }
#endif


#if 0  /* pn: skipped for cider */
    /* Test for valid type. */
    if (!Complex) {
	*pError = spPANIC;
	return NULL;
    }
#endif

    /* Create Matrix. */
    AllocatedSize = MAX( Size, MINIMUM_ALLOCATED_SIZE );
    SizePlusOne = (unsigned)(AllocatedSize + 1);

    if ((Matrix = SP_MALLOC(struct MatrixFrame, 1)) == NULL) {
	*pError = spNO_MEMORY;
	return NULL;
    }

    /* Initialize matrix */
    Matrix->ID = SPARSE_ID;
    Matrix->Complex = Complex;
    Matrix->PreviousMatrixWasComplex = Complex;
    Matrix->Factored = NO;
    Matrix->Elements = 0;
    Matrix->Error = *pError;
    Matrix->Originals = 0;
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

    RecordAllocation( Matrix, Matrix );
    if (Matrix->Error == spNO_MEMORY) goto MemoryError;

    /* Take out the trash. */
    Matrix->TrashCan.Real = 0.0;
    Matrix->TrashCan.Imag = 0.0;
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
    {
	Matrix->IntToExtRowMap[I] = I;
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
    for (I = 1; I <= AllocatedSize; I++) {
	Matrix->ExtToIntColMap[I] = -1;
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
    spDestroy(Matrix);
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
spcGetElement(MatrixPtr Matrix)
{
    ElementPtr  pElements;

    /* Begin `spcGetElement'. */

#if !COMBINE || STRIP || LINT
    /* Allocate block of MatrixElements if necessary. */
    if (Matrix->ElementsRemaining == 0) {
	pElements = SP_MALLOC(struct MatrixElement, ELEMENTS_PER_ALLOCATION);
        RecordAllocation( Matrix, pElements );
        if (Matrix->Error == spNO_MEMORY) return NULL;
        Matrix->ElementsRemaining = ELEMENTS_PER_ALLOCATION;
        Matrix->NextAvailElement = pElements;
    }
#endif

#if COMBINE || STRIP || LINT
    if (Matrix->ElementsRemaining == 0)
    {
	pListNode = Matrix->LastElementListNode;

	/* First see if there are any stripped elements left. */
        if (pListNode->Next != NULL) {
	    Matrix->LastElementListNode = pListNode = pListNode->Next;
            Matrix->ElementsRemaining = pListNode->NumberOfElementsInList;
            Matrix->NextAvailElement = pListNode->pElementList;
        } else {
	    /* Allocate block of elements. */
            pElements = SP_MALLOC(struct MatrixElement, ELEMENTS_PER_ALLOCATION);
            RecordAllocation( Matrix, pElements );
            if (Matrix->Error == spNO_MEMORY) return NULL;
            Matrix->ElementsRemaining = ELEMENTS_PER_ALLOCATION;
            Matrix->NextAvailElement = pElements;

	    /* Allocate an element list structure. */
            pListNode->Next = SP_MALLOC(struct ElementListNodeStruct,1);
            RecordAllocation( Matrix, pListNode->Next );
            if (Matrix->Error == spNO_MEMORY)
		return NULL;
            Matrix->LastElementListNode = pListNode = pListNode->Next;

            pListNode->pElementList = pElements;
            pListNode->NumberOfElementsInList = ELEMENTS_PER_ALLOCATION;
            pListNode->Next = NULL;
        }
    }
#endif

    /* Update Element counter and return pointer to Element. */
    Matrix->ElementsRemaining--;
    return Matrix->NextAvailElement++;

}








/*
 *  ELEMENT ALLOCATION INITIALIZATION
 *
 *  This routine allocates space for matrix fill-ins and an initial
 *  set of elements.  Besides being faster than allocating space for
 *  elements one at a time, it tends to keep the fill-ins physically
 *  close to the other matrix elements in the computer memory.  This
 *  keeps virtual memory paging to a minimum.
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
 *  spNO_MEMORY */

static void
InitializeElementBlocks(MatrixPtr Matrix, int InitialNumberOfElements,
			int NumberOfFillinsExpected)
{
    ElementPtr  pElement;

    /* Begin `InitializeElementBlocks'. */

    /* Allocate block of MatrixElements for elements. */
    pElement = SP_MALLOC(struct MatrixElement, InitialNumberOfElements);
    RecordAllocation( Matrix, pElement );
    if (Matrix->Error == spNO_MEMORY) return;
    Matrix->ElementsRemaining = InitialNumberOfElements;
    Matrix->NextAvailElement = pElement;

    /* Allocate an element list structure. */
    Matrix->FirstElementListNode = SP_MALLOC(struct ElementListNodeStruct,1);
    RecordAllocation( Matrix, Matrix->FirstElementListNode );
    if (Matrix->Error == spNO_MEMORY) return;
    Matrix->LastElementListNode = Matrix->FirstElementListNode;

    Matrix->FirstElementListNode->pElementList = pElement;
    Matrix->FirstElementListNode->NumberOfElementsInList =
	    InitialNumberOfElements;
    Matrix->FirstElementListNode->Next = NULL;

    /* Allocate block of MatrixElements for fill-ins. */
    pElement = SP_MALLOC(struct MatrixElement, NumberOfFillinsExpected);
    RecordAllocation( Matrix, pElement );
    if (Matrix->Error == spNO_MEMORY) return;
    Matrix->FillinsRemaining = NumberOfFillinsExpected;
    Matrix->NextAvailFillin = pElement;

    /* Allocate a fill-in list structure. */
    Matrix->FirstFillinListNode = SP_MALLOC(struct FillinListNodeStruct,1);
    RecordAllocation( Matrix, Matrix->FirstFillinListNode );
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
 *  This routine allocates space for matrix fill-ins. It requests
 *  large blocks of storage from the system and doles out individual
 *  elements as required.  This technique, as opposed to allocating
 *  elements individually, tends to speed the allocation process.
 *
 *  >>> Returned:
 *  A pointer to the fill-in.
 *
 *  >>> Arguments:
 *  Matrix  <input>  (MatrixPtr)
 *      Pointer to matrix.
 *
 *  >>> Possible errors:
 *  spNO_MEMORY */

ElementPtr
spcGetFillin(MatrixPtr Matrix)
{
    /* Begin `spcGetFillin'. */

#if !STRIP || LINT
    if (Matrix->FillinsRemaining == 0)
        return spcGetElement( Matrix );
#endif
#if STRIP || LINT

    if (Matrix->FillinsRemaining == 0) {
	pListNode = Matrix->LastFillinListNode;

	/* First see if there are any stripped fill-ins left. */
        if (pListNode->Next != NULL) {
	    Matrix->LastFillinListNode = pListNode = pListNode->Next;
            Matrix->FillinsRemaining = pListNode->NumberOfFillinsInList;
            Matrix->NextAvailFillin = pListNode->pFillinList;
        } else {
	    /* Allocate block of fill-ins. */
            pFillins = SP_MALLOC(struct MatrixElement, ELEMENTS_PER_ALLOCATION);
            RecordAllocation( Matrix, pFillins );
            if (Matrix->Error == spNO_MEMORY) return NULL;
            Matrix->FillinsRemaining = ELEMENTS_PER_ALLOCATION;
            Matrix->NextAvailFillin = pFillins;

	    /* Allocate a fill-in list structure. */
            pListNode->Next = SP_MALLOC(struct FillinListNodeStruct,1);
            RecordAllocation( Matrix, pListNode->Next );
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
 *  This routine is used to record all memory allocations so that the
 *  memory can be freed later.
 *
 *  >>> Arguments:
 *  Matrix  <input>    (MatrixPtr)
 *      Pointer to the matrix.
 *  AllocatedPtr  <input>  (void *)
 *      The pointer returned by tmalloc or calloc.  These pointers are
 *      saved in a list so that they can be easily freed.
 *
 *  >>> Possible errors:
 *  spNO_MEMORY */

static void
RecordAllocation(MatrixPtr Matrix, void *AllocatedPtr )
{
    /* Begin `RecordAllocation'. */
    /* If Allocated pointer is NULL, assume that tmalloc returned a
     * NULL pointer, which indicates a spNO_MEMORY error.  */
    if (AllocatedPtr == NULL) {
	Matrix->Error = spNO_MEMORY;
        return;
    }

    /* Allocate block of MatrixElements if necessary. */
    if (Matrix->RecordsRemaining == 0) {
	AllocateBlockOfAllocationList( Matrix );
        if (Matrix->Error == spNO_MEMORY) {
	    SP_FREE(AllocatedPtr);
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
 *      Pointer to the list that contains the pointers to segments of
 *      memory that were allocated by the operating system for the
 *      current matrix.
 *
 *  >>> Possible errors:
 * spNO_MEMORY */

static void
AllocateBlockOfAllocationList(MatrixPtr Matrix)
{
    int  I;
    AllocationListPtr  ListPtr;

    /* Begin `AllocateBlockOfAllocationList'. */
    /* Allocate block of records for allocation list. */
    ListPtr = SP_MALLOC(struct AllocationRecord, (ELEMENTS_PER_ALLOCATION+1));
    if (ListPtr == NULL) {
	Matrix->Error = spNO_MEMORY;
        return;
    }

    /* String entries of allocation list into singly linked list.
       List is linked such that any record points to the one before
       it. */

    ListPtr->NextRecord = Matrix->TopOfAllocationList;
    Matrix->TopOfAllocationList = ListPtr;
    ListPtr += ELEMENTS_PER_ALLOCATION;
    for (I = ELEMENTS_PER_ALLOCATION; I > 0; I--) {
	ListPtr->NextRecord = ListPtr - 1;
	ListPtr--;
    }

    /* Record allocation of space for allocation list on allocation list. */
    Matrix->TopOfAllocationList->AllocatedPtr = (void *)ListPtr;
    Matrix->RecordsRemaining = ELEMENTS_PER_ALLOCATION;

    return;
}








/*
 *  MATRIX DEALLOCATION
 *
 *  Deallocates pointers and elements of Matrix.
 *
 *  >>> Arguments:
 *  Matrix  <input>  (void *)
 *      Pointer to the matrix frame which is to be removed from memory.
 *
 *  >>> Local variables:
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
spDestroy(MatrixPtr Matrix)
{
    AllocationListPtr  ListPtr, NextListPtr;


    /* Begin `spDestroy'. */
    assert( IS_SPARSE( Matrix ) );

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

    /* Sequentially step through the list of allocated pointers
     * freeing pointers along the way. */
    ListPtr = Matrix->TopOfAllocationList;
    while (ListPtr != NULL) {
	NextListPtr = ListPtr->NextRecord;
	if ((void *) ListPtr == ListPtr->AllocatedPtr) {
	    SP_FREE( ListPtr );
	} else {
	    SP_FREE( ListPtr->AllocatedPtr );
	}
        ListPtr = NextListPtr;
    }
    return;
}







/*
 *  RETURN MATRIX ERROR STATUS
 *
 *  This function is used to determine the error status of the given
 *  matrix.
 *
 *  >>> Returned:
 *      The error status of the given matrix.
 *
 *  >>> Arguments:
 *  Matrix  <input>  (void *)
 *      The matrix for which the error status is desired.  */
int
spError(MatrixPtr Matrix )
{
    /* Begin `spError'. */

    if (Matrix != NULL) {
	assert(Matrix->ID == SPARSE_ID);
        return Matrix->Error;
    } else {
	/* This error may actually be spPANIC, no way to tell. */
	return spNO_MEMORY;
    }
}









/*
 *  WHERE IS MATRIX SINGULAR
 *
 *  This function returns the row and column number where the matrix was
 *  detected as singular or where a zero was detected on the diagonal.
 *
 *  >>> Arguments:
 *  Matrix  <input>  (void *)
 *      The matrix for which the error status is desired.
 *  pRow  <output>  (int *)
 *      The row number.
 *  pCol  <output>  (int *)
 *      The column number.
 */

void
spWhereSingular(MatrixPtr Matrix, int *pRow, int *pCol)
{
    /* Begin `spWhereSingular'. */
    assert( IS_SPARSE( Matrix ) );

    if (Matrix->Error == spSINGULAR || Matrix->Error == spZERO_DIAG)
    {
	*pRow = Matrix->SingularRow;
        *pCol = Matrix->SingularCol;
    }
    else *pRow = *pCol = 0;
    return;
}






/*
 *  MATRIX SIZE
 *
 *  Returns the size of the matrix.  Either the internal or external size of
 *  the matrix is returned.
 *
 *  >>> Arguments:
 *  Matrix  <input>  (void *)
 *      Pointer to matrix.
 *  External  <input>  (int)
 *      If External is set TRUE, the external size , i.e., the value of the
 *      largest external row or column number encountered is returned.
 *      Otherwise the TRUE size of the matrix is returned.  These two sizes
 *      may differ if the TRANSLATE option is set TRUE.
 */

int
spGetSize(MatrixPtr Matrix, int External)
{
    /* Begin `spGetSize'. */
    assert( IS_SPARSE( Matrix ) );

#if TRANSLATE
    if (External)
        return Matrix->ExtSize;
    else
        return Matrix->Size;
#else
    return Matrix->Size;
#endif
}








/*
 *  SET MATRIX COMPLEX OR REAL
 *
 *  Forces matrix to be either real or complex.
 *
 *  >>> Arguments:
 *  Matrix  <input>  (void *)
 *      Pointer to matrix.
 */

void
spSetReal(MatrixPtr Matrix)
{
    /* Begin `spSetReal'. */

    assert( IS_SPARSE( Matrix ));
    Matrix->Complex = NO;
    return;
}


void
spSetComplex(MatrixPtr Matrix)
{
    /* Begin `spSetComplex'. */

    assert( IS_SPARSE( Matrix ));
    Matrix->Complex = YES;
    return;
}









/*
 *  ELEMENT, FILL-IN OR ORIGINAL COUNT
 *
 *  Two functions used to return simple statistics.  Either the number
 *  of total elements, or the number of fill-ins, or the number
 *  of original elements can be returned.
 *
 *  >>> Arguments:
 *  Matrix  <input>  (void *)
 *      Pointer to matrix.
 */

int
spFillinCount(MatrixPtr Matrix)
{
    /* Begin `spFillinCount'. */

    assert( IS_SPARSE( Matrix ) );
    return Matrix->Fillins;
}


int
spElementCount(MatrixPtr Matrix)
{
    /* Begin `spElementCount'. */

    assert( IS_SPARSE( Matrix ) );
    return Matrix->Elements;
}

int
spOriginalCount(MatrixPtr Matrix)
{
    /* Begin `spOriginalCount'. */

    assert( IS_SPARSE( Matrix ) );
    return Matrix->Originals;
}
