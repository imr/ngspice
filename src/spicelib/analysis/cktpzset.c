/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/

/* CKTpzSetup(ckt)
 * iterate through all the various
 * pzSetup functions provided for the circuit elements in the
 * given circuit, setup ...
 */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"

#ifdef KLU
#include <stdlib.h>

static
int
BindCompare (const void *a, const void *b)
{
    BindElement *A, *B ;
    A = (BindElement *)a ;
    B = (BindElement *)b ;

    return ((int)(A->Sparse - B->Sparse)) ;
}
#endif

int
CKTpzSetup(CKTcircuit *ckt, int type)
{
    PZAN *job = (PZAN *) ckt->CKTcurJob;

    SMPmatrix *matrix;
    int error;
    int i, solution_col, balance_col;
    int input_pos, input_neg, output_pos, output_neg;

    NIdestroy(ckt);
    error = NIinit(ckt);
    if (error)
	return(error);
    matrix = ckt->CKTmatrix;

    /* Really awful . . . */
    ckt->CKTnumStates = 0;

    for (i = 0; i < DEVmaxnum; i++) {
        if (DEVices[i] && DEVices[i]->DEVpzSetup != NULL && ckt->CKThead[i] != NULL) {
            error = DEVices[i]->DEVpzSetup (matrix, ckt->CKThead[i],
		ckt, &ckt->CKTnumStates);
            if (error != OK)
	        return(error);
        }
    }

    solution_col = 0;
    balance_col = 0;

    input_pos = job->PZin_pos;
    input_neg = job->PZin_neg;

    if (type == PZ_DO_ZEROS) {
	/* Vo/Ii in Y */
	output_pos = job->PZout_pos;
	output_neg = job->PZout_neg;
    } else if (job->PZinput_type == PZ_IN_VOL) {
	/* Vi/Ii in Y */
	output_pos = job->PZin_pos;
	output_neg = job->PZin_neg;
    } else {
	/* Denominator */
	output_pos = 0;
	output_neg = 0;
	input_pos = 0;
	input_neg = 0;
    }

    if (output_pos) {
	solution_col = output_pos;
	if (output_neg)
	    balance_col = output_neg;
    } else {
	solution_col = output_neg;
	SWAP(int, input_pos, input_neg);
    }

    if (input_pos)
	job->PZdrive_pptr = SMPmakeElt(matrix, input_pos, solution_col);
    else
	job->PZdrive_pptr = NULL;

    if (input_neg)
	job->PZdrive_nptr = SMPmakeElt(matrix, input_neg, solution_col);
    else
	job->PZdrive_nptr = NULL;

    job->PZsolution_col = solution_col;
    job->PZbalance_col = balance_col;

    job->PZnumswaps = 1;

#ifdef KLU
    if (ckt->CKTmatrix->CKTkluMODE)
    {
        fprintf (stderr, "Using KLU as Direct Linear Solver\n") ;

        int i ;
        int n = SMPmatSize (ckt->CKTmatrix) ;
        ckt->CKTmatrix->CKTkluN = n ;

        SMPnnz (ckt->CKTmatrix) ;
        int nz = ckt->CKTmatrix->CKTklunz ;

        ckt->CKTmatrix->CKTkluAp           = TMALLOC (int, n + 1) ;
        ckt->CKTmatrix->CKTkluAi           = TMALLOC (int, nz) ;
        ckt->CKTmatrix->CKTkluAx           = TMALLOC (double, nz) ;
        ckt->CKTmatrix->CKTkluIntermediate = TMALLOC (double, n) ;

        ckt->CKTmatrix->CKTbindStruct      = TMALLOC (BindElement, nz) ;

        ckt->CKTmatrix->CKTdiag_CSC        = TMALLOC (double *, n) ;

        /* Complex Stuff needed for AC Analysis */
        ckt->CKTmatrix->CKTkluAx_Complex = TMALLOC (double, 2 * nz) ;
        ckt->CKTmatrix->CKTkluIntermediate_Complex = TMALLOC (double, 2 * n) ;

        /* Binding Table from Sparse to CSC Format Creation */
        SMPmatrix_CSC (ckt->CKTmatrix) ;

        /* Binding Table Sorting */
        qsort (ckt->CKTmatrix->CKTbindStruct, (size_t)nz, sizeof(BindElement), BindCompare) ;

        /* KLU Pointers Assignment */
        for (i = 0 ; i < DEVmaxnum ; i++)
            if (DEVices [i] && DEVices [i]->DEVbindCSC && ckt->CKThead [i])
                DEVices [i]->DEVbindCSC (ckt->CKThead [i], ckt) ;

        /* ReOrder */
        error = SMPpreOrder (ckt->CKTmatrix) ;
        if (error) {
            fprintf (stderr, "Error during ReOrdering\n") ;
        }

        /* Conversion from Real Matrix to Complex Matrix */
        for (i = 0 ; i < DEVmaxnum ; i++)
            if (DEVices [i] && DEVices [i]->DEVbindCSCComplex && ckt->CKThead [i])
                DEVices [i]->DEVbindCSCComplex (ckt->CKThead [i], ckt) ;

        ckt->CKTmatrix->CKTkluMatrixIsComplex = CKTkluMatrixComplex ;

        /* Input Pos */
        if ((input_pos != 0) && (solution_col != 0))
        {
            double *j ;

            j = job->PZdrive_pptr ;
            job->PZdrive_pptr = ((BindElement *) bsearch (&j, ckt->CKTmatrix->CKTbindStruct, (size_t)nz, sizeof(BindElement), BindCompare))->CSC_Complex ;
        }

        /* Input Neg */
        if ((input_neg != 0) && (solution_col != 0))
        {
            double *j ;

            j = job->PZdrive_nptr ;
            job->PZdrive_nptr = ((BindElement *) bsearch (&j, ckt->CKTmatrix->CKTbindStruct, (size_t)nz, sizeof(BindElement), BindCompare))->CSC_Complex ;
        }
    } else {
        fprintf (stderr, "Using SPARSE 1.3 as Direct Linear Solver\n") ;
    }
#endif

    error = NIreinit(ckt);
    if (error)
	return(error);

    return OK;
}
