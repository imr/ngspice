/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/

/* CKTpzSetup(ckt)
 * iterate through all the various
 * pzSetup functions provided for the circuit elements in the
 * given circuit, setup ...
 */

#include "ngspice.h"
#include "smpdefs.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "sperror.h"


int
CKTpzSetup(CKTcircuit *ckt, int type)
{
  extern SPICEdev **DEVices;
    PZAN *pzan = (PZAN *) ckt->CKTcurJob;
    SMPmatrix *matrix;
    int error;
    int i, temp, solution_col, balance_col;
    int input_pos, input_neg, output_pos, output_neg;

    NIdestroy(ckt);
    error = NIinit(ckt);
    if (error)
	return(error);
    matrix = ckt->CKTmatrix;

    /* Really awful . . . */
    ckt->CKTnumStates = 0;

    for (i = 0; i < DEVmaxnum; i++) {
        if (DEVices[i]->DEVpzSetup != NULL && ckt->CKThead[i] != NULL) {
            error = (*DEVices[i]->DEVpzSetup)(matrix, ckt->CKThead[i],
		ckt, &ckt->CKTnumStates);
            if (error != OK)
	        return(error);
        }
    }

    solution_col = 0;
    balance_col = 0;

    input_pos = pzan->PZin_pos;
    input_neg = pzan->PZin_neg;

    if (type == PZ_DO_ZEROS) {
	/* Vo/Ii in Y */
	output_pos = pzan->PZout_pos;
	output_neg = pzan->PZout_neg;
    } else if (pzan->PZinput_type == PZ_IN_VOL) {
	/* Vi/Ii in Y */
	output_pos = pzan->PZin_pos;
	output_neg = pzan->PZin_neg;
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
	temp = input_pos;
	input_pos = input_neg;
	input_neg = temp;
    }

    if (input_pos)
	pzan->PZdrive_pptr = SMPmakeElt(matrix, input_pos, solution_col);
    else
	pzan->PZdrive_pptr = NULL;

    if (input_neg)
	pzan->PZdrive_nptr = SMPmakeElt(matrix, input_neg, solution_col);
    else
	pzan->PZdrive_nptr = NULL;

    pzan->PZsolution_col = solution_col;
    pzan->PZbalance_col = balance_col;

    pzan->PZnumswaps = 1;

    error = NIreinit(ckt);
    if (error)
	return(error);

    return OK;
}
