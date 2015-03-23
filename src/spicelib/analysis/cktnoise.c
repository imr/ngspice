/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Gary W. Ng
**********/

/*
 * CKTnoise (ckt, mode, operation, data)
 *
 *   This routine is responsible for naming and evaluating all of the
 *   noise sources in the circuit.  It uses a series of subroutines to
 *   name and evaluate the sources associated with each model, and then
 *   it evaluates the noise for the entire circuit.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/noisedef.h"
#include "ngspice/sperror.h"


int
CKTnoise (CKTcircuit *ckt, int mode, int operation, Ndata *data)
{
    NOISEAN *job = (NOISEAN *) ckt->CKTcurJob;

    double outNdens;
    int i;
    IFvalue outData;    /* output variable (points to list of outputs)*/
    IFvalue refVal; /* reference variable (always 0)*/
    int error;

    outNdens = 0.0;

    /* let each device decide how many and what type of noise sources it has */

    for (i=0; i < DEVmaxnum; i++) {
	if ( DEVices[i] && DEVices[i]->DEVnoise && ckt->CKThead[i] ) {
	    error = DEVices[i]->DEVnoise (mode, operation, ckt->CKThead[i],
		ckt,data, &outNdens);
            if (error) return (error);
        }
    }

    switch (operation) {

    case N_OPEN:

	/* take care of the noise for the circuit as a whole */

	switch (mode) {

	case N_DENS:

	    data->namelist = TREALLOC(IFuid, data->namelist, data->numPlots + 1);

	    SPfrontEnd->IFnewUid (ckt, &(data->namelist[data->numPlots++]),
                                  NULL, "onoise_spectrum", UID_OTHER, NULL);

	    data->namelist = TREALLOC(IFuid, data->namelist, data->numPlots + 1);

	    SPfrontEnd->IFnewUid (ckt, &(data->namelist[data->numPlots++]),
                                  NULL, "inoise_spectrum", UID_OTHER, NULL);

	    /* we've added two more plots */

	    data->outpVector =
		TMALLOC(double, data->numPlots);
	    data->squared_value =
		data->squared ? NULL : TMALLOC(char, data->numPlots);
            break;

	case INT_NOIZ:

	    data->namelist = TREALLOC(IFuid, data->namelist, data->numPlots + 1);
	    SPfrontEnd->IFnewUid (ckt, &(data->namelist[data->numPlots++]),
                                  NULL, "onoise_total", UID_OTHER, NULL);

	    data->namelist = TREALLOC(IFuid, data->namelist, data->numPlots + 1);
	    SPfrontEnd->IFnewUid (ckt, &(data->namelist[data->numPlots++]),
                                  NULL, "inoise_total", UID_OTHER, NULL);
	    /* we've added two more plots */

	    data->outpVector =
		TMALLOC(double, data->numPlots);
	    data->squared_value =
		data->squared ? NULL : TMALLOC(char, data->numPlots);
	    break;

        default:
	    return (E_INTERN);
        }

	break;

    case N_CALC:

	switch (mode) {

	case N_DENS:
            if ((job->NStpsSm == 0)
		|| data->prtSummary)
	    {
		data->outpVector[data->outNumber++] = outNdens;
		data->outpVector[data->outNumber++] =
		    (outNdens * data->GainSqInv);

		refVal.rValue = data->freq; /* the reference is the freq */
		if (!data->squared)
		    for (i = 0; i < data->outNumber; i++)
			if (data->squared_value[i])
			    data->outpVector[i] = sqrt(data->outpVector[i]);
		outData.v.numValue = data->outNumber; /* vector number */
		outData.v.vec.rVec = data->outpVector; /* vector of outputs */
		SPfrontEnd->OUTpData (data->NplotPtr, &refVal, &outData);
            }
            break;

	case INT_NOIZ:
	    data->outpVector[data->outNumber++] =  data->outNoiz; 
	    data->outpVector[data->outNumber++] =  data->inNoise;
	    if (!data->squared)
		for (i = 0; i < data->outNumber; i++)
		    if (data->squared_value[i])
			data->outpVector[i] = sqrt(data->outpVector[i]);
	    outData.v.vec.rVec = data->outpVector; /* vector of outputs */
	    outData.v.numValue = data->outNumber; /* vector number */
	    SPfrontEnd->OUTpData (data->NplotPtr, &refVal, &outData);
	    break;

	default:
	    return (E_INTERN);
        }
        break;

    case N_CLOSE:
	SPfrontEnd->OUTendPlot (data->NplotPtr);
	FREE(data->namelist);
	FREE(data->outpVector);
	FREE(data->squared_value);
        break;

    default:
	return (E_INTERN);
    }
    return (OK);
}
