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

#include "ngspice.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "iferrmsg.h"
#include "noisedef.h"
#include "sperror.h"


int
CKTnoise (CKTcircuit *ckt, int mode, int operation, Ndata *data)
{
    double outNdens;
    int i;
    extern SPICEdev **DEVices;
    IFvalue outData;    /* output variable (points to list of outputs)*/
    IFvalue refVal; /* reference variable (always 0)*/
    int error;

    outNdens = 0.0;

    /* let each device decide how many and what type of noise sources it has */

    for (i=0; i < DEVmaxnum; i++) {
	if ( ((*DEVices[i]).DEVnoise != NULL) && (ckt->CKThead[i] != NULL) ) {
	    error = (*((*DEVices[i]).DEVnoise))(mode,operation,ckt->CKThead[i],
		ckt,data, &outNdens);
            if (error) return (error);
        }
    }

    switch (operation) {

    case N_OPEN:

	/* take care of the noise for the circuit as a whole */

	switch (mode) {

	case N_DENS:

	    data->namelist = (IFuid *)trealloc((char *)data->namelist,
		    (data->numPlots + 1)*sizeof(IFuid));

	    (*(SPfrontEnd->IFnewUid))(ckt, &(data->namelist[data->numPlots++]),
		    (IFuid)NULL,"onoise_spectrum",UID_OTHER,(void **)NULL);

	    data->namelist = (IFuid *)trealloc((char *)data->namelist,
		    (data->numPlots + 1)*sizeof(IFuid));

	    (*(SPfrontEnd->IFnewUid))(ckt, &(data->namelist[data->numPlots++]),
		    (IFuid)NULL,"inoise_spectrum",UID_OTHER,(void **)NULL);

	    /* we've added two more plots */

	    data->outpVector =
		(double *)MALLOC(data->numPlots * sizeof(double));
            break;

	case INT_NOIZ:

	    data->namelist = (IFuid *)trealloc((char *)data->namelist,
		    (data->numPlots + 1)*sizeof(IFuid));
	    (*(SPfrontEnd->IFnewUid))(ckt, &(data->namelist[data->numPlots++]),
		(IFuid)NULL,"onoise_total",UID_OTHER,(void **)NULL);

	    data->namelist = (IFuid *)trealloc((char *)data->namelist,
		    (data->numPlots + 1)*sizeof(IFuid));
	    (*(SPfrontEnd->IFnewUid))(ckt, &(data->namelist[data->numPlots++]),
		(IFuid)NULL,"inoise_total",UID_OTHER,(void **)NULL);
	    /* we've added two more plots */

	    data->outpVector =
		(double *) MALLOC(data->numPlots * sizeof(double));
	    break;

        default:
	    return (E_INTERN);
        }

	break;

    case N_CALC:

	switch (mode) {

	case N_DENS:
            if ((((NOISEAN*)ckt->CKTcurJob)->NStpsSm == 0)
		|| data->prtSummary)
	    {
		data->outpVector[data->outNumber++] = outNdens;
		data->outpVector[data->outNumber++] =
		    (outNdens * data->GainSqInv);

		refVal.rValue = data->freq; /* the reference is the freq */
		outData.v.numValue = data->outNumber; /* vector number */
		outData.v.vec.rVec = data->outpVector; /* vector of outputs */
		(*(SPfrontEnd->OUTpData))(data->NplotPtr,&refVal,&outData);
            }
            break;

	case INT_NOIZ:
	    data->outpVector[data->outNumber++] =  data->outNoiz; 
	    data->outpVector[data->outNumber++] =  data->inNoise;
	    outData.v.vec.rVec = data->outpVector; /* vector of outputs */
	    outData.v.numValue = data->outNumber; /* vector number */
	    (*(SPfrontEnd->OUTpData))(data->NplotPtr,&refVal,&outData);
	    break;

	default:
	    return (E_INTERN);
        }
        break;

    case N_CLOSE:
	(*(SPfrontEnd->OUTendPlot))(data->NplotPtr);
	FREE(data->namelist);
	FREE(data->outpVector);
        break;

    default:
	return (E_INTERN);
    }
    return (OK);
}
