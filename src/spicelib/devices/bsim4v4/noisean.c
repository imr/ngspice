/**********
Copyright 2003 Regents of the University of California.  All rights reserved.
Author: 1987 Gary W. Ng
Author: 2000 Weidong Liu
Author: 2001- Xuemei Xi
**********/

#include "ngspice/ngspice.h"
#include "strext.h"
#include "acdefs.h"
#include "ngspice/cktdefs.h"
#include "fteconst.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/noisedef.h"
#include "ngspice/sperror.h"
#ifdef HAS_FLAT_INCLUDES
#include "vsrcdefs.h"
#include "isrcdefs.h"
#else
#include "vsrc/vsrcdefs.h"
#include "isrc/isrcdefs.h"
#endif
#include "util.h"
#include "ngspice/suffix.h"

int
NOISEan (ckt, restart)

CKTcircuit *ckt;
int restart;

{
    register Ndata *data;
    double realVal;
    double imagVal;
    int error;
    int posOutNode;
    int negOutNode;
    int code;
    int step;
    register CKTnode *node;
    IFuid freqUid;
    char *inst;
    double freqTol; /* tolerence parameter for finding final frequency; hack */

    register NOISEAN *job = (NOISEAN*) (ckt->CKTcurJob);
    static char *noacinput =    "noise input source has no AC value";

    posOutNode = ((CKTnode*) (job->output))->number;
    negOutNode = ((CKTnode*) (job->outputRef))->number;

    /* see if the source specified is AC */
    inst = NULL;
    code = CKTtypelook("Vsource");
    if (code != -1) {
        error = CKTfndDev((GENERIC*)ckt,&code,&inst,
                job->input, (GENERIC *)NULL, (IFuid)NULL);
	if (!error && !((VSRCinstance *)inst)->VSRCacGiven) {
	    errMsg = MALLOC(strlen(noacinput)+1);
	    strcpy(errMsg,noacinput);
	    return (E_NOACINPUT);
	}
    }

    code = CKTtypelook("Isource");
    if (code != -1 && inst==NULL) {
        error = CKTfndDev((GENERIC*)ckt,&code,&inst,
                job->input, (GENERIC *)NULL,(IFuid)NULL);
        if (error) {
	    /* XXX ??? */
            (*(SPfrontEnd->IFerror))(ERR_WARNING,
                    "Noise input source %s not in circuit",
                    &job->input);
		return (E_NOTFOUND);
	    }
	if (!((ISRCinstance *)inst)->ISRCacGiven) {
	    errMsg = MALLOC(strlen(noacinput)+1);
	    strcpy(errMsg,noacinput);
	    return (E_NOACINPUT);
	}
    }

    if ( (job->NsavFstp == 0) || restart) {
	switch (job->NstpType) {


        case DECADE:
            job->NfreqDelta = exp(log(10.0)/
                            job->NnumSteps);
            break;

        case OCTAVE:
            job->NfreqDelta = exp(log(2.0)/
	                    job->NnumSteps);
            break;

        case LINEAR:
            job->NfreqDelta = (job->NstopFreq -
                            job->NstartFreq)/
			    (job->NnumSteps+1);
            break;

        default:
            return(E_BADPARM);
        }

	/* error = DCop(ckt); */
	error = CKTop(ckt, (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITJCT,
		(ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITFLOAT,
		ckt->CKTdcMaxIter);

	if (error) return(error);
        data = (Ndata*)MALLOC(sizeof(Ndata));
	step = 0;
	data->freq = job->NstartFreq;
	data->outNoiz = 0.0;
	data->inNoise = 0.0;

	/* the current front-end needs the namelist to be fully
		declared before an OUTpBeginplot */

	(*(SPfrontEnd->IFnewUid))((GENERIC *)ckt,&freqUid,(IFuid)NULL,
		"frequency", UID_OTHER,(GENERIC **)NULL);

	data->numPlots = 0;                /* we don't have any plots  yet */
        error = CKTnoise(ckt,N_DENS,N_OPEN,data);
        if (error) return(error);

	/*
	 * all names in the namelist have been declared. now start the
	 * plot
	 */

	error = (*(SPfrontEnd->OUTpBeginPlot))(ckt,(GENERIC *)(ckt->CKTcurJob),
	   "Noise Spectral Density Curves - (V^2 or A^2)/Hz",
	   freqUid,IF_REAL,data->numPlots,data->namelist,IF_REAL,
	   &(data->NplotPtr));
	if (error) return(error);

        if (job->NstpType != LINEAR) {
	    (*(SPfrontEnd->OUTattributes))((GENERIC *)data->NplotPtr,NULL,
		    OUT_SCALE_LOG, NULL);
	}

    } else {   /* we must have paused before.  pick up where we left off */
	step = job->NsavFstp;
	switch (job->NstpType) {

	case DECADE:
        case OCTAVE:
	    data->freq = job->NstartFreq * exp (step *
		     log (job->NfreqDelta));
            break;

        case LINEAR:
	    data->freq = job->NstartFreq + step *
		     job->NfreqDelta;
            break;

        default:
            return(E_BADPARM);

        }
	job->NsavFstp = 0;
	data->outNoiz = job->NsavOnoise;
	data->inNoise = job->NsavInoise;
    }

    switch (job->NstpType) {
    case DECADE:
    case OCTAVE:
        freqTol = job->NfreqDelta * job->NstopFreq * ckt->CKTreltol;
        break;
    case LINEAR:
        freqTol = job->NfreqDelta * ckt->CKTreltol;
        break;
    default:
        return(E_BADPARM);
    }

    data->lstFreq = data->freq;

    /* do the noise analysis over all frequencies */

    while (data->freq <= job->NstopFreq + freqTol) {
        if( (*(SPfrontEnd->IFpauseTest))() ) {
	    job->NsavFstp = step;   /* save our results */
	    job->NsavOnoise = data->outNoiz; /* up until now     */
	    job->NsavInoise = data->inNoise;
	    return (E_PAUSE);
        }
	ckt->CKTomega = 2.0 * M_PI * data->freq;
	ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEAC;

	/*
	 * solve the original AC system to get the transfer
	 * function between the input and output
	 */

	NIacIter(ckt);
	realVal = *((ckt->CKTrhsOld) + posOutNode)
		- *((ckt->CKTrhsOld) + negOutNode);
	imagVal = *((ckt->CKTirhsOld) + posOutNode)
		- *((ckt->CKTirhsOld) + negOutNode);
	data->GainSqInv = 1.0 / MAX(((realVal*realVal)
		+ (imagVal*imagVal)),N_MINGAIN);
	data->lnGainInv = log(data->GainSqInv);

	/* set up a block of "common" data so we don't have to
	 * recalculate it for every device
	 */

	data->delFreq = data->freq - data->lstFreq;
	data->lnFreq = log(MAX(data->freq,N_MINLOG));
	data->lnLastFreq = log(MAX(data->lstFreq,N_MINLOG));
        data->delLnFreq = data->lnFreq - data->lnLastFreq;

	if ((job->NStpsSm != 0) && ((step % (job->NStpsSm)) == 0)) {
	    data->prtSummary = TRUE;
        } else {
	    data->prtSummary = FALSE;
        }

	/*
	data->outNumber = 1;
	*/

	data->outNumber = 0;
	/* the frequency will NOT be stored in array[0]  as before; instead,
	 * it will be given in refVal.rValue (see later)
	 */

	NInzIter(ckt,posOutNode,negOutNode);   /* solve the adjoint system */

	/* now we use the adjoint system to calculate the noise
	 * contributions of each generator in the circuit
	 */

	error = CKTnoise(ckt,N_DENS,N_CALC,data);
	if (error) return(error);
	data->lstFreq = data->freq;

	/* update the frequency */

	switch (job->NstpType) {

	case DECADE:
	case OCTAVE:
	    data->freq *= job->NfreqDelta;
	    break;

        case LINEAR:
	    data->freq += job->NfreqDelta;
	    break;

	default:
	    return(E_INTERN);
        }
	step++;
    }

    error = CKTnoise(ckt,N_DENS,N_CLOSE,data);
    if (error) return(error);

#ifdef INT_NOISE
    data->numPlots = 0;
    data->outNumber = 0;

    if (job->NstartFreq != job->NstopFreq) {
	error = CKTnoise(ckt,INT_NOIZ,N_OPEN,data);

	if (error) return(error);

	(*(SPfrontEnd->OUTpBeginPlot))(ckt,(GENERIC *)(ckt->CKTcurJob),
	       "Integrated Noise - V^2 or A^2",
	       (IFuid)NULL,(int)0,data->numPlots,data->namelist,IF_REAL,
	       &(data->NplotPtr));

	error = CKTnoise(ckt,INT_NOIZ,N_CALC,data);
	if (error) return(error);

	error = CKTnoise(ckt,INT_NOIZ,N_CLOSE,data);
	if (error) return(error);
    }
#endif

    FREE(data);
    return(OK);
}
