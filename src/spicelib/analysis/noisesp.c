/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Gary W. Ng
Modified: 2001 AlansFixes
**********/

/* Patch to noisean.c by Richard D. McRoberts.
 * Patched with modifications from Weidong Liu (2000)
 * Patched with modifications ftom Weidong Liu
 * in bsim4.1.0 code
 */

#include "ngspice/ngspice.h"
#include "ngspice/acdefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/cpextern.h"
#include "ngspice/noisedef.h"
#include "ngspice/sperror.h"
#include "ngspice/sim.h"
#include "vsrc/vsrcdefs.h"
#include "isrc/isrcdefs.h"

// fixme
//   ugly hack to work around missing api to specify the "type" of signals
extern int fixme_onoise_type;
extern int fixme_inoise_type;

/* not used */
#ifdef RFSPICE_

int
NOISEsp (CKTcircuit *ckt, int restart)
{
    /* variable must be static, for continuation of interrupted (Ctrl-C),
    longer lasting noise anlysis */
    static Ndata *data;

    double realVal;
    double imagVal;
    int error;
    int posOutNode;
    int negOutNode;
    int step;
    IFuid freqUid;
    double freqTol; /* tolerence parameter for finding final frequency; hack */
    int i, src_type;

    NOISEAN *job = (NOISEAN *) ckt->CKTcurJob;
    GENinstance *inst = CKTfndDev(ckt, job->input);
    bool frequequal = AlmostEqualUlps(job->NstartFreq, job->NstopFreq, 3);

    posOutNode = (job->output) -> number;
    negOutNode = (job->outputRef) -> number;

    if (job->NnumSteps < 1) {
        SPfrontEnd->IFerrorf(ERR_WARNING,
                             "Number of steps for noise measurement has to be larger than 0,\n    but currently is %d\n",
                             job->NnumSteps);
        return(E_PARMVAL);
    } else if ((job->NnumSteps == 1) && (job->NstpType == LINEAR)) {
        if (!frequequal) {
            job->NstopFreq = job->NstartFreq;
            SPfrontEnd->IFerrorf(ERR_WARNING,
                                 "Noise measurement at a single frequency %g only!\n",
                                 job->NstartFreq);
        }
    } else {
        if (frequequal) {
            job->NstopFreq = job->NstartFreq;
            job->NnumSteps = 1;
            SPfrontEnd->IFerrorf(ERR_WARNING,
                                 "Noise measurement at a single frequency %g only!\n",
                                 job->NstartFreq);
        }
    }
    /* see if the source specified is AC */
    {
        bool ac_given = FALSE;

        if (!inst || inst->GENmodPtr->GENmodType < 0) {
            SPfrontEnd->IFerrorf (ERR_WARNING,
                                  "Noise input source %s not in circuit",
                                  job->input);
            return E_NOTFOUND;
        }

        if (inst->GENmodPtr->GENmodType == CKTtypelook("Vsource")) {
            ac_given = ((VSRCinstance *)inst) -> VSRCacGiven;
            src_type = SV_VOLTAGE;
        } else if(inst->GENmodPtr->GENmodType == CKTtypelook("Isource")) {
            ac_given = ((ISRCinstance *)inst) -> ISRCacGiven;
            src_type = SV_CURRENT;
        } else {
            SPfrontEnd->IFerrorf (ERR_WARNING,
                                  "Noise input source %s is not of proper type",
                                  job->input);
            return E_NOTFOUND;
        }

        if (!ac_given) {
            SPfrontEnd->IFerrorf (ERR_WARNING,
                                  "Noise input source %s has no AC value",
                                  job->input);
            return E_NOACINPUT;
        }
    }

    if ( (job->NsavFstp == 0.0) || restart) { /* va, NsavFstp is double */
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
            if (job->NnumSteps == 1)
                job->NfreqDelta = 0;
            else
                job->NfreqDelta = (job->NstopFreq -
                                   job->NstartFreq) / (job->NnumSteps - 1);
            break;

        default:
            return(E_BADPARM);
        }

        /* error = DCop(ckt); */
        error = CKTop(ckt, (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITJCT,
                      (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITFLOAT,
                      ckt->CKTdcMaxIter);

        if (error) return(error);

        /* Patch to noisean.c by Richard D. McRoberts. */
        ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITSMSIG;
        error = CKTload(ckt);
        if(error) return(error);

        data = TMALLOC(Ndata, 1);
        step = 0;
        data->freq = job->NstartFreq;
        data->outNoiz = 0.0;
        data->inNoise = 0.0;
        data->squared = cp_getvar("sqrnoise", CP_BOOL, NULL, 0) ? 1 : 0;

        /* the current front-end needs the namelist to be fully
           declared before an OUTpBeginplot */

        SPfrontEnd->IFnewUid (ckt, &freqUid, NULL, "frequency", UID_OTHER, NULL);

        data->numPlots = 0;                /* we don't have any plots  yet */
        error = CKTnoise(ckt,N_DENS,N_OPEN,data);
        if (error) return(error);

        /*
         * all names in the namelist have been declared. now start the
         * plot
         */

        if (src_type == SV_VOLTAGE)
            fixme_inoise_type =
                data->squared ? SV_SQR_VOLTAGE_DENSITY : SV_VOLTAGE_DENSITY;
        else
            fixme_inoise_type =
                data->squared ? SV_SQR_CURRENT_DENSITY : SV_CURRENT_DENSITY;

        fixme_onoise_type =
            data->squared ? SV_SQR_VOLTAGE_DENSITY : SV_VOLTAGE_DENSITY;

        if (!data->squared)
            for (i = 0; i < data->numPlots; i++)
                data->squared_value[i] =
                    ciprefix("inoise", data->namelist[i]) ||
                    ciprefix("onoise", data->namelist[i]);

        error = SPfrontEnd->OUTpBeginPlot (ckt, ckt->CKTcurJob,
                                           data->squared
                                           ? "Noise Spectral Density Curves - (V^2 or A^2)/Hz"

                                           : "Noise Spectral Density Curves",
                                           freqUid, IF_REAL,
                                           data->numPlots, data->namelist, IF_REAL,
                                           &(data->NplotPtr));
        if (error) return(error);

        if (job->NstpType != LINEAR) {
            SPfrontEnd->OUTattributes (data->NplotPtr, NULL, OUT_SCALE_LOG, NULL);
        }

    } else {   /* we must have paused before.  pick up where we left off */
        step = (int)(job->NsavFstp);
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
        /* saj resume rawfile fix*/
        error = SPfrontEnd->OUTpBeginPlot (NULL, NULL,
                                           NULL,
                                           NULL, 0,
                                           666, NULL, 666,
                                           &(data->NplotPtr));
        /*saj*/
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
        if(SPfrontEnd->IFpauseTest()) {
            job->NsavFstp = step;   /* save our results */
            job->NsavOnoise = data->outNoiz; /* up until now     */
            job->NsavInoise = data->inNoise;
            return (E_PAUSE);
        }
        ckt->CKTomega = 2.0 * M_PI * data->freq;
        ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEAC | MODEACNOISE;
        ckt->noise_input = inst;

        /*
         * solve the original AC system to get the transfer
         * function between the input and output
         */

        NIacIter(ckt);
        realVal = ckt->CKTrhsOld [posOutNode]
                  - ckt->CKTrhsOld [negOutNode];
        imagVal = ckt->CKTirhsOld [posOutNode]
                  - ckt->CKTirhsOld [negOutNode];
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

        if ((job->NnumSteps == 1) && (job->NstpType == LINEAR))
            break;
    }

    error = CKTnoise(ckt,N_DENS,N_CLOSE,data);
    if (error) return(error);

    data->numPlots = 0;
    data->outNumber = 0;

    if (job->NstartFreq != job->NstopFreq) {
        error = CKTnoise(ckt,INT_NOIZ,N_OPEN,data);

        if (error) return(error);

        if (src_type == SV_VOLTAGE)
            fixme_inoise_type =
                data->squared ? SV_SQR_VOLTAGE : SV_VOLTAGE;
        else
            fixme_inoise_type =
                data->squared ? SV_SQR_CURRENT : SV_CURRENT;

        fixme_onoise_type =
            data->squared ? SV_SQR_VOLTAGE : SV_VOLTAGE;

        if (!data->squared)
            for (i = 0; i < data->numPlots; i++)
                data->squared_value[i] =
                    ciprefix("inoise", data->namelist[i]) ||
                    ciprefix("onoise", data->namelist[i]);

        SPfrontEnd->OUTpBeginPlot (ckt, ckt->CKTcurJob,
                                   data->squared
                                   ? "Integrated Noise - V^2 or A^2"
                                   : "Integrated Noise",
                                   NULL, 0,
                                   data->numPlots, data->namelist, IF_REAL,
                                   &(data->NplotPtr));

        error = CKTnoise(ckt,INT_NOIZ,N_CALC,data);
        if (error) return(error);

        error = CKTnoise(ckt,INT_NOIZ,N_CLOSE,data);
        if (error) return(error);
    }

    FREE(data);
    return(OK);
}

#endif