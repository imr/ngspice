/*
**** 
* Alessio Cacciatori 2021
****
*/
#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/spdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"

#ifdef XSPICE
#include "ngspice/evt.h"
#include "ngspice/enh.h"
/* gtri - add - wbk - 12/19/90 - Add headers */ 
#include "ngspice/mif.h"
#include "ngspice/evtproto.h"
#include "ngspice/ipctiein.h"
/* gtri - end - wbk */
#endif




#ifdef RFSPICE
#include "vsrc/vsrcext.h"
#include "../maths/dense/dense.h"

#define INIT_STATS() \
do { \
    startTime  = SPfrontEnd->IFseconds();       \
    startdTime = ckt->CKTstat->STATdecompTime;  \
    startsTime = ckt->CKTstat->STATsolveTime;   \
    startlTime = ckt->CKTstat->STATloadTime;    \
    startkTime = ckt->CKTstat->STATsyncTime;    \
} while(0)

#define UPDATE_STATS(analysis) \
do { \
    ckt->CKTcurrentAnalysis = analysis; \
    ckt->CKTstat->STATacTime += SPfrontEnd->IFseconds() - startTime; \
    ckt->CKTstat->STATacDecompTime += ckt->CKTstat->STATdecompTime - startdTime; \
    ckt->CKTstat->STATacSolveTime += ckt->CKTstat->STATsolveTime - startsTime; \
    ckt->CKTstat->STATacLoadTime += ckt->CKTstat->STATloadTime - startlTime; \
    ckt->CKTstat->STATacSyncTime += ckt->CKTstat->STATsyncTime - startkTime; \
} while(0)



int
CKTspnoise(CKTcircuit * ckt, int mode, int operation, Ndata * data)
{
    NOISEAN* job = (NOISEAN*)ckt->CKTcurJob;

    double outNdens;
    int i;
    IFvalue outData;    /* output variable (points to list of outputs)*/
    IFvalue refVal; /* reference variable (always 0)*/
    int error;

    outNdens = 0.0;

    /* let each device decide how many and what type of noise sources it has */

    for (i = 0; i < DEVmaxnum; i++) {
        if (DEVices[i] && DEVices[i]->DEVnoise && ckt->CKThead[i]) {
            error = DEVices[i]->DEVnoise(mode, operation, ckt->CKThead[i],
                ckt, data, &outNdens);
            if (error) return (error);
        }
    }

    switch (operation) {

    case N_OPEN:

        /* take care of the noise for the circuit as a whole */

        switch (mode) {

        case N_DENS:

            data->namelist = TREALLOC(IFuid, data->namelist, data->numPlots + 1);

            SPfrontEnd->IFnewUid(ckt, &(data->namelist[data->numPlots++]),
                NULL, "onoise_spectrum", UID_OTHER, NULL);

            data->namelist = TREALLOC(IFuid, data->namelist, data->numPlots + 1);

            SPfrontEnd->IFnewUid(ckt, &(data->namelist[data->numPlots++]),
                NULL, "inoise_spectrum", UID_OTHER, NULL);

            /* we've added two more plots */

            data->outpVector =
                TMALLOC(double, data->numPlots);
            data->squared_value =
                data->squared ? NULL : TMALLOC(char, data->numPlots);
            break;

        case INT_NOIZ:

            data->namelist = TREALLOC(IFuid, data->namelist, data->numPlots + 1);
            SPfrontEnd->IFnewUid(ckt, &(data->namelist[data->numPlots++]),
                NULL, "onoise_total", UID_OTHER, NULL);

            data->namelist = TREALLOC(IFuid, data->namelist, data->numPlots + 1);
            SPfrontEnd->IFnewUid(ckt, &(data->namelist[data->numPlots++]),
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
                SPfrontEnd->OUTpData(data->NplotPtr, &refVal, &outData);
            }
            break;

        case INT_NOIZ:
            data->outpVector[data->outNumber++] = data->outNoiz;
            data->outpVector[data->outNumber++] = data->inNoise;
            if (!data->squared)
                for (i = 0; i < data->outNumber; i++)
                    if (data->squared_value[i])
                        data->outpVector[i] = sqrt(data->outpVector[i]);
            outData.v.vec.rVec = data->outpVector; /* vector of outputs */
            outData.v.numValue = data->outNumber; /* vector number */
            SPfrontEnd->OUTpData(data->NplotPtr, &refVal, &outData);
            break;

        default:
            return (E_INTERN);
        }
        break;

    case N_CLOSE:
        SPfrontEnd->OUTendPlot(data->NplotPtr);
        FREE(data->namelist);
        FREE(data->outpVector);
        FREE(data->squared_value);
        break;

    default:
        return (E_INTERN);
    }
    return (OK);
}


void
NInspIter(CKTcircuit * ckt, int posDrive, int negDrive)
{
    int i;

    /* clear out the right hand side vector */

    for (i = 0; i <= SMPmatSize(ckt->CKTmatrix); i++) {
        ckt->CKTrhs[i] = 0.0;
        ckt->CKTirhs[i] = 0.0;
    }

    ckt->CKTrhs[posDrive] = 1.0;     /* apply unit current excitation */
    ckt->CKTrhs[negDrive] = -1.0;
    SMPcaSolve(ckt->CKTmatrix, ckt->CKTrhs, ckt->CKTirhs, ckt->CKTrhsSpare,
        ckt->CKTirhsSpare);

    ckt->CKTrhs[0] = 0.0;
    ckt->CKTirhs[0] = 0.0;
}


int
SPan(CKTcircuit *ckt, int restart)
{


    SPAN *job = (SPAN *) ckt->CKTcurJob;

    double freq;
    double freqTol; /* tolerence parameter for finding final frequency */
    double startdTime;
    double startsTime;
    double startlTime;
    double startkTime;
    double startTime;
    int error;
    int numNames;
    int i;
    IFuid *nameList;  /* va: tmalloc'ed list of names */
    IFuid freqUid;
    static runDesc *spPlot = NULL;
    runDesc *plot = NULL;

    double* rhswoPorts = NULL;
    double* irhswoPorts = NULL;
    int* portPosNodes = NULL;
    int* portNegNodes = NULL;


    if (ckt->CKTportCount == 0)
    {
        fprintf(stderr, "No RF Port is present\n");
        return (E_PARMVAL);
    }



    if (ckt->CKTAmat != NULL) freecmat(ckt->CKTAmat);
    if (ckt->CKTBmat != NULL) freecmat(ckt->CKTBmat);
    if (ckt->CKTSmat != NULL) freecmat(ckt->CKTSmat);

    ckt->CKTAmat = newcmat(ckt->CKTportCount, ckt->CKTportCount, 0.0, 0.0);
    if (ckt->CKTAmat == NULL)
        return (E_NOMEM);
    ckt->CKTBmat = newcmat(ckt->CKTportCount, ckt->CKTportCount, 0.0, 0.0);
    if (ckt->CKTBmat == NULL)
        return (3);

    ckt->CKTSmat = newcmat(ckt->CKTportCount, ckt->CKTportCount, 0.0, 0.0);
    if (ckt->CKTSmat == NULL)
        return (E_NOMEM);

#ifdef XSPICE
/* gtri - add - wbk - 12/19/90 - Add IPC stuff and anal_init and anal_type */

    /* Tell the beginPlot routine what mode we're in */

    // For now, let's keep this as IPC_ANAL_AC (TBD)
    g_ipc.anal_type = IPC_ANAL_AC;

    /* Tell the code models what mode we're in */
    g_mif_info.circuit.anal_type = MIF_DC;
    g_mif_info.circuit.anal_init = MIF_TRUE;

/* gtri - end - wbk */
#endif

    /* start at beginning */
    if (job->SPsaveFreq == 0 || restart) {
        if (job->SPnumberSteps < 1)
            job->SPnumberSteps = 1;

        switch (job->SPstepType) {

        case DECADE:
            if (job->SPstartFreq <= 0) {
                fprintf(stderr, "ERROR: AC startfreq <= 0\n");
                return E_PARMVAL;
            }
            job->SPfreqDelta =
                exp(log(10.0)/job->SPnumberSteps);
            break;
        case OCTAVE:
            if (job->SPstartFreq <= 0) {
                fprintf(stderr, "ERROR: AC startfreq <= 0\n");
                return E_PARMVAL;
            }
            job->SPfreqDelta =
                exp(log(2.0)/job->SPnumberSteps);
            break;
        case LINEAR:
            if (job->SPnumberSteps-1 > 1)
                job->SPfreqDelta =
                    (job->SPstopFreq -
                     job->SPstartFreq) /
                    (job->SPnumberSteps - 1);
            else
            /* Patch from: Richard McRoberts
            * This patch is for a rather pathological case:
            * a linear step with only one point */
                job->SPfreqDelta = 0;
            break;
        default:
            return(E_BADPARM);
    }
#ifdef XSPICE
/* gtri - begin - wbk - Call EVTop if event-driven instances exist */

    if(ckt->evt->counts.num_insts != 0) {
        error = EVTop(ckt,
            (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITJCT,
            (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITFLOAT,
            ckt->CKTdcMaxIter,
            MIF_TRUE);
        EVTdump(ckt, IPC_ANAL_DCOP, 0.0);
        EVTop_save(ckt, MIF_TRUE, 0.0);
    }
    else 
#endif 
    /* If no event-driven instances, do what SPICE normally does */
    if (!ckt->CKTnoopac) { /* skip OP if option NOOPAC is set and circuit is linear */
        error = CKTop(ckt,
            (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITJCT,
            (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITFLOAT,
            ckt->CKTdcMaxIter);

        if(error){
            fprintf(stdout,"\nAC operating point failed -\n");
            CKTncDump(ckt);
            return(error);
        }
    }
    else
        fprintf(stdout,"\n Linear circuit, option noopac given: no OP analysis\n");
		
#ifdef XSPICE
/* gtri - add - wbk - 12/19/90 - Add IPC stuff */

    /* Send the operating point results for Mspice compatibility */
    if(g_ipc.enabled) 
    {
        /* Call CKTnames to get names of nodes/branches used by 
            BeginPlot */
        /* Probably should free nameList after this block since 
            called again... */
        error = CKTnames(ckt,&numNames,&nameList);
        if(error) return(error);

        /* We have to do a beginPlot here since the data to return is
         * different for the DCOP than it is for the AC analysis.  
         * Moreover the begin plot has not even been done yet at this 
         * point... 
         */
        SPfrontEnd->OUTpBeginPlot (ckt, ckt->CKTcurJob,
                                   ckt->CKTcurJob->JOBname,
                                   NULL, IF_REAL,
                                   numNames, nameList, IF_REAL,
                                   &spPlot);
        txfree(nameList);

        ipc_send_dcop_prefix();
        CKTdump(ckt, 0.0, spPlot);
        ipc_send_dcop_suffix();

        SPfrontEnd->OUTendPlot (spPlot);
    }
/* gtri - end - wbk */
#endif

        ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITSMSIG;
        error = CKTload(ckt);
        if(error) return(error);

        error = CKTnames(ckt,&numNames,&nameList);
        if(error) return(error);

	if (ckt->CKTkeepOpInfo) {
	    /* Dump operating point. */
            error = SPfrontEnd->OUTpBeginPlot (ckt, ckt->CKTcurJob,
                                               "AC Operating Point",
                                               NULL, IF_REAL,
                                               numNames, nameList, IF_REAL,
                                               &plot);
	    if(error) return(error);
	    CKTdump(ckt, 0.0, plot);
	    SPfrontEnd->OUTendPlot (plot);
	    plot = NULL;
	}
      
        unsigned int extraSPdataLength =  ckt->CKTportCount * ckt->CKTportCount;
        nameList = (IFuid*)TREALLOC(IFuid, nameList, numNames + extraSPdataLength);


        // Create UIDs
        for (unsigned int dest = 1; dest <= ckt->CKTportCount; dest++)
            for (unsigned int j = 1; j <= ckt->CKTportCount; j++)
            {
                char tmpBuf[32];
                sprintf(tmpBuf, "S_%d_%d", dest, j);

                SPfrontEnd->IFnewUid(ckt, &(nameList[numNames++]), NULL, tmpBuf, UID_OTHER, NULL);
            }

        SPfrontEnd->IFnewUid (ckt, &freqUid, NULL, "frequency", UID_OTHER, NULL);
        error = SPfrontEnd->OUTpBeginPlot (ckt, ckt->CKTcurJob,
                                           ckt->CKTcurJob->JOBname,
                                           freqUid, IF_REAL,
                                           numNames, nameList, IF_COMPLEX,
                                           &spPlot);



	tfree(nameList);		
	if(error) return(error);

        if (job->SPstepType != LINEAR) {
	    SPfrontEnd->OUTattributes (spPlot, NULL, OUT_SCALE_LOG, NULL);
	}
        freq = job->SPstartFreq;

    } else {    /* continue previous analysis */
        freq = job->SPsaveFreq;
        job->SPsaveFreq = 0; /* clear the 'old' frequency */
	/* fix resume? saj, indeed !*/
        error = SPfrontEnd->OUTpBeginPlot (NULL, NULL,
                                           NULL,
                                           NULL, 0,
                                           666, NULL, 666,
                                           &spPlot);
	/* saj*/    
    }
        
    switch (job->SPstepType) {
    case DECADE:
    case OCTAVE:
        freqTol = job->SPfreqDelta *
            job->SPstopFreq * ckt->CKTreltol;
        break;
    case LINEAR:
        freqTol = job->SPfreqDelta * ckt->CKTreltol;
        break;
    default:
        return(E_BADPARM);
    }


#ifdef XSPICE
/* gtri - add - wbk - 12/19/90 - Set anal_init and anal_type */

    g_mif_info.circuit.anal_init = MIF_TRUE;

    /* Tell the code models what mode we're in */
    g_mif_info.circuit.anal_type = MIF_AC;

/* gtri - end - wbk */
#endif

    INIT_STATS();

    ckt->CKTcurrentAnalysis = DOING_AC;

    ckt->CKTactivePort = 0;
    /* main loop through all scheduled frequencies */
    while (freq <= job->SPstopFreq + freqTol) {

        unsigned int activePort = 0;
        //
        if (SPfrontEnd->IFpauseTest()) {
            /* user asked us to pause via an interrupt */
            job->SPsaveFreq = freq;
            return(E_PAUSE);
        }
        ckt->CKTomega = 2.0 * M_PI * freq;

        /* Update opertating point, if variable 'hertz' is given */
        if (ckt->CKTvarHertz) {
#ifdef XSPICE
            /* Call EVTop if event-driven instances exist */

            if (ckt->evt->counts.num_insts != 0) {
                error = EVTop(ckt,
                    (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITJCT,
                    (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITFLOAT,
                    ckt->CKTdcMaxIter,
                    MIF_TRUE);
                EVTdump(ckt, IPC_ANAL_DCOP, 0.0);
                EVTop_save(ckt, MIF_TRUE, 0.0);
            }
            else
#endif 
                // If no event-driven instances, do what SPICE normally does
                error = CKTop(ckt,
                    (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITJCT,
                    (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITFLOAT,
                    ckt->CKTdcMaxIter);

            if (error) {
                fprintf(stdout, "\nAC operating point failed -\n");
                CKTncDump(ckt);
                return(error);
            }
            ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITSMSIG;
            error = CKTload(ckt);
            if (error) return(error);
        }
        // Let's sweep thru all available ports to build Y matrix
        // Y_ij = I_i / V_j | V_k!=j = 0
        // (we have only to modify rhs)

        int vsrcLookupType = CKTtypelook("Vsource");
        int vsrcRoot = -1;
        
        // Get VSRCs root model
        if (ckt->CKTVSRCid == -1)
        {
            for (i = 0; i < DEVmaxnum; i++) {
                if (DEVices[i] && DEVices[i]->DEVacLoad && ckt->CKThead[i] && ckt->CKThead[i]->GENmodType == vsrcLookupType) {

                    vsrcRoot = i;
                    break;
                }
            }
            if (vsrcRoot == -1)
                return (E_NOMOD);

            ckt->CKTVSRCid = vsrcRoot;
        }
        else
            vsrcRoot = ckt->CKTVSRCid;

        if (rhswoPorts == NULL)
            rhswoPorts = (double*)TREALLOC(double, rhswoPorts, ckt->CKTmaxEqNum);
        if (irhswoPorts == NULL)
            irhswoPorts = (double*)TREALLOC(double, irhswoPorts, ckt->CKTmaxEqNum);

        ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODESP;
        // Pre-load everything but RF Ports (these will be updated in the next cycle).
        error = NIspPreload(ckt);
        if (error) return (error);
        
//        error = VSRCsaveNPData(ckt->CKThead[vsrcRoot]);
//        if (error) return (error);

        //Keep a backup copy
        memcpy(rhswoPorts, ckt->CKTrhs,  ckt->CKTmaxEqNum * sizeof(double));
        memcpy(rhswoPorts, ckt->CKTirhs, ckt->CKTmaxEqNum * sizeof(double));

        for (activePort = 1; activePort <= ckt->CKTportCount; activePort++)
        {
            // Copy the backup RHS into CKT's RHS
            memcpy(ckt->CKTrhs, rhswoPorts, ckt->CKTmaxEqNum * sizeof(double));
            memcpy(ckt->CKTirhs, irhswoPorts, ckt->CKTmaxEqNum * sizeof(double));
            ckt->CKTactivePort = activePort;

            // Update only VSRCs
            error = VSRCspupdate(ckt->CKThead[vsrcRoot], ckt);
            if (error)
            {
                tfree(rhswoPorts);
                tfree(irhswoPorts);
                return(error);
            }

            error = NIspSolve(ckt);
            if (error) {
                tfree(rhswoPorts);
                tfree(irhswoPorts);
                UPDATE_STATS(DOING_AC);
                return(error);
            }


#ifdef WANT_SENSE2
            if (ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & ACSEN)) {
                long save;
                int save1;

                save = ckt->CKTmode;
                ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITSMSIG;
                save1 = ckt->CKTsenInfo->SENmode;
                ckt->CKTsenInfo->SENmode = ACSEN;
                if (freq == job->SPstartFreq) {
                    ckt->CKTsenInfo->SENacpertflag = 1;
                }
                else {
                    ckt->CKTsenInfo->SENacpertflag = 0;
                }
                error = CKTsenAC(ckt);
                if (error)
                    return (error);
                ckt->CKTmode = save;
                ckt->CKTsenInfo->SENmode = save1;
            }
#endif

            // We have done 1 activated port.
            error = CKTspCalcPowerWave(ckt);


        } //active ports cycle

        // Now we can calculate the full S-Matrix
        CKTspCalcSMatrix(ckt);

#ifdef XSPICE
        /* gtri - modify - wbk - 12/19/90 - Send IPC stuff */

        if (g_ipc.enabled)
            ipc_send_data_prefix(freq);

        error = CKTspDump(ckt, freq, spPlot);

        if (g_ipc.enabled)
            ipc_send_data_suffix();

        /* gtri - modify - wbk - 12/19/90 - Send IPC stuff */
#else
        error = CKTspDump(ckt, freq, spPlot);
#endif	
        if (error) {
            UPDATE_STATS(DOING_AC);
            tfree(rhswoPorts);
            tfree(irhswoPorts);
            return(error);
        }


        /*
        * Now go with noise cycle, if required
        */

#ifdef NOISE_AVAILABLE

        // To be completed
        if (job->SPdoNoise)
        {
            if (portPosNodes == NULL)
            {
                portPosNodes = TMALLOC(int, ckt->CKTportCount);
                portNegNodes = TMALLOC(int, ckt->CKTportCount);
                VSRCgetActivePortNodes(ckt->CKThead[vsrcRoot], ckt, portPosNodes, portNegNodes);
            }

            static Ndata* data;

            double realVal;
            double imagVal;
            int error;
            int posOutNode;
            int negOutNode;
            //Keep a backup copy
            memcpy(rhswoPorts, ckt->CKTrhs, ckt->CKTmaxEqNum * sizeof(double));
            memcpy(rhswoPorts, ckt->CKTirhs, ckt->CKTmaxEqNum * sizeof(double));

            for (activePort = 0; activePort < ckt->CKTportCount; activePort++)
            {
                /* the frequency will NOT be stored in array[0]  as before; instead,
                 * it will be given in refVal.rValue (see later)
                 */
                 // Copy the backup RHS into CKT's RHS
                memcpy(ckt->CKTrhs, rhswoPorts, ckt->CKTmaxEqNum * sizeof(double));
                memcpy(ckt->CKTirhs, irhswoPorts, ckt->CKTmaxEqNum * sizeof(double));
                ckt->CKTactivePort = activePort+1;

                posOutNode = portPosNodes[activePort];
                negOutNode = portNegNodes[activePort];
                NInspIter(ckt, posOutNode, negOutNode);   /* solve the adjoint system */

                /* now we use the adjoint system to calculate the noise
                 * contributions of each generator in the circuit
                 */

                error = CKTspnoise(ckt, N_DENS, N_CALC, data);
                if (error)
                {
                    tfree(portPosNodes); tfree(portNegNodes);
                    return(error);
                }
            }
        }
#endif
        /*  increment frequency */

        switch (job->SPstepType) {
        case DECADE:
        case OCTAVE:

            /* inserted again 14.12.2001  */
#ifdef HAS_PROGREP
        {
            double endfreq = job->SPstopFreq;
            double startfreq = job->SPstartFreq;
            endfreq = log(endfreq);
            if (startfreq == 0.0)
                startfreq = 1e-12;
            startfreq = log(startfreq);

            if (freq > 0.0)
                SetAnalyse("sp", (int)((log(freq) - startfreq) * 1000.0 / (endfreq - startfreq)));
        }
#endif

        freq *= job->SPfreqDelta;
        if (job->SPfreqDelta == 1) goto endsweep;
        break;
        case LINEAR:

#ifdef HAS_PROGREP
        {
            double endfreq = job->SPstopFreq;
            double startfreq = job->SPstartFreq;
            SetAnalyse("sp", (int)((freq - startfreq) * 1000.0 / (endfreq - startfreq)));
        }
#endif

        freq += job->SPfreqDelta;
        if (job->SPfreqDelta == 0) goto endsweep;
        break;
        default:
            tfree(rhswoPorts);
            tfree(irhswoPorts);
            tfree(portPosNodes); tfree(portNegNodes);
            return(E_INTERN);

        }
    }
endsweep:
    SPfrontEnd->OUTendPlot (spPlot);
    spPlot = NULL;
    UPDATE_STATS(0);
    tfree(rhswoPorts);
    tfree(irhswoPorts);
    tfree(portPosNodes); tfree(portNegNodes);
    return(0);
}


#endif
