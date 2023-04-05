/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified 2001: AlansFixes
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/acdefs.h"
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
ACan(CKTcircuit* ckt, int restart)
{
    ACAN* job = (ACAN*)ckt->CKTcurJob;

    double freq;
    double freqTol; /* tolerence parameter for finding final frequency */
    double startdTime;
    double startsTime;
    double startlTime;
    double startkTime;
    double startTime;
    int error;
    int numNames;
    IFuid* nameList;  /* va: tmalloc'ed list of names */
    IFuid freqUid;
    static runDesc* acPlot = NULL;
    runDesc* plot = NULL;


#ifdef XSPICE
    /* gtri - add - wbk - 12/19/90 - Add IPC stuff and anal_init and anal_type */

        /* Tell the beginPlot routine what mode we're in */
    g_ipc.anal_type = IPC_ANAL_AC;

    /* Tell the code models what mode we're in */
    g_mif_info.circuit.anal_type = MIF_DC;
    g_mif_info.circuit.anal_init = MIF_TRUE;

    /* gtri - end - wbk */
#endif

    /* start at beginning */
    if (job->ACsaveFreq == 0 || restart) {
        if (job->ACnumberSteps < 1)
            job->ACnumberSteps = 1;

        switch (job->ACstepType) {

        case DECADE:
            if (job->ACstartFreq <= 0) {
                fprintf(stderr, "ERROR: AC startfreq <= 0\n");
                return E_PARMVAL;
            }
            double num_steps = floor(fabs(log10(job->ACstopFreq / job->ACstartFreq)) * job->ACnumberSteps);
            job->ACfreqDelta = exp((log(job->ACstopFreq / job->ACstartFreq)) / num_steps);

            break;
        case OCTAVE:
            if (job->ACstartFreq <= 0) {
                fprintf(stderr, "ERROR: AC startfreq <= 0\n");
                return E_PARMVAL;
            }
            job->ACfreqDelta =
                exp(log(2.0) / job->ACnumberSteps);
            break;
        case LINEAR:
            if (job->ACnumberSteps - 1 > 1)
                job->ACfreqDelta =
                (job->ACstopFreq -
                    job->ACstartFreq) /
                (job->ACnumberSteps - 1);
            else
                /* Patch from: Richard McRoberts
                * This patch is for a rather pathological case:
                * a linear step with only one point */
                job->ACfreqDelta = 0;
            break;
        default:
            return(E_BADPARM);
        }
#ifdef XSPICE
        /* gtri - begin - wbk - Call EVTop if event-driven instances exist */

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
            /* If no event-driven instances, do what SPICE normally does */
            if (!ckt->CKTnoopac) { /* skip OP if option NOOPAC is set and circuit is linear */
                error = CKTop(ckt,
                    (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITJCT,
                    (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITFLOAT,
                    ckt->CKTdcMaxIter);

                if (error) {
                    fprintf(stdout, "\nAC operating point failed -\n");
                    CKTncDump(ckt);
                    return(error);
                }
            }
            else
                fprintf(stdout, "\n Linear circuit, option noopac given: no OP analysis\n");

#ifdef XSPICE
        /* gtri - add - wbk - 12/19/90 - Add IPC stuff */

            /* Send the operating point results for Mspice compatibility */
        if (g_ipc.enabled)
        {
            /* Call CKTnames to get names of nodes/branches used by
                BeginPlot */
                /* Probably should free nameList after this block since
                    called again... */
            error = CKTnames(ckt, &numNames, &nameList);
            if (error) return(error);

            /* We have to do a beginPlot here since the data to return is
             * different for the DCOP than it is for the AC analysis.
             * Moreover the begin plot has not even been done yet at this
             * point...
             */
            SPfrontEnd->OUTpBeginPlot(ckt, ckt->CKTcurJob,
                ckt->CKTcurJob->JOBname,
                NULL, IF_REAL,
                numNames, nameList, IF_REAL,
                &acPlot);
            txfree(nameList);

            ipc_send_dcop_prefix();
            CKTdump(ckt, 0.0, acPlot);
            ipc_send_dcop_suffix();

            SPfrontEnd->OUTendPlot(acPlot);
        }
        /* gtri - end - wbk */
#endif

        ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITSMSIG;
        error = CKTload(ckt);
        if (error) return(error);

        error = CKTnames(ckt, &numNames, &nameList);
        if (error) return(error);

        if (ckt->CKTkeepOpInfo) {
            /* Dump operating point. */
            error = SPfrontEnd->OUTpBeginPlot(ckt, ckt->CKTcurJob,
                "AC Operating Point",
                NULL, IF_REAL,
                numNames, nameList, IF_REAL,
                &plot);
            if (error) return(error);
            CKTdump(ckt, 0.0, plot);
            SPfrontEnd->OUTendPlot(plot);
            plot = NULL;
        }

        SPfrontEnd->IFnewUid(ckt, &freqUid, NULL, "frequency", UID_OTHER, NULL);
        error = SPfrontEnd->OUTpBeginPlot(ckt, ckt->CKTcurJob,
            ckt->CKTcurJob->JOBname,
            freqUid, IF_REAL,
            numNames, nameList, IF_COMPLEX,
            &acPlot);
        tfree(nameList);
        if (error) return(error);

        if (job->ACstepType != LINEAR) {
            SPfrontEnd->OUTattributes(acPlot, NULL, OUT_SCALE_LOG, NULL);
        }
        freq = job->ACstartFreq;

    }
    else {    /* continue previous analysis */
        freq = job->ACsaveFreq;
        job->ACsaveFreq = 0; /* clear the 'old' frequency */
        /* fix resume? saj, indeed !*/
        error = SPfrontEnd->OUTpBeginPlot(NULL, NULL,
            NULL,
            NULL, 0,
            666, NULL, 666,
            &acPlot);
        /* saj*/
    }

    switch (job->ACstepType) {
    case DECADE:
    case OCTAVE:
        freqTol = job->ACfreqDelta *
            job->ACstopFreq * ckt->CKTreltol;
        break;
    case LINEAR:
        freqTol = job->ACfreqDelta * ckt->CKTreltol;
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

    /* main loop through all scheduled frequencies */
    while (freq <= job->ACstopFreq + freqTol) {
        if (SPfrontEnd->IFpauseTest()) {
            /* user asked us to pause via an interrupt */
            job->ACsaveFreq = freq;
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

        ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEAC;
        error = NIacIter(ckt);
        if (error) {
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
            if (freq == job->ACstartFreq) {
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

#ifdef XSPICE
        /* gtri - modify - wbk - 12/19/90 - Send IPC stuff */

        if (g_ipc.enabled)
            ipc_send_data_prefix(freq);

        error = CKTacDump(ckt, freq, acPlot);

        if (g_ipc.enabled)
            ipc_send_data_suffix();

        /* gtri - modify - wbk - 12/19/90 - Send IPC stuff */
#else
        error = CKTacDump(ckt, freq, acPlot);
#endif	
        if (error) {
            UPDATE_STATS(DOING_AC);
            return(error);
        }

        /*  increment frequency */

        switch (job->ACstepType) {
        case DECADE:
        case OCTAVE:

            /* inserted again 14.12.2001  */
#ifdef HAS_PROGREP
        {
            double endfreq = job->ACstopFreq;
            double startfreq = job->ACstartFreq;
            endfreq = log(endfreq);
            if (startfreq == 0.0)
                startfreq = 1e-12;
            startfreq = log(startfreq);

            if (freq > 0.0)
                SetAnalyse("ac", (int)((log(freq) - startfreq) * 1000.0 / (endfreq - startfreq)));
        }
#endif

        freq *= job->ACfreqDelta;
        if (job->ACfreqDelta == 1) goto endsweep;
        break;
        case LINEAR:

#ifdef HAS_PROGREP
        {
            double endfreq = job->ACstopFreq;
            double startfreq = job->ACstartFreq;
            SetAnalyse("ac", (int)((freq - startfreq) * 1000.0 / (endfreq - startfreq)));
        }
#endif

        freq += job->ACfreqDelta;
        if (job->ACfreqDelta == 0) goto endsweep;
        break;
        default:
            return(E_INTERN);

        }

    }
endsweep:
    SPfrontEnd->OUTendPlot(acPlot);
    acPlot = NULL;
    UPDATE_STATS(0);
    return(0);
}


/* CKTacLoad(ckt)
 * this is a driver program to iterate through all the various
 * ac load functions provided for the circuit elements in the
 * given circuit
 */


int
CKTacLoad(CKTcircuit* ckt)
{
    int i;
    int size;
    int error;
    double startTime;

    startTime = SPfrontEnd->IFseconds();
    size = SMPmatSize(ckt->CKTmatrix);
    for (i = 0; i <= size; i++) {
        ckt->CKTrhs[i] = 0;
        ckt->CKTirhs[i] = 0;
    }
    SMPcClear(ckt->CKTmatrix);

    for (i = 0; i < DEVmaxnum; i++) {
        if (DEVices[i] && DEVices[i]->DEVacLoad && ckt->CKThead[i]) {
            error = DEVices[i]->DEVacLoad(ckt->CKThead[i], ckt);
            if (error) return(error);
        }
    }

#ifdef XSPICE
    /* gtri - begin - Put resistors to ground at all nodes. */
     /* Value of resistor is set by new "rshunt" option.     */

    if (ckt->enh->rshunt_data.enabled) {
        for (i = 0; i < ckt->enh->rshunt_data.num_nodes; i++) {
            *(ckt->enh->rshunt_data.diag[i]) +=
                ckt->enh->rshunt_data.gshunt;
        }
    }

    /* gtri - end - Put resistors to ground at all nodes */



    /* gtri - add - wbk - 11/26/90 - reset the MIF init flags */

    /* init is set by CKTinit and should be true only for first load call */
    g_mif_info.circuit.init = MIF_FALSE;

    /* anal_init is set by CKTdoJob and is true for first call */
    /* of a particular analysis type */
    g_mif_info.circuit.anal_init = MIF_FALSE;

    /* gtri - end - wbk - 11/26/90 */
#endif


    ckt->CKTstat->STATloadTime += SPfrontEnd->IFseconds() - startTime;
    return(OK);
}
