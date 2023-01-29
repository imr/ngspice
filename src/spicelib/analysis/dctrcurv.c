/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 1999 Paolo Nenzi
**********/

#include "ngspice/ngspice.h"

#include "vsrc/vsrcdefs.h"
#include "isrc/isrcdefs.h"
#include "res/resdefs.h"

#include "ngspice/cktdefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/fteext.h"

#ifdef XSPICE
#include "ngspice/evt.h"
#include "ngspice/mif.h"
#include "ngspice/evtproto.h"
#include "ngspice/ipctiein.h"
#endif

#include "ngspice/devdefs.h"

#ifdef HAS_PROGREP
static double actval, actdiff;
#endif


int
DCtrCurv(CKTcircuit *ckt, int restart)
{
    TRCV *job = (TRCV *) ckt->CKTcurJob;

    int i;
    double *temp;
    int converged;
    int rcode;
    int vcode;
    int icode;
    int j;
    int error;
    IFuid varUid;
    IFuid *nameList;
    int numNames;
    int firstTime = 1;
    static runDesc *plot = NULL;

#ifdef WANT_SENSE2
    long save;
#ifdef SENSDEBUG
    if (ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & DCSEN)) {
        printf("\nDC Sensitivity Results\n\n");
        CKTsenPrint(ckt);
    }
#endif
#endif

    rcode = CKTtypelook("Resistor");
    vcode = CKTtypelook("Vsource");
    icode = CKTtypelook("Isource");

    if (!restart && job->TRCVnestState >= 0) {
        /* continuing */
        i = job->TRCVnestState;
        /* resume to work? saj*/
        error = SPfrontEnd->OUTpBeginPlot (NULL, NULL,
                                           NULL,
                                           NULL, 0,
                                           666, NULL, 666,
                                           &plot);
        goto resume;
    }

    ckt->CKTtime = 0;
    ckt->CKTdelta = job->TRCVvStep[0];
    ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCTRANCURVE | MODEINITJCT;
    ckt->CKTorder = 1;

    /* Save the state of the circuit */
    for (j = 0; j < 7; j++)
        ckt->CKTdeltaOld[j] = ckt->CKTdelta;

    for (i = 0; i <= job->TRCVnestLevel; i++) {

        if (rcode >= 0) {
            /* resistances are in this version, so use them */
            RESinstance *here;
            RESmodel *model;

            for (model = (RESmodel *)ckt->CKThead[rcode]; model; model = RESnextModel(model))
                for (here = RESinstances(model); here; here = RESnextInstance(here))
                    if (here->RESname == job->TRCVvName[i]) {
                        job->TRCVvElt[i]  = (GENinstance *)here;
                        job->TRCVvSave[i] = here->RESresist;
                        job->TRCVgSave[i] = here->RESresGiven;
                        job->TRCVvType[i] = rcode;
                        here->RESresist   = job->TRCVvStart[i];
                        here->RESresGiven = 1;
                        CKTtemp(ckt);
                        goto found;
                    }
        }

        if (vcode >= 0) {
            /* voltage sources are in this version, so use them */
            VSRCinstance *here;
            VSRCmodel *model;

            for (model = (VSRCmodel *)ckt->CKThead[vcode]; model; model = VSRCnextModel(model))
                for (here = VSRCinstances(model); here; here = VSRCnextInstance(here))
                    if (here->VSRCname == job->TRCVvName[i]) {
                        job->TRCVvElt[i]  = (GENinstance *)here;
                        job->TRCVvSave[i] = here->VSRCdcValue;
                        job->TRCVgSave[i] = here->VSRCdcGiven;
                        job->TRCVvType[i] = vcode;
                        here->VSRCdcValue = job->TRCVvStart[i];
                        here->VSRCdcGiven = 1;
                        goto found;
                    }
        }

        if (icode >= 0) {
            /* current sources are in this version, so use them */
            ISRCinstance *here;
            ISRCmodel *model;

            for (model = (ISRCmodel *)ckt->CKThead[icode]; model; model = ISRCnextModel(model))
                for (here = ISRCinstances(model); here; here = ISRCnextInstance(here))
                    if (here->ISRCname == job->TRCVvName[i]) {
                        job->TRCVvElt[i]  = (GENinstance *)here;
                        job->TRCVvSave[i] = here->ISRCdcValue;
                        job->TRCVgSave[i] = here->ISRCdcGiven;
                        job->TRCVvType[i] = icode;
                        here->ISRCdcValue = job->TRCVvStart[i];
                        here->ISRCdcGiven = 1;
                        goto found;
                    }
        }

        if (cieq(job->TRCVvName[i], "temp")) {
            job->TRCVvSave[i] = ckt->CKTtemp; /* Saves the old circuit temperature */
            job->TRCVvType[i] = TEMP_CODE;    /* Set the sweep type code */
            ckt->CKTtemp = job->TRCVvStart[i] + CONSTCtoK; /* Set the new circuit temp */
            inp_evaluate_temper(ft_curckt);
            CKTtemp(ckt);
            goto found;
        }

        SPfrontEnd->IFerrorf (ERR_FATAL,
                "DC Transfer Function: Voltage source, current source, or "
                "resistor named \"%s\" is not in the circuit",
                job->TRCVvName[i]);
        return(E_NODEV);

    found:;
    }

#ifdef HAS_PROGREP
    actval = job->TRCVvStart[job->TRCVnestLevel];
    actdiff = job->TRCVvStart[job->TRCVnestLevel] - job->TRCVvStop[job->TRCVnestLevel];
#endif

#ifdef XSPICE
/* gtri - add - wbk - 12/19/90 - Add IPC stuff and anal_init and anal_type */

    /* Tell the beginPlot routine what mode we're in */
    g_ipc.anal_type = IPC_ANAL_DCTRCURVE;

    /* Tell the code models what mode we're in */
    g_mif_info.circuit.anal_type = MIF_DC;

    g_mif_info.circuit.anal_init = MIF_TRUE;

/* gtri - end - wbk */
#endif

    error = CKTnames(ckt, &numNames, &nameList);
    if (error)
        return(error);

    if (job->TRCVvType[0] == vcode)
        SPfrontEnd->IFnewUid (ckt, &varUid, NULL, "v-sweep", UID_OTHER, NULL);
    else if (job->TRCVvType[0] == icode)
        SPfrontEnd->IFnewUid (ckt, &varUid, NULL, "i-sweep", UID_OTHER, NULL);
    else if (job->TRCVvType[0] == TEMP_CODE)
        SPfrontEnd->IFnewUid (ckt, &varUid, NULL, "temp-sweep", UID_OTHER, NULL);
    else if (job->TRCVvType[0] == rcode)
        SPfrontEnd->IFnewUid (ckt, &varUid, NULL, "res-sweep", UID_OTHER, NULL);
    else
        SPfrontEnd->IFnewUid (ckt, &varUid, NULL, "?-sweep", UID_OTHER, NULL);

    error = SPfrontEnd->OUTpBeginPlot (ckt, ckt->CKTcurJob,
                                       ckt->CKTcurJob->JOBname,
                                       varUid, IF_REAL,
                                       numNames, nameList, IF_REAL,
                                       &plot);
    tfree(nameList);

    if (error)
        return(error);

    /* initialize CKTsoaCheck `warn' counters */
    if (ckt->CKTsoaCheck)
        error = CKTsoaInit();

    /* now have finished the initialization - can start doing hard part */

    i = 0;

 resume:

    for (;;) {

        if (job->TRCVvType[i] == vcode) { /* voltage source */
            if (SGN(job->TRCVvStep[i]) *
                (((VSRCinstance*)(job->TRCVvElt[i]))->VSRCdcValue -
                 job->TRCVvStop[i]) >
                DBL_EPSILON * 1e+03)
            {
                i++;
                firstTime = 1;
                ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCTRANCURVE | MODEINITJCT;
                if (i > job->TRCVnestLevel)
                    break;
                goto nextstep;
            }
        } else if (job->TRCVvType[i] == icode) { /* current source */
            if (SGN(job->TRCVvStep[i]) *
                (((ISRCinstance*)(job->TRCVvElt[i]))->ISRCdcValue -
                 job->TRCVvStop[i]) >
                DBL_EPSILON * 1e+03)
            {
                i++;
                firstTime = 1;
                ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCTRANCURVE | MODEINITJCT;
                if (i > job->TRCVnestLevel)
                    break;
                goto nextstep;
            }
        } else if (job->TRCVvType[i] == rcode) { /* resistance */
            if (SGN(job->TRCVvStep[i]) *
                (((RESinstance*)(job->TRCVvElt[i]))->RESresist -
                 job->TRCVvStop[i]) >
                DBL_EPSILON * 1e+03)
            {
                i++;
                firstTime = 1;
                ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCTRANCURVE | MODEINITJCT;
                if (i > job->TRCVnestLevel)
                    break;
                goto nextstep;
            }
        } else if (job->TRCVvType[i] == TEMP_CODE) { /* temp sweep */
            if (SGN(job->TRCVvStep[i]) *
                ((ckt->CKTtemp - CONSTCtoK) - job->TRCVvStop[i]) >
                DBL_EPSILON * 1e+03)
            {
                i++;
                firstTime = 1;
                ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCTRANCURVE | MODEINITJCT;
                if (i > job->TRCVnestLevel)
                    break;
                goto nextstep;
            }
        }

        while (--i >= 0)
            if (job->TRCVvType[i] == vcode) { /* voltage source */
                ((VSRCinstance *)(job->TRCVvElt[i]))->VSRCdcValue =
                    job->TRCVvStart[i];
            } else if (job->TRCVvType[i] == icode) { /* current source */
                ((ISRCinstance *)(job->TRCVvElt[i]))->ISRCdcValue =
                    job->TRCVvStart[i];
            } else if (job->TRCVvType[i] == TEMP_CODE) {
                ckt->CKTtemp = job->TRCVvStart[i] + CONSTCtoK;
                inp_evaluate_temper(ft_curckt);
                CKTtemp(ckt);
            } else if (job->TRCVvType[i] == rcode) {
                ((RESinstance *)(job->TRCVvElt[i]))->RESresist =
                    job->TRCVvStart[i];
                RESupdate_conduct((RESinstance *)(job->TRCVvElt[i]), FALSE);
                DEVices[rcode]->DEVload(job->TRCVvElt[i]->GENmodPtr, ckt);
            }

        /* Rotate state vectors. */
        temp = ckt->CKTstates[ckt->CKTmaxOrder + 1];
        for (j = ckt->CKTmaxOrder; j >= 0; j--)
            ckt->CKTstates[j + 1] = ckt->CKTstates[j];
        ckt->CKTstate0 = temp;

        /* do operation */
#ifdef XSPICE
/* gtri - begin - wbk - Do EVTop if event instances exist */
        if (ckt->evt->counts.num_insts == 0) {
            /* If no event-driven instances, do what SPICE normally does */
#endif
            converged = NIiter(ckt, ckt->CKTdcTrcvMaxIter);
            if (converged != 0) {
                converged = CKTop(ckt,
                                  (ckt->CKTmode & MODEUIC) | MODEDCTRANCURVE | MODEINITJCT,
                                  (ckt->CKTmode & MODEUIC) | MODEDCTRANCURVE | MODEINITFLOAT,
                                  ckt->CKTdcMaxIter);
                if (converged != 0)
                    return(converged);
            }
#ifdef XSPICE
        }
        else {
            /* else do new algorithm */

            /* first get the current step in the analysis */
            if (job->TRCVvType[0] == vcode) {
                g_mif_info.circuit.evt_step =
                    ((VSRCinstance *)(job->TRCVvElt[0]))->VSRCdcValue;
            } else if (job->TRCVvType[0] == icode) {
                g_mif_info.circuit.evt_step =
                    ((ISRCinstance *)(job->TRCVvElt[0]))->ISRCdcValue;
            } else if (job->TRCVvType[0] == rcode) {
                g_mif_info.circuit.evt_step =
                    ((RESinstance*)(job->TRCVvElt[0]->GENmodPtr))->RESresist;
            } else if (job->TRCVvType[0] == TEMP_CODE) {
                g_mif_info.circuit.evt_step =
                    ckt->CKTtemp - CONSTCtoK;
            }

            /* if first time through, call EVTop immediately and save event results */
            if (firstTime) {
                converged = EVTop(ckt,
                                  (ckt->CKTmode & MODEUIC) | MODEDCTRANCURVE | MODEINITJCT,
                                  (ckt->CKTmode & MODEUIC) | MODEDCTRANCURVE | MODEINITFLOAT,
                                  ckt->CKTdcMaxIter,
                                  MIF_TRUE);
                EVTdump(ckt, IPC_ANAL_DCOP, g_mif_info.circuit.evt_step);
                EVTop_save(ckt, MIF_FALSE, g_mif_info.circuit.evt_step);
                if (converged != 0)
                    return(converged);
            }
            /* else, call NIiter first with mode = MODEINITPRED */
            /* to attempt quick analog solution.  Then call all hybrids and call */
            /* EVTop only if event outputs have changed, or if non-converged */
            else {
                converged = NIiter(ckt, ckt->CKTdcTrcvMaxIter);
                EVTcall_hybrids(ckt);
                if ((converged != 0) || (ckt->evt->queue.output.num_changed != 0)) {
                    converged = EVTop(ckt,
                                      (ckt->CKTmode & MODEUIC) | MODEDCTRANCURVE | MODEINITJCT,
                                      (ckt->CKTmode & MODEUIC) | MODEDCTRANCURVE | MODEINITFLOAT,
                                      ckt->CKTdcMaxIter,
                                      MIF_FALSE);
                    EVTdump(ckt, IPC_ANAL_DCTRCURVE, g_mif_info.circuit.evt_step);
                    EVTop_save(ckt, MIF_FALSE, g_mif_info.circuit.evt_step);
                    if (converged != 0)
                        return(converged);
                }
            }
        }
/* gtri - end - wbk - Do EVTop if event instances exist */
#endif

        ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCTRANCURVE | MODEINITPRED;
        if (job->TRCVvType[0] == vcode)
            ckt->CKTtime = ((VSRCinstance *)(job->TRCVvElt[0]))->VSRCdcValue;
        else if (job->TRCVvType[0] == icode)
            ckt->CKTtime = ((ISRCinstance *)(job->TRCVvElt[0]))->ISRCdcValue;
        else if (job->TRCVvType[0] == rcode)
            ckt->CKTtime = ((RESinstance *)(job->TRCVvElt[0]))->RESresist;
        else if (job->TRCVvType[0] == TEMP_CODE)
            ckt->CKTtime = ckt->CKTtemp - CONSTCtoK;

#ifdef XSPICE
/* gtri - add - wbk - 12/19/90 - Add IPC stuff */

        /* If first time through, call CKTdump to output Operating Point info */
        /* for Mspice compatibility */

        if (((g_ipc.enabled) || wantevtdata) && firstTime) {
            ipc_send_dcop_prefix();
            CKTdump(ckt, 0.0, plot);
            ipc_send_dcop_suffix();
        }

/* gtri - end - wbk */
#endif

#ifdef WANT_SENSE2
/*
  if (!ckt->CKTsenInfo) printf("sensitivity structure does not exist\n");
*/
        if (ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & DCSEN)) {
            int senmode;

#ifdef SENSDEBUG
            if (job->TRCVvType[0] == vcode) { /* voltage source */
                printf("Voltage Source Value : %.5e V\n",
                       ((VSRCinstance*) (job->TRCVvElt[0]))->VSRCdcValue);
            }
            if (job->TRCVvType[0] == icode) { /* current source */
                printf("Current Source Value : %.5e A\n",
                       ((ISRCinstance*)(job->TRCVvElt[0]))->ISRCdcValue);
            }
            if (job->TRCVvType[0] == rcode) { /* resistance */
                printf("Current Resistance Value : %.5e Ohm\n",
                       ((RESinstance*)(job->TRCVvElt[0]->GENmodPtr))->RESresist);
            }
            if (job->TRCVvType[0] == TEMP_CODE) { /* Temperature */
                printf("Current Circuit Temperature : %.5e C\n",
                       ckt->CKTtemp - CONSTCtoK);
            }
#endif

            senmode = ckt->CKTsenInfo->SENmode;
            save = ckt->CKTmode;
            ckt->CKTsenInfo->SENmode = DCSEN;
            error = CKTsenDCtran(ckt);
            if (error)
                return(error);

            ckt->CKTmode = save;
            ckt->CKTsenInfo->SENmode = senmode;
        }
#endif

#ifdef XSPICE
/* gtri - modify - wbk - 12/19/90 - Send IPC delimiters */

        if (g_ipc.enabled)
            ipc_send_data_prefix(ckt->CKTtime);
#endif

        CKTdump(ckt,ckt->CKTtime,plot);

        if (ckt->CKTsoaCheck)
            error = CKTsoaCheck(ckt);

#ifdef XSPICE
        if (g_ipc.enabled)
            ipc_send_data_suffix();

/* gtri - end - wbk */
#endif

        if (firstTime) {
            firstTime = 0;
            memcpy(ckt->CKTstate1, ckt->CKTstate0,
                   (size_t) ckt->CKTnumStates * sizeof(double));
        }

        i = 0;

    nextstep:;

        if (job->TRCVvType[i] == vcode) { /* voltage source */
            ((VSRCinstance*)(job->TRCVvElt[i]))->VSRCdcValue +=
                job->TRCVvStep[i];
        } else if (job->TRCVvType[i] == icode) { /* current source */
            ((ISRCinstance*)(job->TRCVvElt[i]))->ISRCdcValue +=
                job->TRCVvStep[i];
        } else if (job->TRCVvType[i] == rcode) { /* resistance */
            ((RESinstance*)(job->TRCVvElt[i]))->RESresist +=
                job->TRCVvStep[i];
            RESupdate_conduct((RESinstance *)(job->TRCVvElt[i]), FALSE);
            DEVices[rcode]->DEVload(job->TRCVvElt[i]->GENmodPtr, ckt);
        } else if (job->TRCVvType[i] == TEMP_CODE) { /* temperature */
            ckt->CKTtemp += job->TRCVvStep[i];

            /* FIXME: Do the Temp check already here for the first time.
               If the stop criterion is fulfilled, discard Temp evaluation, because
               CKTtemp may report errors if a large extra Temp step is exercized. */
            if (SGN(job->TRCVvStep[i]) *
                ((ckt->CKTtemp - CONSTCtoK) - job->TRCVvStop[i]) > DBL_EPSILON * 1e+03) {
//                ckt->CKTtemp -= job->TRCVvStep[i]; // Undo the large step
//                ckt->CKTtemp += SGN(job->TRCVvStep[i]) * DBL_EPSILON * 2e+03; // Add just a small step
                continue; // Skip model evaluation
            }

            inp_evaluate_temper(ft_curckt);
            CKTtemp(ckt);
        }

        if (SPfrontEnd->IFpauseTest()) {
            /* user asked us to pause, so save state */
            job->TRCVnestState = i;
            return(E_PAUSE);
        }

#ifdef HAS_PROGREP
        if (i == job->TRCVnestLevel) {
            actval += job->TRCVvStep[job->TRCVnestLevel];
            SetAnalyse("dc", abs((int)((actval - job->TRCVvStart[job->TRCVnestLevel]) * 1000. / actdiff)));
        }
#endif

    }

    /* all done, lets put everything back */

    for (i = 0; i <= job->TRCVnestLevel; i++)
        if (job->TRCVvType[i] == vcode) {   /* voltage source */
            ((VSRCinstance*)(job->TRCVvElt[i]))->VSRCdcValue = job->TRCVvSave[i];
            ((VSRCinstance*)(job->TRCVvElt[i]))->VSRCdcGiven = (job->TRCVgSave[i] != 0);
        } else  if (job->TRCVvType[i] == icode) { /*current source */
            ((ISRCinstance*)(job->TRCVvElt[i]))->ISRCdcValue = job->TRCVvSave[i];
            ((ISRCinstance*)(job->TRCVvElt[i]))->ISRCdcGiven = (job->TRCVgSave[i] != 0);
        } else  if (job->TRCVvType[i] == rcode) { /* Resistance */
            ((RESinstance*)(job->TRCVvElt[i]))->RESresist = job->TRCVvSave[i];
            ((RESinstance*)(job->TRCVvElt[i]))->RESresGiven = (job->TRCVgSave[i] != 0);
            RESupdate_conduct((RESinstance *)(job->TRCVvElt[i]), TRUE);
            DEVices[rcode]->DEVload(job->TRCVvElt[i]->GENmodPtr, ckt);
        } else if (job->TRCVvType[i] == TEMP_CODE) {
            ckt->CKTtemp = job->TRCVvSave[i];
            inp_evaluate_temper(ft_curckt);
            CKTtemp(ckt);
        }

    SPfrontEnd->OUTendPlot (plot);

    return(OK);
}
