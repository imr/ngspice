/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000  AlansFixes
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/ifsim.h"

#ifdef XSPICE
/* gtri - add - wbk - 12/19/90 - Add headers */
#include "ngspice/mif.h"
#include "ngspice/evt.h"
#include "ngspice/evtproto.h"
#include "ngspice/ipctiein.h"
/* gtri - end - wbk */
#endif

int
DCop(CKTcircuit *ckt, int notused)
{
#ifdef WANT_SENSE2
    int i, senmode, size;
    long save;
#endif

    int converged;
    int error;
    IFuid *nameList; /* va: tmalloc'ed list */
    int numNames;
    runDesc *plot = NULL;

    NG_IGNORE(notused);
  
#ifdef XSPICE
/* gtri - add - wbk - 12/19/90 - Add IPC stuff and initialize anal_init and anal_type */

    /* Tell the beginPlot routine what mode we're in */
    g_ipc.anal_type = IPC_ANAL_DCOP;

    /* Tell the code models what mode we're in */
    g_mif_info.circuit.anal_type = MIF_DC;

    g_mif_info.circuit.anal_init = MIF_TRUE;

/* gtri - end - wbk */
#endif  

    error = CKTnames(ckt,&numNames,&nameList);
    if(error) return(error);
    error = SPfrontEnd->OUTpBeginPlot (ckt, ckt->CKTcurJob,
                                       ckt->CKTcurJob->JOBname,
                                       NULL, IF_REAL,
                                       numNames, nameList, IF_REAL,
                                       &plot);
    tfree(nameList); /* va: nameList not used any longer, it was a memory leak */
    if(error) return(error);

    /* initialize CKTsoaCheck `warn' counters */
    if (ckt->CKTsoaCheck)
        error = CKTsoaInit();

#ifdef XSPICE
/* gtri - begin - wbk - 6/10/91 - Call EVTop if event-driven instances exist */
    if(ckt->evt->counts.num_insts != 0) {
        /* use new DCOP algorithm */
        converged = EVTop(ckt,
                    (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITJCT,
                    (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITFLOAT,
                    ckt->CKTdcMaxIter,
                    MIF_TRUE);
        EVTdump(ckt, IPC_ANAL_DCOP, 0.0);
	
        EVTop_save(ckt, MIF_TRUE, 0.0);
	/* gtri - end - wbk - 6/10/91 - Call EVTop if event-driven instances exist */
	} else
        /* If no event-driven instances, do what SPICE normally does */
#endif
    converged = CKTop(ckt,
            (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITJCT,
            (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITFLOAT,
            ckt->CKTdcMaxIter);

    if(converged != 0) {
        fprintf(stdout,"\nDC solution failed -\n");
        CKTncDump(ckt);
        return(converged);
    }

    ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITSMSIG;

#ifdef WANT_SENSE2
    if(ckt->CKTsenInfo && ((ckt->CKTsenInfo->SENmode&DCSEN) || 
            (ckt->CKTsenInfo->SENmode&ACSEN)) ){
#ifdef SENSDEBUG
         printf("\nDC Operating Point Sensitivity Results\n\n");
         CKTsenPrint(ckt);
#endif /* SENSDEBUG */
         senmode = ckt->CKTsenInfo->SENmode;
         save = ckt->CKTmode;
         ckt->CKTsenInfo->SENmode = DCSEN;
         size = SMPmatSize(ckt->CKTmatrix);
         for(i = 1; i<=size ; i++){
             ckt->CKTrhsOp[i] = ckt->CKTrhsOld[i];
         }
         error = CKTsenDCtran(ckt);
         if (error)
             return(error);

         ckt->CKTmode = save;
         ckt->CKTsenInfo->SENmode = senmode;

    }
#endif
    converged = CKTload(ckt);
#ifdef XSPICE
/* gtri - modify - wbk - 12/19/90 - Send IPC data delimiters */

    if(g_ipc.enabled)
        ipc_send_dcop_prefix();

    CKTdump(ckt, 0.0, plot);

    if (ckt->CKTsoaCheck)
        error = CKTsoaCheck(ckt);

    if(g_ipc.enabled)
        ipc_send_dcop_suffix();

/* gtri - end - wbk */
#else
    if(converged == 0) {
	   CKTdump(ckt, 0.0, plot);
           if (ckt->CKTsoaCheck)
              error = CKTsoaCheck(ckt);
         } else {
           fprintf(stderr,"error: circuit reload failed.\n");
         }
#endif
    SPfrontEnd->OUTendPlot (plot);
    return(converged);
}
