/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modifed: 2000 AlansFixes
**********/
#include <stdlib.h>
#include <config.h>
#include "memory.h"
#include <cktdefs.h>
#include <devdefs.h>
#include <sperror.h>
#include <fteext.h>
#include <ifsim.h>
#include <dev.h>

#ifdef XSPICE
/* gtri - add - wbk - 11/26/90 - add include for MIF global data */
#include "mif.h"
/* gtri - end - wbk - 11/26/90 */
#endif

int
CKTinit(void **ckt)		/* new circuit to create */
{
    extern void load_alldevs(void);
    int i;
    CKTcircuit *sckt;
    *ckt = (void *) tmalloc(sizeof(CKTcircuit));
    sckt = (CKTcircuit *)(*ckt);
    if (sckt == NULL)
	return(E_NOMEM);
/* gtri - begin - dynamically allocate the array of model lists */
/* CKThead used to be statically sized in CKTdefs.h, but has been changed */
/* to a ** pointer */
    (sckt)->CKThead = (GENmodel **)MALLOC(DEVmaxnum * sizeof(GENmodel *));
    if((sckt)->CKThead == NULL) return(E_NOMEM);
/* gtri - end   - dynamically allocate the array of model lists */


	
    for (i = 0; i < DEVmaxnum; i++)
        sckt->CKThead[i] = (GENmodel *) NULL;

    sckt->CKTmaxEqNum = 1;
    sckt->CKTnodes = (CKTnode *) NULL;
    sckt->CKTlastNode = (CKTnode *) NULL;
    sckt->CKTmatrix = NULL;

    sckt->CKTgmin = 1e-12;
    sckt->CKTgshunt=0;
    sckt->CKTabstol = 1e-12;
    sckt->CKTreltol = 1e-3;
    sckt->CKTchgtol = 1e-14;
    sckt->CKTvoltTol = 1e-6;
    sckt->CKTtrtol = 7;
    sckt->CKTbypass = 0;
    sckt->CKTisSetup = 0;
    sckt->CKTtranMaxIter = 10;
    sckt->CKTdcMaxIter = 100;
    sckt->CKTdcTrcvMaxIter = 50;
    sckt->CKTintegrateMethod = TRAPEZOIDAL;
    sckt->CKTorder = 1;
    sckt->CKTmaxOrder = 2;
    sckt->CKTpivotAbsTol = 1e-13;
    sckt->CKTpivotRelTol = 1e-3;
    sckt->CKTtemp = 300.15;
    sckt->CKTnomTemp = 300.15;
    sckt->CKTdefaultMosM = 1;
    sckt->CKTdefaultMosL = 1e-4;
    sckt->CKTdefaultMosW = 1e-4;
    sckt->CKTdefaultMosAD = 0;
    sckt->CKTdefaultMosAS = 0;
    sckt->CKTsrcFact=1;
    sckt->CKTdiagGmin=0;
    sckt->CKTstat = (STATistics *) tmalloc(sizeof(STATistics));
    sckt->CKTtroubleNode = 0;
    sckt->CKTtroubleElt = NULL;
    sckt->CKTtimePoints = NULL;
    if (sckt->CKTstat == (STATistics *)NULL)
	return E_NOMEM;
    sckt->CKTnodeDamping = 0;
    sckt->CKTabsDv = 0.5;
    sckt->CKTrelDv = 2.0;

#ifdef XSPICE
/* gtri - begin - wbk - allocate/initialize substructs */

    /* Allocate evt data structure */
    (sckt)->evt = (void *) MALLOC(sizeof(Evt_Ckt_Data_t));
    if(! (sckt)->evt)
        return(E_NOMEM);

    /* Initialize options data */
    (sckt)->evt->options.op_alternate = MIF_TRUE;

    /* Allocate enh data structure */
    (sckt)->enh = (void *) MALLOC(sizeof(Enh_Ckt_Data_t));
    if(! (sckt)->enh)
        return(E_NOMEM);

    /* Initialize non-zero, non-NULL data */
    (sckt)->enh->breakpoint.current = 1.0e30;
    (sckt)->enh->breakpoint.last = 1.0e30;
    (sckt)->enh->ramp.ramptime = 0.0;
    (sckt)->enh->conv_limit.enabled = MIF_TRUE;
    (sckt)->enh->conv_limit.step = 0.25;
    (sckt)->enh->conv_limit.abs_step = 0.1;
    (sckt)->enh->rshunt_data.enabled = MIF_FALSE;

/* gtri - end - wbk - allocate/initialize substructs */

/* gtri - add - wbk - 01/12/91 - initialize g_mif_info */
    g_mif_info.circuit.init =      MIF_TRUE;
    g_mif_info.circuit.anal_init = MIF_TRUE;
    g_mif_info.circuit.anal_type = MIF_DC;
    g_mif_info.instance          = NULL;
    g_mif_info.ckt               = sckt;
    g_mif_info.errmsg            = NULL;
    g_mif_info.auto_partial.global = MIF_FALSE;
    g_mif_info.auto_partial.local = MIF_FALSE;
/* gtri - end - wbk - 01/12/91 */
#endif
    return OK;
}
