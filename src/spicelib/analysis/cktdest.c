/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

    /* CKTdestroy(ckt)
     * this is a driver program to iterate through all the various
     * destroy functions provided for the circuit elements in the
     * given circuit 
     */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"

#ifdef XSPICE
#include "ngspice/evtproto.h"
#include "ngspice/mif.h"
#include "ngspice/enh.h"
#endif

int
CKTdestroy(CKTcircuit *ckt)
{
    int i;
    CKTnode *node;
    CKTnode *nnode;

    if (!ckt)
        return (E_NOTFOUND);


#ifdef WANT_SENSE2
    if(ckt->CKTsenInfo){
         if(ckt->CKTrhsOp) FREE(ckt->CKTrhsOp);
         if(ckt->CKTsenRhs) FREE(ckt->CKTsenRhs);
         if(ckt->CKTseniRhs) FREE(ckt->CKTseniRhs);
         SENdestroy(ckt->CKTsenInfo);
    }
#endif

    for (i = 0; i < DEVmaxnum; i++)
        if (DEVices[i]) {
            GENmodel *model = ckt->CKThead[i];
            while (model) {
                GENmodel *next_model = model->GENnextModel;
                GENinstance *inst = model->GENinstances;
                while (inst) {
                    GENinstance *next_inst = inst->GENnextInstance;
                    if (DEVices[i]->DEVdelete)
                        DEVices[i]->DEVdelete(inst);
                    GENinstanceFree(inst);
                    inst = next_inst;
                }
                if (DEVices[i]->DEVmodDelete)
                    DEVices[i]->DEVmodDelete(model);
                GENmodelFree(model);
                model = next_model;
            }
            if (DEVices[i]->DEVdestroy)
                DEVices[i]->DEVdestroy();
        }

    for(i=0;i<=ckt->CKTmaxOrder+1;i++){
        FREE(ckt->CKTstates[i]);
    }
    if(ckt->CKTmatrix) {
        SMPdestroy(ckt->CKTmatrix);
        ckt->CKTmatrix = NULL;
    }
    FREE(ckt->CKTbreaks);
    for(node = ckt->CKTnodes; node; ) {
        nnode = node->next;
        FREE(node);
        node = nnode;
    }
    ckt->CKTnodes = NULL;
    ckt->CKTlastNode = NULL;

    /* LTRA code addition */
    if (ckt->CKTtimePoints != NULL)
        FREE(ckt->CKTtimePoints);

    FREE(ckt->CKTrhs);
    FREE(ckt->CKTrhsOld);
    FREE(ckt->CKTrhsSpare);
    FREE(ckt->CKTirhs);
    FREE(ckt->CKTirhsOld);
    FREE(ckt->CKTirhsSpare);

#ifdef PREDICTOR
    if(ckt->CKTpred) FREE(ckt->CKTpred);
    for( i=0;i<8;i++) {
        if(ckt->CKTsols[i]) FREE(ckt->CKTsols[i]);
    }
#endif

    FREE(ckt->CKTstat->STATdevNum);
    FREE(ckt->CKTstat);
    FREE(ckt->CKThead);

#ifdef XSPICE
    EVTdest(ckt->evt);
    if (ckt->enh->rshunt_data.enabled)
        FREE(ckt->enh->rshunt_data.diag);
    FREE(ckt->enh);
    FREE(ckt->evt);
#endif

    nghash_free(ckt->DEVnameHash, NULL, NULL);
    nghash_free(ckt->MODnameHash, NULL, NULL);

#ifdef RFSPICE
    FREE(ckt->CKTrfPorts);
    freecmat(ckt->CKTAmat); ckt->CKTAmat = NULL;
    freecmat(ckt->CKTBmat); ckt->CKTBmat = NULL;
    freecmat(ckt->CKTSmat); ckt->CKTSmat = NULL;
    freecmat(ckt->CKTYmat); ckt->CKTYmat = NULL;
    freecmat(ckt->CKTZmat); ckt->CKTZmat = NULL;
    freecmat(ckt->CKTNoiseCYmat); ckt->CKTNoiseCYmat = NULL;
    freecmat(ckt->CKTadjointRHS); ckt->CKTadjointRHS = NULL;
#endif

    FREE(ckt);

#ifdef XSPICE
    g_mif_info.ckt = NULL;
#endif

    return(OK);
}
