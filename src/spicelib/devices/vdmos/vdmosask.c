/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "vdmosdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/*ARGSUSED*/
int
VDMOSask(CKTcircuit *ckt, GENinstance *inst, int which, IFvalue *value,
        IFvalue *select)
{
    VDMOSinstance *here = (VDMOSinstance*)inst;
    static char *msg = "Current and power not available for ac analysis";
    NG_IGNORE(select);

    switch(which) {
        case VDMOS_TEMP:
            value->rValue = here->VDMOStemp - CONSTCtoK;
            return(OK);
        case VDMOS_DTEMP:
            value->rValue = here->VDMOSdtemp;
            return(OK);
        case VDMOS_CAPGS:
            value->rValue = 2*  *(ckt->CKTstate0 + here->VDMOScapgs);
            return(OK);
        case VDMOS_CAPGD:
            value->rValue = 2* *(ckt->CKTstate0 + here->VDMOScapgd);
            return(OK);
        case VDMOS_CAPDS:
            value->rValue = here->VDIOcap;
            return(OK);
        case VDMOS_M:
            value->rValue = here->VDMOSm;
            return(OK);
        case VDMOS_OFF:
            value->iValue = here->VDMOSoff;
            return(OK);
        case VDMOS_THERMAL:
            value->iValue = here->VDMOSthermal;
            return(OK);
        case VDMOS_IC_VDS:
            value->rValue = here->VDMOSicVDS;
            return(OK);
        case VDMOS_IC_VGS:
            value->rValue = here->VDMOSicVGS;
            return(OK);
        case VDMOS_DNODE:
            value->iValue = here->VDMOSdNode;
            return(OK);
        case VDMOS_GNODE:
            value->iValue = here->VDMOSgNode;
            return(OK);
        case VDMOS_SNODE:
            value->iValue = here->VDMOSsNode;
            return(OK);
        case VDMOS_TNODE:
            value->iValue = here->VDMOStempNode;
            return(OK);
        case VDMOS_SNODEPRIME:
            value->iValue = here->VDMOSsNodePrime;
            return(OK);
        case VDMOS_SOURCECONDUCT:
            value->rValue = here->VDMOSsourceConductance;
            return(OK);
        case VDMOS_SOURCERESIST:
            if (here->VDMOSsNodePrime != here->VDMOSsNode)
                value->rValue = 1.0 / here->VDMOSsourceConductance;
            else
                value->rValue = 0.0;
            return(OK);
        case VDMOS_DRAINCONDUCT:
            value->rValue = here->VDMOSdrainConductance;
            return(OK);
        case VDMOS_DRAINRESIST:
            if (here->VDMOSdNodePrime != here->VDMOSdNode)
                value->rValue = 1.0 / here->VDMOSdrainConductance;
            else
                value->rValue = 0.0;
            return(OK);
        case VDMOS_VON:
            value->rValue = here->VDMOSvon;
            return(OK);
        case VDMOS_CD:
            value->rValue = here->VDMOScd;
            return(OK);
        case VDMOS_GM:
            value->rValue = here->VDMOSgm;
            return(OK);
        case VDMOS_GDS:
            value->rValue = here->VDMOSgds;
            return(OK);
        case VDMOS_VGS:
            value->rValue = *(ckt->CKTstate0 + here->VDMOSvgs);
            return(OK);
        case VDMOS_VDS:
            value->rValue = *(ckt->CKTstate0 + here->VDMOSvds);
            return(OK);
        case VDMOS_QGS:
            value->rValue = *(ckt->CKTstate0 + here->VDMOSqgs);
            return(OK);
        case VDMOS_CQGS:
            value->rValue = *(ckt->CKTstate0 + here->VDMOScqgs);
            return(OK);
        case VDMOS_QGD:
            value->rValue = *(ckt->CKTstate0 + here->VDMOSqgd);
            return(OK);
        case VDMOS_CQGD:
            value->rValue = *(ckt->CKTstate0 + here->VDMOScqgd);
            return(OK);
        case VDMOS_CDIO:
            value->rValue = *(ckt->CKTstate0 + here->VDIOcurrent);
            return(OK);
        case VDMOS_CG :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = TMALLOC(char, strlen(msg) + 1);
                errRtn = "VDMOSask.c";
                strcpy(errMsg,msg);
                return(E_ASKCURRENT);
            } else if (ckt->CKTcurrentAnalysis & (DOING_DCOP | DOING_TRCV)) {
                value->rValue = 0;
            } else if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && 
                    (ckt->CKTmode & MODETRANOP)) {
                value->rValue = 0;
            } else {
                value->rValue = *(ckt->CKTstate0 + here->VDMOScqgd) + *(ckt->CKTstate0 + 
                        here->VDMOScqgs);
            }
            return(OK);
        case VDMOS_CS :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = TMALLOC(char, strlen(msg) + 1);
                errRtn = "VDMOSask.c";
                strcpy(errMsg,msg);
                return(E_ASKCURRENT);
            } else {
                value->rValue = -here->VDMOScd;
                if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && 
                        !(ckt->CKTmode & MODETRANOP)) {
                    value->rValue -= *(ckt->CKTstate0 + here->VDMOScqgd) +
                            *(ckt->CKTstate0 + here->VDMOScqgs);
                }
            }
            return(OK);
        case VDMOS_POWER :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = TMALLOC(char, strlen(msg) + 1);
                errRtn = "VDMOSask.c";
                strcpy(errMsg,msg);
                return(E_ASKPOWER);
            } else {
                value->rValue = fabs(here->VDMOScd * 
                                     (*(ckt->CKTrhsOld + here->VDMOSdNode) - 
                                      *(ckt->CKTrhsOld + here->VDMOSsNode)));

                if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && 
                        !(ckt->CKTmode & MODETRANOP)) {
                    value->rValue += fabs(*(ckt->CKTstate0 + here->VDMOScqgd) *
                                          (*(ckt->CKTrhsOld + here->VDMOSgNode) -
                                           *(ckt->CKTrhsOld + here->VDMOSdNode)));
                    value->rValue += fabs(*(ckt->CKTstate0 + here->VDMOScqgs) *
                                          (*(ckt->CKTrhsOld + here->VDMOSgNode) -
                                           *(ckt->CKTrhsOld + here->VDMOSsNode)));
                }

                value->rValue += fabs(*(ckt->CKTstate0 + here->VDIOcurrent) * 
                                      (*(ckt->CKTrhsOld + here->VDMOSdNode) -
                                       *(ckt->CKTrhsOld + here->VDMOSsNode)));
            }
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

