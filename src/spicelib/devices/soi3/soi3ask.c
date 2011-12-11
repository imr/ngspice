/**********
STAG version 2.7
Copyright 2000 owned by the United Kingdom Secretary of State for Defence
acting through the Defence Evaluation and Research Agency.
Developed by :     Jim Benson,
                   Department of Electronics and Computer Science,
                   University of Southampton,
                   United Kingdom.
With help from :   Nele D'Halleweyn, Ketan Mistry, Bill Redman-White, and Craig Easson.

Based on STAG version 2.1
Developed by :     Mike Lee,
With help from :   Bernard Tenbroek, Bill Redman-White, Mike Uren, Chris Edwards
                   and John Bunyan.
Acknowledgements : Rupert Howes and Pete Mole.
**********/

/********** 
Modified by Paolo Nenzi 2002
ngspice integration
**********/

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "soi3defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/*ARGSUSED*/
int
SOI3ask(CKTcircuit *ckt, GENinstance *inst, int which, IFvalue *value, 
        IFvalue *select)
{
    SOI3instance *here = (SOI3instance*)inst;

    NG_IGNORE(select);

    /* The line below belong to the parts of the case statement that are also
     * commented out */
/*     double vr; */
/*     double vi; */
/*     double sr; */
/*     double si; */
/*     double vm; */
/*     static char *msg = "Current and power not available for ac analysis"; */

    switch(which) {
        case SOI3_L:
            value->rValue = here->SOI3l;
                return(OK);
        case SOI3_W:
            value->rValue = here->SOI3w;
                return(OK);
	case SOI3_M:
            value->rValue = here->SOI3m;
                return(OK);
	 case SOI3_AS:
            value->rValue = here->SOI3as;
            	 return(OK);
        case SOI3_AD:
            value->rValue = here->SOI3ad;
            	 return(OK);
        case SOI3_AB:
            value->rValue = here->SOI3ab;
            	 return(OK);
        case SOI3_NRS:
            value->rValue = here->SOI3sourceSquares;
                return(OK);
        case SOI3_NRD:
            value->rValue = here->SOI3drainSquares;
                return(OK);
        case SOI3_OFF:
            value->rValue = here->SOI3off;
                return(OK);
        case SOI3_IC_VDS:
            value->rValue = here->SOI3icVDS;
                return(OK);
        case SOI3_IC_VGFS:
            value->rValue = here->SOI3icVGFS;
                return(OK);
        case SOI3_IC_VGBS:
            value->rValue = here->SOI3icVGBS;
                return(OK);
        case SOI3_IC_VBS:
            value->rValue = here->SOI3icVBS;
                return(OK);
        case SOI3_TEMP:
            value->rValue = here->SOI3temp-CONSTCtoK;
            return(OK);
        case SOI3_RT:
            value->rValue = here->SOI3rt;
	    value->rValue /= here->SOI3m;
            return(OK);
        case SOI3_CT:
            value->rValue = here->SOI3ct;
	    value->rValue *= here->SOI3m;
            return(OK);
        case SOI3_DNODE:
            value->iValue = here->SOI3dNode;
            return(OK);
        case SOI3_GFNODE:
            value->iValue = here->SOI3gfNode;
            return(OK);
        case SOI3_SNODE:
            value->iValue = here->SOI3sNode;
            return(OK);
        case SOI3_GBNODE:
            value->iValue = here->SOI3gbNode;
            return(OK);
        case SOI3_BNODE:
            value->iValue = here->SOI3bNode;
            return(OK);
        case SOI3_DNODEPRIME:
            value->iValue = here->SOI3dNodePrime;
            return(OK);
        case SOI3_SNODEPRIME:
            value->iValue = here->SOI3sNodePrime;
            return(OK);
        case SOI3_TNODE:
            value->iValue = here->SOI3toutNode;
            return(OK);
        case SOI3_BRANCH:
            value->iValue = here->SOI3branch;
            return(OK);
        case SOI3_SOURCECONDUCT:
            value->rValue = here->SOI3sourceConductance;
	    value->rValue *= here->SOI3m;
            return(OK);
        case SOI3_DRAINCONDUCT:
            value->rValue = here->SOI3drainConductance;
	    value->rValue *= here->SOI3m;
            return(OK);
        case SOI3_VON:
            value->rValue = here->SOI3tVto;
            return(OK);
        case SOI3_VFBF:
            value->rValue = here->SOI3tVfbF;
            return(OK);
        case SOI3_VDSAT:
            value->rValue = here->SOI3vdsat;
            return(OK);
        case SOI3_SOURCEVCRIT:
            value->rValue = here->SOI3sourceVcrit;
            return(OK);
        case SOI3_DRAINVCRIT:
            value->rValue = here->SOI3drainVcrit;
            return(OK);
        case SOI3_ID:
            value->rValue = here->SOI3id;
	    value->rValue *= here->SOI3m;
            return(OK);
        case SOI3_IBS:
            value->rValue = here->SOI3ibs;
	    value->rValue *= here->SOI3m;
            return(OK);
        case SOI3_IBD:
            value->rValue = here->SOI3ibd;
	    value->rValue *= here->SOI3m;
            return(OK);
        case SOI3_GMBS:
            value->rValue = here->SOI3gmbs;
	    value->rValue *= here->SOI3m;
            return(OK);
        case SOI3_GMF:
            value->rValue = here->SOI3gmf;
	    value->rValue *= here->SOI3m;
            return(OK);
        case SOI3_GMB:
            value->rValue = here->SOI3gmb;
	    value->rValue *= here->SOI3m;
            return(OK);
        case SOI3_GDS:
            value->rValue = here->SOI3gds;
	    value->rValue *= here->SOI3m;
            return(OK);
        case SOI3_GBD:
            value->rValue = here->SOI3gbd;
	    value->rValue *= here->SOI3m;
            return(OK);
        case SOI3_GBS:
            value->rValue = here->SOI3gbs;
	    value->rValue *= here->SOI3m;
            return(OK);
        case SOI3_CAPBD:
            value->rValue = here->SOI3capbd;
	    value->rValue *= here->SOI3m;
            return(OK);
        case SOI3_CAPBS:
            value->rValue = here->SOI3capbs;
	    value->rValue *= here->SOI3m;
            return(OK);
        case SOI3_CAPZEROBIASBD:
            value->rValue = here->SOI3Cbd;
	    value->rValue *= here->SOI3m;
            return(OK);
        case SOI3_CAPZEROBIASBS:
            value->rValue = here->SOI3Cbs;
	    value->rValue *= here->SOI3m;
            return(OK);
        case SOI3_VBD:
            value->rValue = *(ckt->CKTstate0 + here->SOI3vbd);
            return(OK);
        case SOI3_VBS:
            value->rValue = *(ckt->CKTstate0 + here->SOI3vbs);
            return(OK);
        case SOI3_VGFS:
            value->rValue = *(ckt->CKTstate0 + here->SOI3vgfs);
            return(OK);
        case SOI3_VGBS:
            value->rValue = *(ckt->CKTstate0 + here->SOI3vgbs);
            return(OK);
        case SOI3_VDS:
            value->rValue = *(ckt->CKTstate0 + here->SOI3vds);
            return(OK);
        case SOI3_QGF:
            value->rValue = *(ckt->CKTstate0 + here->SOI3qgf);
	    value->rValue *= here->SOI3m;
            return(OK);
        case SOI3_IQGF:
            value->rValue = *(ckt->CKTstate0 + here->SOI3iqgf);
	    value->rValue *= here->SOI3m;
            return(OK);
        case SOI3_QD:
            value->rValue = *(ckt->CKTstate0 + here->SOI3qd);
	    value->rValue *= here->SOI3m;
            return(OK);
        case SOI3_IQD:
            value->rValue = *(ckt->CKTstate0 + here->SOI3iqd);
	    value->rValue *= here->SOI3m;
            return(OK);
        case SOI3_QS:
            value->rValue = *(ckt->CKTstate0 + here->SOI3qs);
	    value->rValue *= here->SOI3m;
            return(OK);
        case SOI3_IQS:
            value->rValue = *(ckt->CKTstate0 + here->SOI3iqs);
	    value->rValue *= here->SOI3m;
            return(OK);
        case SOI3_CGFGF:
            value->rValue = *(ckt->CKTstate0 + here->SOI3cgfgf);
	    value->rValue *= here->SOI3m;
            return (OK);
        case SOI3_CGFD:
            value->rValue = *(ckt->CKTstate0 + here->SOI3cgfd);
	    value->rValue *= here->SOI3m;
            return (OK);
        case SOI3_CGFS:
            value->rValue = *(ckt->CKTstate0 + here->SOI3cgfs);
	    value->rValue *= here->SOI3m;
            return (OK);
        case SOI3_CGFDELTAT:
            value->rValue = *(ckt->CKTstate0 + here->SOI3cgfdeltaT);
	    value->rValue *= here->SOI3m;
            return (OK);
	case SOI3_CGFGB:
            value->rValue = *(ckt->CKTstate0 + here->SOI3cgfgb);
	    value->rValue *= here->SOI3m;
	    return (OK);
        case SOI3_CDGF:
            value->rValue = *(ckt->CKTstate0 + here->SOI3cdgf);
	    value->rValue *= here->SOI3m;
            return (OK);
        case SOI3_CDD:
            value->rValue = *(ckt->CKTstate0 + here->SOI3cdd);
	    value->rValue *= here->SOI3m;
            return (OK);
        case SOI3_CDS:
            value->rValue = *(ckt->CKTstate0 + here->SOI3cds);
	    value->rValue *= here->SOI3m;
            return (OK);
        case SOI3_CDDELTAT:
            value->rValue = *(ckt->CKTstate0 + here->SOI3cddeltaT);
	    value->rValue *= here->SOI3m;
            return (OK);
	case SOI3_CDGB:
            value->rValue = *(ckt->CKTstate0 + here->SOI3cdgb);
	    value->rValue *= here->SOI3m;
            return (OK);
        case SOI3_CSGF:
            value->rValue = *(ckt->CKTstate0 + here->SOI3csgf);
	    value->rValue *= here->SOI3m;
            return (OK);
        case SOI3_CSD:
            value->rValue = *(ckt->CKTstate0 + here->SOI3csd);
	    value->rValue *= here->SOI3m;
            return (OK);
        case SOI3_CSS:
            value->rValue = *(ckt->CKTstate0 + here->SOI3css);
	    value->rValue *= here->SOI3m;
            return (OK);
        case SOI3_CSDELTAT:
            value->rValue = *(ckt->CKTstate0 + here->SOI3csdeltaT);
	    value->rValue *= here->SOI3m;
            return (OK);
	case SOI3_CSGB:
            value->rValue = *(ckt->CKTstate0 + here->SOI3csgb);
	    value->rValue *= here->SOI3m;
            return (OK);
        case SOI3_CGBGF:
            value->rValue = *(ckt->CKTstate0 + here->SOI3cgbgf);
	    value->rValue *= here->SOI3m;
            return (OK);
        case SOI3_CGBD:
            value->rValue = *(ckt->CKTstate0 + here->SOI3cgbd);
	    value->rValue *= here->SOI3m;
            return (OK);
        case SOI3_CGBS:
            value->rValue = *(ckt->CKTstate0 + here->SOI3cgbs);
	    value->rValue *= here->SOI3m;
            return (OK);
        case SOI3_CGBDELTAT:
            value->rValue = *(ckt->CKTstate0 + here->SOI3cgbdeltaT);
	    value->rValue *= here->SOI3m;
            return (OK);
        case SOI3_CGBGB:
            value->rValue = *(ckt->CKTstate0 + here->SOI3cgbgb);
	    value->rValue *= here->SOI3m;
            return (OK);
        case SOI3_QBD:
            value->rValue = *(ckt->CKTstate0 + here->SOI3qbd);
	    value->rValue *= here->SOI3m;
            return(OK);
        case SOI3_IQBD:
            value->rValue = *(ckt->CKTstate0 + here->SOI3iqbd);
	    value->rValue *= here->SOI3m;
            return(OK);
        case SOI3_QBS:
            value->rValue = *(ckt->CKTstate0 + here->SOI3qbs);
	    value->rValue *= here->SOI3m;
            return(OK);
        case SOI3_IQBS:
            value->rValue = *(ckt->CKTstate0 + here->SOI3iqbs);
	    value->rValue *= here->SOI3m;
            return(OK);
/* extra stuff for newer model - msll Jan96 */
        case SOI3_VFBB:
            value->rValue = here->SOI3tVfbB;
            return(OK);
        case SOI3_RT1:
            value->rValue = here->SOI3rt1;
	    value->rValue /= here->SOI3m;
            return(OK);
        case SOI3_CT1:
            value->rValue = here->SOI3ct1;
	    value->rValue *= here->SOI3m;
            return(OK);
        case SOI3_RT2:
            value->rValue = here->SOI3rt2;
	    value->rValue /= here->SOI3m;
            return(OK);
        case SOI3_CT2:
            value->rValue = here->SOI3ct2;
	    value->rValue *= here->SOI3m;
            return(OK);
        case SOI3_RT3:
            value->rValue = here->SOI3rt3;
	    value->rValue /= here->SOI3m;
            return(OK);
        case SOI3_CT3:
            value->rValue = here->SOI3ct3;
	    value->rValue *= here->SOI3m;
            return(OK);
        case SOI3_RT4:
            value->rValue = here->SOI3rt4;
	    value->rValue /= here->SOI3m;
            return(OK);
        case SOI3_CT4:
            value->rValue = here->SOI3ct4;
	    value->rValue *= here->SOI3m;
            return(OK);

/*
        case SOI3_L_SENS_DC:
            if(ckt->CKTsenInfo && here->SOI3sens_l){
               value->rValue = *(ckt->CKTsenInfo->SEN_Sap[select->iValue + 1]+
                       here->SOI3senParmNo);
            }
            return(OK);
        case SOI3_L_SENS_REAL:
            if(ckt->CKTsenInfo && here->SOI3sens_l){
               value->rValue = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                       here->SOI3senParmNo);
            }
            return(OK);
        case SOI3_L_SENS_IMAG:
            if(ckt->CKTsenInfo && here->SOI3sens_l){
               value->rValue = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                       here->SOI3senParmNo);
            }
            return(OK);
        case SOI3_L_SENS_MAG:
            if(ckt->CKTsenInfo && here->SOI3sens_l){
                vr = *(ckt->CKTrhsOld + select->iValue + 1);
                vi = *(ckt->CKTirhsOld + select->iValue + 1);
                vm = sqrt(vr*vr + vi*vi);
                if(vm == 0){
                    value->rValue = 0;
                    return(OK);
                }
                sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->SOI3senParmNo);
                si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->SOI3senParmNo);
                value->rValue = (vr * sr + vi * si)/vm;
            }
            return(OK);
        case SOI3_L_SENS_PH:
            if(ckt->CKTsenInfo && here->SOI3sens_l){
                vr = *(ckt->CKTrhsOld + select->iValue + 1);
                vi = *(ckt->CKTirhsOld + select->iValue + 1);
                vm = vr*vr + vi*vi;
                if(vm == 0){
                    value->rValue = 0;
                    return(OK);
                }
                sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->SOI3senParmNo);
                si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->SOI3senParmNo);
                value->rValue =  (vr * si - vi * sr)/vm;
            }
            return(OK);
        case SOI3_L_SENS_CPLX:
            if(ckt->CKTsenInfo && here->SOI3sens_l){
                value->cValue.real=
                        *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->SOI3senParmNo);
                value->cValue.imag=
                        *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->SOI3senParmNo);
            }
            return(OK);
        case SOI3_W_SENS_DC:
            if(ckt->CKTsenInfo && here->SOI3sens_w){
                value->rValue = *(ckt->CKTsenInfo->SEN_Sap[select->iValue + 1]+
                        here->SOI3senParmNo + here->SOI3sens_l);
            }
            return(OK);
        case SOI3_W_SENS_REAL:
            if(ckt->CKTsenInfo && here->SOI3sens_w){
                value->rValue = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->SOI3senParmNo + here->SOI3sens_l);
            }
             return(OK);
        case SOI3_W_SENS_IMAG:
            if(ckt->CKTsenInfo && here->SOI3sens_w){
                value->rValue = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->SOI3senParmNo + here->SOI3sens_l);
            }
            return(OK);
        case SOI3_W_SENS_MAG:
            if(ckt->CKTsenInfo && here->SOI3sens_w){
                vr = *(ckt->CKTrhsOld + select->iValue + 1);
                vi = *(ckt->CKTirhsOld + select->iValue + 1);
                vm = sqrt(vr*vr + vi*vi);
                if(vm == 0){
                    value->rValue = 0;
                    return(OK);
                }
                sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->SOI3senParmNo + here->SOI3sens_l);
                si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->SOI3senParmNo + here->SOI3sens_l);
                value->rValue = (vr * sr + vi * si)/vm;
            }
            return(OK);
        case SOI3_W_SENS_PH:
            if(ckt->CKTsenInfo && here->SOI3sens_w){
                vr = *(ckt->CKTrhsOld + select->iValue + 1);
                vi = *(ckt->CKTirhsOld + select->iValue + 1);
                vm = vr*vr + vi*vi;
                if(vm == 0){
                    value->rValue = 0;
                    return(OK);
                }
                sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->SOI3senParmNo + here->SOI3sens_l);
                si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->SOI3senParmNo + here->SOI3sens_l);
                value->rValue =  (vr * si - vi * sr)/vm;
            }
                    return(OK);
        case SOI3_W_SENS_CPLX:
            if(ckt->CKTsenInfo && here->SOI3sens_w){
                value->cValue.real=
                        *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
                        here->SOI3senParmNo + here->SOI3sens_l);
                value->cValue.imag=
                        *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
                        here->SOI3senParmNo + here->SOI3sens_l);
            }
            return(OK);                           */

/*
        case SOI3_IS :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = TMALLOC(char, strlen(msg) + 1);
                errRtn = "SOI3ask.c";
                strcpy(errMsg,msg);
                return(E_ASKCURRENT);
            } else {
                value->rValue = -here->SOI3id;
                value->rValue -= here->SOI3ibd + here->SOI3ibs -
                        *(ckt->CKTstate0 + here->SOI3iqgfb);
                if ((ckt->CKTcurrentAnalysis & DOING_TRAN) &&
                        !(ckt->CKTmode & MODETRANOP)) {
                    value->rValue -= *(ckt->CKTstate0 + here->SOI3iqgfb) +
                            *(ckt->CKTstate0 + here->SOI3iqgfd) +
                            *(ckt->CKTstate0 + here->SOI3iqgfs);
                }
            }
            return(OK);
        case SOI3_IB :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = TMALLOC(char, strlen(msg) + 1);
                errRtn = "SOI3ask.c";
                strcpy(errMsg,msg);
                return(E_ASKCURRENT);
            } else {
                value->rValue = here->SOI3ibd + here->SOI3ibs -
                        *(ckt->CKTstate0 + here->SOI3iqgfb);
            }
            return(OK);
        case SOI3_IGF :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = TMALLOC(char, strlen(msg) + 1);
                errRtn = "SOI3ask.c";
                strcpy(errMsg,msg);
                return(E_ASKCURRENT);
            } else if (ckt->CKTcurrentAnalysis & (DOING_DCOP | DOING_TRCV)) {
                value->rValue = 0;
            } else if ((ckt->CKTcurrentAnalysis & DOING_TRAN) &&
                    (ckt->CKTmode & MODETRANOP)) {
                value->rValue = 0;
            } else {
                value->rValue =   *(ckt->CKTstate0 + here->SOI3iqgfb) +
                        *(ckt->CKTstate0 + here->SOI3iqgfd) + *(ckt->CKTstate0 +
                        here->SOI3iqgfs);
            }
            return(OK);
        case SOI3_IGB :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = TMALLOC(char, strlen(msg) + 1);
                errRtn = "SOI3ask.c";
                strcpy(errMsg,msg);
                return(E_ASKCURRENT);
            } else if (ckt->CKTcurrentAnalysis & (DOING_DCOP | DOING_TRCV)) {
                value->rValue = 0;
            } else if ((ckt->CKTcurrentAnalysis & DOING_TRAN) &&
                    (ckt->CKTmode & MODETRANOP)) {
                value->rValue = 0;
            } else {
                value->rValue =  *(ckt->CKTstate0 + here->SOI3iqgfb) +
                        *(ckt->CKTstate0 + here->SOI3iqgfd) + *(ckt->CKTstate0 +
                        here->SOI3iqgfs);
            }
            return(OK);
        case SOI3_POWER :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = TMALLOC(char, strlen(msg) + 1);
                errRtn = "SOI3ask.c";
                strcpy(errMsg,msg);
                return(E_ASKPOWER);
            } else {
                double temp;

                value->rValue = here->SOI3id *
                        *(ckt->CKTrhsOld + here->SOI3dNode);
                value->rValue += ((here->SOI3ibd + here->SOI3ibs) -
                        *(ckt->CKTstate0 + here->SOI3iqgfb)) *
                        *(ckt->CKTrhsOld + here->SOI3bNode);
                if ((ckt->CKTcurrentAnalysis & DOING_TRAN) &&
                        !(ckt->CKTmode & MODETRANOP)) {
                    value->rValue += (*(ckt->CKTstate0 + here->SOI3iqgfb) +
                            *(ckt->CKTstate0 + here->SOI3iqgfd) +
                            *(ckt->CKTstate0 + here->SOI3iqgfs)) *
                            *(ckt->CKTrhsOld + here->SOI3gfNode);
                }
                temp = -here->SOI3id;
                temp -= here->SOI3ibd + here->SOI3ibs ;
                if ((ckt->CKTcurrentAnalysis & DOING_TRAN) &&
                        !(ckt->CKTmode & MODETRANOP)) {
                    temp -= *(ckt->CKTstate0 + here->SOI3iqgfb) +
                            *(ckt->CKTstate0 + here->SOI3iqgfd) +
                            *(ckt->CKTstate0 + here->SOI3iqgfs);
                }
                value->rValue += temp * *(ckt->CKTrhsOld + here->SOI3sNode);
            }
            return(OK);
*/
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

