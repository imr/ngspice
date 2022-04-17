/*
 * Author: 2000 Wladek Grabinski; EKV v2.6 Model Upgrade
 * Author: 1997 Eckhard Brass;    EKV v2.5 Model Implementation
 *     (C) 1990 Regents of the University of California. Spice3 Format
 */

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ekvdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/*ARGSUSED*/
int
EKVask(
CKTcircuit *ckt,
GENinstance *inst,
int which,
IFvalue *value,
IFvalue *select)
{
	EKVinstance *here = (EKVinstance*)inst;
	double vr;
	double vi;
	double sr;
	double si;
	double vm;
	static char *msg = "Current and power not available for ac analysis";

	switch(which) {
	case EKV_TEMP:
		value->rValue = here->EKVtemp-CONSTCtoK;
		return(OK);
	case EKV_CGS:
		value->rValue = *(ckt->CKTstate0 + here->EKVcapgs);
		return(OK);
	case EKV_CGD:
		value->rValue = *(ckt->CKTstate0 + here->EKVcapgd);
		return(OK);
	case EKV_L:
		value->rValue = here->EKVl;
		return(OK);
	case EKV_W:
		value->rValue = here->EKVw;
		return(OK);
	case EKV_AS:
		value->rValue = here->EKVsourceArea;
		return(OK);
	case EKV_AD:
		value->rValue = here->EKVdrainArea;
		return(OK);
	case EKV_PS:
		value->rValue = here->EKVsourcePerimiter;
		return(OK);
	case EKV_PD:
		value->rValue = here->EKVdrainPerimiter;
		return(OK);
	case EKV_NRS:
		value->rValue = here->EKVsourceSquares;
		return(OK);
	case EKV_NRD:
		value->rValue = here->EKVdrainSquares;
		return(OK);
	case EKV_OFF:
		value->rValue = here->EKVoff;
		return(OK);
	case EKV_TVTO:
		value->rValue = here->EKVtVto;
		return(OK);
	case EKV_TPHI:
		value->rValue = here->EKVtPhi;
		return(OK);
	case EKV_TKP:
		value->rValue = here->EKVtkp;
		return(OK);
	case EKV_TUCRIT:
		value->rValue = here->EKVtucrit;
		return(OK);
	case EKV_TIBB:
		value->rValue = here->EKVtibb;
		return(OK);
	case EKV_TRS:
		value->rValue = here->EKVtrs;
		return(OK);
	case EKV_TRD:
		value->rValue = here->EKVtrd;
		return(OK);
	case EKV_TRSH:
		value->rValue = here->EKVtrsh;
		return(OK);
	case EKV_TRSC:
		value->rValue = here->EKVtrsc;
		return(OK);
	case EKV_TRDC:
		value->rValue = here->EKVtrdc;
		return(OK);
	case EKV_TIS:
		value->rValue = here->EKVtSatCur;
		return(OK);
	case EKV_TJS:
		value->rValue = here->EKVtSatCurDens;
		return(OK);
	case EKV_TJSW:
		value->rValue = here->EKVtjsw;
		return(OK);
	case EKV_TPB:
		value->rValue = here->EKVtBulkPot;
		return(OK);
	case EKV_TPBSW:
		value->rValue = here->EKVtpbsw;
		return(OK);
	case EKV_TCBD:
		value->rValue = here->EKVtCbd;
		return(OK);
	case EKV_TCBS:
		value->rValue = here->EKVtCbs;
		return(OK);
	case EKV_TCJ:
		value->rValue = here->EKVtCj;
		return(OK);
	case EKV_TCJSW:
		value->rValue = here->EKVtCjsw;
		return(OK);
	case EKV_TAF:
		value->rValue = here->EKVtaf;
		return(OK);
	case EKV_ISUB:
		value->rValue = here->EKVisub;
		return(OK);
	case EKV_VP:
		value->rValue = here->EKVvp;
		return(OK);
	case EKV_SLOPE:
		value->rValue = here->EKVslope;
		return(OK);
	case EKV_IF:
		value->rValue = here->EKVif;
		return(OK);
	case EKV_IR:
		value->rValue = here->EKVir;
		return(OK);
	case EKV_IRPRIME:
		value->rValue = here->EKVirprime;
		return(OK);
	case EKV_TAU:
		value->rValue = here->EKVtau;
		return(OK);
	case EKV_IC_VBS:
		value->rValue = here->EKVicVBS;
		return(OK);
	case EKV_IC_VDS:
		value->rValue = here->EKVicVDS;
		return(OK);
	case EKV_IC_VGS:
		value->rValue = here->EKVicVGS;
		return(OK);
	case EKV_DNODE:
		value->iValue = here->EKVdNode;
		return(OK);
	case EKV_GNODE:
		value->iValue = here->EKVgNode;
		return(OK);
	case EKV_SNODE:
		value->iValue = here->EKVsNode;
		return(OK);
	case EKV_BNODE:
		value->iValue = here->EKVbNode;
		return(OK);
	case EKV_DNODEPRIME:
		value->iValue = here->EKVdNodePrime;
		return(OK);
	case EKV_SNODEPRIME:
		value->iValue = here->EKVsNodePrime;
		return(OK);
	case EKV_SOURCECONDUCT:
		value->rValue = here->EKVsourceConductance;
		return(OK);
	case EKV_SOURCERESIST:
		if (here->EKVsNodePrime != here->EKVsNode)
			value->rValue = 1.0/here->EKVsourceConductance;
		else value->rValue = 0.0;
		return(OK);
	case EKV_DRAINCONDUCT:
		value->rValue = here->EKVdrainConductance;
		return(OK);
	case EKV_DRAINRESIST:
		if (here->EKVdNodePrime != here->EKVdNode)
			value->rValue = 1.0/here->EKVdrainConductance;
		else value->rValue = 0.0;
		return(OK);
	case EKV_VON:
		value->rValue = here->EKVvon;
		return(OK);
	case EKV_VGEFF:
		value->rValue = here->EKVvgeff;
		return(OK);
	case EKV_VDSAT:
		value->rValue = here->EKVvdsat;
		return(OK);
	case EKV_VGPRIME:
		value->rValue = here->EKVvgprime;
		return(OK);
	case EKV_VGSTAR:
		value->rValue = here->EKVvgstar;
		return(OK);
	case EKV_SOURCEVCRIT:
		value->rValue = here->EKVsourceVcrit;
		return(OK);
	case EKV_DRAINVCRIT:
		value->rValue = here->EKVdrainVcrit;
		return(OK);
	case EKV_CD:
		value->rValue = here->EKVcd;
		return(OK);
	case EKV_CBS:
		value->rValue = here->EKVcbs;
		return(OK);
	case EKV_CBD:
		value->rValue = here->EKVcbd;
		return(OK);
	case EKV_GMBS:
		value->rValue = here->EKVgmbs;
		return(OK);
	case EKV_GM:
		value->rValue = here->EKVgm;
		return(OK);
	case EKV_GMS:
		value->rValue = here->EKVgms;
		return(OK);
	case EKV_GDS:
		value->rValue = here->EKVgds;
		return(OK);
	case EKV_GBD:
		value->rValue = here->EKVgbd;
		return(OK);
	case EKV_GBS:
		value->rValue = here->EKVgbs;
		return(OK);
	case EKV_CAPBD:
		value->rValue = here->EKVcapbd;
		return(OK);
	case EKV_CAPBS:
		value->rValue = here->EKVcapbs;
		return(OK);
	case EKV_CAPZEROBIASBD:
		value->rValue = here->EKVCbd;
		return(OK);
	case EKV_CAPZEROBIASBDSW:
		value->rValue = here->EKVCbdsw;
		return(OK);
	case EKV_CAPZEROBIASBS:
		value->rValue = here->EKVCbs;
		return(OK);
	case EKV_CAPZEROBIASBSSW:
		value->rValue = here->EKVCbssw;
		return(OK);
	case EKV_VBD:
		value->rValue = *(ckt->CKTstate0 + here->EKVvbd);
		return(OK);
	case EKV_VBS:
		value->rValue = *(ckt->CKTstate0 + here->EKVvbs);
		return(OK);
	case EKV_VGS:
		value->rValue = *(ckt->CKTstate0 + here->EKVvgs);
		return(OK);
	case EKV_VDS:
		value->rValue = *(ckt->CKTstate0 + here->EKVvds);
		return(OK);
	case EKV_CAPGS:
		value->rValue = *(ckt->CKTstate0 + here->EKVcapgs);
		return(OK);
	case EKV_QGS:
		value->rValue = *(ckt->CKTstate0 + here->EKVqgs);
		return(OK);
	case EKV_CQGS:
		value->rValue = *(ckt->CKTstate0 + here->EKVcqgs);
		return(OK);
	case EKV_CAPGD:
		value->rValue = *(ckt->CKTstate0 + here->EKVcapgd);
		return(OK);
	case EKV_QGD:
		value->rValue = *(ckt->CKTstate0 + here->EKVqgd);
		return(OK);
	case EKV_CQGD:
		value->rValue = *(ckt->CKTstate0 + here->EKVcqgd);
		return(OK);
	case EKV_CAPGB:
		value->rValue = *(ckt->CKTstate0 + here->EKVcapgb);
		return(OK);
	case EKV_QGB:
		value->rValue = *(ckt->CKTstate0 + here->EKVqgb);
		return(OK);
	case EKV_CQGB:
		value->rValue = *(ckt->CKTstate0 + here->EKVcqgb);
		return(OK);
	case EKV_QBD:
		value->rValue = *(ckt->CKTstate0 + here->EKVqbd);
		return(OK);
	case EKV_CQBD:
		value->rValue = *(ckt->CKTstate0 + here->EKVcqbd);
		return(OK);
	case EKV_QBS:
		value->rValue = *(ckt->CKTstate0 + here->EKVqbs);
		return(OK);
	case EKV_CQBS:
		value->rValue = *(ckt->CKTstate0 + here->EKVcqbs);
		return(OK);
	case EKV_L_SENS_DC:
		if(ckt->CKTsenInfo && here->EKVsens_l){
			value->rValue = *(ckt->CKTsenInfo->SEN_Sap[select->iValue + 1]+
			    here->EKVsenParmNo);
		}
		return(OK);
	case EKV_L_SENS_REAL:
		if(ckt->CKTsenInfo && here->EKVsens_l){
			value->rValue = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
			    here->EKVsenParmNo);
		}
		return(OK);
	case EKV_L_SENS_IMAG:
		if(ckt->CKTsenInfo && here->EKVsens_l){
			value->rValue = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
			    here->EKVsenParmNo);
		}
		return(OK);
	case EKV_L_SENS_MAG:
		if(ckt->CKTsenInfo && here->EKVsens_l){
			vr = *(ckt->CKTrhsOld + select->iValue + 1);
			vi = *(ckt->CKTirhsOld + select->iValue + 1);
			vm = sqrt(vr*vr + vi*vi);
			if(vm == 0){
				value->rValue = 0;
				return(OK);
			}
			sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
			    here->EKVsenParmNo);
			si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
			    here->EKVsenParmNo);
			value->rValue = (vr * sr + vi * si)/vm;
		}
		return(OK);
	case EKV_L_SENS_PH:
		if(ckt->CKTsenInfo && here->EKVsens_l){
			vr = *(ckt->CKTrhsOld + select->iValue + 1);
			vi = *(ckt->CKTirhsOld + select->iValue + 1);
			vm = vr*vr + vi*vi;
			if(vm == 0){
				value->rValue = 0;
				return(OK);
			}
			sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
			    here->EKVsenParmNo);
			si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
			    here->EKVsenParmNo);
			value->rValue =  (vr * si - vi * sr)/vm;
		}
		return(OK);
	case EKV_L_SENS_CPLX:
		if(ckt->CKTsenInfo && here->EKVsens_l){
			value->cValue.real= 
			    *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
			    here->EKVsenParmNo);
			value->cValue.imag= 
			    *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
			    here->EKVsenParmNo);
		}
		return(OK);
	case EKV_W_SENS_DC:
		if(ckt->CKTsenInfo && here->EKVsens_w){
			value->rValue = *(ckt->CKTsenInfo->SEN_Sap[select->iValue + 1]+
			    here->EKVsenParmNo + here->EKVsens_l);
		}
		return(OK);
	case EKV_W_SENS_REAL:
		if(ckt->CKTsenInfo && here->EKVsens_w){
			value->rValue = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
			    here->EKVsenParmNo + here->EKVsens_l);
		}
		return(OK);
	case EKV_W_SENS_IMAG:
		if(ckt->CKTsenInfo && here->EKVsens_w){
			value->rValue = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
			    here->EKVsenParmNo + here->EKVsens_l);
		}
		return(OK);
	case EKV_W_SENS_MAG:
		if(ckt->CKTsenInfo && here->EKVsens_w){
			vr = *(ckt->CKTrhsOld + select->iValue + 1);
			vi = *(ckt->CKTirhsOld + select->iValue + 1);
			vm = sqrt(vr*vr + vi*vi);
			if(vm == 0){
				value->rValue = 0;
				return(OK);
			}
			sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
			    here->EKVsenParmNo + here->EKVsens_l);
			si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
			    here->EKVsenParmNo + here->EKVsens_l);
			value->rValue = (vr * sr + vi * si)/vm;
		}
		return(OK);
	case EKV_W_SENS_PH:
		if(ckt->CKTsenInfo && here->EKVsens_w){
			vr = *(ckt->CKTrhsOld + select->iValue + 1);
			vi = *(ckt->CKTirhsOld + select->iValue + 1);
			vm = vr*vr + vi*vi;
			if(vm == 0){
				value->rValue = 0;
				return(OK);
			}
			sr = *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
			    here->EKVsenParmNo + here->EKVsens_l);
			si = *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
			    here->EKVsenParmNo + here->EKVsens_l);
			value->rValue =  (vr * si - vi * sr)/vm;
		}
		return(OK);
	case EKV_W_SENS_CPLX:
		if(ckt->CKTsenInfo && here->EKVsens_w){
			value->cValue.real= 
			    *(ckt->CKTsenInfo->SEN_RHS[select->iValue + 1]+
			    here->EKVsenParmNo + here->EKVsens_l);
			value->cValue.imag= 
			    *(ckt->CKTsenInfo->SEN_iRHS[select->iValue + 1]+
			    here->EKVsenParmNo + here->EKVsens_l);
		}
		return(OK);
	case EKV_CB :
		if (ckt->CKTcurrentAnalysis & DOING_AC) {
			errMsg = TMALLOC(char, strlen(msg)+1);
			errRtn = "EKVask.c";
			strcpy(errMsg,msg);
			return(E_ASKCURRENT);
		} else {
			value->rValue = here->EKVcbd + here->EKVcbs - *(ckt->CKTstate0
			    + here->EKVcqgb);
		}
		return(OK);
	case EKV_CG :
		if (ckt->CKTcurrentAnalysis & DOING_AC) {
			errMsg = TMALLOC(char, strlen(msg)+1);
			errRtn = "EKVask.c";
			strcpy(errMsg,msg);
			return(E_ASKCURRENT);
		} else if (ckt->CKTcurrentAnalysis & (DOING_DCOP | DOING_TRCV)) {
			value->rValue = 0;
		} else if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && 
		    (ckt->CKTmode & MODETRANOP)) {
			value->rValue = 0;
		} else {
			value->rValue = *(ckt->CKTstate0 + here->EKVcqgb) +
			    *(ckt->CKTstate0 + here->EKVcqgd) + *(ckt->CKTstate0 + 
			    here->EKVcqgs);
		}
		return(OK);
	case EKV_CS :
		if (ckt->CKTcurrentAnalysis & DOING_AC) {
			errMsg = TMALLOC(char, strlen(msg)+1);
			errRtn = "EKVask.c";
			strcpy(errMsg,msg);
			return(E_ASKCURRENT);
		} else {
			value->rValue = -here->EKVcd;
			value->rValue -= here->EKVcbd + here->EKVcbs -
			    *(ckt->CKTstate0 + here->EKVcqgb);
			if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && 
			    !(ckt->CKTmode & MODETRANOP)) {
				value->rValue -= *(ckt->CKTstate0 + here->EKVcqgb) + 
				    *(ckt->CKTstate0 + here->EKVcqgd) +
				    *(ckt->CKTstate0 + here->EKVcqgs);
			}
		}
		return(OK);
	case EKV_POWER :
		if (ckt->CKTcurrentAnalysis & DOING_AC) {
			errMsg = TMALLOC(char, strlen(msg)+1);
			errRtn = "EKVask.c";
			strcpy(errMsg,msg);
			return(E_ASKPOWER);
		} else {
			double temp;

			value->rValue = here->EKVcd * 
			    *(ckt->CKTrhsOld + here->EKVdNode);
			value->rValue += (here->EKVcbd + here->EKVcbs -
			    *(ckt->CKTstate0 + here->EKVcqgb)) *
			    *(ckt->CKTrhsOld + here->EKVbNode);
			if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && 
			    !(ckt->CKTmode & MODETRANOP)) {
				value->rValue += (*(ckt->CKTstate0 + here->EKVcqgb) + 
				    *(ckt->CKTstate0 + here->EKVcqgd) +
				    *(ckt->CKTstate0 + here->EKVcqgs)) *
				    *(ckt->CKTrhsOld + here->EKVgNode);
			}
			temp = -here->EKVcd;
			temp -= here->EKVcbd + here->EKVcbs ;
			if ((ckt->CKTcurrentAnalysis & DOING_TRAN) && 
			    !(ckt->CKTmode & MODETRANOP)) {
				temp -= *(ckt->CKTstate0 + here->EKVcqgb) + 
				    *(ckt->CKTstate0 + here->EKVcqgd) + 
				    *(ckt->CKTstate0 + here->EKVcqgs);
			}
			value->rValue += temp * *(ckt->CKTrhsOld + here->EKVsNode);
		}
		return(OK);
	default:
		return(E_BADPARM);
	}
	/* NOTREACHED */
}
