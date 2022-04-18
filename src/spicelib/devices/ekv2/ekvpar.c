/*
 * Author: 2000 Wladek Grabinski; EKV v2.6 Model Upgrade
 * Author: 1997 Eckhard Brass;    EKV v2.5 Model Implementation
 *     (C) 1990 Regents of the University of California. Spice3 Format
 */
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "ekvdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/* ARGSUSED */
int
EKVparam(
int param,
IFvalue *value,
GENinstance *inst,
IFvalue *select)
{
	EKVinstance *here = (EKVinstance *)inst;
	NG_IGNORE(select);
	switch(param) {
	case EKV_TEMP:
		here->EKVtemp = value->rValue+CONSTCtoK;
		here->EKVtempGiven = TRUE;
		return(OK);
	case EKV_W:
		here->EKVw = value->rValue;
		here->EKVwGiven = TRUE;
		return(OK);
	case EKV_L:
		here->EKVl = value->rValue;
		here->EKVlGiven = TRUE;
		return(OK);
	case EKV_AS:
		here->EKVsourceArea = value->rValue;
		here->EKVsourceAreaGiven = TRUE;
		return(OK);
	case EKV_AD:
		here->EKVdrainArea = value->rValue;
		here->EKVdrainAreaGiven = TRUE;
		return(OK);
	case EKV_PS:
		here->EKVsourcePerimiter = value->rValue;
		here->EKVsourcePerimiterGiven = TRUE;
		return(OK);
	case EKV_PD:
		here->EKVdrainPerimiter = value->rValue;
		here->EKVdrainPerimiterGiven = TRUE;
		return(OK);
	case EKV_NRS:
		here->EKVsourceSquares = value->rValue;
		here->EKVsourceSquaresGiven = TRUE;
		return(OK);
	case EKV_NRD:
		here->EKVdrainSquares = value->rValue;
		here->EKVdrainSquaresGiven = TRUE;
		return(OK);
	case EKV_OFF:
		here->EKVoff = (value->iValue != 0);
		return(OK);
	case EKV_IC_VBS:
		here->EKVicVBS = value->rValue;
		here->EKVicVBSGiven = TRUE;
		return(OK);
	case EKV_IC_VDS:
		here->EKVicVDS = value->rValue;
		here->EKVicVDSGiven = TRUE;
		return(OK);
	case EKV_IC_VGS:
		here->EKVicVGS = value->rValue;
		here->EKVicVGSGiven = TRUE;
		return(OK);
	case EKV_IC:
		switch(value->v.numValue){
		case 3:
			here->EKVicVBS = *(value->v.vec.rVec+2);
			here->EKVicVBSGiven = TRUE;
			/* FALLTHROUGH */
		case 2:
			here->EKVicVGS = *(value->v.vec.rVec+1);
			here->EKVicVGSGiven = TRUE;
			/* FALLTHROUGH */
		case 1:
			here->EKVicVDS = *(value->v.vec.rVec);
			here->EKVicVDSGiven = TRUE;
			return(OK);
		default:
			return(E_BADPARM);
		}
		break;
	case EKV_L_SENS:
		if(value->iValue) {
			here->EKVsenParmNo = 1;
			here->EKVsens_l = 1;
		}
		return(OK);
	case EKV_W_SENS:
		if(value->iValue) {
			here->EKVsenParmNo = 1;
			here->EKVsens_w = 1;
		}
		return(OK);
	default:
		return(E_BADPARM);
	}
}
