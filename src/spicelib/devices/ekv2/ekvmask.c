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
EKVmAsk(
CKTcircuit *ckt,
GENmodel *inst,
int which,
IFvalue *value)
{
	EKVmodel *model = (EKVmodel *)inst;
	NG_IGNORE(ckt);
	switch(which) {
	case EKV_MOD_TNOM:
		value->rValue = model->EKVtnom-CONSTCtoK;
		return(OK);
	case EKV_MOD_EKVINT:
		value->rValue = model->EKVekvint;
		return(OK);
	case EKV_MOD_VTO:
		value->rValue = model->EKVvt0;
		return(OK);
	case EKV_MOD_KP:
		value->rValue = model->EKVkp;
		return(OK);
	case EKV_MOD_GAMMA:
		value->rValue = model->EKVgamma;
		return(OK);
	case EKV_MOD_PHI:
		value->rValue = model->EKVphi;
		return(OK);
	case EKV_MOD_COX:
		value->rValue = model->EKVcox;
		return(OK);
	case EKV_MOD_XJ:
		value->rValue = model->EKVxj;
		return(OK);
	case EKV_MOD_THETA:
		value->rValue = model->EKVtheta;
		return(OK);
	case EKV_MOD_E0:
		value->rValue = model->EKVe0;
		return(OK);
	case EKV_MOD_UCRIT:
		value->rValue = model->EKVucrit;
		return(OK);
	case EKV_MOD_DW:
		value->rValue = model->EKVdw;
		return(OK);
	case EKV_MOD_DL:
		value->rValue = model->EKVdl;
		return(OK);
	case EKV_MOD_LAMBDA:
		value->rValue = model->EKVlambda;
		return(OK);
	case EKV_MOD_WETA:
		value->rValue = model->EKVweta;
		return(OK);
	case EKV_MOD_LETA:
		value->rValue = model->EKVleta;
		return(OK);
	case EKV_MOD_IBA:
		value->rValue = model->EKViba;
		return(OK);
	case EKV_MOD_IBB:
		value->rValue = model->EKVibb;
		return(OK);
	case EKV_MOD_IBN:
		value->rValue = model->EKVibn;
		return(OK);
	case EKV_MOD_Q0:
		value->rValue = model->EKVq0;
		return(OK);
	case EKV_MOD_LK:
		value->rValue = model->EKVlk;
		return(OK);
	case EKV_MOD_TCV:
		value->rValue = model->EKVtcv;
		return(OK);
	case EKV_MOD_BEX:
		value->rValue = model->EKVbex;
		return(OK);
	case EKV_MOD_UCEX:
		value->rValue = model->EKVucex;
		return(OK);
	case EKV_MOD_IBBT:
		value->rValue = model->EKVibbt;
		return(OK);
	case EKV_MOD_NQS:
		value->rValue = model->EKVnqs;
		return(OK);
	case EKV_MOD_SATLIM:
		value->rValue = model->EKVsatlim;
		return(OK);
	case EKV_MOD_KF:
		value->rValue = model->EKVfNcoef;
		break;
	case EKV_MOD_AF:
		value->rValue = model->EKVfNexp;
		break;
	case EKV_MOD_IS:
		value->rValue = model->EKVjctSatCur;
		return(OK);
	case EKV_MOD_JS:
		value->rValue = model->EKVjctSatCurDensity;
		return(OK);
	case EKV_MOD_JSW:
		value->rValue = model->EKVjsw;
		return(OK);
	case EKV_MOD_N:
		value->rValue = model->EKVn;
		return(OK);
	case EKV_MOD_CBD:
		value->rValue = model->EKVcapBD;
		return(OK);
	case EKV_MOD_CBS:
		value->rValue = model->EKVcapBS;
		return(OK);
	case EKV_MOD_CJ:
		value->rValue = model->EKVbulkCapFactor;
		return(OK);
	case EKV_MOD_CJSW:
		value->rValue = model->EKVsideWallCapFactor;
		return(OK);
	case EKV_MOD_MJ:
		value->rValue = model->EKVbulkJctBotGradingCoeff;
		return(OK);
	case EKV_MOD_MJSW:
		value->rValue = model->EKVbulkJctSideGradingCoeff;
		return(OK);
	case EKV_MOD_FC:
		value->rValue = model->EKVfwdCapDepCoeff;
		return(OK);
	case EKV_MOD_PB:
		value->rValue = model->EKVbulkJctPotential;
		return(OK);
	case EKV_MOD_PBSW:
		value->rValue = model->EKVpbsw;
		return(OK);
	case EKV_MOD_TT:
		value->rValue = model->EKVtt;
		return(OK);
	case EKV_MOD_CGSO:
		value->rValue = model->EKVgateSourceOverlapCapFactor;
		return(OK);
	case EKV_MOD_CGDO:
		value->rValue = model->EKVgateDrainOverlapCapFactor;
		return(OK);
	case EKV_MOD_CGBO:
		value->rValue = model->EKVgateBulkOverlapCapFactor;
		return(OK);
	case EKV_MOD_RD:
		value->rValue = model->EKVdrainResistance;
		return(OK);
	case EKV_MOD_RS:
		value->rValue = model->EKVsourceResistance;
		return(OK);
	case EKV_MOD_RSH:
		value->rValue = model->EKVsheetResistance;
		return(OK);
	case EKV_MOD_RSC:
		value->rValue = model->EKVrsc;
		return(OK);
	case EKV_MOD_RDC:
		value->rValue = model->EKVrdc;
		return(OK);
	case EKV_MOD_XTI:
		value->rValue = model->EKVxti;
		return(OK);
	case EKV_MOD_TR1:
		value->rValue = model->EKVtr1;
		return(OK);
	case EKV_MOD_TR2:
		value->rValue = model->EKVtr2;
		return(OK);
	case EKV_MOD_NLEVEL:
		value->rValue = model->EKVnlevel;
		return(OK);
	case EKV_MOD_TYPE:
		if (model->EKVtype > 0)
			value->sValue = "nmos";
		else
			value->sValue = "pmos";
		return(OK);
	default:
		return(E_BADPARM);
	}
	/* NOTREACHED */
	return(OK);
}
