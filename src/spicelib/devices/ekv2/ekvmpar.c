/*
 * Author: 2000 Wladek Grabinski; EKV v2.6 Model Upgrade
 * Author: 1997 Eckhard Brass;    EKV v2.5 Model Implementation
 *     (C) 1990 Regents of the University of California. Spice3 Format
 */

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "ekvdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
EKVmParam(
int param,
IFvalue *value,
GENmodel *inModel)
{
	EKVmodel *model = (EKVmodel *)inModel;
	switch(param) {
	case EKV_MOD_TNOM:
		model->EKVtnom = value->rValue+CONSTCtoK;
		model->EKVtnomGiven = TRUE;
		return(OK);
	case EKV_MOD_EKVINT:
		model->EKVekvint = value->rValue;
		model->EKVekvintGiven = TRUE;
		return(OK);
	case EKV_MOD_VTO:
		model->EKVvt0 = value->rValue;
		model->EKVvt0Given = TRUE;
		return(OK);
	case EKV_MOD_KP:
		model->EKVkp = value->rValue;
		model->EKVkpGiven = TRUE;
		return(OK);
	case EKV_MOD_GAMMA:
		model->EKVgamma = value->rValue;
		model->EKVgammaGiven = TRUE;
		return(OK);
	case EKV_MOD_PHI:
		model->EKVphi = value->rValue;
		model->EKVphiGiven = TRUE;
		return(OK);
	case EKV_MOD_LAMBDA:
		model->EKVlambda = value->rValue;
		model->EKVlambdaGiven = TRUE;
		return(OK);
	case EKV_MOD_COX:
		model->EKVcox = value->rValue;
		model->EKVcoxGiven = TRUE;
		return(OK);
	case EKV_MOD_XJ:
		model->EKVxj = value->rValue;
		model->EKVxjGiven = TRUE;
		return(OK);
	case EKV_MOD_THETA:
		model->EKVtheta = value->rValue;
		model->EKVthetaGiven = TRUE;
		return(OK);
	case EKV_MOD_E0:
		model->EKVe0 = value->rValue;
		model->EKVe0Given = TRUE;
		return(OK);
	case EKV_MOD_UCRIT:
		model->EKVucrit = value->rValue;
		model->EKVucritGiven = TRUE;
		return(OK);
	case EKV_MOD_DW:
		model->EKVdw = value->rValue;
		model->EKVdwGiven = TRUE;
		return(OK);
	case EKV_MOD_DL:
		model->EKVdl = value->rValue;
		model->EKVdlGiven = TRUE;
		return(OK);
	case EKV_MOD_WETA:
		model->EKVweta = value->rValue;
		model->EKVwetaGiven = TRUE;
		return(OK);
	case EKV_MOD_LETA:
		model->EKVleta = value->rValue;
		model->EKVletaGiven = TRUE;
		return(OK);
	case EKV_MOD_IBA:
		model->EKViba = value->rValue;
		model->EKVibaGiven = TRUE;
		return(OK);
	case EKV_MOD_IBB:
		model->EKVibb = value->rValue;
		model->EKVibbGiven = TRUE;
		return(OK);
	case EKV_MOD_IBN:
		model->EKVibn = value->rValue;
		model->EKVibnGiven = TRUE;
		return(OK);
	case EKV_MOD_Q0:
		model->EKVq0 = value->rValue;
		model->EKVq0Given = TRUE;
		return(OK);
	case EKV_MOD_LK:
		model->EKVlk = value->rValue;
		model->EKVlkGiven = TRUE;
		return(OK);
	case EKV_MOD_TCV:
		model->EKVtcv = value->rValue;
		model->EKVtcvGiven = TRUE;
		return(OK);
	case EKV_MOD_BEX:
		model->EKVbex = value->rValue;
		model->EKVbexGiven = TRUE;
		return(OK);
	case EKV_MOD_UCEX:
		model->EKVucex = value->rValue;
		model->EKVucexGiven = TRUE;
		return(OK);
	case EKV_MOD_IBBT:
		model->EKVibbt = value->rValue;
		model->EKVibbtGiven = TRUE;
		return(OK);
	case EKV_MOD_NQS:
		model->EKVnqs = value->rValue;
		model->EKVnqsGiven = TRUE;
		return(OK);
	case EKV_MOD_SATLIM:
		model->EKVsatlim = value->rValue;
		model->EKVsatlimGiven = TRUE;
		return(OK);
	case EKV_MOD_KF:
		model->EKVfNcoef = value->rValue;
		model->EKVfNcoefGiven = TRUE;
		return(OK);
	case EKV_MOD_AF:
		model->EKVfNexp = value->rValue;
		model->EKVfNexpGiven = TRUE;
		return(OK);

	case EKV_MOD_IS:
		model->EKVjctSatCur = value->rValue;
		model->EKVjctSatCurGiven = TRUE;
		return(OK);
	case EKV_MOD_JS:
		model->EKVjctSatCurDensity = value->rValue;
		model->EKVjctSatCurDensityGiven = TRUE;
		return(OK);
	case EKV_MOD_JSW:
		model->EKVjsw = value->rValue;
		model->EKVjswGiven = TRUE;
		return(OK);
	case EKV_MOD_N:
		model->EKVn = value->rValue;
		model->EKVnGiven = TRUE;
		return(OK);
	case EKV_MOD_CBD:
		model->EKVcapBD = value->rValue;
		model->EKVcapBDGiven = TRUE;
		return(OK);
	case EKV_MOD_CBS:
		model->EKVcapBS = value->rValue;
		model->EKVcapBSGiven = TRUE;
		return(OK);
	case EKV_MOD_CJ:
		model->EKVbulkCapFactor = value->rValue;
		model->EKVbulkCapFactorGiven = TRUE;
		return(OK);
	case EKV_MOD_CJSW:
		model->EKVsideWallCapFactor = value->rValue;
		model->EKVsideWallCapFactorGiven = TRUE;
		return(OK);
	case EKV_MOD_MJ:
		model->EKVbulkJctBotGradingCoeff = value->rValue;
		model->EKVbulkJctBotGradingCoeffGiven = TRUE;
		return(OK);
	case EKV_MOD_MJSW:
		model->EKVbulkJctSideGradingCoeff = value->rValue;
		model->EKVbulkJctSideGradingCoeffGiven = TRUE;
		return(OK);
	case EKV_MOD_FC:
		model->EKVfwdCapDepCoeff = value->rValue;
		model->EKVfwdCapDepCoeffGiven = TRUE;
		return(OK);
	case EKV_MOD_PB:
		model->EKVbulkJctPotential = value->rValue;
		model->EKVbulkJctPotentialGiven = TRUE;
		return(OK);
	case EKV_MOD_PBSW:
		model->EKVpbsw = value->rValue;
		model->EKVpbswGiven = TRUE;
		return(OK);
	case EKV_MOD_TT:
		model->EKVtt = value->rValue;
		model->EKVttGiven = TRUE;
		return(OK);
	case EKV_MOD_CGSO:
		model->EKVgateSourceOverlapCapFactor = value->rValue;
		model->EKVgateSourceOverlapCapFactorGiven = TRUE;
		return(OK);
	case EKV_MOD_CGDO:
		model->EKVgateDrainOverlapCapFactor = value->rValue;
		model->EKVgateDrainOverlapCapFactorGiven = TRUE;
		return(OK);
	case EKV_MOD_CGBO:
		model->EKVgateBulkOverlapCapFactor = value->rValue;
		model->EKVgateBulkOverlapCapFactorGiven = TRUE;
		return(OK);
	case EKV_MOD_RD:
		model->EKVdrainResistance = value->rValue;
		model->EKVdrainResistanceGiven = TRUE;
		return(OK);
	case EKV_MOD_RS:
		model->EKVsourceResistance = value->rValue;
		model->EKVsourceResistanceGiven = TRUE;
		return(OK);
	case EKV_MOD_RSH:
		model->EKVsheetResistance = value->rValue;
		model->EKVsheetResistanceGiven = TRUE;
		return(OK);
	case EKV_MOD_RSC:
		model->EKVrsc = value->rValue;
		model->EKVrscGiven = TRUE;
		return(OK);
	case EKV_MOD_RDC:
		model->EKVrdc = value->rValue;
		model->EKVrdcGiven = TRUE;
		return(OK);
	case EKV_MOD_XTI:
		model->EKVxti = value->rValue;
		model->EKVxtiGiven = TRUE;
		return(OK);
	case EKV_MOD_TR1:
		model->EKVtr1 = value->rValue;
		model->EKVtr1Given = TRUE;
		return(OK);
	case EKV_MOD_TR2:
		model->EKVtr2 = value->rValue;
		model->EKVtr2Given = TRUE;
		return(OK);
	case EKV_MOD_NLEVEL:
		model->EKVnlevel = value->rValue;
		model->EKVnlevelGiven = TRUE;
		return(OK);
	case EKV_MOD_NMOS:
		if(value->iValue) {
			model->EKVtype = 1;
			model->EKVtypeGiven = TRUE;
		}
		return(OK);
	case EKV_MOD_PMOS:
		if(value->iValue) {
			model->EKVtype = -1;
			model->EKVtypeGiven = TRUE;
		}
		return(OK);
	default:
		return(E_BADPARM);
	}
	return(OK);
}
