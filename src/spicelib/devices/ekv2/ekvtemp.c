/*
 * Author: 2000 Wladek Grabinski; EKV v2.6 Model Upgrade
 * Author: 1997 Eckhard Brass;    EKV v2.5 Model Implementation
 *     (C) 1990 Regents of the University of California. Spice3 Format
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ekvdefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
EKVtemp(
GENmodel *inModel,
CKTcircuit *ckt)
{
	EKVmodel *model = (EKVmodel *)inModel;
	EKVinstance *here;

	double eg,egnom;
	double ratio,ratio2;
	double cap_factor;
	double vt,vtnom;
	double r_factor;
	double cur_factor;
	double dtemp;
	double t,tnom;
	double phi_term;

	/* loop through all the resistor models */
	for( ; model != NULL; model = model->EKVnextModel) {

		/* perform model defaulting */
		if (!model->EKVtnomGiven)
			model->EKVtnom = ckt->CKTnomTemp;

		if (model->EKVphi<0.1) model->EKVphi=0.1;

		if (model->EKVbulkJctPotential<0.1) model->EKVbulkJctPotential=0.1;

		if (model->EKVpbsw<0.1) model->EKVpbsw=0.1;

		if (model->EKVfwdCapDepCoeff>=1.0) model->EKVfwdCapDepCoeff=0.99;

		tnom  = model->EKVtnom;
		vtnom = tnom*CONSTKoverQ;
		egnom = 1.16-(7.02e-4*tnom*tnom)/(tnom+1108.0);

		/* now model parameter preprocessing */

		/* loop through all instances of the model */
		for(here = model->EKVinstances; here!= NULL; 
		    here = here->EKVnextInstance) {

			double czbd;    /* zero voltage bulk-drain capacitance */
			double czbdsw;  /* zero voltage bulk-drain sidewall capacitance */
			double czbs;    /* zero voltage bulk-source capacitance */
			double czbssw;  /* zero voltage bulk-source sidewall capacitance */
			double arg;     /* 1 - fc */
			double sarg;    /* (1-fc) ^^ (-mj) */
			double sargsw;  /* (1-fc) ^^ (-mjsw) */

			if(!here->EKVdrainAreaGiven)
				here->EKVdrainArea = ckt->CKTdefaultMosAD;
			if(!here->EKVlGiven)
				here->EKVl = ckt->CKTdefaultMosL;
			if(!here->EKVsourceAreaGiven)
				here->EKVsourceArea = ckt->CKTdefaultMosAS;
			if(!here->EKVwGiven)
				here->EKVw = ckt->CKTdefaultMosW;

			if(here->EKVl+model->EKVdl <=0)
				(*(SPfrontEnd->IFerror))(ERR_WARNING,
				    "%s: effective channel length less than zero",
				    &(model->EKVmodName));

			if(here->EKVw+model->EKVdw <=0)
				(*(SPfrontEnd->IFerror))(ERR_WARNING,
				    "%s: effective channel width less than zero",
				    &(model->EKVmodName));

			/* perform the parameter defaulting */

			if(!here->EKVtempGiven)
				here->EKVtemp = ckt->CKTtemp;

			t  = here->EKVtemp;
			vt = t*CONSTKoverQ;
			eg = 1.16-(7.02e-4*t*t)/(t+1108.0);

			ratio  = t/tnom;
			ratio2 = (t-tnom)/tnom;
			dtemp  = t-tnom;

			here->EKVtVto   = model->EKVvt0
			    -model->EKVtype*model->EKVtcv*dtemp;

			here->EKVtkp    = model->EKVkp   *pow(ratio,model->EKVbex);
			here->EKVtucrit = model->EKVucrit*pow(ratio,model->EKVucex);

			phi_term = -3*vt*log(ratio)-egnom*ratio+eg;

			here->EKVtPhi = model->EKVphi*ratio+phi_term;
			here->EKVtibb = model->EKVibb*(1.0+model->EKVibbt*dtemp);

			cur_factor = pow(ratio,model->EKVxti)*exp(egnom/vtnom-eg/vt);

			here->EKVtSatCur     = model->EKVjctSatCur        * cur_factor;
			here->EKVtSatCurDens = model->EKVjctSatCurDensity * cur_factor;
			here->EKVtjsw        = model->EKVjsw              * cur_factor;

			here->EKVtBulkPot = model->EKVbulkJctPotential*ratio+phi_term;
			here->EKVtpbsw    = model->EKVpbsw            *ratio+phi_term;

			cap_factor = 1.0+model->EKVbulkJctBotGradingCoeff*
			    (0.0004*dtemp+1-here->EKVtBulkPot/model->EKVbulkJctPotential);

			here->EKVtCbd = model->EKVcapBD         * cap_factor;
			here->EKVtCbs = model->EKVcapBS         * cap_factor;
			here->EKVtCj  = model->EKVbulkCapFactor * cap_factor;

			cap_factor = 1.0+model->EKVbulkJctSideGradingCoeff*
			    (0.0004*dtemp+1-here->EKVtBulkPot/model->EKVbulkJctPotential);

			here->EKVtCjsw = model->EKVsideWallCapFactor * cap_factor;

			r_factor = 1.0+ratio2*(model->EKVtr1+ratio2*model->EKVtr2);

			here->EKVtrs  = model->EKVsourceResistance * r_factor;
			here->EKVtrd  = model->EKVdrainResistance  * r_factor;
			here->EKVtrsh = model->EKVsheetResistance  * r_factor;
			here->EKVtrsc = model->EKVrsc              * r_factor;
			here->EKVtrdc = model->EKVrdc              * r_factor;

			here->EKVtaf  = model->EKVfNexp
			    *here->EKVtBulkPot/model->EKVbulkJctPotential;

			if( (here->EKVtSatCurDens == 0) ||
			    (here->EKVdrainArea == 0) ||
			    (here->EKVsourceArea == 0) ) {
				here->EKVsourceVcrit = here->EKVdrainVcrit =
				    vt*log(vt/(CONSTroot2*here->EKVtSatCur));
			} else {
				here->EKVdrainVcrit =
				    vt * log( vt / (CONSTroot2 *
				    here->EKVtSatCurDens * here->EKVdrainArea));
				here->EKVsourceVcrit =
				    vt * log( vt / (CONSTroot2 *
				    here->EKVtSatCurDens * here->EKVsourceArea));
			}

			if(model->EKVcapBDGiven) {
				czbd = here->EKVtCbd;
			} else {
				if(model->EKVbulkCapFactorGiven) {
					czbd=here->EKVtCj*here->EKVdrainArea;
				} else czbd=0;
			}
			if(model->EKVsideWallCapFactorGiven) {
				czbdsw= here->EKVtCjsw * here->EKVdrainPerimiter;
			} else czbdsw=0;

			arg = 1-model->EKVfwdCapDepCoeff;
			sarg = exp( (-model->EKVbulkJctBotGradingCoeff) * log(arg) );
			sargsw = exp( (-model->EKVbulkJctSideGradingCoeff) * log(arg) );
			here->EKVCbd = czbd;
			here->EKVCbdsw = czbdsw;
			here->EKVf2d = czbd*(1-model->EKVfwdCapDepCoeff*
			    (1+model->EKVbulkJctBotGradingCoeff))* sarg/arg
			    +  czbdsw*(1-model->EKVfwdCapDepCoeff*
			    (1+model->EKVbulkJctSideGradingCoeff))*
			    sargsw/arg;
			here->EKVf3d = czbd * model->EKVbulkJctBotGradingCoeff * sarg/arg/
			    here->EKVtBulkPot
			    + czbdsw * model->EKVbulkJctSideGradingCoeff * sargsw/arg /
			    here->EKVtBulkPot;
			here->EKVf4d = czbd*here->EKVtBulkPot*(1-arg*sarg)/
			    (1-model->EKVbulkJctBotGradingCoeff)
			    + czbdsw*here->EKVtBulkPot*(1-arg*sargsw)/
			    (1-model->EKVbulkJctSideGradingCoeff)
			    -here->EKVf3d/2*
			    (here->EKVtDepCap*here->EKVtDepCap)
			    -here->EKVtDepCap * here->EKVf2d;
			if(model->EKVcapBSGiven)
				czbs=here->EKVtCbs;
			else
				if(model->EKVbulkCapFactorGiven)
					czbs=here->EKVtCj*here->EKVsourceArea;
				else czbs=0;

			if(model->EKVsideWallCapFactorGiven)
				czbssw = here->EKVtCjsw * here->EKVsourcePerimiter;
			else czbssw=0;

			arg    = 1-model->EKVfwdCapDepCoeff;
			sarg   = exp( (-model->EKVbulkJctBotGradingCoeff) * log(arg) );
			sargsw = exp( (-model->EKVbulkJctSideGradingCoeff) * log(arg) );

			here->EKVCbs = czbs;
			here->EKVCbssw = czbssw;
			here->EKVf2s = czbs*(1-model->EKVfwdCapDepCoeff*
			    (1+model->EKVbulkJctBotGradingCoeff))* sarg/arg
			    +  czbssw*(1-model->EKVfwdCapDepCoeff*
			    (1+model->EKVbulkJctSideGradingCoeff))*
			    sargsw/arg;
			here->EKVf3s = czbs * model->EKVbulkJctBotGradingCoeff * sarg/arg/
			    here->EKVtBulkPot
			    + czbssw * model->EKVbulkJctSideGradingCoeff * sargsw/arg /
			    here->EKVtBulkPot;
			here->EKVf4s = czbs*here->EKVtBulkPot*(1-arg*sarg)/
			    (1-model->EKVbulkJctBotGradingCoeff)
			    + czbssw*here->EKVtBulkPot*(1-arg*sargsw)/
			    (1-model->EKVbulkJctSideGradingCoeff)
			    -here->EKVf3s/2*
			    (here->EKVtDepCap*here->EKVtDepCap)
			    -here->EKVtDepCap * here->EKVf2s;

			if(model->EKVdrainResistanceGiven)
				if(here->EKVtrd != 0)
					here->EKVdrainConductance = 1/here->EKVtrd;
				else here->EKVdrainConductance = 0;
			else if (model->EKVsheetResistanceGiven)
				if(here->EKVtrsh != 0)
					here->EKVdrainConductance = 
					    1/(here->EKVtrsh*here->EKVdrainSquares);
				else here->EKVdrainConductance = 0;
			else here->EKVdrainConductance = 0;

			if(model->EKVsourceResistanceGiven)
				if(here->EKVtrs != 0)
					here->EKVsourceConductance = 1/here->EKVtrs;
				else here->EKVsourceConductance = 0;
			else if (model->EKVsheetResistanceGiven)
				if(here->EKVtrsh != 0)
					here->EKVsourceConductance = 
					    1/(here->EKVtrsh*here->EKVsourceSquares);
				else here->EKVsourceConductance = 0;
			else here->EKVsourceConductance = 0;
		}
	}
	return(OK);
}
