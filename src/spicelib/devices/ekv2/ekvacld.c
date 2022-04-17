/*
 * Author: 2000 Wladek Grabinski; EKV v2.6 Model Upgrade
 * Author: 1997 Eckhard Brass;    EKV v2.5 Model Implementation
 *     (C) 1990 Regents of the University of California. Spice3 Format
 */
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ekvdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
EKVacLoad(
GENmodel *inModel,
CKTcircuit *ckt)
{
	EKVmodel *model = (EKVmodel*)inModel;
	EKVinstance *here;
	int xnrm;
	int xrev;
	double xgs;
	double xgd;
	double xgb;
	double xbd;
	double xbs;
	double capgs;
	double capgd;
	double capgb;
	double GateBulkOverlapCap;
	double GateDrainOverlapCap;
	double GateSourceOverlapCap;
	double EffectiveLength;
	double EffectiveWidth;

	for( ; model != NULL; model = model->EKVnextModel) {
		for(here = model->EKVinstances; here!= NULL;
		    here = here->EKVnextInstance) {

			if (here->EKVmode < 0) {
				xnrm=0;
				xrev=1;
			} else {
				xnrm=1;
				xrev=0;
			}
			/*
             *     meyer's model parameters
             */
			EffectiveLength=here->EKVl+model->EKVdl;
			EffectiveWidth =here->EKVw+model->EKVdw;

			GateSourceOverlapCap = model->EKVgateSourceOverlapCapFactor * 
			    EffectiveWidth;
			GateDrainOverlapCap = model->EKVgateDrainOverlapCapFactor * 
			    EffectiveWidth;
			GateBulkOverlapCap = model->EKVgateBulkOverlapCapFactor * 
			    EffectiveLength;
			capgs = ( *(ckt->CKTstate0+here->EKVcapgs)+ 
			    *(ckt->CKTstate0+here->EKVcapgs) +
			    GateSourceOverlapCap );
			capgd = ( *(ckt->CKTstate0+here->EKVcapgd)+ 
			    *(ckt->CKTstate0+here->EKVcapgd) +
			    GateDrainOverlapCap );
			capgb = ( *(ckt->CKTstate0+here->EKVcapgb)+ 
			    *(ckt->CKTstate0+here->EKVcapgb) +
			    GateBulkOverlapCap );
			xgs = capgs * ckt->CKTomega;
			xgd = capgd * ckt->CKTomega;
			xgb = capgb * ckt->CKTomega;
			xbd  = here->EKVcapbd * ckt->CKTomega;
			xbs  = here->EKVcapbs * ckt->CKTomega;
			/*
             *    load matrix
             */

			*(here->EKVGgPtr +1) += xgd+xgs+xgb;
			*(here->EKVBbPtr +1) += xgb+xbd+xbs;
			*(here->EKVDPdpPtr +1) += xgd+xbd;
			*(here->EKVSPspPtr +1) += xgs+xbs;
			*(here->EKVGbPtr +1) -= xgb;
			*(here->EKVGdpPtr +1) -= xgd;
			*(here->EKVGspPtr +1) -= xgs;
			*(here->EKVBgPtr +1) -= xgb;
			*(here->EKVBdpPtr +1) -= xbd;
			*(here->EKVBspPtr +1) -= xbs;
			*(here->EKVDPgPtr +1) -= xgd;
			*(here->EKVDPbPtr +1) -= xbd;
			*(here->EKVSPgPtr +1) -= xgs;
			*(here->EKVSPbPtr +1) -= xbs;
			*(here->EKVDdPtr) += here->EKVdrainConductance;
			*(here->EKVSsPtr) += here->EKVsourceConductance;
			*(here->EKVBbPtr) += here->EKVgbd+here->EKVgbs;
			*(here->EKVDPdpPtr) += here->EKVdrainConductance+
			    here->EKVgds+here->EKVgbd+
			    xrev*(here->EKVgm+here->EKVgmbs);
			*(here->EKVSPspPtr) += here->EKVsourceConductance+
			    here->EKVgds+here->EKVgbs+
			    xnrm*(here->EKVgm+here->EKVgmbs);
			*(here->EKVDdpPtr) -= here->EKVdrainConductance;
			*(here->EKVSspPtr) -= here->EKVsourceConductance;
			*(here->EKVBdpPtr) -= here->EKVgbd;
			*(here->EKVBspPtr) -= here->EKVgbs;
			*(here->EKVDPdPtr) -= here->EKVdrainConductance;
			*(here->EKVDPgPtr) += (xnrm-xrev)*here->EKVgm;
			*(here->EKVDPbPtr) += -here->EKVgbd+(xnrm-xrev)*here->EKVgmbs;
			*(here->EKVDPspPtr) -= here->EKVgds+
			    xnrm*(here->EKVgm+here->EKVgmbs);
			*(here->EKVSPgPtr) -= (xnrm-xrev)*here->EKVgm;
			*(here->EKVSPsPtr) -= here->EKVsourceConductance;
			*(here->EKVSPbPtr) -= here->EKVgbs+(xnrm-xrev)*here->EKVgmbs;
			*(here->EKVSPdpPtr) -= here->EKVgds+
			    xrev*(here->EKVgm+here->EKVgmbs);

		}
	}
	return(OK);
}
