/*
 * Author: 2000 Wladek Grabinski; EKV v2.6 Model Upgrade
 * Author: 1997 Eckhard Brass;    EKV v2.5 Model Implementation
 *     (C) 1990 Regents of the University of California. Spice3 Format
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ekvdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
EKVconvTest(
GENmodel *inModel,
CKTcircuit *ckt)
{
	EKVmodel *model = (EKVmodel*)inModel;
	EKVinstance *here;
	double delvbs;
	double delvbd;
	double delvgs;
	double delvds;
	double delvgd;
	double cbhat;
	double cdhat;
	double vbs;
	double vbd;
	double vgs;
	double vds;
	double vgd;
	double vgdo;
	double tol;

	for( ; model != NULL; model = EKVnextModel(model)) {
		for(here = EKVinstances(model); here!= NULL;
		    here = EKVnextInstance(here)) {

			vbs = model->EKVtype * ( 
			    *(ckt->CKTrhs+here->EKVbNode) -
			    *(ckt->CKTrhs+here->EKVsNodePrime));
			vgs = model->EKVtype * ( 
			    *(ckt->CKTrhs+here->EKVgNode) -
			    *(ckt->CKTrhs+here->EKVsNodePrime));
			vds = model->EKVtype * ( 
			    *(ckt->CKTrhs+here->EKVdNodePrime) -
			    *(ckt->CKTrhs+here->EKVsNodePrime));
			vbd=vbs-vds;
			vgd=vgs-vds;
			vgdo = *(ckt->CKTstate0 + here->EKVvgs) -
			    *(ckt->CKTstate0 + here->EKVvds);
			delvbs = vbs - *(ckt->CKTstate0 + here->EKVvbs);
			delvbd = vbd - *(ckt->CKTstate0 + here->EKVvbd);
			delvgs = vgs - *(ckt->CKTstate0 + here->EKVvgs);
			delvds = vds - *(ckt->CKTstate0 + here->EKVvds);
			delvgd = vgd-vgdo;

			/* these are needed for convergence testing */

			if (here->EKVmode >= 0) {
				cdhat=
				    here->EKVcd-
				    here->EKVgbd * delvbd +
				    here->EKVgmbs * delvbs +
				    here->EKVgm * delvgs + 
				    here->EKVgds * delvds ;
			} else {
				cdhat=
				    here->EKVcd -
				    ( here->EKVgbd -
				    here->EKVgmbs) * delvbd -
				    here->EKVgm * delvgd + 
				    here->EKVgds * delvds ;
			}
			cbhat=
			    here->EKVcbs +
			    here->EKVcbd +
			    here->EKVgbd * delvbd +
			    here->EKVgbs * delvbs ;
			/*
             *  check convergence
             */
			tol=ckt->CKTreltol*MAX(fabs(cdhat),fabs(here->EKVcd))+
			    ckt->CKTabstol;
			if (fabs(cdhat-here->EKVcd) >= tol) {
				ckt->CKTnoncon++;
				ckt->CKTtroubleElt = (GENinstance *) here;
				return(OK); /* no reason to continue, we haven't converged */
			} else {
				tol=ckt->CKTreltol*
				    MAX(fabs(cbhat),fabs(here->EKVcbs+here->EKVcbd))+
				    ckt->CKTabstol;
				if (fabs(cbhat-(here->EKVcbs+here->EKVcbd)) > tol) {
					ckt->CKTnoncon++;
					ckt->CKTtroubleElt = (GENinstance *) here;
					return(OK); /* no reason to continue, we haven't converged*/
				}
			}
		}
	}
	return(OK);
}
