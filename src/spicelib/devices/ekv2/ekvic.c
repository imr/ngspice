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
EKVgetic(
GENmodel *inModel,
CKTcircuit *ckt)
{
	EKVmodel *model = (EKVmodel *)inModel;
	EKVinstance *here;
	/*
     * grab initial conditions out of rhs array.   User specified, so use
     * external nodes to get values
     */

	for( ; model ; model = model->EKVnextModel) {
		for(here = model->EKVinstances; here ; here = here->EKVnextInstance) {
			if(!here->EKVicVBSGiven) {
				here->EKVicVBS = 
				    *(ckt->CKTrhs + here->EKVbNode) - 
				    *(ckt->CKTrhs + here->EKVsNode);
			}
			if(!here->EKVicVDSGiven) {
				here->EKVicVDS = 
				    *(ckt->CKTrhs + here->EKVdNode) - 
				    *(ckt->CKTrhs + here->EKVsNode);
			}
			if(!here->EKVicVGSGiven) {
				here->EKVicVGS = 
				    *(ckt->CKTrhs + here->EKVgNode) - 
				    *(ckt->CKTrhs + here->EKVsNode);
			}
		}
	}
	return(OK);
}
