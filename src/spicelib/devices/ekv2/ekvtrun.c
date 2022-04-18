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
EKVtrunc(
GENmodel *inModel,
CKTcircuit *ckt,
double *timeStep)
{
	EKVmodel *model = (EKVmodel *)inModel;
	EKVinstance *here;

	for( ; model != NULL; model = EKVnextModel(model)) {
		for(here=EKVinstances(model);here!=NULL;here = EKVnextInstance(here)){
			CKTterr(here->EKVqgs,ckt,timeStep);
			CKTterr(here->EKVqgd,ckt,timeStep);
			CKTterr(here->EKVqgb,ckt,timeStep);
		}
	}
	return(OK);
}
