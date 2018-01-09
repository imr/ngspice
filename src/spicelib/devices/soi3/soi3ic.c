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
#include "ngspice/cktdefs.h"
#include "soi3defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
SOI3getic(GENmodel *inModel, CKTcircuit *ckt)
{
    SOI3model *model = (SOI3model *)inModel;
    SOI3instance *here;
    /*
     * grab initial conditions out of rhs array.   User specified, so use
     * external nodes to get values
     */

    for( ; model ; model = SOI3nextModel(model)) {
        for(here = SOI3instances(model); here ; here = SOI3nextInstance(here)) {
	    
	    if(!here->SOI3icVBSGiven) {
                here->SOI3icVBS =
                        *(ckt->CKTrhs + here->SOI3bNode) -
                        *(ckt->CKTrhs + here->SOI3sNode);
            }
            if(!here->SOI3icVDSGiven) {
                here->SOI3icVDS =
                        *(ckt->CKTrhs + here->SOI3dNode) -
                        *(ckt->CKTrhs + here->SOI3sNode);
            }
            if(!here->SOI3icVGFSGiven) {
                here->SOI3icVGFS =
                        *(ckt->CKTrhs + here->SOI3gfNode) -
                        *(ckt->CKTrhs + here->SOI3sNode);
            }
            if(!here->SOI3icVGBSGiven) {
                here->SOI3icVGBS =
                        *(ckt->CKTrhs + here->SOI3gbNode) -
                        *(ckt->CKTrhs + here->SOI3sNode);
            }
        }
    }
    return(OK);
}
