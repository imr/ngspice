/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

    /* CKTacDump(ckt,freq,file)
     * this is a simple program to dump the complex rhs vector 
     * into the rawfile.
     */

#include "ngspice.h"
#include "smpdefs.h"
#include "cktdefs.h"
#include "iferrmsg.h"
#include "ifsim.h"



int
CKTacDump(CKTcircuit *ckt, double freq, void *plot)
{
    double *rhsold;
    double *irhsold;
    int i;
    IFcomplex *data;
    IFvalue freqData;
    IFvalue valueData;

    rhsold = ckt->CKTrhsOld;
    irhsold = ckt->CKTirhsOld;
    freqData.rValue = freq;
    valueData.v.numValue = ckt->CKTmaxEqNum-1;
    data = (IFcomplex *) MALLOC((ckt->CKTmaxEqNum-1)*sizeof(IFcomplex));
    valueData.v.vec.cVec = data;
    for (i=0;i<ckt->CKTmaxEqNum-1;i++) {
        data[i].real = rhsold[i+1];
        data[i].imag = irhsold[i+1];
    }
    (*(SPfrontEnd->OUTpData))(plot,&freqData,&valueData);
    FREE(data);
    return(OK);
}
