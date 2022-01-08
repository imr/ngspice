/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

    /* CKTspDump(ckt,freq,file)
     * this is a simple program to dump the complex rhs vector 
     * into the rawfile.
     */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/ifsim.h"
#include "vsrc/vsrcdefs.h"

unsigned int CKTmatrixIndex(CKTcircuit* ckt, unsigned int source, unsigned int dest)
{
    return  source * ckt->CKTportCount + dest;
};

int CKTspCalcSMatrix(CKTcircuit* ckt)
{
    CMat* Ainv = cinverse(ckt->CKTAmat);
    if (Ainv == NULL) return (E_NOMEM);
    cmultiplydest(ckt->CKTBmat, Ainv, ckt->CKTSmat);
    freecmat(Ainv);
    return (OK);
}

int CKTspCalcPowerWave(CKTcircuit* ckt)
{
    double* rhsold = ckt->CKTrhsOld;
    double* irhsold = ckt->CKTirhsOld;
    int col = ckt->CKTactivePort - 1;
    for (int port = 0; port < ckt->CKTportCount; port++)
    {
        VSRCinstance* pSrc = (VSRCinstance*)(ckt->CKTrfPorts[port]);
        int row = pSrc->VSRCportNum - 1;
        double zi = pSrc->VSRCportZ0;
        double iReal = -rhsold[pSrc->VSRCbranch];
        double iImag = -irhsold[pSrc->VSRCbranch];

        double vReal =  rhsold[pSrc->VSRCposNode] -  rhsold[pSrc->VSRCnegNode];
        double vImag = irhsold[pSrc->VSRCposNode] - irhsold[pSrc->VSRCnegNode];
        // Forward wave (a) of i-th port, real (r) and imag (i)
        cplx a;
        a.re = pSrc->VSRCki * (vReal + zi * iReal);
        a.im = pSrc->VSRCki * (vImag + zi * iImag);
        
        // Scattered wave (b) of i-th port, real (r) and imag (i)
        cplx b;
        b.re = pSrc->VSRCki * (vReal - zi * iReal);
        b.im = pSrc->VSRCki * (vImag - zi * iImag);

        // fill in A and B matrices
        setc(ckt->CKTAmat, row, col, a);
        setc(ckt->CKTBmat, row, col, b);
        
    }
    return (OK);
}

int
CKTspDump(CKTcircuit *ckt, double freq, runDesc *plot)
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
    unsigned int extraSPdataCount =   ckt->CKTportCount * ckt->CKTportCount;
    valueData.v.numValue = ckt->CKTmaxEqNum - 1 + extraSPdataCount;

    data = TMALLOC(IFcomplex, ckt->CKTmaxEqNum - 1 + extraSPdataCount);
    valueData.v.vec.cVec = data;
    for (i=0;i<ckt->CKTmaxEqNum-1;i++) {
        data[i].real = rhsold[i+1];
        data[i].imag = irhsold[i+1];
    }
    
    if (ckt->CKTrfPorts )
    {
        // Cycle thru all ports
        for (unsigned int pdest = 0; pdest < ckt->CKTportCount; pdest++)
        {
            for (unsigned int psource = 0; psource < ckt->CKTportCount; psource++)
            {
                unsigned int nPlot = ckt->CKTmaxEqNum - 1 + CKTmatrixIndex(ckt, pdest, psource);
                cplx sij = ckt->CKTSmat->d[pdest][psource];
                data[nPlot].real = sij.re;
                data[nPlot].imag = sij.im;
            }
        }
    }
    

    SPfrontEnd->OUTpData(plot, &freqData, &valueData);

    FREE(data);
    return(OK);
}
