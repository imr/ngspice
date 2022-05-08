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

#ifdef RFSPICE

extern CMat* eyem;
extern CMat* zref;
extern CMat* gn;
extern CMat* gninv;

extern double NF;
extern double Rn;
extern cplx   Sopt;
extern double Fmin;


int CKTmatrixIndex(CKTcircuit* ckt, int source, int dest)
{
    return  source * ckt->CKTportCount + dest;
};

int CKTspCalcSMatrix(CKTcircuit* ckt)
{
    CMat* Ainv = cinverse(ckt->CKTAmat);
    if (Ainv == NULL) return (E_NOMEM);
    cmultiplydest(ckt->CKTBmat, Ainv, ckt->CKTSmat);
    freecmat(Ainv);
    // Calculate Y matrix
    CMat* temp = cmultiply(ckt->CKTSmat, zref);
    CMat* temp2 = csum(temp, zref);
    CMat* temp3 = cmultiply(temp2, gn);
    
    CMat* temp4 = cminus(eyem, ckt->CKTSmat);
    CMat* temp5 = cinverse(temp4);
    
    cmultiplydest(temp5, temp3, temp);
    cmultiplydest(gninv, temp, ckt->CKTZmat);

    cinversedest(ckt->CKTZmat, ckt->CKTYmat);

    freecmat(temp);
    freecmat(temp2);
    freecmat(temp3);
    freecmat(temp4);
    freecmat(temp5);        

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
CKTspDump(CKTcircuit *ckt, double freq, runDesc *plot, int doNoise)
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
    int extraSPdataCount =  3* ckt->CKTportCount * ckt->CKTportCount;
    valueData.v.numValue = ckt->CKTmaxEqNum - 1 + extraSPdataCount;

    int datasize = ckt->CKTmaxEqNum - 1 + extraSPdataCount;
    
    // Add Cy matrix, NF, Rn, SOpt, NFmin
    if (doNoise)
    {
        datasize += ckt->CKTportCount * ckt->CKTportCount;
        if (ckt->CKTportCount == 2)  // account for NF, Sopt, NFmin, Rn
            datasize += 4;
    }   

    data = TMALLOC(IFcomplex, datasize);
    valueData.v.vec.cVec = data;
    for (i=0;i<ckt->CKTmaxEqNum-1;i++) {
        data[i].real = rhsold[i+1];
        data[i].imag = irhsold[i+1];
    }
    
    if (ckt->CKTrfPorts )
    {
        int nPlot = ckt->CKTmaxEqNum - 1 ;
        // Cycle thru all ports
        for (int pdest = 0; pdest < ckt->CKTportCount; pdest++)
        {
            for (int psource = 0; psource < ckt->CKTportCount; psource++)
            {
                cplx sij = ckt->CKTSmat->d[pdest][psource];
                data[nPlot].real = sij.re;
                data[nPlot].imag = sij.im;
                nPlot++; 
            }
        }

        // Put Y data
        for (int pdest = 0; pdest < ckt->CKTportCount; pdest++)
        {
            for (int psource = 0; psource < ckt->CKTportCount; psource++)
            {
                //unsigned int nPlot = ckt->CKTmaxEqNum - 1 + CKTmatrixIndex(ckt, pdest, psource);
                cplx yij = ckt->CKTYmat->d[pdest][psource];
                data[nPlot].real = yij.re;
                data[nPlot].imag = yij.im;
                nPlot++;
            }
        }

        // Put Z data
        for (int pdest = 0; pdest < ckt->CKTportCount; pdest++)
        {
            for (int psource = 0; psource < ckt->CKTportCount; psource++)
            {
                //unsigned int nPlot = ckt->CKTmaxEqNum - 1 + CKTmatrixIndex(ckt, pdest, psource);
                cplx zij = ckt->CKTZmat->d[pdest][psource];
                data[nPlot].real = zij.re;
                data[nPlot].imag = zij.im;
                nPlot++;
            }
        }

        if (doNoise)
        {
            // Put Cy data
            for (int pdest = 0; pdest < ckt->CKTportCount; pdest++)
            {
                for (int psource = 0; psource < ckt->CKTportCount; psource++)
                {
                    //unsigned int nPlot = ckt->CKTmaxEqNum - 1 + CKTmatrixIndex(ckt, pdest, psource);
                    cplx CYij = ckt->CKTNoiseCYmat->d[pdest][psource];
                    data[nPlot].real = CYij.re;
                    data[nPlot].imag = CYij.im;
                    nPlot++;
                }
            }

            if (ckt->CKTportCount == 2)
            {
                // If we have two ports, put also  NF, Sopt, NFmin, Rn
                data[nPlot].real = NF;
                data[nPlot].imag = 0.0;
                nPlot++;

                data[nPlot].real = Sopt.re;
                data[nPlot].imag = Sopt.im;
                nPlot++;

                data[nPlot].real = Fmin;
                data[nPlot].imag = 0.0;
                nPlot++;

                data[nPlot].real = Rn;
                data[nPlot].imag = 0.0;
                nPlot++;
            }


        }
    }


    

    SPfrontEnd->OUTpData(plot, &freqData, &valueData);

    FREE(data);
    return(OK);
}

#endif
