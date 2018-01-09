/***  B4SOI 12/16/2010 Released by Tanvir Morshed  ***/


/**********
 * Copyright 2010 Regents of the University of California.  All rights reserved.
 * Authors: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
 * Authors: 1999-2004 Pin Su, Hui Wan, Wei Jin, b3soipzld.c
 * Authors: 2005- Hui Wan, Xuemei Xi, Ali Niknejad, Chenming Hu.
 * Authors: 2009- Wenwei Yang, Chung-Hsun Lin, Ali Niknejad, Chenming Hu.
 * File: b4soipzld.c
 * Modified by Hui Wan, Xuemei Xi 11/30/2005
 * Modified by Wenwei Yang, Chung-Hsun Lin, Darsen Lu 03/06/2009
 * Modified by Tanvir Morshed 09/22/2009
 * Modified by Tanvir Morshed 12/31/2009
 **********/

#include "ngspice/ngspice.h"

#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/sperror.h"
#include "b4soidef.h"
#include "ngspice/suffix.h"

int
B4SOIpzLoad(
GENmodel *inModel,
CKTcircuit *ckt,
SPcomplex *s)
{
register B4SOImodel *model = (B4SOImodel*)inModel;
register B4SOIinstance *here;
double xcggb, xcgdb, xcgsb, xcbgb, xcbdb, xcbsb, xcddb, xcssb, xcdgb;
double gdpr, gspr, gds, gbd, gbs, capbd=0.0, capbs=0.0, xcsgb, xcdsb, xcsdb;
double cggb, cgdb, cgsb, cbgb, cbdb, cbsb, cddb, cdgb, cdsb;
double GSoverlapCap, GDoverlapCap, GBoverlapCap=0.0;
double FwdSum, RevSum, Gm, Gmbs;

double m;

    NG_IGNORE(ckt);

    for (; model != NULL; model = B4SOInextModel(model)) 
    {    for (here = B4SOIinstances(model); here!= NULL;
              here = B4SOInextInstance(here)) 
         {
            if (here->B4SOImode >= 0) 
            {   Gm = here->B4SOIgm;
                Gmbs = here->B4SOIgmbs;
                FwdSum = Gm + Gmbs;
                RevSum = 0.0;
                cggb = here->B4SOIcggb;
                cgsb = here->B4SOIcgsb;
                cgdb = here->B4SOIcgdb;

                cbgb = here->B4SOIcbgb;
                cbsb = here->B4SOIcbsb;
                cbdb = here->B4SOIcbdb;

                cdgb = here->B4SOIcdgb;
                cdsb = here->B4SOIcdsb;
                cddb = here->B4SOIcddb;
            }
            else
            {   Gm = -here->B4SOIgm;
                Gmbs = -here->B4SOIgmbs;
                FwdSum = 0.0;
                RevSum = -Gm - Gmbs;
                cggb = here->B4SOIcggb;
                cgsb = here->B4SOIcgdb;
                cgdb = here->B4SOIcgsb;

                cbgb = here->B4SOIcbgb;
                cbsb = here->B4SOIcbdb;
                cbdb = here->B4SOIcbsb;

                cdgb = -(here->B4SOIcdgb + cggb + cbgb);
                cdsb = -(here->B4SOIcddb + cgsb + cbsb);
                cddb = -(here->B4SOIcdsb + cgdb + cbdb);
            }
            gdpr=here->B4SOIdrainConductance;
            gspr=here->B4SOIsourceConductance;
            gds= here->B4SOIgds;
            gbd= here->B4SOIgjdb;
            gbs= here->B4SOIgjsb;
#ifdef BULKCODE
            capbd= here->B4SOIcapbd;
            capbs= here->B4SOIcapbs;
#endif
            GSoverlapCap = here->B4SOIcgso;
            GDoverlapCap = here->B4SOIcgdo;
#ifdef BULKCODE
            GBoverlapCap = here->pParam->B4SOIcgbo;
#endif

            xcdgb = (cdgb - GDoverlapCap);
            xcddb = (cddb + capbd + GDoverlapCap);
            xcdsb = cdsb;
            xcsgb = -(cggb + cbgb + cdgb + GSoverlapCap);
            xcsdb = -(cgdb + cbdb + cddb);
            xcssb = (capbs + GSoverlapCap - (cgsb+cbsb+cdsb));
            xcggb = (cggb + GDoverlapCap + GSoverlapCap + GBoverlapCap);
            xcgdb = (cgdb - GDoverlapCap);
            xcgsb = (cgsb - GSoverlapCap);
            xcbgb = (cbgb - GBoverlapCap);
            xcbdb = (cbdb - capbd);
            xcbsb = (cbsb - capbs);

            m = here->B4SOIm;

            *(here->B4SOIGgPtr ) += m * xcggb * s->real;
            *(here->B4SOIGgPtr +1) += m * xcggb * s->imag;
            *(here->B4SOIBbPtr ) += m * (-xcbgb-xcbdb-xcbsb) * s->real;
            *(here->B4SOIBbPtr +1) += m * (-xcbgb-xcbdb-xcbsb) * s->imag;
            *(here->B4SOIDPdpPtr ) += m * xcddb * s->real;
            *(here->B4SOIDPdpPtr +1) += m * xcddb * s->imag;
            *(here->B4SOISPspPtr ) += m * xcssb * s->real;
            *(here->B4SOISPspPtr +1) += m * xcssb * s->imag;
            *(here->B4SOIGbPtr ) += m * (-xcggb-xcgdb-xcgsb) * s->real;
            *(here->B4SOIGbPtr +1) += m * (-xcggb-xcgdb-xcgsb) * s->imag;
            *(here->B4SOIGdpPtr ) += m * xcgdb * s->real;
            *(here->B4SOIGdpPtr +1) += m * xcgdb * s->imag;
            *(here->B4SOIGspPtr ) += m * xcgsb * s->real;
            *(here->B4SOIGspPtr +1) += m * xcgsb * s->imag;
            *(here->B4SOIBgPtr ) += m * xcbgb * s->real;
            *(here->B4SOIBgPtr +1) += m * xcbgb * s->imag;
            *(here->B4SOIBdpPtr ) += m * xcbdb * s->real;
            *(here->B4SOIBdpPtr +1) += m * xcbdb * s->imag;
            *(here->B4SOIBspPtr ) += m * xcbsb * s->real;
            *(here->B4SOIBspPtr +1) += m * xcbsb * s->imag;
            *(here->B4SOIDPgPtr ) += m * xcdgb * s->real;
            *(here->B4SOIDPgPtr +1) += m * xcdgb * s->imag;
            *(here->B4SOIDPbPtr ) += m * (-xcdgb-xcddb-xcdsb) * s->real;
            *(here->B4SOIDPbPtr +1) += m * (-xcdgb-xcddb-xcdsb) * s->imag;
            *(here->B4SOIDPspPtr ) += m * xcdsb * s->real;
            *(here->B4SOIDPspPtr +1) += m * xcdsb * s->imag;
            *(here->B4SOISPgPtr ) += m * xcsgb * s->real;
            *(here->B4SOISPgPtr +1) += m * xcsgb * s->imag;
            *(here->B4SOISPbPtr ) += m * (-xcsgb-xcsdb-xcssb) * s->real;
            *(here->B4SOISPbPtr +1) += m * (-xcsgb-xcsdb-xcssb) * s->imag;
            *(here->B4SOISPdpPtr ) += m * xcsdb * s->real;
            *(here->B4SOISPdpPtr +1) += m * xcsdb * s->imag;
            *(here->B4SOIDdPtr) += m * gdpr;
            *(here->B4SOISsPtr) += m * gspr;
            *(here->B4SOIBbPtr) += m * (gbd+gbs);
            *(here->B4SOIDPdpPtr) += m * (gdpr+gds+gbd+RevSum);
            *(here->B4SOISPspPtr) += m * (gspr+gds+gbs+FwdSum);
            *(here->B4SOIDdpPtr) -= m * gdpr;
            *(here->B4SOISspPtr) -= m * gspr;
            *(here->B4SOIBdpPtr) -= m * gbd;
            *(here->B4SOIBspPtr) -= m * gbs;
            *(here->B4SOIDPdPtr) -= m * gdpr;
            *(here->B4SOIDPgPtr) += m * Gm;
            *(here->B4SOIDPbPtr) -= m * (gbd - Gmbs);
            *(here->B4SOIDPspPtr) -= m * (gds + FwdSum);
            *(here->B4SOISPgPtr) -= m * Gm;
            *(here->B4SOISPsPtr) -= m * gspr;
            *(here->B4SOISPbPtr) -= m * (gbs + Gmbs);
            *(here->B4SOISPdpPtr) -= m * (gds + RevSum);

        }
    }
    return(OK);
}


