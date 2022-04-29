/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vsrcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

#ifdef RFSPICE


int VSRCspinit(GENmodel* inModel, CKTcircuit* ckt, CMat* zref, CMat* gn, CMat* gninv)
{
    if (!(ckt->CKTmode & MODESP) && !(ckt->CKTcurrentAnalysis & DOING_SP))
        return (OK);
    VSRCmodel* model = (VSRCmodel*)inModel;
    VSRCinstance* here;

    for (; model != NULL; model = VSRCnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = VSRCinstances(model); here != NULL;
            here = VSRCnextInstance(here)) {

            if (here->VSRCisPort)
            {
                int i = here->VSRCportNum - 1;
                zref->d[i][i].re = here->VSRCportZ0;
                gn->d[i][i].re = 2.0 * here->VSRCki;
                gninv->d[i][i].re = 1.0 / gn->d[i][i].re;
            }
        }
    }
    return (OK);
}

int VSRCspupdate(GENmodel* inModel, CKTcircuit* ckt)
{
    if (!(ckt->CKTmode & MODESP))
        return (OK);
    VSRCmodel* model = (VSRCmodel*)inModel;
    VSRCinstance* here;

    for (; model != NULL; model = VSRCnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = VSRCinstances(model); here != NULL;
            here = VSRCnextInstance(here)) {

            if (here->VSRCisPort)
            {
                double acReal = here->VSRCportNum == ckt->CKTactivePort ? 1.0 : 0.0;

                *(ckt->CKTrhs + (here->VSRCbranch)) += acReal;
            }
        }
    }
    return (OK);
}



int VSRCgetActivePorts(GENmodel* inModel, CKTcircuit* ckt, VSRCinstance** ports)
{
    if (!(ckt->CKTmode & MODESP))
        return (OK);
    for (int n = 0; n < ckt->CKTportCount; n++)
        ports[n] = NULL;

    VSRCmodel* model = (VSRCmodel*)inModel;
    VSRCinstance* here;

    for (; model != NULL; model = VSRCnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = VSRCinstances(model); here != NULL;
            here = VSRCnextInstance(here)) {

            if (here->VSRCisPort)
            {
                int id = here->VSRCportNum - 1;
                ports[id] = here;
            }
        }
    }
    return (OK);
}
#endif

int
VSRCacLoad(GENmodel* inModel, CKTcircuit* ckt)
{
    VSRCmodel* model = (VSRCmodel*)inModel;
    VSRCinstance* here;

    for (; model != NULL; model = VSRCnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = VSRCinstances(model); here != NULL;
            here = VSRCnextInstance(here)) {

            double acReal, acImag;


#ifdef RFSPICE
            double g0 = 0;
            acReal = 0.0;
            acImag = 0.0;
            /*
            * TBD: Verify that MODESPNOISE require also noise from source port
            * In any case, activate only the required noise input to build
            * noise matrix properly
            */
            if ((ckt->CKTmode & MODEACNOISE) || (ckt->CKTmode & MODESPNOISE))
            {
                if ((GENinstance*)here == ckt->noise_input) {
                    acReal = 1.0;
                    acImag = 0.0;
                }
            }
            else
                if (ckt->CKTmode & MODESP)  // In SP Analysis, shut down all AC sources off except the current port
                {
                    // During SP load, no port must excite the circuit before the SPAN procedure. 
                    // They will be turned on & off in the inner cycle of SPAN.
                    // Also, AC Source must be switched off (<--- Check this against ADS).
                    {
                        acReal = 0.0;
                        acImag = 0.0;
                    }
                }
                else // AC Analysis
                {
                    /*
                    * For AC analysis, AC mag is leading parameter for ports.
                    */
                    acReal = here->VSRCacReal;
                    acImag = here->VSRCacImag;
                };

            *(here->VSRCposIbrPtr) += 1.0;
            *(here->VSRCnegIbrPtr) -= 1.0;
            *(here->VSRCibrPosPtr) += 1.0;
            *(here->VSRCibrNegPtr) -= 1.0;
            *(ckt->CKTrhs + (here->VSRCbranch)) += acReal;
            *(ckt->CKTirhs + (here->VSRCbranch)) += acImag;

            if (here->VSRCisPort)
            {
                g0 = here->VSRCportY0;
                *(here->VSRCposPosPtr) += g0;
                *(here->VSRCnegNegPtr) += g0;
                *(here->VSRCposNegPtr) -= g0;
                *(here->VSRCnegPosPtr) -= g0;
            }
#else
            if (ckt->CKTmode & MODEACNOISE) {
                if ((GENinstance*)here == ckt->noise_input) {
                    acReal = 1.0;
                    acImag = 0.0;
                }
                else {
                    acReal = 0.0;
                    acImag = 0.0;
                }
            }
            else {
                acReal = here->VSRCacReal;
                acImag = here->VSRCacImag;
            }

            *(here->VSRCposIbrPtr) += 1.0;
            *(here->VSRCnegIbrPtr) -= 1.0;
            *(here->VSRCibrPosPtr) += 1.0;
            *(here->VSRCibrNegPtr) -= 1.0;
            *(ckt->CKTrhs + (here->VSRCbranch)) += acReal;
            *(ckt->CKTirhs + (here->VSRCbranch)) += acImag;
#endif
        }
    }

    return(OK);
}
