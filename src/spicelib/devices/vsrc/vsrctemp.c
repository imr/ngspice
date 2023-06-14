/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "vsrcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/*ARGSUSED*/
int
VSRCtemp(GENmodel *inModel, CKTcircuit *ckt)
        /* Pre-process voltage source parameters
         */
{
    VSRCmodel *model = (VSRCmodel *) inModel;
    VSRCinstance *here;
    double radians;

    NG_IGNORE(ckt);

#ifdef RFSPICE
    ckt->CKTportCount = 0;
    int* portIDs;
    int  prevPort;
#endif

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = VSRCnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = VSRCinstances(model); here != NULL ;
                here=VSRCnextInstance(here)) {

            if(here->VSRCacGiven && !here->VSRCacMGiven) {
                here->VSRCacMag = 1;
            }
            if(here->VSRCacGiven && !here->VSRCacPGiven) {
                here->VSRCacPhase = 0;
            }
            if (!here->VSRCdcGiven && !here->VSRCfuncTGiven) {
                /* no DC value, no transient value */
                SPfrontEnd->IFerrorf(ERR_INFO,
                    "%s: has no value, DC 0 assumed",
                    here->VSRCname);
            }
            else if (here->VSRCdcGiven && here->VSRCfuncTGiven
                     && here->VSRCfunctionType != TRNOISE
                     && here->VSRCfunctionType != TRRANDOM
                     && here->VSRCfunctionType != EXTERNAL) {
                /* DC value and transient time 0 values given */
                double time0value;
                /* determine transient time 0 value */
                if (here->VSRCfunctionType == AM || here->VSRCfunctionType == PWL)
                    time0value = here->VSRCcoeffs[1];
                else
                    time0value = here->VSRCcoeffs[0];
                /* No warning issued if DC value and transient time 0 value are the same */
                if (!AlmostEqualUlps(time0value, here->VSRCdcValue, 3)) {
                    SPfrontEnd->IFerrorf(ERR_INFO,
                        "%s: dc value used for op instead of transient time=0 value.",
                        here->VSRCname);
                }
            }
            radians = here->VSRCacPhase * M_PI / 180.0;
            here->VSRCacReal = here->VSRCacMag * cos(radians);
            here->VSRCacImag = here->VSRCacMag * sin(radians);
#ifdef RFSPICE
            // To have a power port, we need to define its index value
            // AND a proper port impedance
            if (here->VSRCportNumGiven)
            {
                if (!here->VSRCportZ0Given)
                    here->VSRCportZ0 = 50.0;

                here->VSRCisPort = here->VSRCportZ0 > 0.0;
            }
            else
                here->VSRCisPort = FALSE;

            if (here->VSRCisPort)
            {
                if (!here->VSRCportFreqGiven)
                    here->VSRCportFreq = 1.0e9;
                if (!here->VSRCportPowerGiven)
                    here->VSRCportPower = 0.001; // 1mW (0dBm) default RF power
                if (!here->VSRCportPhaseGiven)
                    here->VSRCportPhase = 0.0;

                here->VSRC2pifreq = 2.0 * M_PI * here->VSRCportFreq;
                here->VSRCVAmplitude = sqrt(here->VSRCportPower * 4.0 * here->VSRCportZ0);
                here->VSRCportY0 = 1.0 / here->VSRCportZ0;
                here->VSRCportPhaseRad = here->VSRCportPhase * M_PI / 180.0;
                here->VSRCki = 0.5 / sqrt(here->VSRCportZ0);

                ckt->CKTportCount++;
                ckt->CKTrfPorts = (GENinstance**)TREALLOC(GENinstance*, ckt->CKTrfPorts, ckt->CKTportCount);
                ckt->CKTrfPorts[ckt->CKTportCount - 1] = (GENinstance*)here;

                // Reorder ports according to their PortNum
                unsigned int done = 0;
                while (!done)
                {
                    int nMax = ckt->CKTportCount - 1;
                    done = 1;
                    for (int n = 0; n < nMax; n++)
                    {
                        VSRCinstance* a = (VSRCinstance*)ckt->CKTrfPorts[n];
                        VSRCinstance* b = (VSRCinstance*)ckt->CKTrfPorts[n + 1];
                        if (a->VSRCportNum > b->VSRCportNum)
                        {
                            // Swap a and b. Restart
                            done = 0;
                            ckt->CKTrfPorts[n] = (GENinstance*)b;
                            ckt->CKTrfPorts[n + 1] = (GENinstance*)a;
                            break;
                        }
                    }
                }
            }
#endif
        }
    }

#ifdef RFSPICE
    portIDs = (int*)malloc((size_t)ckt->CKTportCount * sizeof(int));
    if (portIDs == NULL)
        return (E_NOMEM);

    int curport = 0;

    // Sweep thru all ports to check for correct indexing

    /*  loop through all the voltage source models */
    for (model = (VSRCmodel*)inModel; model != NULL; model = VSRCnextModel(model)) {
        /* loop through all the instances of the model */
        for (here = VSRCinstances(model); here != NULL;
            here = VSRCnextInstance(here)) {

            if (!here->VSRCisPort) continue;

            int curId = here->VSRCportNum;
            // If port Index > port Count then we have either a duplicate number or a missing number
            if (curId > ckt->CKTportCount)
            {
                SPfrontEnd->IFerrorf(ERR_FATAL,
                    "%s: incorrect port ordering",
                    here->VSRCname);
                free(portIDs);
                return (E_BADPARM);
            }


            // Check if we have already defined the "curId"
            for (prevPort = 0; prevPort < curport; prevPort++)
            {
                if (portIDs[prevPort] == curId)
                {
                    SPfrontEnd->IFerrorf(ERR_FATAL,
                        "%s: duplicate port Index",
                        here->VSRCname);
                    free(portIDs);
                    return (E_BADPARM);
                }
            }

            portIDs[curport++] = curId;
        }
    }

    free(portIDs);

#endif
    return(OK);
}
