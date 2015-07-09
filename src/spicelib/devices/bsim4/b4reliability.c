/**********
Author: 2015 Francesco Lannutti - July 2015
**********/

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "bsim4def.h"
#include "ngspice/sperror.h"

#include <gsl/gsl_fit.h>

static int
BSIM4reliability_internal (BSIM4instance *here, CKTcircuit *ckt, unsigned int mode)
{
    BSIM4model *model ;
    double delta, vds, vgs, von ;
    int NowIsON, ret ;

    model = here->BSIM4modPtr ;

    // Determine if the transistor is ON or OFF
    vds = ckt->CKTstate0 [here->BSIM4vds] ;
    vgs = ckt->CKTstate0 [here->BSIM4vgs] ;
    von = here->BSIM4von ;
    if (vds >= 0)
    {
//        printf ("VDS >= 0\tBSIM4type: %d\tBSIM4instance: %s\tVgs: %-.9g\tVon: %-.9g\t", model->BSIM4type, here->BSIM4name, vgs, von) ;
        if (vgs > von)
        {
            if (here->BSIM4rgateMod == 3)
            {
                double vges, vgms ;
                vges = ckt->CKTstate0 [here->BSIM4vges] ;
                vgms = ckt->CKTstate0 [here->BSIM4vgms] ;
                if ((vges > von) && (vgms > von))
                {
//                    printf ("Acceso!\n") ;
                    NowIsON = 1 ;
                } else {
//                    printf ("Spento!\n") ;
                    NowIsON = 0 ;
                }
            } else if ((here->BSIM4rgateMod == 1) || (here->BSIM4rgateMod == 2)) {
                double vges ;
                vges = ckt->CKTstate0 [here->BSIM4vges] ;
                if (vges > von)
                {
//                    printf ("Acceso!\n") ;
                    NowIsON = 1 ;
                } else {
//                    printf ("Spento!\n") ;
                    NowIsON = 0 ;
                }
            } else {
//                printf ("Acceso!\n") ;
                NowIsON = 1 ;
            }
        } else {
//            printf ("Spento!\n") ;
            NowIsON = 0 ;
        }
    } else {
        double vgd ;
        vgd = vgs - vds ;
//        printf ("VDS <  0\tBSIM4type: %d\tBSIM4instance: %s\tVgd: %-.9g\tVon: %-.9g\t", model->BSIM4type, here->BSIM4name, vgd, von) ;
        if (vgd > von)
        {
            if (here->BSIM4rgateMod == 3)
            {
                double vges, vged, vgms, vgmd ;
                vges = ckt->CKTstate0 [here->BSIM4vges] ;
                vged = vges - vds ;
                vgms = ckt->CKTstate0 [here->BSIM4vgms] ;
                vgmd = vgms - vds ;
                if ((vged > von) && (vgmd > von))
                {
//                    printf ("Acceso!\n") ;
                    NowIsON = 1 ;
                } else {
//                    printf ("Spento!\n") ;
                    NowIsON = 0 ;
                }
            } else if ((here->BSIM4rgateMod == 1) || (here->BSIM4rgateMod == 2)) {
                double vges, vged ;
                vges = ckt->CKTstate0 [here->BSIM4vges] ;
                vged = vges - vds ;
                if (vged > von)
                {
//                    printf ("Acceso!\n") ;
                    NowIsON = 1 ;
                } else {
//                    printf ("Spento!\n") ;
                    NowIsON = 0 ;
                }
            } else {
//                printf ("Acceso!\n") ;
                NowIsON = 1 ;
            }
        } else {
//            printf ("Spento!\n") ;
            NowIsON = 0 ;
        }
    }

    // If it's the first time, initialize 'here->relStruct->IsON'
    if (here->relStruct->IsON == -1)
    {
        here->relStruct->IsON = NowIsON ;
    }

    if (mode == 0)
    {
        if (NowIsON)
        {
            if (here->relStruct->IsON == 1)
            {
                // Until now, the device was ON - Do NOTHING
                delta = -1 ;
            } else if (here->relStruct->IsON == 0) {
                // Until now, the device was OFF - Calculate recovery
                delta = ckt->CKTtime - here->relStruct->time ;

                // Update time and flag - Stress begins
                here->relStruct->time = ckt->CKTtime ;
                here->relStruct->IsON = 1 ;

                // Calculate Aging - Giogio Liatis' Model
                ret = RELMODELcalculateAging ((GENinstance *)here, here->BSIM4modPtr->BSIM4modType, delta, 0) ;
                if (ret == 1)
                {
                    return (E_INTERN) ;
                }
            } else {
                fprintf (stderr, "Reliability Analysis Error\n") ;
            }
        } else {
            if (here->relStruct->IsON == 1)
            {
                // Until now, the device was ON - Calculate stress
                delta = ckt->CKTtime - here->relStruct->time ;

                // Update time and flag - Recovery begins
                here->relStruct->time = ckt->CKTtime ;
                here->relStruct->IsON = 0 ;

                // Calculate Aging - Giorgio Liatis' Model
                ret = RELMODELcalculateAging ((GENinstance *)here, here->BSIM4modPtr->BSIM4modType, delta, 1) ;
                if (ret == 1)
                {
                    return (E_INTERN) ;
                }
            } else if (here->relStruct->IsON == 0) {
                // Until now, the device was OFF - Do NOTHING
                delta = -1 ;
            } else {
                fprintf (stderr, "Reliability Analysis Error\n") ;
            }
        }
    } else if (mode == 1) {
        // In this mode, it doesn't matter if NOW the device is in stress or in recovery, since it's the latest timestep
        if (here->relStruct->IsON == 1)
        {
            // Calculate stress
            delta = ckt->CKTtime - here->relStruct->time ;

            // Update time and flag - Maybe Optional
            here->relStruct->time = ckt->CKTtime ;
            here->relStruct->IsON = 1 ;

            // Calculate Aging - Giorgio Liatis' Model
            ret = RELMODELcalculateAging ((GENinstance *)here, here->BSIM4modPtr->BSIM4modType, delta, 1) ;
            if (ret == 1)
            {
                return (E_INTERN) ;
            }
        } else if (here->relStruct->IsON == 0) {
            // Calculate recovery
            delta = ckt->CKTtime - here->relStruct->time ;

            // Update time and flag - Maybe Optional
            here->relStruct->time = ckt->CKTtime ;
            here->relStruct->IsON = 0 ;

            // Calculate Aging - Giorgio Liatis' Model
            ret = RELMODELcalculateAging ((GENinstance *)here, here->BSIM4modPtr->BSIM4modType, delta, 0) ;
            if (ret == 1)
            {
                return (E_INTERN) ;
            }
        } else {
            fprintf (stderr, "Reliability Analysis Error\n") ;
        }
        printf ("\tTime: %-.9gs\t\t", ckt->CKTtime) ;
        printf ("DeltaVth: %-.9gmV\t\t", here->relStruct->deltaVth * 1000) ;
        printf ("Device Name: %s\t\t", here->BSIM4name) ;
        printf ("Device Type: %s\n\n", model->BSIM4modName) ;


        /* Calculate fitting */
        double c0, c1, cov00, cov01, cov11, chisq ;

        /* Count how many deltaVth we have */
        unsigned int i ;
        RELMODELrelList *current ;

        i = 0 ;
        current = here->relStruct->deltaVthList ;
        while (current != NULL)
        {
            i++ ;
            current = current->next ;
        }

        /* Assign list members to vectors */
        double *timeFit, *deltaVthFit ;

        timeFit = TMALLOC (double, i) ;
        deltaVthFit = TMALLOC (double, i) ;

        i = 0 ;
        current = here->relStruct->deltaVthList ;
        while (current != NULL)
        {
            timeFit [i] = current->time ;
            deltaVthFit [i] = exp (current->deltaVth) ;
            printf ("Time: %-.9g\n", current->time) ;
            printf ("DeltaVth: %-.9g\n", current->deltaVth) ;
            printf ("DeltaVth (exp): %-.9g\n", exp (current->deltaVth)) ;
            i++ ;
            current = current->next ;
        }

        /* Execute Fitting */
        gsl_fit_linear (timeFit + 1, 1, deltaVthFit + 1, 1, i - 1, &c0, &c1, &cov00, &cov01, &cov11, &chisq) ;

        /* Perform Extrapolation */
        double yFit, yFitErr ;
        gsl_fit_linear_est (2e-6, c0, c1, cov00, cov01, cov11, &yFit, &yFitErr) ;

        printf ("\n\tFitting -> %-.9gs\t%-.9gmV\n\n", 31536000.0, log (yFit) * 1000) ;
    } else {
        fprintf (stderr, "Reliability Analysis Error\n") ;
    }

    return 0 ;
}

int
BSIM4reliability (GENmodel *inModel, CKTcircuit *ckt, unsigned int mode)
{
    BSIM4model *model = (BSIM4model *)inModel ;
    BSIM4instance *here ;
    GENrelmodelDeviceElem *elem ;

    /*  loop through all the BSIM4 device models */
    for ( ; model != NULL ; model = model->BSIM4nextModel)
    {
        if (model->BSIM4type == PMOS)
        {
            if (model->BSIM4relmodelDeviceList != NULL)
            {
                for (elem = model->BSIM4relmodelDeviceList ; elem != NULL ; elem = elem->next)
                {
                    here = (BSIM4instance *)(CKTfndDev (ckt, elem->device_name)) ;
                    if (!here)
                    {
                        SPfrontEnd->IFerrorf (ERR_WARNING, "Error: Cannot find the %s instance of the %s model", elem->device_name, model->BSIM4modName) ;
                    } else {
                        BSIM4reliability_internal (here, ckt, mode) ;
                    }
                }
            } else {
                /* loop through all the instances of the model */
                for (here = model->BSIM4instances ; here != NULL ; here=here->BSIM4nextInstance)
                {
                    BSIM4reliability_internal (here, ckt, mode) ;
                }
            }
        }
    }

    return (OK) ;
}
