/**********
Author: 2015 Francesco Lannutti - July 2015
**********/

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "bsim4def.h"
#include "ngspice/sperror.h"

static int
listInsert (BSIM4vgsList **list, double vgs)
{
    BSIM4vgsList *current, *previous ;

    /* Loop until the end of the list */
    previous = NULL ;
    current = *list ;
    while (current != NULL)
    {
        previous = current ;
        current = current->next ;
    }

    /* Insert the new element into the list */
    if (previous == NULL)
    {
        *list = TMALLOC (BSIM4vgsList, 1) ;
        current = *list ;
    } else {
        current = TMALLOC (BSIM4vgsList, 1) ;
        previous->next = current ;
    }

    /* Populate the element */
    current->vgs = vgs ;
    current->next = NULL ;

    return 0 ;
}

static int
listPurge (BSIM4vgsList **list)
{
    BSIM4vgsList *current, *previous ;

    current = *list ;
    while (current != NULL) {
        previous = current ;
        current = current->next ;
        free (previous) ;
    }
    *list = NULL ;

    return 0 ;
}

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
        /* Insert 'vgs' into the list for the later average calculation */
        listInsert (&(here->vgsList), vgs) ;
//        listInsert (&(here->vgsList), ckt->CKTrhsOld [here->BSIM4gNodePrime]) ;

//        printf ("VDS >= 0\tBSIM4type: %d\tBSIM4instance: %s\tVgs: %-.9g\tVon: %-.9g\n", model->BSIM4type, here->BSIM4name, vgs, von) ;
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

        /* Insert 'vgd' into the list for the later average calculation */
        listInsert (&(here->vgsList), vgd) ;
//        listInsert (&(here->vgsList), ckt->CKTrhsOld [here->BSIM4gNodePrime]) ;

//        printf ("VDS <  0\tBSIM4type: %d\tBSIM4instance: %s\tVgd: %-.9g\tVon: %-.9g\n", model->BSIM4type, here->BSIM4name, vgd, von) ;
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
                ret = RELMODELcalculateAging ((GENinstance *)here, here->BSIM4modPtr->BSIM4modType, ckt, delta, 1e-12, 0) ;
                if (ret == 1)
                {
                    return (E_INTERN) ;
                }

                // Update the semiperiod counter
                here->relStruct->semiPeriods++ ;

                // Free the vgs average list
                listPurge (&(here->vgsList)) ;
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
                ret = RELMODELcalculateAging ((GENinstance *)here, here->BSIM4modPtr->BSIM4modType, ckt, delta, 1e-12, 1) ;
                if (ret == 1)
                {
                    return (E_INTERN) ;
                }

                // Update the semiperiod counter
                here->relStruct->semiPeriods++ ;

                // Free the vgs average list
                listPurge (&(here->vgsList)) ;
            } else if (here->relStruct->IsON == 0) {
                // Until now, the device was OFF - Do NOTHING
                delta = -1 ;
            } else {
                fprintf (stderr, "Reliability Analysis Error\n") ;
            }
        }
    } else if (mode == 1) {
        if (NowIsON)
        {
            if (here->relStruct->IsON == 1)
            {
                // Until now, the device was ON - Calculate stress
                delta = ckt->CKTtime - here->relStruct->time ;

                // Update time and flag - Recovery begins
                here->relStruct->time = ckt->CKTtime ;
                here->relStruct->IsON = 1 ;

                // Calculate Aging - Giorgio Liatis' Model
                ret = RELMODELcalculateAging ((GENinstance *)here, here->BSIM4modPtr->BSIM4modType, ckt, delta, 1e-12, 1) ;
                if (ret == 1)
                {
                    return (E_INTERN) ;
                }

                // Update the semiperiod counter
                here->relStruct->semiPeriods++ ;
            } else if (here->relStruct->IsON == 0) {
                // Until now, the device was OFF - Calculate recovery
                delta = ckt->CKTtime - here->relStruct->time ;

                // Update time and flag - Stress begins
                here->relStruct->time = ckt->CKTtime ;
                here->relStruct->IsON = 1 ;

                // Calculate Aging - Giogio Liatis' Model
                ret = RELMODELcalculateAging ((GENinstance *)here, here->BSIM4modPtr->BSIM4modType, ckt, delta, 1e-12, 0) ;
                if (ret == 1)
                {
                    return (E_INTERN) ;
                }

                // Update the semiperiod counter
                here->relStruct->semiPeriods++ ;
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
                ret = RELMODELcalculateAging ((GENinstance *)here, here->BSIM4modPtr->BSIM4modType, ckt, delta, 1e-12, 1) ;
                if (ret == 1)
                {
                    return (E_INTERN) ;
                }

                // Update the semiperiod counter
                here->relStruct->semiPeriods++ ;
            } else if (here->relStruct->IsON == 0) {
                // Until now, the device was OFF - Calculate recovery
                delta = ckt->CKTtime - here->relStruct->time ;

                // Update time and flag - Stress begins
                here->relStruct->time = ckt->CKTtime ;
                here->relStruct->IsON = 0 ;

                // Calculate Aging - Giogio Liatis' Model
                ret = RELMODELcalculateAging ((GENinstance *)here, here->BSIM4modPtr->BSIM4modType, ckt, delta, 1e-12, 0) ;
                if (ret == 1)
                {
                    return (E_INTERN) ;
                }

                // Update the semiperiod counter
                here->relStruct->semiPeriods++ ;
            } else {
                fprintf (stderr, "Reliability Analysis Error\n") ;
            }
        }

        if (here->relStruct->deltaVth > 0) {
            printf ("DEVICE OK!!!\tTime: %-.9gs\t\t", ckt->CKTtime) ;
            printf ("DeltaVth: %-.9gmV\t\t", here->relStruct->deltaVth * 1000) ;
            printf ("Device Name: %s\t\t", here->BSIM4name) ;
            printf ("Device Type: %s\n\n", model->BSIM4modName) ;
        } else if (here->relStruct->deltaVth < 0) {
            printf ("\n\n\n\nWarning: PROBLEMATIC DEVICE!!!\tTime: %-.9gs\t\t", ckt->CKTtime) ;
            printf ("DeltaVth: %-.9gmV\t\t", here->relStruct->deltaVth * 1000) ;
            printf ("Device Name: %s\t\t", here->BSIM4name) ;
            printf ("Device Type: %s\n\n\n\n\n\n", model->BSIM4modName) ;
        } else {
            printf ("THIS DEVICE IS OFF!!!\tTime: %-.9gs\t\t", ckt->CKTtime) ;
            printf ("DeltaVth: %-.9gmV\t\t", here->relStruct->deltaVth * 1000) ;
            printf ("Device Name: %s\t\t", here->BSIM4name) ;
            printf ("Device Type: %s\n\n", model->BSIM4modName) ;
        }


        /* Calculate fitting */

        if (here->relStruct->semiPeriods > 1)
        {
            /* The model behavior is periodic - Use Fourier basis fitting */

            double *deltaVthFit, *timeFit ;
            RELMODELrelList *current ;
            unsigned int i, number_of_periods ;

            /* Count how many deltaVth we have */
            i = 0 ;
            current = here->relStruct->deltaVthList ;
            while (current != NULL)
            {
                i++ ;
                current = current->next ;
            }

            /* Assign list members to vectors */
            timeFit = TMALLOC (double, i) ;
            deltaVthFit = TMALLOC (double, i) ;

            i = 0 ;
            current = here->relStruct->deltaVthList ;
            while (current != NULL)
            {
                timeFit [i] = current->time ;
                deltaVthFit [i] = current->deltaVth ;
                i++ ;
                current = current->next ;
            }

            number_of_periods = here->relStruct->semiPeriods / 2 ;

            /* Assign the extrapolated DeltaVth to the model */
            RELMODELcalculateFitting (i, number_of_periods, ckt->CKTtargetFitting, timeFit, deltaVthFit, &here->relStruct->deltaVth) ;

            FREE (timeFit) ;
            FREE (deltaVthFit) ;
        } else {
            if (here->relStruct->deltaVth > 0)
            {
                ret = RELMODELcalculateAging ((GENinstance *)here, here->BSIM4modPtr->BSIM4modType, ckt, 315360000.0, 0, 1) ;
                if (ret == 1)
                {
                    return (E_INTERN) ;
                }
            }

            printf ("\n\nExtrapolation at %-.9g years:\n\t\t\t\tDeltaVth: %-.9gmV\n\n\n\n", ckt->CKTtargetFitting, here->relStruct->deltaVth * 1000) ;
        }

        // Free the vgs average list
        listPurge (&(here->vgsList)) ;

        if (here->relStruct->deltaVth > 0) {
            model->number_of_aged_instances++ ;
            model->total_deltaVth += here->relStruct->deltaVth ;
        }
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

            if (mode == 1) {
                printf ("Number of aged BSIM4 PMOS instances: %u\n\twith a mean DeltaVth of: %-.9gmV\n\n\n",
                        model->number_of_aged_instances, model->total_deltaVth * 1000 / model->number_of_aged_instances) ;
            }
        }
    }

    return (OK) ;
}
