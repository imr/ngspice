/**********
Author: 2015 Francesco Lannutti - July 2015
**********/

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "bsim4def.h"
#include "ngspice/sperror.h"

int
BSIM4reliability (GENmodel *inModel, CKTcircuit *ckt, unsigned int mode)
{
    BSIM4model *model = (BSIM4model *)inModel ;
    BSIM4instance *here ;
    double delta, vds, vgs, von ;
    int NowIsON, ret ;

    /*  loop through all the BSIM4 device models */
    for ( ; model != NULL ; model = model->BSIM4nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM4instances ; here != NULL ; here=here->BSIM4nextInstance)
        {
            if (model->BSIM4type == PMOS)
            {
                // Determine if the transistor is ON or OFF
                vds = ckt->CKTstate0 [here->BSIM4vds] ;
                vgs = ckt->CKTstate0 [here->BSIM4vgs] ;
                von = model->BSIM4type * here->BSIM4von ;
                if (vds >= 0)
                {
//                    printf ("VDS >= 0\tBSIM4type: %d\tBSIM4instance: %s\tVgs: %-.9g\tVon: %-.9g\t", model->BSIM4type, here->BSIM4name, vgs, von) ;
                    if (vgs > von)
                    {
                        if (here->BSIM4rgateMod == 3)
                        {
                            double vges, vgms ;
                            vges = ckt->CKTstate0 [here->BSIM4vges] ;
                            vgms = ckt->CKTstate0 [here->BSIM4vgms] ;
                            if ((vges > von) && (vgms > von))
                            {
//                                printf ("Acceso!\n") ;
                                NowIsON = 1 ;
                            } else {
//                                printf ("Spento!\n") ;
                                NowIsON = 0 ;
                            }
                        } else if ((here->BSIM4rgateMod == 1) || (here->BSIM4rgateMod == 2)) {
                            double vges ;
                            vges = ckt->CKTstate0 [here->BSIM4vges] ;
                            if (vges > von)
                            {
//                                printf ("Acceso!\n") ;
                                NowIsON = 1 ;
                            } else {
//                                printf ("Spento!\n") ;
                                NowIsON = 0 ;
                            }
                        } else {
//                            printf ("Acceso!\n") ;
                            NowIsON = 1 ;
                        }
                    } else {
//                        printf ("Spento!\n") ;
                        NowIsON = 0 ;
                    }
                } else {
                    double vgd ;
                    vgd = vgs - vds ;
//                    printf ("VDS <  0\tBSIM4type: %d\tBSIM4instance: %s\tVgd: %-.9g\tVon: %-.9g\t", model->BSIM4type, here->BSIM4name, vgd, von) ;
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
//                                printf ("Acceso!\n") ;
                                NowIsON = 1 ;
                            } else {
//                                printf ("Spento!\n") ;
                                NowIsON = 0 ;
                            }
                        } else if ((here->BSIM4rgateMod == 1) || (here->BSIM4rgateMod == 2)) {
                            double vges, vged ;
                            vges = ckt->CKTstate0 [here->BSIM4vges] ;
                            vged = vges - vds ;
                            if (vged > von)
                            {
//                                printf ("Acceso!\n") ;
                                NowIsON = 1 ;
                            } else {
//                                printf ("Spento!\n") ;
                                NowIsON = 0 ;
                            }
                        } else {
//                            printf ("Acceso!\n") ;
                            NowIsON = 1 ;
                        }
                    } else {
//                        printf ("Spento!\n") ;
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

                            // Calculate Aging - Giogio Liatis' Model
                            ret = RELMODELcalculateAging ((GENinstance *)here, here->BSIM4modPtr->BSIM4modType, delta, 0) ;
                            if (ret == 1)
                            {
                                return (E_INTERN) ;
                            }

                            // Update time and flag - Stress begins
                            here->relStruct->time = ckt->CKTtime ;
                            here->relStruct->IsON = 1 ;
                        } else {
                            fprintf (stderr, "Reliability Analysis Error\n") ;
                        }
                    } else {
                        if (here->relStruct->IsON == 1)
                        {
                            // Until now, the device was ON - Calculate stress
                            delta = ckt->CKTtime - here->relStruct->time ;

                            // Calculate Aging - Giorgio Liatis' Model
                            ret = RELMODELcalculateAging ((GENinstance *)here, here->BSIM4modPtr->BSIM4modType, delta, 1) ;
                            if (ret == 1)
                            {
                                return (E_INTERN) ;
                            }

                            // Update time and flag - Recovery begins
                            here->relStruct->time = ckt->CKTtime ;
                            here->relStruct->IsON = 0 ;
                        } else if (here->relStruct->IsON == 0) {
                            // Until now, the device was OFF - Do NOTHING
                            delta = -1 ;
                        } else {
                            fprintf (stderr, "Reliability Analysis Error\n") ;
                        }
                    }
                } else if (mode == 1) {
                    // In this mode, it doesn't matter if NOW the device is in stress or in recovery, since it's the last timestep
                    if (here->relStruct->IsON == 1)
                    {
                        // Calculate stress
                        delta = ckt->CKTtime - here->relStruct->time ;
                        ret = RELMODELcalculateAging ((GENinstance *)here, here->BSIM4modPtr->BSIM4modType, delta, 1) ;
                        if (ret == 1)
                        {
                            return (E_INTERN) ;
                        }

                        // Update time and flag - Maybe Optional
                        here->relStruct->time = ckt->CKTtime ;
                        here->relStruct->IsON = 1 ;
                    } else if (here->relStruct->IsON == 0) {
                        // Calculate recovery
                        delta = ckt->CKTtime - here->relStruct->time ;
                        ret = RELMODELcalculateAging ((GENinstance *)here, here->BSIM4modPtr->BSIM4modType, delta, 0) ;
                        if (ret == 1)
                        {
                            return (E_INTERN) ;
                        }

                        // Update time and flag - Maybe Optional
                        here->relStruct->time = ckt->CKTtime ;
                        here->relStruct->IsON = 0 ;
                    } else {
                        fprintf (stderr, "Reliability Analysis Error\n") ;
                    }
                    printf ("\tTime: %-.9gs\t\t", ckt->CKTtime) ;
                    printf ("DeltaVth: %-.9gmV\t\t", here->relStruct->deltaVth * 1000) ;
                    printf ("Device Name: %s\t\t", here->BSIM4name) ;
                    printf ("Device Type: %s\n\n", model->BSIM4modName) ;
                } else {
                    fprintf (stderr, "Reliability Analysis Error\n") ;
                }
            }
        }
    }

    return (OK) ;
}
