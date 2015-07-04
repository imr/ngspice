/**********
Author: 2015 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "bsim4def.h"
#include "ngspice/sperror.h"

static int
calculate_aging
(
    BSIM4instance *here,
    double t_aging,
    unsigned int stress_or_recovery
)
{
    double K_b, T, h_cut, q, Nts, T_hk, Nt, eps_0, eps_hk, eps_SiO2, m_star, W, tau_0, beta, tau_e, beta1 ;

    double A ;

    K_b = 8.6e-5 ;
    T = 300 ;
    h_cut = 1.05e-34 ;
    q = 1.6e-19 ;
    Nts = 2e13 ;
    T_hk = 2 ;
    Nt = pow ((sqrt (Nts)), 3) * 1e-21 ;
    eps_0 = 8.85e-21 ;
    eps_hk = 25 ;
    eps_SiO2 = 3.9 ;
    m_star = 0.1 * 9.11e-31 ;
    W = 1.5 * 1.6e-19 ;
    tau_0 = 1e-11 ;
    beta = 0.373 ;
    tau_e = 0.85e-9 ;
    beta1 = 0.112 ;

    A = (q / (4 * eps_0 * eps_hk)) * pow ((h_cut / (2 * sqrt (2 * m_star * W)) * 1e9), 2) ;

    if (stress_or_recovery)
    {
        if (h_cut / (2 * sqrt (2 * m_star * W)) * log (1 + pow (((t_aging + here->BSIM4reliability->t_star) / tau_0), beta)) * 1e9 <= 2)
        {
            here->BSIM4reliability->deltaVth = Nt * A * pow (log (1 + pow (((t_aging + here->BSIM4reliability->t_star) / tau_0), beta)), 2) ;
        } else {
            here->BSIM4reliability->deltaVth = pow ((q / (4 * eps_0 * eps_hk)) * Nt * T_hk, 2) ;
        }
    } else {
        here->BSIM4reliability->deltaVth = here->BSIM4reliability->deltaVth * log (1 + (1.718 / (1 + pow ((t_aging / tau_e), beta1)))) ;
    }

    if (!stress_or_recovery)
    {
        here->BSIM4reliability->t_star = pow ((exp (sqrt (here->BSIM4reliability->deltaVth / (Nt * A))) - 1), (1 / beta)) * tau_0 ;
    }

    return 0 ;
}

int
BSIM4reliability (GENmodel *inModel, CKTcircuit *ckt, unsigned int mode)
{
    BSIM4model *model = (BSIM4model *)inModel ;
    BSIM4instance *here ;
    double delta, vds, vgs, von ;
    int NowIsON ;

    /*  loop through all the BSIM4 device models */
    for ( ; model != NULL ; model = model->BSIM4nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM4instances ; here != NULL ; here=here->BSIM4nextInstance)
        {
            if (model->BSIM4type == -1)
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

                // If it's the first time, initialize 'here->BSIM4reliability->IsON'
                if (here->BSIM4reliability->IsON == -1)
                {
                    here->BSIM4reliability->IsON = NowIsON ;
                }

                if (mode == 0)
                {
                    if (NowIsON)
                    {
                        if (here->BSIM4reliability->IsON == 1)
                        {
                            // Until now, the device was ON - Do NOTHING
                            delta = -1 ;
                        } else if (here->BSIM4reliability->IsON == 0) {
                            // Until now, the device was OFF - Calculate recovery
                            delta = ckt->CKTtime - here->BSIM4reliability->time ;

                            // Calculate Aging - Giogio Liatis' Model
                            calculate_aging (here, delta, 0) ;

                            // Update time and flag - Stress begins
                            here->BSIM4reliability->time = ckt->CKTtime ;
                            here->BSIM4reliability->IsON = 1 ;
                        } else {
                            fprintf (stderr, "Reliability Analysis Error\n") ;
                        }
                    } else {
                        if (here->BSIM4reliability->IsON == 1)
                        {
                            // Until now, the device was ON - Calculate stress
                            delta = ckt->CKTtime - here->BSIM4reliability->time ;

                            // Calculate Aging - Giorgio Liatis' Model
                            calculate_aging (here, delta, 1) ;

                            // Update time and flag - Recovery begins
                            here->BSIM4reliability->time = ckt->CKTtime ;
                            here->BSIM4reliability->IsON = 0 ;
                        } else if (here->BSIM4reliability->IsON == 0) {
                            // Until now, the device was OFF - Do NOTHING
                            delta = -1 ;
                        } else {
                            fprintf (stderr, "Reliability Analysis Error\n") ;
                        }
                    }
                } else if (mode == 1) {
                    // In this mode, it doesn't matter if NOW the device is in stress or in recovery, since it's the last timestep
                    if (here->BSIM4reliability->IsON == 1)
                    {
                        // Calculate stress
                        delta = ckt->CKTtime - here->BSIM4reliability->time ;
                        calculate_aging (here, delta, 1) ;

                        // Update time and flag - Maybe Optional
                        here->BSIM4reliability->time = ckt->CKTtime ;
                        here->BSIM4reliability->IsON = 1 ;
                    } else if (here->BSIM4reliability->IsON == 0) {
                        // Calculate recovery
                        delta = ckt->CKTtime - here->BSIM4reliability->time ;
                        calculate_aging (here, delta, 0) ;

                        // Update time and flag - Maybe Optional
                        here->BSIM4reliability->time = ckt->CKTtime ;
                        here->BSIM4reliability->IsON = 0 ;
                    } else {
                        fprintf (stderr, "Reliability Analysis Error\n") ;
                    }
                    printf ("\tTime: %-.9gs\t\t", ckt->CKTtime) ;
                    printf ("DeltaVth: %-.9gmV\t\t", here->BSIM4reliability->deltaVth * 1000) ;
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
