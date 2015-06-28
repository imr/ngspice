/**********
Author: 2015 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "mos1defs.h"
#include "ngspice/sperror.h"

static int
calculate_aging
(
    MOS1instance *here,
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
        if (h_cut / (2 * sqrt (2 * m_star * W)) * log (1 + pow (((t_aging + here->MOS1reliability->t_star) / tau_0), beta)) * 1e9 <= 2)
        {
            here->MOS1reliability->deltaVth = Nt * A * pow (log (1 + pow (((t_aging + here->MOS1reliability->t_star) / tau_0), beta)), 2) ;
        } else {
            here->MOS1reliability->deltaVth = pow ((q / (4 * eps_0 * eps_hk)) * Nt * T_hk, 2) ;
        }
    } else {
        here->MOS1reliability->deltaVth = here->MOS1reliability->deltaVth * log (1 + (1.718 / (1 + pow ((t_aging / tau_e), beta1)))) ;
    }

    if (!stress_or_recovery)
    {
        here->MOS1reliability->t_star = pow ((exp (sqrt (here->MOS1reliability->deltaVth / (Nt * A))) - 1), (1 / beta)) * tau_0 ;
    }

    return 0 ;
}

int
MOS1reliability (GENmodel *inModel, CKTcircuit *ckt, unsigned int mode)
{
    MOS1model *model = (MOS1model *)inModel ;
    MOS1instance *here ;
    double delta, vds, vgs, von ;
    int NowIsON ;

    /*  loop through all the MOS1 device models */
    for ( ; model != NULL ; model = model->MOS1nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MOS1instances ; here != NULL ; here=here->MOS1nextInstance)
        {
            if (model->MOS1type == -1)
            {
                vds = ckt->CKTstate0 [here->MOS1vds] ;
                vgs = ckt->CKTstate0 [here->MOS1vgs] ;
                von = model->MOS1type * here->MOS1von ;
                if (vds >= 0)
                {
                    printf ("VDS >= 0\tMOS1type: %d\tMOS1instance: %s\tVgs: %-.9g\tVon: %-.9g\t", model->MOS1type, here->MOS1name, vgs, von) ;
                    if (vgs > von)
                    {
                        printf ("Acceso!\n") ;
                        NowIsON = 1 ;
                    } else {
                        printf ("Spento!\n") ;
                        NowIsON = 0 ;
                    }
                } else {
                    double vgd ;
                    vgd = vgs - vds ;
                    printf ("VDS <  0\tMOS1type: %d\tMOS1instance: %s\tVgd: %-.9g\tVon: %-.9g\t", model->MOS1type, here->MOS1name, vgd, von) ;
                    if (vgd > von)
                    {
                        printf ("Acceso!\n") ;
                        NowIsON = 1 ;
                    } else {
                        printf ("Spento!\n") ;
                        NowIsON = 0 ;
                    }
                }

                // If it's the first time, initialize 'here->MOS1reliability->IsON'
                if (here->MOS1reliability->IsON == -1)
                {
                    here->MOS1reliability->IsON = NowIsON ;
                }

                if (mode == 0)
                {
                    if (NowIsON)
                    {
                        if (here->MOS1reliability->IsON == 1)
                        {
                            // Until now, the device was ON - Do NOTHING
                            delta = -1 ;
                        } else if (here->MOS1reliability->IsON == 0) {
                            // Until now, the device was OFF - Calculate recovery
                            delta = ckt->CKTtime - here->MOS1reliability->time ;

                            // Calculate Aging - Giogio Liatis' Model
                            calculate_aging (here, delta, 0) ;

                            // Update time and flag - Stress begins
                            here->MOS1reliability->time = ckt->CKTtime ;
                            here->MOS1reliability->IsON = 1 ;
                        } else {
                            fprintf (stderr, "Reliability Analysis Error\n") ;
                        }
                    } else {
                        if (here->MOS1reliability->IsON == 1)
                        {
                            // Until now, the device was ON - Calculate stress
                            delta = ckt->CKTtime - here->MOS1reliability->time ;

                            // Calculate Aging - Giorgio Liatis' Model
                            calculate_aging (here, delta, 1) ;

                            // Update time and flag - Recovery begins
                            here->MOS1reliability->time = ckt->CKTtime ;
                            here->MOS1reliability->IsON = 0 ;
                        } else if (here->MOS1reliability->IsON == 0) {
                            // Until now, the device was OFF - Do NOTHING
                            delta = -1 ;
                        } else {
                            fprintf (stderr, "Reliability Analysis Error\n") ;
                        }
                    }
                } else if (mode == 1) {
                    // In this mode, it doesn't matter if NOW the device is in stress or in recovery, since it's the last timestep
                    if (here->MOS1reliability->IsON == 1)
                    {
                        // Calculate stress
                        delta = ckt->CKTtime - here->MOS1reliability->time ;
                        calculate_aging (here, delta, 1) ;

                        // Update time and flag - Maybe Optional
                        here->MOS1reliability->time = ckt->CKTtime ;
                        here->MOS1reliability->IsON = 1 ;
                    } else if (here->MOS1reliability->IsON == 0) {
                        // Calculate recovery
                        delta = ckt->CKTtime - here->MOS1reliability->time ;
                        calculate_aging (here, delta, 0) ;

                        // Update time and flag - Maybe Optional
                        here->MOS1reliability->time = ckt->CKTtime ;
                        here->MOS1reliability->IsON = 0 ;
                    } else {
                        fprintf (stderr, "Reliability Analysis Error\n") ;
                    }
                } else {
                    fprintf (stderr, "Reliability Analysis Error\n") ;
                }
                printf ("Time: %-.9gs\t", here->MOS1reliability->time) ;
                printf ("DeltaVth: %-.9gmV\t", here->MOS1reliability->deltaVth * 1000) ;
                printf ("IsON: %u\t", here->MOS1reliability->IsON) ;
                printf ("t_star: %-.9gs\n\n\n", here->MOS1reliability->t_star) ;
            }
        }
    }

    return (OK) ;
}
