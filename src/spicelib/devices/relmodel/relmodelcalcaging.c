/**********
Author: Francesco Lannutti - July 2015
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "relmodeldefs.h"
#include "../bsim4/bsim4def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

//#define CONSTepsZero (8.854214871e-12)   /* epsilon zero F/m */
//#define CONSTepsSiO2 (3.4531479969e-11)  /* epsilon SiO2 F/m */

static int
listInsert (RELMODELrelList **list, double time, double deltaVth)
{
    RELMODELrelList *current ;
    RELMODELrelList *previous ;

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
        *list = TMALLOC (RELMODELrelList, 1) ;
        current = *list ;
    } else {
        current = TMALLOC (RELMODELrelList, 1) ;
        previous->next = current ;
    }

    /* Populate the element */
    current->time = time ;
    current->deltaVth = deltaVth ;
    current->next = NULL ;

    return 0 ;
}

int
RELMODELcalculateAging (GENinstance *inInstance, int modType, CKTcircuit *ckt, double t_aging, double t_step, unsigned int stress_or_recovery)
{
    BSIM4instance *here ;
    RELMODELmodel *relmodel ;

    if (modType == 10)
    {
        here = (BSIM4instance *)inInstance ;
    } else {
        printf ("Error: The Reliability Analysis isn't supported for this model: %d\n\n\n", modType) ;
        return 1 ;
    }

    relmodel = (RELMODELmodel *)(here->BSIM4modPtr->BSIM4relmodelModel) ;
    if (relmodel == NULL)
    {
        printf ("Error: The Reliability Model hasn't been declared or it hasn't been connected to the device model (%s) through the '.appendmodel' command\n\n\n", here->BSIM4modPtr->BSIM4modName) ;
        return 1 ;
    }

    if (relmodel->RELMODELtype == 1) {
        // Giorgio Liatis' Model

        double A, i, Nt, R ;
        double b, qFI ;

        Nt = pow ((sqrt (relmodel->RELMODELnts)), 3) * 1e-21 ;
        A = (CHARGE / (4 * CONSTepsZero * 1e-9 * relmodel->RELMODELeps_hk)) * pow ((relmodel->RELMODELh_cut / (2 * sqrt (2 * relmodel->RELMODELm_star * relmodel->RELMODELw)) * 1e9), 2) ;

        if (t_step == 0)
        {
            // Extrapolation for 10 years when there is only stress
            here->relStruct->t_star = 0 ;
            if (relmodel->RELMODELh_cut / (2 * sqrt (2 * relmodel->RELMODELm_star * relmodel->RELMODELw)) * log (1 + pow (((t_aging + here->relStruct->t_star) / relmodel->RELMODELtau_0), relmodel->RELMODELbeta)) * 1e9 <= 2)
            {
                here->relStruct->deltaVth = Nt * A * pow (log (1 + pow (((t_aging + here->relStruct->t_star) / relmodel->RELMODELtau_0), relmodel->RELMODELbeta)), 2) ;
            } else {
                here->relStruct->deltaVth = (CHARGE / (4 * CONSTepsZero * 1e-9 * relmodel->RELMODELeps_hk)) * Nt * pow (here->BSIM4modPtr->BSIM4toxe * 1e9, 2) ;
            }
        } else {
            for (i = 0 ; i < t_aging ; i += t_step)
            {
                if (stress_or_recovery)
                {
                    if (relmodel->RELMODELh_cut / (2 * sqrt (2 * relmodel->RELMODELm_star * relmodel->RELMODELw)) * log (1 + pow (((i + here->relStruct->t_star) / relmodel->RELMODELtau_0), relmodel->RELMODELbeta)) * 1e9 <= 2)
                    {
                        here->relStruct->deltaVth = Nt * A * pow (log (1 + pow (((i + here->relStruct->t_star) / relmodel->RELMODELtau_0), relmodel->RELMODELbeta)), 2) ;
                    } else {
                        here->relStruct->deltaVth = (CHARGE / (4 * CONSTepsZero * 1e-9 * relmodel->RELMODELeps_hk)) * Nt * pow (here->BSIM4modPtr->BSIM4toxe * 1e9, 2) ;
                    }
                    here->relStruct->deltaVthMax = here->relStruct->deltaVth ;
                } else {
                    // Without Temperature Dependency
//                    here->relStruct->deltaVth = here->relStruct->deltaVthMax * log (1 + (1.718 / (1 + pow ((i / relmodel->RELMODELtau_e), relmodel->RELMODELbeta1)))) ;

                    // With Temperature Dependency
//                    b = 5.621 * 1.2e6 ;
                    b = 0.706 ;
                    qFI = 1.03 ;
                    R = b * pow (ckt->CKTtemp, 2) * exp (-qFI / (relmodel->RELMODELk_b * ckt->CKTtemp)) ; // R = b * pow (T, 2) * exp (-qFI/kT) ;
                    here->relStruct->deltaVth = (1 - R) * here->relStruct->deltaVthMax * (1 - log (1 + (1.718 / (1 + pow ((i / relmodel->RELMODELtau_e), relmodel->RELMODELbeta1))))) ;
                }

                // Insert 'here->relStruct->deltaVth' into the list for the later fitting
                listInsert (&(here->relStruct->deltaVthList), here->relStruct->offsetTime + i, here->relStruct->deltaVth) ;
            }
            here->relStruct->offsetTime += i ;
        }

        if (!stress_or_recovery)
        {
            here->relStruct->t_star = pow ((exp (sqrt (here->relStruct->deltaVth / (Nt * A))) - 1), (1 / relmodel->RELMODELbeta)) * relmodel->RELMODELtau_0 ;
        }
    } else if (relmodel->RELMODELtype == 2) {
        // Model taken from the following article for 65nm devices with 1.4nm plasma-nitrided oxide
        // A compact model for NBTI degradation and recovery under use-profile variations and its application to aging analysis of digital integrated circuits

        BSIM4vgsList *current ;
        double A, delta_Vth_fast, delta_Vth_slow, i, vgs_average ;

//        V = 2 ;

        if (stress_or_recovery) {
            // Calculate the Average Vgs
            current = here->vgsList ;
            vgs_average = 0 ;
            i = 0 ;
            while (current != NULL)
            {
                vgs_average += fabs (current->vgs) ;
                i++ ;
                current = current->next ;
            }
            here->relStruct->Vstress = vgs_average / i ;
//            printf ("Device Type: %s\t\t", here->BSIM4modPtr->BSIM4modName) ;
//            printf ("V: %-.9g\t\t", V) ;
//            printf ("Vgs: %-.9g\t\t", ckt->CKTstate0 [here->BSIM4vgs]) ;
//            printf ("Vges: %-.9g\t\t", ckt->CKTstate0 [here->BSIM4vges]) ;
//            printf ("Vds: %-.9g\n\n", ckt->CKTstate0 [here->BSIM4vds]) ;
        }

        if (t_step == 0)
        {
            // Extrapolation for 10 years when there is only stress
            A = relmodel->RELMODELk_new * exp (-relmodel->RELMODELe0_new / (relmodel->RELMODELkb_new * ckt->CKTtemp)) * exp (relmodel->RELMODELb_new * here->relStruct->Vstress / (here->BSIM4modPtr->BSIM4toxe * 1e9 * relmodel->RELMODELkb_new * ckt->CKTtemp)) ;
            delta_Vth_fast = A * log (1 + ((t_aging / relmodel->RELMODELtau_c_fast_new))) ;
            delta_Vth_slow = A * log (1 + ((t_aging / relmodel->RELMODELtau_c_slow_new))) ;
            here->relStruct->deltaVth = relmodel->RELMODELalpha_new * delta_Vth_fast + (1 - relmodel->RELMODELalpha_new) * delta_Vth_slow ;

            // Saturation at 10% of the Average Vgs - This is not in the original aging model
            if (here->relStruct->deltaVth > here->relStruct->Vstress * 0.5)
            {
                here->relStruct->deltaVth = here->relStruct->Vstress * 0.5 ;
            }
        } else {
            for (i = 0 ; i < t_aging ; i += t_step)
            {
                if (stress_or_recovery)
                {
                    relmodel->RELMODELt_s_new += i ;
                } else {
                    relmodel->RELMODELt_r_new += i ;
                }
                A = relmodel->RELMODELk_new * exp (-relmodel->RELMODELe0_new / (relmodel->RELMODELkb_new * ckt->CKTtemp)) * exp (relmodel->RELMODELb_new * here->relStruct->Vstress / (here->BSIM4modPtr->BSIM4toxe * 1e9 * relmodel->RELMODELkb_new * ckt->CKTtemp)) ;
                delta_Vth_fast = A * log (1 + ((relmodel->RELMODELt_s_new / relmodel->RELMODELtau_c_fast_new) / (1 + pow (relmodel->RELMODELt_r_new / relmodel->RELMODELtau_e_fast_new, relmodel->RELMODELbeta_new)))) ;
                delta_Vth_slow = A * log (1 + ((relmodel->RELMODELt_s_new / relmodel->RELMODELtau_c_slow_new) / (1 + pow (relmodel->RELMODELt_r_new / relmodel->RELMODELtau_e_slow_new, relmodel->RELMODELbeta_new)))) ;
                here->relStruct->deltaVth = relmodel->RELMODELalpha_new * delta_Vth_fast + (1 - relmodel->RELMODELalpha_new) * delta_Vth_slow ;

                // Insert 'here->relStruct->deltaVth' into the list for the later fitting
                listInsert (&(here->relStruct->deltaVthList), here->relStruct->offsetTime + i, here->relStruct->deltaVth) ;
            }
            here->relStruct->offsetTime += i ;
        }
    }

    return 0 ;
}
