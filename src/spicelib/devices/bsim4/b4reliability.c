/**********
Author: Francesco Lannutti - August 2014
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4def.h"
#include "ngspice/sperror.h"

int
BSIM4agingAdd (GENmodel *inModel, int step)
{
    BSIM4model *model = (BSIM4model*)inModel ;
    BSIM4instance *here ;

    /* loop through all the BSIM4 device models */
    for (; model != NULL ; model = model->BSIM4nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM4instances ; here != NULL ; here = here->BSIM4nextInstance)
        {
            here->BSIM4delvto = here->BSIM4agingDelvto [step] ;
        }
    }

    return (OK) ;
}

int
BSIM4agingSetup (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM4model *model = (BSIM4model*)inModel ;
    BSIM4instance *here ;
    unsigned int i ;
    double vges ;

    const double K1_2 = 81e53 ;                // [C^(-1/2) * m^(-5/2)] //------------------fittato   8----->9 ->squared
    const double T0 = 0.05e10 ;                // [s/m^2] //------------fittato  1---->0.05
    const double E_01 = 0.335e9 ;              // [V/m] //
    const double Ea = (0.04 * 0.1602136e-18) ; // [J] // Activation energy   ---------------- Fittato    0.49-------------------->0.04
    const double X1 = 0.9 ;                    // [dimensionless] // Process parameter
    const double X2 = 0.5 ;                    // [dimensionless] // Process parameter

    double T_clk = 10e-9 ; // [s] // Clock period (from netlist)
    double alfa = 0.5 ;    // [dimensionless]  // Duty cycle (from paper)

    double f = 0.333333333333333333333333333333333333333 ;

    double Kv ;
    double C ;         // Temperature dependant constant
    double beta ;      // Recovery factor
    double delta_Vth ; // Voltage threshold shift

    double r ;

    C = 1 / T0 * exp (-Ea / (CONSTboltz * ckt->CKTtemp)) ;

    ckt->CKTagingN = (int)((ckt->CKTagingTotalTime - ckt->CKTagingStartTime) / ckt->CKTagingStep) ;

    /* loop through all the BSIM4 device models */
    for (; model != NULL ; model = model->BSIM4nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM4instances ; here != NULL ; here = here->BSIM4nextInstance)
        {
            here->BSIM4agingDelvto = TMALLOC (double, (size_t)ckt->CKTagingN) ;

            //vgs = model->BSIM4type * (ckt->CKTrhsOld [here->BSIM4gNodePrime] - ckt->CKTrhsOld [here->BSIM4sNodePrime]) ;
            vges = model->BSIM4type * (ckt->CKTrhsOld [here->BSIM4gNodeExt] - ckt->CKTrhsOld [here->BSIM4sNodePrime]) ;
//            printf ("\n\nName: %s\nVth: %-.9g\nVges: %-.9g\nToxe: %-.9g\n", here->BSIM4name, model->BSIM4type * here->BSIM4von, model->BSIM4type * vges, model->BSIM4toxe) ;

            Kv = (CHARGE * CHARGE * CHARGE * model->BSIM4toxe * model->BSIM4toxe) / (CONSTepsSiO2 * CONSTepsSiO2) * K1_2 * (model->BSIM4type * vges - model->BSIM4type * here->BSIM4von) * sqrt (C) * exp (2 * (model->BSIM4type * vges - model->BSIM4type * here->BSIM4von) / (model->BSIM4toxe * E_01)) ;

            /* loop through all the aging steps */
            i = 0 ;
            for (r = ckt->CKTagingStartTime + ckt->CKTagingStep ; r <= ckt->CKTagingTotalTime ; r += ckt->CKTagingStep)
            {
                beta = 1 - (2 * X1 * model->BSIM4toxe + sqrt (X2 * C * (1 - alfa) * T_clk)) / (2 * model->BSIM4toxe + sqrt (C * r)) ;
                delta_Vth = pow ((sqrt ((pow (Kv, 2)) * alfa * T_clk) / (1 - pow (beta, 3))), f) ;
                here->BSIM4agingDelvto [i] = delta_Vth ;
//                double Vth ;
//                Vth = model->BSIM4type * here->BSIM4von - here->BSIM4agingDelvto [i] ;
//                printf ("Delta Vth equals to %-.9g for %.1f second(s)\t\t", here->BSIM4agingDelvto [i], r) ;
//                printf ("Vth equals to %-.9g for %.1f second(s)\n", Vth, r) ;
                i++ ;
            }
        }
    }
    printf ("\n\n") ;

    return (OK) ;
}
