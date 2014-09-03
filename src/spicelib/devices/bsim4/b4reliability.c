/**********
Author: Francesco Lannutti - August 2014
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4def.h"
#include "../relmodel/relmodeldefs.h"
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
            if (model->BSIM4type == PMOS)
            {
                here->BSIM4delvto = here->BSIM4agingDelvto [step] ;
            }
        }
    }

    return (OK) ;
}

int
BSIM4agingSetup (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM4model *model = (BSIM4model*)inModel ;
    BSIM4instance *here ;
    RELMODELmodel *relmodelModel ;
    unsigned int i ;

//    const double K1_2 = 81e53 ;                // [C^(-1/2) * m^(-5/2)] //------------------fittato   8----->9 ->squared
//    const double T0 = 1e10 ;                   // [s/m^2] //------------fittato  1---->0.05
//    const double E_01 = 0.335e9 ;              // [V/m] //
//    const double Ea = (0.13 * 0.1602136e-18) ; // [J] // Activation energy   ---------------- Fittato    0.49-------------------->0.04
//    const double X1 = 1 ;                    // [dimensionless] // Process parameter
//    const double X2 = 0.5 ;                    // [dimensionless] // Process parameter

//    double T_clk = 1e-9 ; // [s] // Clock period (from netlist)
//    double alfa = 0.5 ;    // [dimensionless]  // Duty cycle (from paper)

    double f = 0.333333333333333333333333333333333333333 ;

    double vges ;
    double Kv ;
    double C ;         // Temperature dependant constant
    double beta ;      // Recovery factor
    double delta_Vth ; // Voltage threshold shift

    double r ;

    ckt->CKTagingN = (int)((ckt->CKTagingTotalTime - ckt->CKTagingStartTime) / ckt->CKTagingStep) ;

    /* loop through all the BSIM4 device models */
    for (; model != NULL ; model = model->BSIM4nextModel)
    {
        if (model->BSIM4type == PMOS)
        {
            relmodelModel = (RELMODELmodel *)(model->BSIM4relmodelModel) ;

            C = 1 / relmodelModel->RELMODELt0 * exp (-relmodelModel->RELMODELea * 0.1602136e-18 / (CONSTboltz * ckt->CKTtemp)) ;

            /* loop through all the instances of the model */
            for (here = model->BSIM4instances ; here != NULL ; here = here->BSIM4nextInstance)
            {
                here->BSIM4agingDelvto = TMALLOC (double, (size_t)ckt->CKTagingN) ;

//                vgs = model->BSIM4type * (ckt->CKTrhsOld [here->BSIM4gNodePrime] - ckt->CKTrhsOld [here->BSIM4sNodePrime]) ;
//                vges = model->BSIM4type * (ckt->CKTrhsOld [here->BSIM4gNodeExt] - ckt->CKTrhsOld [here->BSIM4sNodePrime]) ;
                vges = model->BSIM4type * (ckt->CKTrhsOld [here->BSIM4gNodeExt] - ckt->CKTrhsOld [here->BSIM4sNode]) ;
                printf ("\n\nName: %s\nVth: %-.9g\nVges: %-.9g\nToxe: %-.9g\n", here->BSIM4name, here->BSIM4von, model->BSIM4type * vges, model->BSIM4toxe) ;

                Kv = (CHARGE * CHARGE * CHARGE * model->BSIM4toxe * model->BSIM4toxe) / (CONSTepsSiO2 * CONSTepsSiO2) * relmodelModel->RELMODELk1_2 * (model->BSIM4type * vges - model->BSIM4type * here->BSIM4von) * sqrt (C) * exp (2 * (model->BSIM4type * vges - model->BSIM4type * here->BSIM4von) / (model->BSIM4toxe * relmodelModel->RELMODELe_01)) ;

                /* loop through all the aging steps */
                i = 0 ;
                for (r = ckt->CKTagingStartTime + ckt->CKTagingStep ; r <= ckt->CKTagingTotalTime ; r += ckt->CKTagingStep)
                {
                    beta = 1 - (2 * relmodelModel->RELMODELx1 * model->BSIM4toxe + sqrt (relmodelModel->RELMODELx2 * C * (1 - relmodelModel->RELMODELalfa) * relmodelModel->RELMODELt_clk)) / (2 * model->BSIM4toxe + sqrt (C * r)) ;
                    delta_Vth = pow ((sqrt (Kv * Kv * relmodelModel->RELMODELalfa * relmodelModel->RELMODELt_clk) / (1 - beta * beta * beta)), f) ;
                    here->BSIM4agingDelvto [i] = delta_Vth ;
                    double Vth ;
                    Vth = here->BSIM4von + here->BSIM4agingDelvto [i] ;
//                    printf ("Delta Vth equals to %-.9g for %.1f second(s)\t\t", here->BSIM4agingDelvto [i], r) ;
                    printf ("Vth = %-.9lf\t%.1f seconds\n", Vth, r) ;
                    printf ("delvto = %-.9lf\t%.1f seconds\n", here->BSIM4agingDelvto [i], r) ;
                    i++ ;
                }
                printf ("\n\n") ;
            }
        }
    }

    return (OK) ;
}
