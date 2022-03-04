/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE d_pwm/d_pwm.h

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503-405
The ngspice team

AUTHORS                      

    25 Jul 1991     Jeffrey P. Murray
    02 Mar 2022     Holger Vogt


MODIFICATIONS   

    30 Sept 1991    Jeffrey P. Murray
                                   

SUMMARY

    This file contains the header information for the d_pwm
    code model.


INTERFACES       

    FILE                 ROUTINE CALLED     

    N/A                  N/A
                         


REFERENCED FILES

    N/A
                     

NON-STANDARD FEATURES

    NONE

===============================================================================*/
/*
         Structures, etc. for d_pwm oscillator model.     
         7/25/90                                     
         Last Modified 7/25/91               J.P.Murray
                       3/02/22               H. Vogt         */

/*=======================================================================*/

/*=== INCLUDE FILES =====================================================*/
                                    
#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <string.h>




/*=== CONSTANTS =========================================================*/


/****  Error Messages ****/
char *d_pwm_allocation_error = "\n**** Error ****\nD_PWM: Error allocating VCO block storage \n";
char *d_pwm_array_error = "\n**** Error ****\nD_PWM: Size of control array different than duty cycle array \n";
char *d_pwm_negative_dc_error = "\n**** Error ****\nD_PWM: The extrapolated value for duty cycle\nhas been found to be negative... \n Lower duty cycle level has been clamped to 0.0  \n";
char *d_pwm_positive_dc_error = "\n**** Error ****\nD_PWM: The extrapolated value for duty cycle\nhas been found to be > 1... \n Upper duty cycle level has been clamped to 1.0  \n";






/*=== MACROS ============================================================*/


  
/*=== LOCAL VARIABLES & TYPEDEFS ========================================*/                         

    
           
/*=== FUNCTION PROTOTYPE DEFINITIONS ====================================*/


/*=======================================================================*/
                                    

