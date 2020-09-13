/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE d_osc/d_osc.h

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503-405
               

AUTHORS                      

    25 Jul 1991     Jeffrey P. Murray


MODIFICATIONS   

    30 Sept 1991    Jeffrey P. Murray
                                   

SUMMARY

    This file contains the header information for the d_osc
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
         Structures, etc. for d_osc oscillator model.     
         7/25/90                                     
         Last Modified 7/25/91               J.P.Murray */

/*=======================================================================*/

/*=== INCLUDE FILES =====================================================*/
                                    
#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <string.h>




/*=== CONSTANTS =========================================================*/


/****  Error Messages ****/
char *d_osc_allocation_error = "\n**** Error ****\nD_OSC: Error allocating VCO block storage \n";
char *d_osc_array_error = "\n**** Error ****\nD_OSC: Size of control array different than frequency array \n";
char *d_osc_negative_freq_error = "\n**** Error ****\nD_OSC: The extrapolated value for frequency\nhas been found to be negative... \n Lower frequency level has been clamped to 0.0 Hz \n";






/*=== MACROS ============================================================*/


  
/*=== LOCAL VARIABLES & TYPEDEFS ========================================*/                         

    
           
/*=== FUNCTION PROTOTYPE DEFINITIONS ====================================*/


/*=======================================================================*/
                                    

