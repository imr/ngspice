/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE sine/sin.h

Copyright 1991
Georgia Tech Research Corporation, Atlanta, Ga. 30332
All Rights Reserved

PROJECT A-8503-405


AUTHORS

    20 Mar 1991     Harry Li


MODIFICATIONS

     2 Oct 1991    Jeffrey P. Murray


SUMMARY

    This file contains additional header information for the
    sine code model.


INTERFACES

    FILE                 ROUTINE CALLED

    N/A                  N/A


REFERENCED FILES

    NONE


NON-STANDARD FEATURES

    NONE

===============================================================================*/

/*=== INCLUDE FILES ====================*/




/*=== CONSTANTS ========================*/

#define PI 3.141592654;

#define INT1 1

char *allocation_error = "\n**** Error ****\nSINE: Error allocating sine block storage \n";
char *limit_error = "\n**** Error ****\nSINE: Smoothing domain value too large \n";
char *sine_freq_clamp = "\n**** Warning ****\nSINE: Extrapolated frequency limited to 1e-16 Hz \n";
char *array_error = "\n**** Error ****\nSINE: Size of control array different than frequency array \n";



/*=== MACROS ===========================*/




/*=== LOCAL VARIABLES & TYPEDEFS =======*/




/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/
