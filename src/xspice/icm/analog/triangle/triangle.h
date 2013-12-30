/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE triangle/triangle.h

Copyright 1991
Georgia Tech Research Corporation, Atlanta, Ga. 30332
All Rights Reserved

PROJECT A-8503-405


AUTHORS

    12 Apr 1991     Harry Li


MODIFICATIONS

     2 Oct 1991    Jeffrey P. Murray


SUMMARY

    This file contains additional header information for the
    triangle (controlled trianglewave oscillator) code model.


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

char *triangle_allocation_error = "\n**** Error ****\nTRIANGLE: Error allocating triangle block storage \n";
char *triangle_freq_clamp = "\n**** Warning ****\nTRIANGLE: Extrapolated Minimum Frequency Set to 1e-16 Hz \n";
char *triangle_array_error = "\n**** Error ****\nTRIANGLE: Size of control array different than frequency array \n";


#define INT1 1
#define T1 2
#define T2 3
#define T3 4



/*=== MACROS ===========================*/




/*=== LOCAL VARIABLES & TYPEDEFS =======*/




/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/




