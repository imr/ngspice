/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE square/square.h

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
    square (controlled squarewave oscillator) code model.


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

char *square_allocation_error = "\n**** Error ****\nSQUARE: Error allocating square block storage \n";
char *square_limit_error = "\n**** Error ****\nSQUARE: Smoothing domain value too large \n";
char *square_freq_clamp = "\n**** WARNING  ****\nSQUARE: Frequency extrapolation limited to 1e-16 \n";
char *square_array_error = "\n**** Error ****\nSQUARE: Size of control array different than frequency array \n";


#define INT1 1
#define T1 2
#define T2 3
#define T3 4
#define T4 5



/*=== MACROS ===========================*/




/*=== LOCAL VARIABLES & TYPEDEFS =======*/




/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/





