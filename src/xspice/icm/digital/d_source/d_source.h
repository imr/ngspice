/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE d_source/cfunc.mod

Copyright 1991
Georgia Tech Research Corporation, Atlanta, Ga. 30332
All Rights Reserved

PROJECT A-8503-405
               

AUTHORS                      

    13 Jun 1991     Jeffrey P. Murray


MODIFICATIONS   

     8 Aug 1991    Jeffrey P. Murray
    30 Sep 1991    Jeffrey P. Murray
                                   

SUMMARY

    This file contains the header information used by the 
    d_source code model.


INTERFACES       

    FILE                 ROUTINE CALLED     

    N/A                  N/A
                         


REFERENCED FILES

    NONE
                     

NON-STANDARD FEATURES

    NONE

===============================================================================*/

/*=== INCLUDE FILES ====================*/

#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <string.h>

                                      

/*=== CONSTANTS ========================*/

#define OK 0
#define FAIL 1                                        



/*=== MACROS ===========================*/



  
/*=== LOCAL VARIABLES & TYPEDEFS =======*/                         

typedef char line_t[82]; /* A SPICE size line. <= 80 characters plus '\n\0' */

    
           
/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/




                   
/*=============================================================================*/
