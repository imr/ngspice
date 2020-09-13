/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE summer/cfunc.mod

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503-405
               

AUTHORS                      

     9 Apr 1991     Jeffrey P. Murray


MODIFICATIONS   

     2 Oct 1991    Jeffrey P. Murray
                                   

SUMMARY

    This file contains the model-specific routines used to
    functionally describe the summer code model.


INTERFACES       

    FILE                 ROUTINE CALLED     

    N/A                  N/A


REFERENCED FILES

    Inputs from and outputs to ARGS structure.
                     

NON-STANDARD FEATURES

    NONE

===============================================================================*/

/*=== INCLUDE FILES ====================*/


                                      

/*=== CONSTANTS ========================*/




/*=== MACROS ===========================*/



  
/*=== LOCAL VARIABLES & TYPEDEFS =======*/                         


    
           
/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/




                   
/*==============================================================================

FUNCTION void cm_summer()

AUTHORS                      

     9 Apr 1991     Jeffrey P. Murray

MODIFICATIONS   

     2 Oct 1991    Jeffrey P. Murray

SUMMARY

    This function implements the summer code model.

INTERFACES       

    FILE                 ROUTINE CALLED     

    N/A                  N/A


RETURNED VALUE
    
    Returns inputs and outputs via ARGS structure.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== CM_SUMMER ROUTINE ===*/
                                                   

void cm_summer(ARGS) 

{
    int i;                 /* generic loop counter index */
	int size;              /* number of inputs */

    double accumulate;     /* sum of all the (inputs times their
							  respective gains plus their offset). */
	double final_gain;     /* output gain stage */
	double in_gain_temp;   /* temporary variable used to calculate
							  accumulate */

    Mif_Complex_t ac_gain;


    size = PORT_SIZE(in);               /* Note that port size */
    final_gain = PARAM(out_gain);        /* and out_gain are read only */
                                        /* once...saves access time. */
    if(ANALYSIS != MIF_AC) {       /* DC & Transient */         
        accumulate = 0.0;
        for (i=0; i<size; i++) {
            in_gain_temp = PARAM(in_gain[i]);  /* Ditto for in_gain[i] */
            accumulate = accumulate + in_gain_temp * 
                            (INPUT(in[i]) + PARAM(in_offset[i]));
            PARTIAL(out,in[i]) = in_gain_temp * final_gain;
        }
        OUTPUT(out) = accumulate * final_gain + PARAM(out_offset);
    }

    else {                     /* AC Analysis */
        for (i=0; i<size; i++) {
            ac_gain.real = PARAM(in_gain[i]) * final_gain; 
            ac_gain.imag = 0.0;
            AC_GAIN(out,in[i]) = ac_gain;
        }
    }
}
