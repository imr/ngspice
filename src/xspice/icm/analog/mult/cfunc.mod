/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE mult/cfunc.mod

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503-405
               

AUTHORS                      

    20 Mar 1991     Jeffrey P. Murray


MODIFICATIONS   

     2 Oct 1991    Jeffrey P. Murray
                                   

SUMMARY

    This file contains the model-specific routines used to
    functionally describe the mult code model.


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

FUNCTION void cm_mult()

AUTHORS                      

    20 Mar 1991     Jeffrey P. Murray


MODIFICATIONS   

     2 Oct 1991    Jeffrey P. Murray

SUMMARY

    This function implements the mult code model.

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

/*=== CM_MULT ROUTINE ===*/
                                                   

void cm_mult(ARGS) 

{
    int i;     /* generic loop counter index */
	int size;  /* number of input ports */

    double accumulate_gain;  /* product of all the gains */
	double accumulate_in;    /* product of all (inputs + offsets) */
	double final_gain;       /* output gain */

    Mif_Complex_t ac_gain;


    size = PORT_SIZE(in);               /* Note that port size */
    final_gain = PARAM(out_gain);        /* and out_gain are read only */
                                        /* once...saves access time. */


    /* Calculate multiplication of inputs and gains for   */
    /* all types of analyes....                           */

    accumulate_gain = 1.0;
    accumulate_in = 1.0;

    for (i=0; i<size; i++) {
        accumulate_gain = accumulate_gain *    /* Multiply all input gains */
                             PARAM(in_gain[i]); 
        accumulate_in = accumulate_in *        /* Multiply offset input values */
                           (INPUT(in[i]) + PARAM(in_offset[i]));
    }


    if(ANALYSIS != MIF_AC) {                /* DC & Transient */         

        for (i=0; i<size; i++) {   /* Partials are product of all gains and */
                                   /*   inputs divided by each individual   */
                                   /*      input value.                     */

            if (0.0 != accumulate_in) {  /* make sure that no division by zero
                                            will occur....                      */
                PARTIAL(out,in[i]) = (accumulate_in/(INPUT(in[i]) +
                         PARAM(in_offset[i]))) * accumulate_gain * final_gain;
            }
            else {                       /* otherwise, set partial to zero.     */
                PARTIAL(out,in[i]) = 0.0;
            }

        }

        OUTPUT(out) = accumulate_in * accumulate_gain * final_gain + 
                                          PARAM(out_offset);
    }

    else {                              /* AC Analysis */
 
        for (i=0; i<size; i++) {   /* Partials are product of all gains and */
                                   /*   inputs divided by each individual   */
                                   /*      input value.                     */
            ac_gain.real = (accumulate_in/(INPUT(in[i]) +
                     PARAM(in_offset[i]))) * accumulate_gain * final_gain;
            ac_gain.imag = 0.0;
            AC_GAIN(out,in[i]) = ac_gain;
        }
    }
}
