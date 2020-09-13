/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE slew/cfunc.mod

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503-405
               

AUTHORS                      

    15 Apr 1991     Harry Li


MODIFICATIONS   

     2 Oct 1991    Jeffrey P. Murray
                                   

SUMMARY

    This file contains the model-specific routines used to
    functionally describe the slew (slew rate) code model.


INTERFACES       

    FILE                 ROUTINE CALLED     

    CM.c                 void *cm_analog_alloc()
                         void *cm_analog_get_ptr()


REFERENCED FILES

    Inputs from and outputs to ARGS structure.
                     

NON-STANDARD FEATURES

    NONE

===============================================================================*/

/*=== INCLUDE FILES ====================*/

#include "slew.h"

                                      

/*=== CONSTANTS ========================*/




/*=== MACROS ===========================*/



  
/*=== LOCAL VARIABLES & TYPEDEFS =======*/                         


    
           
/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/




                   
/*==============================================================================

FUNCTION void cm_slew()

AUTHORS                      

    15 Apr 1991     Harry Li


MODIFICATIONS   

     2 Oct 1991    Jeffrey P. Murray

SUMMARY

    This function implements the slew code model.

INTERFACES       

    FILE                 ROUTINE CALLED     

    CM.c                 void *cm_analog_alloc()
                         void *cm_analog_get_ptr()


RETURNED VALUE
    
    Returns inputs and outputs via ARGS structure.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== CM_SLEW ROUTINE ===*/

/****************************************************************
*                                                               *
*  This model describes a single input, single output slew      *
*  rate limited block.  The user may specify the positive       *
*  and negative slew rates.                                     *
*                                                               *
*  Note that the model makes no provision for output high and   *
*  low values.  That is assumed to be handled by another model. *
*                                                               *
****************************************************************/

void cm_slew(ARGS)   

{
    double *ins;        /* input value                            */
	double *in_old;     /* previous input value                   */
	double *outs;       /* output value                           */
	double *out_old;    /* previous output value                  */
	double pout_pin;    /* partial derivative--output wrt input   */
	double delta;       /* change in time from previous iteration */
	double slope_rise;  /* positive going slew rate               */
	double slope_fall;  /* negative going slew rate               */
    double out_slew;    /* "slewed" output value                  */
	double slope;       /* slope of the input wrt time            */ 

    Mif_Complex_t ac_gain;
                                                   
   /** Retrieve frequently used parameters (used by all analyses)... **/

   if (INIT == 1) { 

		/* First pass...allocate storage for previous state.   */
    
		cm_analog_alloc(INT1,sizeof(double));   
		cm_analog_alloc(INT4,sizeof(double));
		cm_analog_alloc(INT2,sizeof(double));
		cm_analog_alloc(INT3,sizeof(double));

    }

    if (ANALYSIS == MIF_DC) {  

    /* DC analysis, get old values */ 

            ins= (double *) cm_analog_get_ptr(INT1,0);
			outs= (double *) cm_analog_get_ptr(INT4,0);
            in_old = (double *) cm_analog_get_ptr(INT1,0);
			out_old = (double *) cm_analog_get_ptr(INT4,0);

            *ins = *in_old = INPUT(in);   
            *outs = *out_old = *ins;

			/***    so, return a zero d/dt value. ***/
            pout_pin = 1.0;

        OUTPUT(out) = INPUT(in);          
        PARTIAL(out,in) = 1; 

    }else

    if (ANALYSIS == MIF_TRAN) {     /**** DC & Transient Analyses ****/

        /** Retrieve frequently used parameters... **/

        slope_rise = PARAM(rise_slope);
        slope_fall = PARAM(fall_slope);

        /* Allocation not necessary...retrieve previous values */

         ins = (double *) cm_analog_get_ptr(INT1,0);   /* Set out pointer to current 
                                                            time storage */    
         in_old = (double *) cm_analog_get_ptr(INT1,1);  /*  Set old-output-state pointer */
		 outs = (double *) cm_analog_get_ptr(INT4,0);
         out_old = (double *) cm_analog_get_ptr(INT4,1);   /* Set old-output-state pointer 
                                              previous time storage */    
       

        if ( TIME == 0.0 ) {         /*** Test to see if this is the first ***/
                                     /***    timepoint calculation...if    ***/

            *ins = *in_old = INPUT(in);   
            *outs = *out_old = *ins;   /* input = output, d/dt = 1 */ 
            pout_pin = 1.0;

        }else{      

     /* determine the slope of the input */
		delta = TIME - T(1); 
     	*ins = INPUT(in);
		slope = (*ins - *in_old)/delta;

		if(slope >= 0){

			out_slew = *out_old + slope_rise*delta;

			if(*ins < (*out_old - slope_fall*delta)){

			/* If the input had a negative slope (and the output
			   was slewing) and then changed direction to a positive 
			   slope and the "slewed" response hasn't caught up 
			   to the input yet (input < slewed output), then 
			   continue negative slewing until the slewed output 
			   meets the positive sloping input */

				*outs = *out_old - slope_fall*delta;
				pout_pin = 0;

			}else

            /* Two conditions for slewing, if the slope is greater
			   than the positive slew rate, or if the input slope
			   is less than the positive slew rate and the slewed output
			   is less than the input.  This second condition occurs
			   if the input levels off and the slewed output hasn't
			   caught up to the input yet */

			if((slope > slope_rise) || ((slope < slope_rise) && (out_slew <= *ins))){  
			/* SLEWING ! */
					*outs = out_slew; 
				    pout_pin = 0;

			}else{
                   /* No slewing, output=input */
					*outs = *ins;   
					pout_pin = 1;
                             	
				}

        }else{	  /* this ends the positive slope stuff */ 

			out_slew = *out_old - slope_fall*delta;

			if(*ins > (*out_old + slope_rise*delta)){

			/* If the input had a positive slope (and the output
			   was slewing) and then changed direction to a negative 
			   slope and the "slewed" response hasn't caught up 
			   to the input yet (input > slewed output), then 
			   continue positive slewing until the slewed output 
			   meets the negative sloping input */

				*outs = *out_old + slope_rise*delta;
				pout_pin = 0;

			}else

            /* Two conditions for slewing, if the negative slope is 
			   greater than the neg. slew rate, or if the neg. input 
			   slope is less than the negative slew rate and the 
			   slewed output is greater than the input.  This second 
			   condition occurs if the input levels off and the 
			   slewed output hasn't caught up to the input yet */

			if((-slope > slope_fall) || ((-slope < slope_fall) && (out_slew > *ins))){  /* SLEWING ! */
					*outs = out_slew;
					pout_pin = 0;

			}else{

					*outs = *ins;
					pout_pin = 1;

			}


		}

   }
        /** Output values for DC & Transient **/

        OUTPUT(out) = *outs;          
        PARTIAL(out,in) = pout_pin; 


    }else{                    /**** AC Analysis...output (0.0,s*gain) ****/
        ac_gain.real = 1.0;
        ac_gain.imag= 0; 
        AC_GAIN(out,in) = ac_gain;
    }
}





