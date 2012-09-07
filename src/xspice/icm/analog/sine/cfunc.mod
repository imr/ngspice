/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE sine/cfunc.mod

Copyright 1991
Georgia Tech Research Corporation, Atlanta, Ga. 30332
All Rights Reserved

PROJECT A-8503-405


AUTHORS

    20 Mar 1991     Harry Li


MODIFICATIONS

     2 Oct 1991    Jeffrey P. Murray
     7 Sep 2012    Holger Vogt


SUMMARY

    This file contains the model-specific routines used to
    functionally describe the sine (controlled sine-wave
    oscillator) code model.


INTERFACES

    FILE                 ROUTINE CALLED

    CMmacros.h           cm_message_send();

    CM.c                 void *cm_analog_alloc()
                         void *cm_analog_get_ptr()


REFERENCED FILES

    Inputs from and outputs to ARGS structure.


NON-STANDARD FEATURES

    NONE

===============================================================================*/

/*=== INCLUDE FILES ====================*/

#include "sin.h"
#include <math.h>




/*=== CONSTANTS ========================*/




/*=== MACROS ===========================*/




/*=== LOCAL VARIABLES & TYPEDEFS =======*/


typedef struct {

    double   *control;   /* the storage array for the
                            control vector (cntl_array)   */

    double   *freq;   /* the storage array for the
                         frequency vector (freq_array)   */
} Local_Data_t;


/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/





/*==============================================================================

FUNCTION void cm_sine()

AUTHORS

    20 Mar 1991     Harry Li

MODIFICATIONS

     2 Oct 1991    Jeffrey P. Murray
     7 Sep 2012    Holger Vogt

SUMMARY

    This function implements the sine (controlled sinewave
    oscillator) code model.

INTERFACES

    FILE                 ROUTINE CALLED

    CMmacros.h           cm_message_send();

    CM.c                 void *cm_analog_alloc()
                         void *cm_analog_get_ptr()

RETURNED VALUE

    Returns inputs and outputs via ARGS structure.

GLOBAL VARIABLES

    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/
#include <stdlib.h>

/*=== CM_SINE ROUTINE ===*/

void cm_sine(ARGS)  /* structure holding parms,
                                   inputs, outputs, etc.     */
{
        int i;             /* generic loop counter index                    */
		int cntl_size;     /* control array size                            */
		int freq_size;     /* frequency array size                          */

    	double *x;         /* pointer to the control array values           */
	    double *y;         /* pointer to the frequency array values         */
		double cntl_input; /* control input                                 */
		/*double out;*/        /* output value                                  */
		double dout_din;   /* partial derivative of output wrt control in   */
		double output_low; /* output low value                              */
		double output_hi;  /* output high value                             */
        double *phase;     /* pointer to the instantaneous phase value      */
		double *phase1;    /* pointer to the previous value for the phase   */
		double freq=0.0;       /* frequency of the sine wave                    */
		double center;     /* dc offset for the sine wave                   */
		double peak;       /* peak voltage value for the wave               */
		double radian;     /* phase value in radians                        */

    Mif_Complex_t ac_gain;

    Local_Data_t *loc;        /* Pointer to local static data, not to be included
                                       in the state vector */


    /**** Retrieve frequently used parameters... ****/


    cntl_size = PARAM_SIZE(cntl_array);
    freq_size = PARAM_SIZE(freq_array);
    output_low = PARAM(out_low);
    output_hi = PARAM(out_high);

	if(cntl_size != freq_size){
		cm_message_send(array_error);
		return;
 	}

  if(INIT == 1){

	cm_analog_alloc(INT1,sizeof(double));


    /*** allocate static storage for *loc ***/
    STATIC_VAR (locdata) = calloc (1 , sizeof ( Local_Data_t ));
    loc = STATIC_VAR (locdata);


    /* Allocate storage for breakpoint domain & freq. range values */
    x = loc->control = (double *) calloc((size_t) cntl_size, sizeof(double));
    if (!x) {
        cm_message_send(allocation_error);
        return;
    }
    y = loc->freq = (double *) calloc((size_t) freq_size, sizeof(double));
    if (!y) {
        cm_message_send(allocation_error);
        return;
    }


  }

  if(ANALYSIS == MIF_DC){

		OUTPUT(out) = (output_hi + output_low)/2;
		PARTIAL(out,cntl_in) = 0;
		phase = (double *) cm_analog_get_ptr(INT1,0);
		*phase = 0;

  }else

  if(ANALYSIS == MIF_TRAN){

	phase = (double *) cm_analog_get_ptr(INT1,0);
	phase1 = (double *) cm_analog_get_ptr(INT1,1);

	loc = STATIC_VAR (locdata);
	x = loc->control;
	y = loc->freq;

    /* Retrieve x and y values. */
    for (i=0; i<cntl_size; i++) {
        *(x+i) = PARAM(cntl_array[i]);
        *(y+i) = PARAM(freq_array[i]);
    }



    /* Retrieve cntl_input value. */
    cntl_input = INPUT(cntl_in);


    /* Determine segment boundaries within which cntl_input resides */
				/*** cntl_input below lowest cntl_voltage ***/
    if (cntl_input <= *x) {
            dout_din = (*(y+1) - *y)/(*(x+1) - *x);
             freq = *y + (cntl_input - *x) * dout_din;

	     if(freq <= 0){
	        cm_message_send(sine_freq_clamp);
	        freq = 1e-16;
         }
          /* freq = *y; */
    }
    else
        /*** cntl_input above highest cntl_voltage ***/

        if (cntl_input >= *(x+cntl_size-1)){
            dout_din = (*(y+cntl_size-1) - *(y+cntl_size-2)) /
                          (*(x+cntl_size-1) - *(x+cntl_size-2));
            freq = *(y+cntl_size-1) + (cntl_input - *(x+cntl_size-1)) * dout_din;

        } else { /*** cntl_input within bounds of end midpoints...
                must determine position progressively & then
                calculate required output.                    ***/

            for (i=0; i<cntl_size-1; i++) {

                if ((cntl_input < *(x+i+1)) && (cntl_input >= *(x+i))) {

					/* Interpolate to the correct frequency value */

					freq = ((cntl_input - *(x+i))/(*(x+i+1) - *(x+i)))*
							(*(y+i+1)-*(y+i)) + *(y+i);
                }

            }
        }
/*   calculate the peak value of the wave, the center of the wave, the
	 instantaneous phase and the radian value of the phase */
		peak = (output_hi - output_low)/2;
		center = (output_hi + output_low)/2;
		*phase = *phase1 + freq*(TIME - T(1));
		radian = *phase * 2.0 * PI;

        OUTPUT(out) = peak*sin(radian) + center;
        PARTIAL(out,cntl_in) = 0;

    } else {                      /* Output AC Gain */

        ac_gain.real = 0.0;
        ac_gain.imag= 0.0;
        AC_GAIN(out,cntl_in) = ac_gain;
    }
}

