/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE hyst/cfunc.mod

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503-405
               

AUTHORS                      

    6 Jun 1991     Jeffrey P. Murray


MODIFICATIONS   

     2 Oct 1991    Jeffrey P. Murray
                                   

SUMMARY

    This file contains the model-specific routines used to
    functionally describe the hyst code model.


INTERFACES       

    FILE                 ROUTINE CALLED     

    CMutil.c             void cm_smooth_corner(); 
                             
    CM.c                 void *cm_analog_alloc()
                         void *cm_analog_get_ptr()


REFERENCED FILES

    Inputs from and outputs to ARGS structure.
                     

NON-STANDARD FEATURES

    NONE

===============================================================================*/

/*=== INCLUDE FILES ====================*/

#include "cm_hyst.h"			

                                      

/*=== CONSTANTS ========================*/




/*=== MACROS ===========================*/



  
/*=== LOCAL VARIABLES & TYPEDEFS =======*/                         


    
           
/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/
                   



/*==============================================================================

FUNCTION void hyst()

AUTHORS                      

     2 Oct 1991     Jeffrey P. Murray

MODIFICATIONS   

    NONE

SUMMARY

    This function implements the hyst code model.

INTERFACES       

    FILE                 ROUTINE CALLED     

    CMutil.c             void cm_smooth_corner(); 

    CM.c                 void *cm_analog_alloc()
                         void *cm_analog_get_ptr()

RETURNED VALUE
    
    Returns inputs and outputs via ARGS structure.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/


/*=== CM_HYST ROUTINE ===*/


/*************************************************************************
*                 BEHAVIOR OF HYSTERESIS:                                *
*        out               hyst         hyst                             *
*         ^             ____/\_____  ____/\_____                         *
*         |            /           \/           \                        *
*         |        x_fall_linear                  x_rise_zero            *
*  out_upper_limit- -  *----<-------------<------*------->               *
*         |           /|           /|           /|                       *
*         |          /            /in_high     /                         *
*         |         /  |         /  |         /  |                       *
*         |        /            /          __/                           *
*         |      |/_   |       /    |       /|   |                       *
*         |      /            /            /                             *
*         |     /      |     /      |     /      |                       *
*  <------O----/------------/------------/-----------------------> in    *
*         | | /          | /          | /                                *
*         |  /            /            /                                 *
*  <--------*------->----|---->-------* - - - - out_lower_limit          *
*       x_fall_zero    in_low     x_rise_linear                          *
*         V                                                              *
*                                                                        *
*  input_domain defines "in" increment below & above the "*" points      *
*     shown, within which smoothing of the d(out)/d(in) values           *
*     occurs...this prevents abrupt changes in d(out)/d(in) which        *
*     could prevent the simulator from reaching convergence during       *
*     a transient or DC analysis.                                        *
*                                                                        *
**************************************************************************/

/**************************************************************************/
/*  Usage of cm_smooth_corner:                                                 */
/*                                                                        */
/*  void cm_smooth_corner(double x_input, double x_center, double y_center,    */
/*                   double domain, double lower_slope,                   */
/*                   double upper_slope,double *y_output, double *dy_dx)  */
/*                                                                        */
/**************************************************************************/
                


void cm_hyst(ARGS)  /* structure holding parms, 
                                       inputs, outputs, etc.     */
{
    double        in, /* input to hysteresis block      */
                 out, /* output from hysteresis block   */
              in_low, /* lower input value for hyst=0 at which 
                         the transfer curve changes from constant 
                         to linear                               */
             in_high, /* upper input value for hyst=0 at which 
                         the transfer curve changes from constant 
                         to linear                               */
                hyst, /* the hysteresis value (see above diagram)    */
     out_lower_limit, /* the minimum output value from the block */
     out_upper_limit, /* the maximum output value from the block */
        input_domain, /* the delta value of the input above and
                         below in_low and in_high within which
                         smoothing will be applied to the output 
                         in order to maintain continuous first partial
                         derivatives.                                  */
               slope, /* calculated rise and fall slope for the block  */
            pout_pin, /* partial derivative of output w.r.t. input */
       x_rise_linear, /* = in_low + hyst                           */
         x_rise_zero, /* = in_high + hyst                          */
       x_fall_linear, /* = in_high - hyst                          */
         x_fall_zero; /* = in_low - hyst                           */

    Boolean_t     *hyst_state, /* TRUE => input is on lower leg of 
                                  hysteresis curve, between -infinity
                                  and in_high + hyst.
                                  FALSE => input is on upper leg
                                  of hysteresis curve, between 
                                  in_low - hyst and +infinity      */
              *old_hyst_state; /* previous value of *hyst_state    */

    Mif_Complex_t     ac_gain; /* AC gain */
                                                   




    /** Retrieve frequently used parameters... **/

    in_low = PARAM(in_low);
    in_high = PARAM(in_high);
    hyst = PARAM(hyst);
    out_lower_limit = PARAM(out_lower_limit);
    out_upper_limit = PARAM(out_upper_limit);                         
    input_domain = PARAM(input_domain);

                        


    /** Calculate Hysteresis Linear Region Slopes & Derived Values **/


    /* Define slope of rise and fall lines when not being smoothed */

    slope = (out_upper_limit - out_lower_limit)/(in_high - in_low);  

    x_rise_linear = in_low + hyst;    /* Breakpoint - x rising to 
                                             linear region */
    x_rise_zero = in_high + hyst;     /* Breakpoint - x rising to 
                                             zero-slope (out_upper_limit) */
    x_fall_linear = in_high - hyst;   /* Breakpoint - x falling to 
                                             linear region */
    x_fall_zero = in_low - hyst;      /* Breakpoint - x falling to 
                                             zero-slope (out_lower_limit) */
                                        
    if (PARAM(fraction) == MIF_TRUE)        /* Set range to absolute value */
        input_domain = input_domain * (in_high - in_low);

                                          


    /** Retrieve frequently used inputs... **/

    in = INPUT(in);



    /** Test for INIT; if so, allocate storage, otherwise, retrieve
                               previous timepoint value for output...     **/

    if (INIT==1) {  /* First pass...allocate storage for previous state.   */
                    /* Also, calculate roughly where the current output    */
                    /* will be and use this value to define current state. */

        cm_analog_alloc(TRUE,sizeof(Boolean_t));   

        hyst_state     = (Boolean_t *) cm_analog_get_ptr(TRUE,0);
        old_hyst_state = (Boolean_t *) cm_analog_get_ptr(TRUE,1);

        if (in < x_rise_zero + input_domain) { /* Set state to X_RISING */
            *old_hyst_state = X_RISING;   
        }
        else {                         
            *old_hyst_state = X_FALLING;  
        }
    }
    else {          /* Allocation not necessary...retrieve previous values */

        hyst_state = (Boolean_t *) cm_analog_get_ptr(TRUE,0);  /* Set out pointer to current 
                                                            time storage */    
        old_hyst_state = (Boolean_t *) cm_analog_get_ptr(TRUE,1);  /* Set old-output-state pointer 
                                                   to previous time storage */    
    }

                   

    /** Set *hyst_out = *old_hyst_out, unless changed below...
          we don't need the last iteration value of *hyst_state.  **/

    *hyst_state = *old_hyst_state;




    /*** Calculate value of hyst_state, pout_pin.... ***/

    if (*old_hyst_state == X_RISING) { /* Assume calculations on lower  */
                                       /* hysteresis section (x rising) */

        if ( in <= x_rise_linear - input_domain ) { /* Output @ lower limit */

            out = out_lower_limit;
            pout_pin = 0.0;
        }
        else {
            if ( in <= x_rise_linear + input_domain ) { /* lower smoothing region */
                cm_smooth_corner(in,x_rise_linear,out_lower_limit,input_domain,
                             0.0,slope,&out,&pout_pin);
            }
            else {
                if (in <= x_rise_zero - input_domain) { /* Rising linear region */ 
                    out = (in - x_rise_linear)*slope + out_lower_limit;
                    pout_pin = slope;
                }
                else {
                    if (in <= x_rise_zero + input_domain) { /* Upper smoothing region */ 
                        cm_smooth_corner(in,x_rise_zero,out_upper_limit,input_domain,
                                    slope,0.0,&out,&pout_pin);
                    }
                    else { /* input has transitioned to X_FALLING region... */
                        out = out_upper_limit;
                        pout_pin = 0.0;
                        *hyst_state = X_FALLING;
                    }
                }
            }
        }
    }
    else {    /* Assume calculations on upper hysteresis section (x falling) */

        if ( in >= x_fall_linear + input_domain ) { /* Output @ upper limit */

            out = out_upper_limit;
            pout_pin = 0.0;
        }
        else {
            if ( in >= x_fall_linear - input_domain ) { /* Upper smoothing region */
                cm_smooth_corner(in,x_fall_linear,out_upper_limit,input_domain,
                             slope,0.0,&out,&pout_pin);
            }
            else {
                if (in >= x_fall_zero + input_domain) { /* Falling linear region */ 
                    out = (in - x_fall_zero)*slope + out_lower_limit;
                    pout_pin = slope;
                }
                else {
                    if (in >= x_fall_zero - input_domain) { /* Lower smoothing region */ 
                        cm_smooth_corner(in,x_fall_zero,out_lower_limit,input_domain,
                                    0.0,slope,&out,&pout_pin);
                    }
                    else { /* input has transitioned to X_RISING region... */
                        out = out_lower_limit;
                        pout_pin = 0.0;
                        *hyst_state = X_RISING;
                    }
                }
            }
        }
    }



    if (ANALYSIS != MIF_AC) {     /* DC & Transient Analyses */

        OUTPUT(out) = out;          
        PARTIAL(out,in) = pout_pin; 

    }
    else {                        /* AC Analysis */
        ac_gain.real = pout_pin;
        ac_gain.imag= 0.0;
        AC_GAIN(out,in) = ac_gain;

    }
}





