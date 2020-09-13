/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE climit/cfunc.mod

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503-405
               

AUTHORS                      

    6 June 1991     Jeffrey P. Murray


MODIFICATIONS   

    26 Sept 1991    Jeffrey P. Murray
                                   

SUMMARY

    This file contains the functional description of the adc_bridge
    code model.


INTERFACES       

    FILE                 ROUTINE CALLED     

    CMutil.c             void cm_smooth_corner(); 
                         void cm_smooth_discontinuity();
                         
    CMmacros.h           cm_message_send();                   
                         

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

FUNCTION cm_climit()

AUTHORS                      

    6 June 1991     Jeffrey P. Murray

MODIFICATIONS   

    26 Sept 1991    Jeffrey P. Murray

SUMMARY

    This function implements the climit code model.

INTERFACES       

    FILE                 ROUTINE CALLED     

    CMutil.c             void cm_smooth_corner(); 
                         void cm_smooth_discontinuity();
                         
    CMmacros.h           cm_message_send();                   

RETURNED VALUE
    
    Returns inputs and outputs via ARGS structure.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== CM_CLIMIT ROUTINE ===*/

void cm_climit(ARGS)  /* structure holding parms, 
                                       inputs, outputs, etc.     */
{
    /* Define error message string constants */

    char *climit_range_error = "\n**** ERROR ****\n* CLIMIT function linear range less than zero. *\n";


    double      lower_delta,    /* lower delta value parameter  */
                upper_delta,    /* upper delta value parameter  */
                limit_range,    /* range of output below 
                                   (out_upper_limit - upper_delta) 
                                   or above (out_lower_limit + lower_delta)
                                   within which smoothing will be applied   */
                       gain,    /* gain parameter   */
            threshold_upper,    /* = out_upper_limit - upper_delta  */
            threshold_lower,    /* = out_lower_limit + lower_delta  */
               linear_range,    /* = threshold_upper - threshold_lower  */
            out_lower_limit,    /* output lower limit parameter */
            out_upper_limit,    /* output upper limit parameter */
                        out,    /* originally-calculated output value   */
                limited_out,    /* output value after limiting  */
                   pout_pin,    /* partial derivative of output w.r.t.input */
           pout_pcntl_upper,    /* partial derivative of output w.r.t. 
                                      cntl_upper input  */
           pout_pcntl_lower,    /* partial derivative of output w.r.t. 
                                      cntl_lower input  */
                       junk;    /* dummy variable   */
    
    Mif_Complex_t   ac_gain;    /* AC gain  */



    /* Retrieve frequently used parameters... */

    lower_delta = PARAM(lower_delta);
    upper_delta = PARAM(upper_delta);
    limit_range = PARAM(limit_range);
    gain = PARAM(gain);


    /* Find Upper & Lower Limits */

    out_lower_limit = INPUT(cntl_lower) + lower_delta;
    out_upper_limit = INPUT(cntl_upper) - upper_delta;


    if (PARAM(fraction) == MIF_TRUE)     /* Set range to absolute value */
        limit_range = limit_range * 
              (out_upper_limit - out_lower_limit);



    threshold_upper = out_upper_limit -   /* Set Upper Threshold */
                         limit_range;
    threshold_lower = out_lower_limit +   /* Set Lower Threshold */
                         limit_range;
    linear_range = threshold_upper - threshold_lower;

    
    /* Test the linear region & make sure there IS one... */
    if (linear_range < 0.0) {
        /* This INIT test keeps the models from outputting
        an error message the first time through when all
        the inputs are initialized to zero */
        if( (INIT != 1) && (0.0 != TIME) ){
            cm_message_send(climit_range_error);                   
        }
        limited_out = 0.0;
        pout_pin = 0.0;  
        pout_pcntl_lower = 0.0;
        pout_pcntl_upper = 0.0;
        return;
    } 

    /* Compute Un-Limited Output */
    out = gain * (PARAM(in_offset) + INPUT(in)); 


    if (out < threshold_lower) {       /* Limit Out @ Lower Bound */

        pout_pcntl_upper= 0.0;

        if (out > (out_lower_limit - limit_range)) {  /* Parabolic */
            cm_smooth_corner(out,out_lower_limit,out_lower_limit,
                        limit_range,0.0,1.0,&limited_out,
                        &pout_pin);         
            pout_pin = gain * pout_pin;
            cm_smooth_discontinuity(out,out_lower_limit,1.0,threshold_lower,
                           0.0,&pout_pcntl_lower,&junk);                 
        }
        else {                             /* Hard-Limited Region */
            limited_out = out_lower_limit;
            pout_pin = 0.0;
            pout_pcntl_lower = 1.0;
        }    
    }
    else {
        if (out > threshold_upper) {       /* Limit Out @ Upper Bound */

            pout_pcntl_lower= 0.0;

            if (out < (out_upper_limit+limit_range)) {  /* Parabolic */
                cm_smooth_corner(out,out_upper_limit,out_upper_limit,
                            limit_range,1.0,0.0,&limited_out,
                            &pout_pin);
                pout_pin = gain * pout_pin;
                cm_smooth_discontinuity(out,threshold_upper,0.0,out_upper_limit,
                               1.0,&pout_pcntl_upper,&junk);              
            }
            else {                             /* Hard-Limited Region */
                limited_out = out_upper_limit;
                pout_pin = 0.0;
                pout_pcntl_upper = 1.0;
            }
        }
        else {               /* No Limiting Needed */
            limited_out = out;
            pout_pin = gain;
            pout_pcntl_lower = 0.0;
            pout_pcntl_upper = 0.0;
        }
    }

    if (ANALYSIS != MIF_AC) {     /* DC & Transient Analyses */

        OUTPUT(out) = limited_out;
        PARTIAL(out,in) = pout_pin;
        PARTIAL(out,cntl_lower) = pout_pcntl_lower;
        PARTIAL(out,cntl_upper) = pout_pcntl_upper;

    }
    else {                        /* AC Analysis */
        ac_gain.real = pout_pin;
        ac_gain.imag= 0.0;
        AC_GAIN(out,in) = ac_gain;

        ac_gain.real = pout_pcntl_lower;
        ac_gain.imag= 0.0;
        AC_GAIN(out,cntl_lower) = ac_gain;

        ac_gain.real = pout_pcntl_upper;
        ac_gain.imag= 0.0;
        AC_GAIN(out,cntl_upper) = ac_gain;

    }
}
