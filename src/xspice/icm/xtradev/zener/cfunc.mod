/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE zener/cfunc.mod

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503-405
               

AUTHORS                      

     2 May 1991     Jeffrey P. Murray


MODIFICATIONS   

    18 Sep 1991    Jeffrey P. Murray
     2 Oct 1991    Jeffrey P. Murray
                                   

SUMMARY

    This file contains the model-specific routines used to
    functionally describe the zener code model.


INTERFACES       

    FILE                 ROUTINE CALLED     

    CM.c                 void cm_analog_not_converged()


REFERENCED FILES

    Inputs from and outputs to ARGS structure.
                     

NON-STANDARD FEATURES

    NONE

===============================================================================*/

/*=== INCLUDE FILES ====================*/

#include <math.h>
#include <stdlib.h>

                                      

/*=== CONSTANTS ========================*/




/*=== MACROS ===========================*/



  
/*=== LOCAL VARIABLES & TYPEDEFS =======*/                         


    
           
/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/



static void
cm_zener_callback(ARGS, Mif_Callback_Reason_t reason)
{
    switch (reason) {
        case MIF_CB_DESTROY: {
            double *loc = STATIC_VAR(previous_voltage);
	    if (loc) {
                free(loc);
                STATIC_VAR(previous_voltage) = NULL;
            }
            break;
        }
    }
}

                   
/*==============================================================================

FUNCTION void cm_zener()

AUTHORS                      

     2 May 1991     Jeffrey P. Murray

MODIFICATIONS   

    18 Sep 1991    Jeffrey P. Murray
     2 Oct 1991    Jeffrey P. Murray

SUMMARY

    This function implements the zener code model.

INTERFACES       

    FILE                 ROUTINE CALLED     

    CM.c                 void cm_analog_not_converged()


RETURNED VALUE
    
    Returns inputs and outputs via ARGS structure.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

#include <stdlib.h>

/*=== CM_ZENER ROUTINE ===*/


void cm_zener(ARGS)  /* structure holding parms, 
                                          inputs, outputs, etc.     */
{
    double v_breakdown, /* breakdown voltage parameter  */
           i_breakdown, /* breakdown current parameter  */
           r_breakdown, /* breakdown resistance parameter   */
                 i_rev, /* reverse current parameter    */
                 i_sat, /* saturation current parameter...
                           a.k.a. Io in the forward diode
                           characteristic equation...see below. */
             n_forward, /* forward emission coefficient parameter...
                           a.k.a. "n" in the forward diode
                           characteristic equation...see below. */
                    vt, /* volt-equivalent of temperature, Vt,
                           used in conjunction with n = n_forward 
                           value to describe the forward-voltage
                           diode behavior described as:
 
                           I = Io * (e^(V/n*Vt) - 1.0)  */ 

                 v_1_2, /* Boundary value voltage between region
                           1 (forward diode characteristic) and 
                           region 2 (linear region) */
                     k, /* intermediate value used to find v_2_3 */ 
                 v_2_3, /* Boundary value voltage between region
                           2 (linear region) and region 3 
                           (reverse breakdown region) */
                slope1, /* Slope of endpoint for a two segment model */ 
                slope2, /* Slope of endpoint for a two segment model */
                  temp, /* temporary variable used to calulate the
						   derivatives */
               v_zener, /* input voltage across zener   */
               i_zener, /* current which is allowed to   
                           flow through zener, for a given voltage  */
                    i0,
                    v0,
                     a, /* coefficient used to calculate "c" */
                     b, /* coefficient used to calculate "c" */
                     c, /* A constant to match ordinates at 
                           region 2/3 boundary */
                 deriv, /* partial derivative of the output
                           current w.r.t. the input voltage */
                  diff, /* difference between slope1 and slope2 */
               ord_1_2, /* Compute ordinate at boundary of regions 1 & 2 */
     *previous_voltage, /* Previous voltage value (used for limiting) */
             increment, /* Increment value calculated from the
                           previous_input for v_zener input limiting */
                     g; /* conductance value equal to 
                           i_rev / v_breakdown. This value is
                           used to simulate a reverse-leakage
                           conductance in parallel with the
                           zener characteristic.    */

    Mif_Complex_t ac_gain;  /* AC gain  */
                   

    if (INIT==1) {  /* First pass...allocate storage for previous value... */

        /* Allocate storage for frequencies */
        STATIC_VAR(previous_voltage) = (double *) malloc(sizeof(double));
        previous_voltage = (double *) STATIC_VAR(previous_voltage);

        /* Set previous_voltage value to zero... */
        *previous_voltage = 0.0;

        CALLBACK = cm_zener_callback;
    }  
    else {

        previous_voltage = (double *) STATIC_VAR(previous_voltage);

    }
                       

    /* Retrieve frequently used parameters & inputs... */

    v_breakdown = PARAM(v_breakdown);
    i_breakdown = PARAM(i_breakdown);
    r_breakdown = PARAM(r_breakdown);
    i_rev = PARAM(i_rev);
    i_sat = PARAM(i_sat);
    n_forward = PARAM(n_forward);
                              
    v_zener = INPUT(z);
      



    /** If the limit_switch parameter is set, test the   **/                      
    /** current input against previous value for limiting **/

    if ( MIF_TRUE == PARAM(limit_switch) ) {
        /* Check magnitude of v_zener */
        if ( fabs(*previous_voltage) >= 1.0 ) {
            increment = 0.1 * *previous_voltage;
        }
        else {
            if (v_zener < 0.0) {
                increment = -0.1;
            }
            else {
                increment = 0.1;
            }
        }
                                        
        /* Test v_zener for reasonable change in value since last call.. */
        if ( fabs(v_zener) > ( fabs(*previous_voltage + increment) ) ) {
    
            /* Apply limiting... */
            *previous_voltage = v_zener = *previous_voltage + increment;
    
            cm_analog_not_converged();
            
        }
        else {
    
            *previous_voltage = v_zener;
    
        }
    }




  
    /* Compute voltage at boundary of regions 1 & 2 */

    vt = 0.026;
    v_1_2 = n_forward * vt * log(n_forward * vt / 10.0);


    /* Compute voltage at boundary of regions 2 & 3 */

    k = 1.0 / i_breakdown / r_breakdown;
    v_2_3 = -v_breakdown + log(10.0/i_sat/r_breakdown)/k;



    /* Compare v_1_2 and v_2_3 to determine if a 3 segment model is possible */ 

    if (v_2_3 < v_1_2) {           /* Use a 3 segment model */

        /* Compute v0 for region 3... */
    
        i0 = 1.e-6;
        v0 = -v_breakdown + 1.0/k*log(i_breakdown/i0);


        /* Compute ordinate at boundary of regions 1 & 2 */

        ord_1_2 = i_sat * (exp(v_1_2/n_forward/vt) - 1.0);


        /* Compute a & b coefficients for linear section in region 2 */

        a = i_sat / 10.0;
        b = ord_1_2 - a * v_1_2;


        /* Compute constant to match ordinates at region 2/3 boundary */

        c = a*v_2_3 + b + i0*exp(-k * (v_2_3 - v0));


        /* Compute zener current */

        if (v_zener >= v_1_2) {
            temp = exp(v_zener / n_forward / vt);
            i_zener = i_sat * (temp - 1.0);
            deriv = i_sat / n_forward / vt * temp;
        }
        else {
            if (v_zener >= v_2_3) {
                i_zener = a * v_zener + b;
                deriv = i_sat / 10.0;
            }
            else {
                temp = exp(-k * (v_zener - v0));
                i_zener = -i0 * temp + c;
                deriv = k * i0 * temp;
            }
        }
    }
    else {                        /* Must use a 2 segment model */

        /* Determine i0 for reverse region */

        i0 = i_breakdown / (exp(k * v_breakdown) - 1.0);


        /* Determine the slopes at the region endpoints */

        slope1 = i_sat / n_forward / vt;
        slope2 = i0 * k;              

        
        /* Determine zener current & first partial...           */
        /* Use a linear conductance in one region to match      */
        /* slopes at the boundary.                              */
                                                                  
        if (v_zener >= 0.0) {
            temp = exp(v_zener / n_forward / vt);
            i_zener = i_sat * (temp - 1.0);
            deriv = i_sat / n_forward / vt * temp;
            diff = slope2 - slope1;
            if (diff > 0.0) {
                i_zener = i_zener + diff * v_zener;
                deriv = deriv + diff;
            }
        }
        else {
            temp = exp(-k * v_zener);
            i_zener = -i0 * (temp - 1.0);
            deriv = k * i0 * temp;
            diff = slope1 - slope2;
            if (diff > 0.0) {
                i_zener = i_zener + diff * v_zener;
                deriv = deriv + diff;
            }
        }
    }

    /* Add resistor in parallel to simulate reverse leakage */

    g = i_rev / v_breakdown;
    i_zener = i_zener + g * v_zener;
    deriv = deriv + g;


    if(ANALYSIS != MIF_AC) {        /* Output DC & Transient Values */
        OUTPUT(z) = i_zener;
        PARTIAL(z,z) = deriv;
    }
    else {                      /* Output AC Gain */
        ac_gain.real = deriv;
        ac_gain.imag= 0.0;
        AC_GAIN(z,z) = ac_gain;
    }
} 

