/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE aswitch/cfunc.mod

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503-405
               

AUTHORS                      

    6 June 1991     Jeffrey P. Murray


MODIFICATIONS   

    26 Sept 1991    Jeffrey P. Murray
                                   

SUMMARY

    This file contains the functional description of the aswitch
    code model.


INTERFACES       

    FILE                 ROUTINE CALLED     

    CMmacros.h           cm_message_send();                   


REFERENCED FILES

    Inputs from and outputs to ARGS structure.
                     

NON-STANDARD FEATURES

    NONE

===============================================================================*/

/*=== INCLUDE FILES ====================*/

#include <math.h>

                                      

/*=== CONSTANTS ========================*/




/*=== MACROS ===========================*/



  
/*=== LOCAL VARIABLES & TYPEDEFS =======*/                         


    
           
/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/






                   
/*==============================================================================

FUNCTION cm_aswitch()

AUTHORS                      

    6 June 1991     Jeffrey P. Murray

MODIFICATIONS   

    26 Sept 1991    Jeffrey P. Murray

SUMMARY

    This function implements the aswitch code model.

INTERFACES       

    FILE                 ROUTINE CALLED     

    CMmacros.h           cm_message_send();                   

RETURNED VALUE
    
    Returns inputs and outputs via ARGS structure.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== CM_ASWITCH ROUTINE ===*/



void cm_aswitch(ARGS)  /* structure holding parms, 
                                          inputs, outputs, etc.     */
{
    double cntl_on;      /* voltage above which switch come on */
	double cntl_off;     /* voltage below the switch has resistance roff */ 
	double r_on;         /* on resistance */
	double r_off;        /* off resistance */
	double intermediate; /* intermediate value used to calculate
				the resistance of the switch when the
				controlling voltage is between cntl_on
				and cntl_of */
	double r;            /* value of the resistance of the switch */
    double pi_pvout;     /* partial of the output wrt input       */
	double pi_pcntl;     /* partial of the output wrt control input */

   Mif_Complex_t ac_gain;
                   
	  char *cntl_error = "\n*****ERROR*****\nASWITCH: CONTROL voltage delta less than 1.0e-12\n";
                       

    /* Retrieve frequently used parameters... */

    cntl_on = PARAM(cntl_on);
    cntl_off = PARAM(cntl_off);
    r_on = PARAM(r_on);
    r_off = PARAM(r_off);

    if( r_on < 1.0e-3 ) r_on = 1.0e-3;  /* Set minimum 'ON' resistance */  

    if( (fabs(cntl_on - cntl_off) < 1.0e-12) ) {
        cm_message_send(cntl_error);          
        return;
    }

    if ( PARAM(log) == MIF_TRUE ) {   /* Logarithmic Variation in 'R' */
        intermediate = log(r_off / r_on) / (cntl_on - cntl_off);
        r = r_on * exp(intermediate * (cntl_on - INPUT(cntl_in)));

        if (PARAM(limit) == MIF_TRUE) {
            if(r<r_on) r=r_on;/* minimum resistance limiter */
            if(r>r_off) r=r_off;/* maximum resistance limiter */
        }
        else {
            if(r<=1.0e-9) r=1.0e-9;/* minimum resistance limiter */
        }
        pi_pvout = 1.0 / r;
        pi_pcntl = intermediate * INPUT(out) / r;
    }
    else {                      /* Linear Variation in 'R' */
        intermediate = (r_on - r_off) / (cntl_on - cntl_off);
        r = INPUT(cntl_in) * intermediate + ((r_off*cntl_on - 
                r_on*cntl_off) / (cntl_on - cntl_off));

        if (PARAM(limit) == MIF_TRUE) {
            if(r<r_on) r=r_on;/* minimum resistance limiter */
            if(r>r_off) r=r_off;/* maximum resistance limiter */
        }
        else {
            if(r<=1.0e-9) r=1.0e-9;/* minimum resistance limiter */
        }
        pi_pvout = 1.0 / r;
        pi_pcntl = -intermediate * INPUT(out) / (r*r);
    }                                 

    if(ANALYSIS != MIF_AC) {            /* Output DC & Transient Values  */
        OUTPUT(out) = INPUT(out) / r;      /* Note that the minus   */
        PARTIAL(out,out) = pi_pvout;       /* Signs are required    */
        PARTIAL(out,cntl_in) = pi_pcntl;   /* because current is    */
                                            /* positive flowing INTO */
                                            /* rather than OUT OF a  */
                                            /* component node.       */
    }
    else {                       /*   Output AC Gain Values      */
        ac_gain.real = -pi_pvout;           /* See comment on minus   */
        ac_gain.imag= 0.0;                  /* signs above....        */
        AC_GAIN(out,out) = ac_gain;

        ac_gain.real = -pi_pcntl;
        ac_gain.imag= 0.0;
        AC_GAIN(out,cntl_in) = ac_gain;
    }
} 

