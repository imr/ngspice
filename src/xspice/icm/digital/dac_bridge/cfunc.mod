/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE dac_bridge/cfunc.mod

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503-405
               

AUTHORS                      

    3 Jun 1991     Jeffrey P. Murray


MODIFICATIONS   

    16 Aug 1991    Jeffrey P. Murray
     2 Oct 1991    Jeffrey P. Murray
                                   

SUMMARY

    This file contains the model-specific routines used to
    functionally describe the dac_bridge code model.


INTERFACES       

    FILE                 ROUTINE CALLED     

    CM.c                 void *cm_analog_alloc()
                         void *cm_analog_get_ptr()
                         int cm_analog_set_perm_bkpt()

    CMevt.c              void *cm_event_alloc()
                         void *cm_event_get_ptr()


REFERENCED FILES

    Inputs from and outputs to ARGS structure.
                     

NON-STANDARD FEATURES

    NONE

===============================================================================*/

/*=== INCLUDE FILES ====================*/

#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <string.h>

                                      

/*=== CONSTANTS ========================*/




/*=== MACROS ===========================*/



  
/*=== LOCAL VARIABLES & TYPEDEFS =======*/                         


    
           
/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/




                   
/*==============================================================================

FUNCTION cm_dac_bridge()

AUTHORS                      

    3 Jun 1991     Jeffrey P. Murray

MODIFICATIONS   

    16 Aug 1991    Jeffrey P. Murray
     2 Oct 1991    Jeffrey P. Murray

SUMMARY

    This function implements the dac_bridge code model.

INTERFACES       

    FILE                 ROUTINE CALLED     

    CM.c                 void *cm_analog_alloc()
                         void *cm_analog_get_ptr()
                         int cm_analog_set_perm_bkpt()

    CMevt.c              void *cm_event_alloc()
                         void *cm_event_get_ptr()

RETURNED VALUE
    
    Returns inputs and outputs via ARGS structure.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/


/*=== CM_DAC_BRIDGE ROUTINE ===*/

/************************************************
*      The following is the model for the       *
*   digital-to-analog nodebridge for the        *
*   ATESSE Version 2.0 system.                  *
*                                               *
*   Created 6/3/91                J.P.Murray    *
************************************************/


void cm_dac_bridge(ARGS) 

{
    double  out_low,        /* analog output value corresponding to '0' 
                               digital input 	*/
           out_high,        /* analog output value corresponding to '1' 
                               digital input 	*/
          out_undef,        /* analog output value corresponding to 'U' 
                               digital input 	*/
             t_rise,        /* rise time...used to produce d(out)/d(time) 
                               values for gradual change in analog output.	*/ 
             t_fall,        /* fall time...used to produce d(out)/d(time) 
                               values for gradual change in analog output.	*/ 
               *out,        /* array holding all output values  */
           *out_old,        /* array holding previous output values */            
           fraction,        /* fraction of total rise or fall time to add to
                               current time value for breakpoint calculation */
          level_inc,        /* incremental level value out_high - out_low */
         rise_slope,        /* level_inc divided by t_rise */
         fall_slope,        /* level_inc divided by t_fall */
           time_inc,        /* time increment since last analog call */
               test,        /* testing variable */
        *breakpoint;        /* holding variable to prevent infinite
                               posting of the same breakpoint */



    int           i,        /* generic loop counter index */
	           size;        /* number of input & output ports */
         
                        

    Digital_State_t   *in,       /* base address of array holding all input 
                                    values  */
                  *in_old;       /* array holding previous input values */





    /* determine "width" of the node bridge... */

    size = PORT_SIZE(in);               

            
    /** Read in remaining model parameters **/
                              
    out_low = PARAM(out_low);
    out_high = PARAM(out_high);
    t_rise = PARAM(t_rise);
    t_fall = PARAM(t_fall);


    /* Test to see if out_low and out_high were specified, but */
    /* out_undef was not...                                    */
    /* if so, take out_undef as mean of out_high and out_low.  */

    if (!PARAM_NULL(out_low) && !PARAM_NULL(out_high) && 
         PARAM_NULL(out_undef) ) {
       out_undef = out_low + (out_high - out_low) / 2.0;
    }
    else {
       out_undef = PARAM(out_undef);
    }                                 



    if (INIT) {  /*** Test for INIT == TRUE. If so, allocate storage, etc. ***/


        /* Allocate storage for inputs */
        cm_event_alloc(0, size * (int) sizeof(Digital_State_t));

                      
        /* Allocate storage for outputs */

        /* retrieve previously-allocated discrete input and */
        /* allocate storage for analog output values.       */
                                    
        /* allocate output space and obtain adresses */
        cm_analog_alloc(0, size * (int) sizeof(double));
        cm_analog_alloc(1, sizeof(double));
        
        /* assign discrete addresses */
        in = in_old = (Digital_State_t *) cm_event_get_ptr(0,0);

        /* assign analog addresses */
        out = out_old = (double *) cm_analog_get_ptr(0,0);
        breakpoint = (double *) cm_analog_get_ptr(1,0);


        /* read current input values */
        for (i=0; i<size; i++) {
            in[i] = INPUT_STATE(in[i]);
        }
                               


        /* Output initial analog levels based on input values */

        for (i=0; i<size; i++) { /* assign addresses */

            switch (in[i]) {

                case ZERO: out[i] = out_old[i] = out_low;
                        OUTPUT(out[i]) = out_old[i];
                        break;

                case UNKNOWN: out[i] = out_old[i] = out_undef;
                        OUTPUT(out[i]) = out_old[i];
                        break;

                case ONE: out[i] = out_old[i] = out_high;
                        OUTPUT(out[i]) = out_old[i];
                        break;

            }

            LOAD(in[i]) = PARAM(input_load);

        }
    }

    else {    /*** This is not an initialization pass...read in parameters,
                   retrieve storage addresses and calculate new outputs,
                   if required. ***/



        /** Retrieve previous values... **/


        /* assign discrete addresses */
        in = (Digital_State_t *) cm_event_get_ptr(0,0);
        in_old= (Digital_State_t *) cm_event_get_ptr(0,1);



        /* assign analog addresses */
        out = (double *) cm_analog_get_ptr(0,0);
        out_old = (double *) cm_analog_get_ptr(0,1);
        breakpoint = (double *) cm_analog_get_ptr(1,0);


        /* read current input values */
        for (i=0; i<size; i++) {
            in[i] = INPUT_STATE(in[i]);
        }
    }
    

    switch (CALL_TYPE) {

    case EVENT:  /** discrete call... **/
                  
        /* Test to see if any change has occurred in an input */
        /* since the last digital call...                     */ 

        for (i=0; i<size; i++) {
			
            if (in[i] != in_old[i]) { /* if there has been a change... */

                /* post current time as a breakpoint */

                cm_analog_set_perm_bkpt(TIME);

            }
        }

        break;

    case ANALOG:    /** analog call... **/
                  
        level_inc = out_high - out_low;
        rise_slope = level_inc / t_rise;
        fall_slope = level_inc / t_fall;
                                  

        time_inc = TIME - T(1);

        for (i=0; i<size; i++) {  


            if ( 0.0 == TIME ) {  /*** DC analysis ***/
                                                      
                switch (in[i]) {
             
                case ONE: 
                    out[i] = out_high;
                    break;

                case ZERO:
                    out[i] = out_low;
                    break;

                case UNKNOWN:
                    out[i] = out_undef;
                    break;

                }
            }

            else          /*** Transient Analysis ***/

            if ( in_old[i] == in[i] ) {    /* There has been no change in 
                                              this digital input since the 
                                              last analog call...           */

                switch (in[i]) {
                case ZERO: 
                    if (out_old[i] > out_low) { /* output still dropping */

                        out[i] = out_old[i] - fall_slope*time_inc;
                        if ( out_low > out[i]) out[i]=out_low; 

                    }
                    else { /* output at out_low */              

                        out[i] = out_low;

                    }
                    break;

                case ONE:
                    if (out_old[i] < out_high) { /* output still rising */

                        out[i] = out_old[i] + rise_slope*time_inc;
                        if ( out_high < out[i]) out[i]=out_high; 

                    }
                    else { /* output at out_high */              

                        out[i] = out_high;

                    }
                    break;


                case UNKNOWN:
                    if (out_old[i] < out_undef) {     /* output still rising */

                        out[i] = out_old[i] + (rise_slope * time_inc);
                        if ( out_undef < out[i]) out[i]=out_undef; 

                    }
                    else { 

                        if (out_old[i] > out_undef) { /* output still falling */              
    
                            out[i] = out_old[i] - fall_slope*time_inc;
                            if ( out_undef > out[i]) out[i]=out_undef; 
                        }
                        else {                        /* output at out_undef */
                                                      
                            out[i] = out_undef;
                        }

                    }

                    break;

                }
            }
            else {     /* There HAS been a change in this digital input
                          since the last analog access...need to use the
                          old value of input to complete the breakpoint
                          slope before changing directions... */


                switch (in_old[i]) {

                case ZERO: 
                    if (out_old[i] > out_low) { /* output still dropping */

                        out[i] = out_old[i] - fall_slope*time_inc;
                        if ( out_low > out[i]) out[i]=out_low; 

                    }
                    else { /* output at out_low */              

                        out[i] = out_low;

                    }
                    break;

                case ONE:
                    if (out_old[i] < out_high) { /* output still rising */

                        out[i] = out_old[i] + rise_slope*time_inc;
                        if ( out_high < out[i]) out[i]=out_high; 

                    }
                    else { /* output at out_high */              

                        out[i] = out_high;

                    }
                    break;


                case UNKNOWN:
                    if (out_old[i] < out_undef) {     /* output still rising */

                        out[i] = out_old[i] + (rise_slope * time_inc);
                        if ( out_undef < out[i]) out[i]=out_undef; 

                    }
                    else { 

                        if (out_old[i] > out_undef) { /* output still falling */              
    
                            out[i] = out_old[i] - fall_slope*time_inc;
                            if ( out_undef > out[i]) out[i]=out_undef; 
                        }
                        else {                        /* output at out_undef */
                                                      
                            out[i] = out_undef;
                        }

                    }

                    break;

                }                                             



                /* determine required new breakpoint for the end of
                   the output analog transition & post              */

                switch (in[i]) {     

                case ONE:        /* rising for all outputs */
                        fraction = (out_high - out[i]) / (out_high - out_low);
                        test = TIME + (fraction * t_rise);
                        cm_analog_set_perm_bkpt(test);
                        break;
                
                case UNKNOWN:    /* may be rising or falling */

                        if ( out_undef > out[i] ) { /* rising to U */
                            fraction = (out_undef - out[i]) / (out_high - out_low);
                            test = TIME + (fraction * t_rise);
                            cm_analog_set_perm_bkpt(test);
                        }
                        else {                       /* falling to U */
                            fraction = (out[i] - out_undef) / (out_high - out_low);
                            test = TIME + (fraction * t_fall);
                            cm_analog_set_perm_bkpt(test);
                        }
                        break;

                case ZERO:    /* falling for all outputs */
                        fraction = (out[i] - out_low) / (out_high - out_low);
                        test = TIME + (fraction * t_fall);
                        cm_analog_set_perm_bkpt(test);
                        break;
                }
            }
        }

        /* Output values... */
               
        for (i=0; i<size; i++) {

            OUTPUT(out[i]) = out[i];

        }

        break;

    }

}



