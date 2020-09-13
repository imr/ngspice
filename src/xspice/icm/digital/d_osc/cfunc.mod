/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE d_osc/cfunc.mod

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503-405
               

AUTHORS                      

    24 Jul 1991     Jeffrey P. Murray


MODIFICATIONS   

    23 Aug 1991    Jeffrey P. Murray
    30 Sep 1991    Jeffrey P. Murray
                                   

SUMMARY

    This file contains the model-specific routines used to
    functionally describe the d_osc code model.


INTERFACES       

    FILE                 ROUTINE CALLED     

    CMmacros.h           cm_message_send();                   
                             
    CM.c                 void *cm_analog_alloc()
                         void *cm_analog_get_ptr()

    CMevt.c              void cm_event_queue()


REFERENCED FILES

    Inputs from and outputs to ARGS structure.
                     

NON-STANDARD FEATURES

    NONE

===============================================================================*/

/*=== INCLUDE FILES ====================*/

#include "d_osc.h"    /*    ...contains macros & type defns.
                               for this model.  7/24/91 - JPM */

                                      

/*=== CONSTANTS ========================*/




/*=== MACROS ===========================*/



  
/*=== LOCAL VARIABLES & TYPEDEFS =======*/                         


    
           
/*=== FUNCTION PROTOTYPE DEFINITIONS ===*/




                   
/*==============================================================================

FUNCTION cm_d_osc()

AUTHORS                      

    24 Jul 1991     Jeffrey P. Murray

MODIFICATIONS   

    30 Sep 1991     Jeffrey P. Murray

SUMMARY

    This function implements the d_osc code model.

INTERFACES       

    FILE                 ROUTINE CALLED     

    CMmacros.h           cm_message_send();                   
                             
    CM.c                 void *cm_analog_alloc()
                         void *cm_analog_get_ptr()

    CMevt.c              void cm_event_queue()

RETURNED VALUE
    
    Returns inputs and outputs via ARGS structure.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== CM_D_OSC ROUTINE ===*/

/*************************************************************
*      The following is the model for the controlled digital *
*   oscillator for the ATESSE Version 2.0 system.            *
*                                                            *
*   Created 7/24/91                           J.P.Murray     *
*************************************************************/

/*************************************************************
*                                                            *
*                                                            *
*                          <-----duty_cycle----->            *
*  I                                                         *
*  I                      t2               t3                *
*  I                       \______________/_____             *
*  I                       |                    |            *
*  I                 |     |              |     |            *
*  I                       |                    |            *
*  I                 |     |              |     |            *
*  I                       |                    |            *
*  I                 |     |              |     |            *
*  I-----------------*-----* - - - - - - - - - -*---------   *   
*		   			t1				           t4            *
*                                                            *
*                                                            *
*                     t2 = t1 + rise_delay                   *
*                     t4 = t3 + fall_delay                   *
*                                                            *
*        Note that for the digital model, unlike for the     *
*    analog "square" model, t1 and t3 are stored and         *
*    adjusted values, but t2 & t4 are implied by the         *
*    rise and fall delays of the model, but are otherwise    *
*    not stored values.                     JPM              *
*                                                            *
*************************************************************/

#include <stdlib.h>

void cm_d_osc(ARGS) 
{

    double           *x,    /* analog input value control array */
                     *y,    /* frequency array  */
             cntl_input,    /* control input value  */
                 *phase,    /* instantaneous phase of the model  */
             *phase_old,    /* previous phase of the model   */
                    *t1,    /* pointer to t1 value  */   
                    *t3,    /* pointer to t3 value  */   
              /*time1,*/    /* variable for calculating new time1 value */   
              /*time3,*/    /* variable for calculating new time3 value */   
                   freq = 0.0,    /* instantaneous frequency value    */   
                 dphase,    /* fractional part into cycle */
             duty_cycle,    /* duty_cycle value */
            test_double,    /* testing variable */
                  slope;    /* slope value...used to extrapolate
                               freq values past endpoints.  */


    int               i,    /* generic loop counter index */
              cntl_size,    /* control array size         */
              freq_size;    /* frequency array size       */
         
                        




    /**** Retrieve frequently used parameters... ****/

    cntl_size = PARAM_SIZE(cntl_array);           
    freq_size = PARAM_SIZE(freq_array);           
    duty_cycle = PARAM(duty_cycle);


    /* check and make sure that the control array is the
	   same size as the frequency array */

	if(cntl_size != freq_size){
		cm_message_send(d_osc_array_error);
		return;
 	}

            
    if (INIT) {  /*** Test for INIT == TRUE. If so, allocate storage, etc. ***/


        /* Allocate storage for internal variables */
        cm_analog_alloc(0, sizeof(double));
        cm_analog_alloc(1, sizeof(double));
        cm_analog_alloc(2, sizeof(double));

        /* assign internal variables */
        phase = phase_old = (double *) cm_analog_get_ptr(0,0);

        t1 = (double *) cm_analog_get_ptr(1,0);
                          
        t3 = (double *) cm_analog_get_ptr(2,0);

    }

    else {    /*** This is not an initialization pass...retrieve storage
                   addresses and calculate new outputs, if required. ***/


        /** Retrieve previous values... **/


        /* assign internal variables */
        phase = (double *) cm_analog_get_ptr(0,0);
        phase_old = (double *) cm_analog_get_ptr(0,1);

        t1 = (double *) cm_analog_get_ptr(1,0);
                          
        t3 = (double *) cm_analog_get_ptr(2,0);

    }

                


    switch (CALL_TYPE) {

    case ANALOG:    /** analog call **/
                                                      
        test_double = TIME;                           

        if ( AC == ANALYSIS ) { /* this model does not function 
                                   in AC analysis mode.         */
            
            return; 

        }
        else {

            if ( 0.0 == TIME ) { /* DC analysis */

                /* retrieve & normalize phase value */
                *phase = PARAM(init_phase);
                if ( 0 > *phase ) {
                    *phase = *phase + 360.0;
                } 
                *phase = *phase / 360.0;


                /* set phase value to init_phase */
                *phase_old = *phase;
                            
                /* preset time values to harmless values... */
                *t1 = -1;
                *t3 = -1;
                

            }


            /* Allocate storage for breakpoint domain & freq. range values */
        
            x = (double *) calloc((size_t) cntl_size, sizeof(double));
            if (!x) {
                cm_message_send(d_osc_allocation_error); 
                return;
            }
        
            y = (double *) calloc((size_t) freq_size, sizeof(double));
            if (!y) {
                cm_message_send(d_osc_allocation_error);  
                if(x) free(x);
                return;
            }
        
            /* Retrieve x and y values. */       
            for (i=0; i<cntl_size; i++) {
                x[i] = PARAM(cntl_array[i]);
                y[i] = PARAM(freq_array[i]);
            }                       
                    
            /* Retrieve cntl_input value. */       
            cntl_input = INPUT(cntl_in);
        
            /* Determine segment boundaries within which cntl_input resides */
        				/*** cntl_input below lowest cntl_voltage ***/
            if (cntl_input <= x[0]) {

                slope = (y[1] - y[0])/(x[1] - x[0]);
                freq = y[0] + (cntl_input - x[0]) * slope;

            }
            else 
                /*** cntl_input above highest cntl_voltage ***/
        	
            if (cntl_input >= x[cntl_size-1]) { 

                slope = (y[cntl_size-1] - y[cntl_size-2]) /
                         (x[cntl_size-1] - x[cntl_size-2]);
                freq = y[cntl_size-1] + (cntl_input - x[cntl_size-1]) * slope;
        
            } 
            else { /*** cntl_input within bounds of end midpoints...   
                        must determine position progressively & then 
                        calculate required output.                ***/
                                           
                for (i=0; i<cntl_size-1; i++) {
        
                    if ( (cntl_input < x[i+1]) && (cntl_input >= x[i]) ) { 
                    		
                        /* Interpolate to the correct frequency value */
        
                        freq = ( (cntl_input - x[i]) / (x[i+1] - x[i]) ) * 
                               ( y[i+1]-y[i] ) + y[i]; 
                    } 
                }
            }
            
            /*** If freq < 0.0, clamp to 1e-16 & issue a warning ***/
            if ( 0.0 > freq ) {
                freq = 1.0e-16;
                cm_message_send(d_osc_negative_freq_error);  
            }


            /* calculate the instantaneous phase */
            *phase = *phase_old + freq * (TIME - T(1));
        
            /* dphase is the percent into the cycle for
               the period */	
            dphase = *phase_old - floor(*phase_old);


            /* Calculate the time variables and the output value
               for this iteration */ 	
            
            if((*t1 <= TIME) && (TIME <= *t3)) { /* output high */

                *t3 = T(1) + (1 - dphase)/freq;
            
                if(TIME < *t3) {
                    cm_event_queue(*t3);
                }
            
            			
            } 
            else 

            if((*t3 <= TIME) && (TIME <= *t1)) { /* output low */

                if(dphase > (1.0 - duty_cycle) ) {
            			dphase = dphase - 1.0;
                }
                *t1 = T(1) + ( (1.0 - duty_cycle) - dphase)/freq;
            
                if(TIME < *t1) {

                    cm_event_queue(*t1);

                }
           	}      
            else {

                if(dphase > (1.0 - duty_cycle) ) {
                    dphase = dphase - 1.0;
                }
                *t1 = T(1) + ( (1.0 - duty_cycle) - dphase )/freq;

                if((TIME < *t1) || (T(1) == 0)) {
                    cm_event_queue(*t1);
                }

                *t3 = T(1) + (1 - dphase)/freq;


            }


            
            if(x) free(x);
            if(y) free(y);


        }
        break;
                

    case EVENT:    /** discrete call...lots to do **/


        test_double = TIME;                           

        if ( 0.0 == TIME ) { /* DC analysis...preset values, 
                                as appropriate.... */

            /* retrieve & normalize phase value */
            *phase = PARAM(init_phase);
            if ( 0 > *phase ) {
                *phase = *phase + 360.0;
            } 
            *phase = *phase / 360.0;


            /* set phase value to init_phase */
            *phase_old = *phase;
                        
            /* preset time values to harmless values... */
            *t1 = -1;
            *t3 = -1;
        }



        /* Calculate the time variables and the output value
           for this iteration */ 	
                
        /* Output is always set to STRONG */
        OUTPUT_STRENGTH(out) = STRONG;        



        if( *t1 == TIME ) { /* rising edge */

            OUTPUT_STATE(out) = ONE;        
            OUTPUT_DELAY(out) = PARAM(rise_delay);        

        }
        else {

            if ( *t3 == TIME ) { /* falling edge */

                OUTPUT_STATE(out) = ZERO;        
                OUTPUT_DELAY(out) = PARAM(fall_delay);        
            }

            else { /* no change in output */

                if ( TIME != 0.0 ) {
                    OUTPUT_CHANGED(out) = FALSE;
                }

                if ( (*t1 < TIME) && (TIME < *t3) ) {
                    OUTPUT_STATE(out) = ONE;        
                }
                else {
                    OUTPUT_STATE(out) = ZERO;        
                }
            }
        }

        break;

    }
}             







