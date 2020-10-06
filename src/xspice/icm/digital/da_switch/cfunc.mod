/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE da_switch/cfunc.mod

(C) 2019 Anamosic Ballenegger Design

Based on aswitch code model by Georgia Tech.

Copyright 1991
Georgia Tech Research Corporation, Atlanta, Ga. 30332
All Rights Reserved

PROJECT A-8503-405
           

===============================================================================*/

/*=== INCLUDE FILES ====================*/

#include <math.h>


/*============================================================================*/

void cm_da_switch(ARGS) 

{
    double  
      r, rinv, ron, roff;    /* ON and OFF resistances of the switch */

    Mif_Complex_t ac_gain;


    int           i,        /* generic loop counter index */
	           size;        /* number of input & output ports */
         
                        
    Digital_State_t   *in,       /* base address of array holding all input 
                                    values  */
                  *in_old;       /* array holding previous input values */





    /* determine "width" of the node bridge... */

    size = PORT_SIZE(in);               
    if(PORT_SIZE(out) != size) {
    	cm_message_send("Error: output port size different than input port size\n");
    }
                
    /** Read in remaining parameters **/
                              
    ron = PARAM(r_on);
    roff = PARAM(r_off);

    if (INIT) {  /*** Test for INIT == TRUE. If so, allocate storage, etc. ***/
 	 /* printf("Call cm_da_switch with size=%d, ron=%g, roff=%g\n", size, ron, roff); */

        /* Allocate storage for inputs */
        cm_event_alloc(0, size * (int) sizeof(Digital_State_t));

        /* assign discrete addresses */
        in = in_old = (Digital_State_t *) cm_event_get_ptr(0,0);


        /* read current input values */
        for (i=0; i<size; i++) {
            in[i] = INPUT_STATE(in[i]);
        }
                               


        /* Output initial analog levels based on input values */

        for (i=0; i<size; i++) { /* assign addresses */
	    r = roff;
            switch (in[i]) {

                case ZERO:
			r = roff;
                        break;

                case UNKNOWN:
			r = roff;
                        break;

                case ONE:
			r = ron;
                        break;

            }
	    rinv = 1/r;
	    OUTPUT(out[i]) = INPUT(out[i])*rinv;
            PARTIAL(out[i],out[i]) = rinv;

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
        for (i=0; i<size; i++) {            
	  r = roff;
	  switch (in[i]) {
                case ONE: 
                    r = ron;
                    break;

                case ZERO:
                    r = roff;
                    break;

                case UNKNOWN:
                    r = roff;
                    break;
                }
		
	  rinv = 1/r;
	  
	  /* to do: implement maximum current limiter (with tanh ?) */
	  if(ANALYSIS != MIF_AC) {      /* Output DC & Transient Values  */
        	OUTPUT(out[i]) = INPUT(out[i])*rinv;     
        	PARTIAL(out[i],out[i]) = rinv; 
    	  } else {                      /*   Output AC Gain Values      */
        	ac_gain.real = -rinv;
        	ac_gain.imag= 0.0;
        	AC_GAIN(out[i],out[i]) = ac_gain;

    	  }
    	}
    

        break;

    }

}





