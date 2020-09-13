/*.......1.........2.........3.........4.........5.........6.........7.........8
================================================================================

FILE s_xfer/cfunc.mod

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503-405
               

AUTHORS                      

    17 Mar 1991     Jeffrey P. Murray


MODIFICATIONS   

    18 Apr 1991     Harry Li
    27 Sept 1991    Jeffrey P. Murray
                                   

SUMMARY

    This file contains the functional description of the s-domain
    transfer function (s_xfer) code model.


INTERFACES       

    FILE                 ROUTINE CALLED     

    CMmacros.h           cm_message_send();                   

    CM.c                 void *cm_analog_alloc()
                         void *cm_analog_get_ptr()
                         int  cm_analog_integrate()




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


         


/*=============================================================================

FUNCTION cm_complex_div

AUTHORS                      

    27 Sept 1991     Jeffrey P. Murray

MODIFICATIONS   

    NONE

SUMMARY

    Performs a complex division.

INTERFACES       

    FILE                 ROUTINE CALLED     

    N/A                  N/A


RETURNED VALUE
    
    A Mif_Complex_t value representing the result of the complex division.           

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/
#include <stdlib.h>

/*=== Static CM_COMPLEX_DIV ROUTINE ===*/

/**** Cm_complex_div Function - FAKE ***********/
/*                                             */
/*  Function will not be used in finished      */
/*  system...provides a stub for performing    */
/*  a simple complex division.                 */
/*                           12/3/90 JPM       */
/*                                             */
/***********************************************/

static Mif_Complex_t cm_complex_div(Mif_Complex_t x, Mif_Complex_t y)              
{
double mag_x, phase_x, mag_y, phase_y;

Mif_Complex_t out;
                                     
mag_x = hypot(x.real, x.imag);
phase_x = atan2(x.imag, x.real);

mag_y = hypot(y.real, y.imag);
phase_y = atan2(y.imag, y.real);
               
mag_x = mag_x/mag_y;
phase_x = phase_x - phase_y;

out.real = mag_x * cos(phase_x);
out.imag = mag_x * sin(phase_x);

return out;
}       


                   
/*==============================================================================

FUNCTION cm_s_xfer()

AUTHORS                      

    17 Mar 1991     Jeffrey P. Murray

MODIFICATIONS   

    18 Apr 1991     Harry Li
    27 Sept 1991    Jeffrey P. Murray

SUMMARY

    This function implements the s_xfer code model.

INTERFACES       

    FILE                 ROUTINE CALLED     

    CMmacros.h           cm_message_send();                   

    CM.c                 void *cm_analog_alloc()
                         void *cm_analog_get_ptr()
                         int  cm_analog_integrate()

RETURNED VALUE
    
    Returns inputs and outputs via ARGS structure.

GLOBAL VARIABLES
    
    NONE

NON-STANDARD FEATURES

    NONE

==============================================================================*/

/*=== CM_S_XFER ROUTINE ===*/

/****************************************
* S-Domain Transfer Function -          *
*      Code Body                        *
*                                       *
* Last Modified - 9/27/91        JPM    *
****************************************/

void cm_s_xfer(ARGS)  /* structure holding parms, inputs, outputs, etc.     */
{
    double *out;                 /* pointer to the output */
	double *in;                  /* pointer to the input */
	double in_offset;            /* input offset */
	double *gain;                /* pointer to the gain */
	double **den_coefficient;    /* dynamic array that holds the denominator
									coefficients */
	double **old_den_coefficient;/* dynamic array that holds the old 
									denonminator coefficients */
	double **num_coefficient;    /* dynamic array that holds the numerator
									coefficients */
	double **old_num_coefficient;/* dynamic array that holds the old numerator
									coefficients */
	double factor;               /* gain factor in case the highest
									denominator coefficient is not 1 */
    double **integrator;         /* outputs of the integrators       */
	double **old_integrator;     /* previous integrator outputs      */
	double null;                 /* dummy pointer for use with the
									integrate function               */
	double pout_pin;             /* partial out wrt in               */
	/*double total_gain;*/           /* not used, currently-used with ITP stuff */
    double temp;                 /* temporary variable used with the 
									correct type of AC value */
	double frac;                 /* holds fractional part of a divide */
	double divide_integer;       /* integer part of a modf used in AC */
	double denormalized_freq;    /* denormalization constant...the nominal
                                    corner or center frequencies specified
                                    by the model coefficients will be 
                                    denormalized by this amount. Thus, if
                                    coefficients were obtained which specified
                                    a 1 rad/sec cornere frequency, specifying
                                    a value of 1000.0 for denormalized_freq
                                    will cause the model to shift the corner
                                    freq. to 2.0 * pi * 1000.0 */
	double *old_gain;            /* pointer to the gain if the highest order
								    denominator coefficient is not factored out */ 	

    Mif_Complex_t ac_gain, acc_num, acc_den;
                                                   
    int i;                       /* generic loop counter index */
	int den_size;                /* size of the denominator coefficient array */
	int num_size;                /* size of the numerator coefficient array */ 

    char *num_size_error="\n***ERROR***\nS_XFER: Numerator coefficient array size greater than\ndenominator coefficiant array size.\n";



    /** Retrieve frequently used parameters (used by all analyses)... **/

    in_offset = PARAM(in_offset);
    num_size = PARAM_SIZE(num_coeff);                    
    den_size = PARAM_SIZE(den_coeff);                    
    if ( PARAM_NULL(denormalized_freq) ) {
        denormalized_freq = 1.0;          
    }
    else {
        denormalized_freq = PARAM(denormalized_freq);                    
    }

    if ( num_size > den_size ) {
        cm_message_send(num_size_error);
        return;
    }

    /** Test for INIT; if so, allocate storage, otherwise, retrieve previous       **/
    /** timepoint input values as necessary in subsequent analysis sections...     **/

    if (INIT==1) {  /* First pass...allocate storage for previous values... */
       
        /* Allocate rotational storage for integrator outputs, in & out */


/*****  The following two lines may be unnecessary in the final version *****/

/*  We have to allocate memory and use cm_analog_alloc, because the ITP variables
	are not functional */

        integrator     = (double **) calloc((size_t) den_size, sizeof(double *));
        old_integrator = (double **) calloc((size_t) den_size, sizeof(double *));

        /* Allocate storage for coefficient values */

        den_coefficient     = (double **) calloc((size_t) den_size, sizeof(double *));
        old_den_coefficient = (double **) calloc((size_t) den_size, sizeof(double *));

        num_coefficient     = (double **) calloc((size_t) num_size, sizeof(double *));
        old_num_coefficient = (double **) calloc((size_t) num_size, sizeof(double *));

        for (i=0; i < (2*den_size + num_size + 3); i++)
              cm_analog_alloc(i,sizeof(double));

   /*     ITP_VAR_SIZE(den) = den_size;  */

     /*   gain = (double *) calloc(1,sizeof(double));
        ITP_VAR(total_gain) = gain;
        ITP_VAR_SIZE(total_gain) = 1.0;  */

		// Retrieve pointers
        
        for (i=0; i<den_size; i++) {
            integrator[i]     = (double *) cm_analog_get_ptr(i,0);
            old_integrator[i] = (double *) cm_analog_get_ptr(i,0);
        }

        for(i=den_size;i<2*den_size;i++) {
            den_coefficient[i-den_size]     = (double *) cm_analog_get_ptr(i,0);
            old_den_coefficient[i-den_size] = (double *) cm_analog_get_ptr(i,0);
        }

        for(i=2*den_size;i<2*den_size+num_size;i++) {
            num_coefficient[i-2*den_size]     = (double *) cm_analog_get_ptr(i,0);
            old_num_coefficient[i-2*den_size] = (double *) cm_analog_get_ptr(i,0);
        }

        out = (double *) cm_analog_get_ptr(2*den_size+num_size, 0);
        in  = (double *) cm_analog_get_ptr(2*den_size+num_size+1, 0);

        gain = (double *) cm_analog_get_ptr(2*den_size+num_size+2,0);

    }else { /* Allocation was not necessary...retrieve previous values */
    
        /* Set pointers to storage locations for in, out, and integrators...*/
 
        integrator = (double **) calloc((size_t) den_size, sizeof(double *));
        old_integrator = (double **) calloc((size_t) den_size, sizeof(double *));

        for (i=0; i<den_size; i++) {
            integrator[i] = (double *) cm_analog_get_ptr(i,0);
            old_integrator[i] = (double *) cm_analog_get_ptr(i,1);

        }
        out = (double *) cm_analog_get_ptr(2*den_size+num_size,0);   
        in = (double *) cm_analog_get_ptr(2*den_size+num_size+1,0);   
    
    
        /* Set den_coefficient & gain pointers to ITP values */
        /* for denominator coefficients & gain...      */

        old_den_coefficient = (double **) calloc((size_t) den_size, sizeof(double));  
        den_coefficient = (double **) calloc((size_t) den_size, sizeof(double));  

		for(i=den_size;i<2*den_size;i++){
            old_den_coefficient[i-den_size] = (double *) cm_analog_get_ptr(i,1);
		    den_coefficient[i-den_size] = (double *) cm_analog_get_ptr(i,0);
            *(den_coefficient[i-den_size]) = *(old_den_coefficient[i-den_size]);
		} 

        num_coefficient = (double **) calloc((size_t) num_size, sizeof(double));  
		old_num_coefficient = (double **) calloc((size_t) num_size, sizeof(double));  

		for(i=2*den_size;i<2*den_size+num_size;i++){
		    old_num_coefficient[i-2*den_size] = (double *) cm_analog_get_ptr(i,1);
			num_coefficient[i-2*den_size] = (double *) cm_analog_get_ptr(i,0);
			*(num_coefficient[i-2*den_size]) = *(old_num_coefficient[i-2*den_size]);
		} 

        /* gain has to be stored each time since it could possibly change
		   if the highest order denominator coefficient isn't zero.  This
		   is a hack until the ITP variables work */

        old_gain = (double *) cm_analog_get_ptr(2*den_size+num_size+2,1);  
        gain = (double *) cm_analog_get_ptr(2*den_size+num_size+2,0);  

		*gain = *old_gain;

        /* gain = ITP_VAR(total_gain); */
    
    }


    /** Test for TIME=0.0; if so, initialize...       **/

    if (TIME == 0.0) {  /* First pass...set initial conditions... */
        
        /* Initialize integrators to int_ic condition values... */
        for (i=0; i<den_size-1; i++) {   /* Note...do NOT set the highest   */
                                         /* order value...this represents   */
                                         /* the "calculated" input to the   */
                                         /* actual highest integrator...it  */
                                         /* is NOT a true state variable.   */
            if ( PARAM_NULL(int_ic) ) {
                // *(integrator[i]) = *(old_integrator[i]) = PARAM(int_ic[0]);
				*(integrator[i]) = *(old_integrator[i]) = 0;
            }                                                
            else {
                *(integrator[i]) = *(old_integrator[i]) = 
                                   PARAM(int_ic[den_size - 2 - i]);
            }
        }


        /*** Read in coefficients and denormalize, if required ***/

        for (i=0; i<num_size; i++) {
            *(num_coefficient[i]) = PARAM(num_coeff[num_size - 1 - i]);
            if ( denormalized_freq != 1.0 ) {
                *(num_coefficient[i]) = *(num_coefficient[i]) / 
                                        pow(denormalized_freq,(double) i);
            }
        }

        for (i=0; i<den_size; i++) {
            *(den_coefficient[i]) = PARAM(den_coeff[den_size - 1 - i]);
            if ( denormalized_freq != 1.0 ) {
                *(den_coefficient[i]) = *(den_coefficient[i]) / 
                                        pow(denormalized_freq,(double) i);
            }
        }



        /* Test denominator highest order coefficient...if that value   */
        /* is other than 1.0, then divide all denominator coefficients  */
        /* and the gain by that value...                                */
		// if ( (factor = PARAM(den_coeff[den_size-1])) != 1.0 ) {
		if ( (factor = *den_coefficient[den_size-1]) != 1.0 ) {
            for (i=0; i<den_size; i++) {
                *(den_coefficient[i]) = *(den_coefficient[i]) / factor;
            }
            *gain = PARAM(gain) / factor; 
        }
        else {    /* No division by coefficient necessary... */
                  /* only need to adjust gain value.         */
            *gain = PARAM(gain); 
        }
        
    }
                                 

    /**** DC & Transient Analyses **************************/
    if (ANALYSIS != MIF_AC) {     

        /**** DC Analysis - Not needed JPM 10/29/91 *****************/
/*        if (ANALYSIS == MIF_DC) {    
            
            ?* Test to see if a term exists for the zero-th order
               denom coeff...       

            ?* division by zero if output              
            ?* num_coefficient[0]/den_coefficient[0],  
            ?* so output init. conds. instead...       
            if ( 0.0 == *(den_coefficient[0])) {    
                                            
                                            
                *out = 0.0;
                for (i=0; i<num_size; i++) {
                    *out = *out + ( *(old_integrator[i]) * 
                                    *(num_coefficient[i]) ); 
                }
                *out = *gain * *out;
                pout_pin = *(old_integrator[1]);

            }

            ?* Zero-th order den term != 0.0, so output 
            ?*    num_coeff[0]/den_coeff[0]...          
            else {                      
                                        
                *out = *gain * ( INPUT(in) + 
                                 in_offset) * ( *(num_coefficient[0]) /
                                                *(den_coefficient[0]) );
                pout_pin = 0.0;
            }
        }
 


        else {   
*/


        /**** Transient & DC Analyses ****************************/

        /*** Read input value for current time, and 
             calculate pseudo-input which includes input 
             offset and gain....                         ***/

        *in = *gain * (INPUT(in)+in_offset);



        /*** Obtain the "new" input to the Controller 
             Canonical topology, then propagate through
             the integrators....                         ***/

        /* calculate the "new" input to the first integrator, based on   */
        /* the old values of each integrator multiplied by their         */
        /* respective denominator coefficients and then subtracted       */
        /* from *in....                                                  */
        /* Note that this value, which is similar to a state variable,   */
        /* is stored in *(integrator[den_size-1]).                       */

        *(integrator[den_size-1]) = *in;
        for (i=0; i<den_size-1; i++) {  
            *(integrator[den_size-1]) = 
                          *(integrator[den_size-1]) - 
                          *(old_integrator[i]) * *(den_coefficient[i]);
        }

    
 

       /* Propagate the new input through each integrator in succession. */
        
        for (i=den_size-1; i>0; i--) {  
            cm_analog_integrate(*(integrator[i]),(integrator[i-1]),&null);
        }



        /* Calculate the output based on the new integrator values... */

        *out = 0.0;
        for (i=0; i<num_size; i++) {
            *out = *out + ( *(integrator[i]) * 
                            *(num_coefficient[i]) );
        }
        pout_pin = *(integrator[1]);
        

        /** Output values for DC & Transient **/

        OUTPUT(out) = *out;          
        PARTIAL(out,in) = pout_pin; 
        // cm_analog_auto_partial(); // Removed again. Seems to have problems.

    }

    /**** AC Analysis ************************************/
    else {                    

        /*** Calculate Real & Imaginary portions of AC gain ***/
        /***    at the current RAD_FREQ point...             ***/ 


        /*** Calculate Numerator Real & Imaginary Components... ***/
        
        acc_num.real = 0.0;
        acc_num.imag = 0.0;

        for (i=0; i<num_size; i++) {
            frac = modf(i/2.0, &divide_integer); /* Determine the integer portion    */
                                                 /* of a divide-by-2.0 on the index. */

            if (modf(divide_integer/2.0,&temp) > 0.0 ) { /* Negative coefficient       */
                                                     /* values for this iteration. */
                if (frac > 0.0 ) {  /** Odd Powers of "s" **/
                    acc_num.imag = acc_num.imag - *(num_coefficient[i]) * pow(RAD_FREQ,i) * (*gain); 
                }
                else {                      /** Even Powers of "s" **/
                    acc_num.real = acc_num.real - *(num_coefficient[i]) * pow(RAD_FREQ,i) * (*gain); 
                }
            }
            else {               /* Positive coefficient values for this iteration */
                if (frac> 0.0 ) {  /** Odd Powers of "s" **/
                    acc_num.imag = acc_num.imag + *(num_coefficient[i]) * pow(RAD_FREQ,i) * (*gain); 
                }
                else {                      /** Even Powers of "s" **/
                    acc_num.real = acc_num.real + *(num_coefficient[i]) * pow(RAD_FREQ,i) * (*gain); 
                }
            }
        }

        /*** Calculate Denominator Real & Imaginary Components... ***/
        
        acc_den.real = 0.0;
        acc_den.imag = 0.0;

        for (i=0; i<den_size; i++) {
            frac = modf(i/2.0, &divide_integer);  /* Determine the integer portion    */
                                                 /* of a divide-by-2.0 on the index. */
            if (modf(divide_integer/2.0,&temp) > 0.0 ) { /* Negative coefficient       */
                                                     /* values for this iteration. */
                if (frac > 0.0 ) {  /** Odd Powers of "s" **/
                    acc_den.imag = acc_den.imag - *(den_coefficient[i]) * pow(RAD_FREQ,i); 
                }
                else {                      /** Even Powers of "s" **/
                    acc_den.real = acc_den.real - *(den_coefficient[i]) * pow(RAD_FREQ,i); 
                }
            }
            else {               /* Positive coefficient values for this iteration */
                if (frac > 0.0 ) {  /** Odd Powers of "s" **/
                    acc_den.imag = acc_den.imag + *(den_coefficient[i]) * pow(RAD_FREQ,i); 
                }
                else {                      /** Even Powers of "s" **/
                    acc_den.real = acc_den.real + *(den_coefficient[i]) * pow(RAD_FREQ,i); 
                }
            }
        }
                      
        /* divide numerator values by denominator values */
                                                    
        ac_gain = cm_complex_div(acc_num, acc_den);

        AC_GAIN(out,in) = ac_gain;
    }

	  /* free all allocated memory */
		if(integrator) free(integrator);
		if(old_integrator) free(old_integrator);
		if(den_coefficient) free(den_coefficient);
		if(old_den_coefficient) free(old_den_coefficient);
		if(num_coefficient) free(num_coefficient);
		if(old_num_coefficient) free(old_num_coefficient);
}







