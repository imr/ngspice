/* ===========================================================================
FILE    CMutil.c

MEMBER OF process XSPICE

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503

AUTHORS

    9/12/91  Jeff Murray

MODIFICATIONS

    <date> <person name> <nature of modifications>

SUMMARY

    This file contains functions callable from user code models.
    These functions were written to support code models in the
    XSPICE library, but may be useful in general.

INTERFACES

    cm_smooth_corner()
    cm_smooth_discontinuity()
    cm_smooth_pwl()

    cm_climit_fcn()

    cm_complex_set()
    cm_complex_add()
    cm_complex_subtract()
    cm_complex_multiply()
    cm_complex_divide()
    

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

=========================================================================== */
#include "ngspice/ngspice.h" /* for NaN */
#include <stdio.h>
#include <math.h>
#include "ngspice/cm.h"
 
/* Corner Smoothing Function ************************************
*                                                               *
* The following function smooths the transition between two     *
* slopes into a quadratic (parabolic) curve. The calling        *
* function passes an x,y coordinate representing the            *
* "breakpoint", a smoothing domain value (d), and the slopes at *
* both endpoints, and the x value itself. The situation is      *
* shown below:              B     C                             *
*                     A     |<-d->|            ^ y              *
*            ---------*-----*     |            |                *
*     lower_slope-^   |<-d->|\    |            |                *
*                             \   |            |                *
*                              \  |            *------>x        *
*  At A<x<C, cm_smooth_corner   \ |                             *
*  returns a "y" value which     \|                             *
*  smoothly transitions from      *__                           *
*  f(A) to f(C) in a parabolic     \ | upper_slope              *
*  fashion...the slope of the new   \|                          *
*  function is also returned.        \                          *
*                                                               *
*****************************************************************/

void cm_smooth_corner(
    double x_input,        /* The value of the x input */
    double x_center,       /* The x intercept of the two slopes */
    double y_center,       /* The y intercept of the two slopes */
    double domain,         /* The smoothing domain */
    double lower_slope,    /* The lower slope */
    double upper_slope,    /* The upper slope */
    double *y_output,      /* The smoothed y output */
    double *dy_dx)         /* The partial of y wrt x */
{
    double x_upper,y_upper,a,b,c,dy_dx_temp;

    /* Set up parabolic constants */

    x_upper = x_center + domain;
    y_upper = y_center + (upper_slope * domain);
    a = ((upper_slope - lower_slope) / 4.0) * (1 / domain);
    b = upper_slope - (2.0 * a * x_upper);
    c = y_upper - (a * x_upper * x_upper) - (b * x_upper);

    /* Calculate y value & derivative */

    dy_dx_temp = 2.0*a*x_input + b;  /* Prevents reassignment problems */
                                     /* for x-limiting cases.          */

    *y_output = a*x_input*x_input + b*x_input + c;
    *dy_dx = dy_dx_temp;
}







/* Discontinuity Smoothing Function *****************************
*                                                               *
* The following function smooths the transition between two     *
* values using an x^2 function. The calling function            *
* function passes an x1,y1 coordinate representing the          *
* starting point, an x2,y2 coordinate representing an ending    *
* point, and the x value itself. The situation is shown below:  *
*                                                               *
*            ^ y                        (xu,yu)                 *
*            |                       ---*-----------            *
*            |                   ---    |                       *
*            |                 --                               *
*            |                -         |                       *
*            |               -                                  *
*            |              -           |                       *
*            |            --                                    *
*            | (xl,yl) ---              |                       *
*        ----|----*---                                          *
*            |    |                     |                       *
*            O------------------------------------->x           *
*****************************************************************/

void cm_smooth_discontinuity(
    double x_input,           /* The x value at which to compute y */
    double x_lower,           /* The x value of the lower corner */
    double y_lower,           /* The y value of the lower corner */
    double x_upper,           /* The x value of the upper corner */
    double y_upper,           /* The y value of the upper corner */
    double *y_output,         /* The computed smoothed y value */
    double *dy_dx)            /* The partial of y wrt x */
{
    double x_center,y_center,a,b,c,center_slope;


    /* Derive x_center, y_center & center_slope values */
    x_center = (x_upper + x_lower) / 2.0;
    y_center = (y_upper + y_lower) / 2.0;
    center_slope = 2.0 * (y_upper - y_lower) / (x_upper - x_lower);


    if (x_input < x_lower) {    /* x_input @ lower level */
        *y_output = y_lower;
        *dy_dx = 0.0;
    }
    else {
        if (x_input < x_center) {   /*  x_input in lower transition */
            a = center_slope / (x_upper - x_lower); 
            b = center_slope - 2.0 * a * x_center;
            c = y_center - a * x_center * x_center - b * x_center;
            *y_output = a * x_input * x_input + b * x_input + c;
            *dy_dx = 2.0 * a * x_input + b;
        }
        else {                      /*  x_input in upper transition */
            if (x_input < x_upper) {
                a = -center_slope / (x_upper - x_lower); 
                b = -2.0 * a * x_upper;
                c = y_upper - a * x_upper * x_upper - b * x_upper;
                *y_output = a * x_input * x_input + b * x_input + c;
                *dy_dx = 2.0 * a * x_input + b;
            }
            else {       /* x_input @ upper level */
                *y_output = y_upper;
                *dy_dx = 0.0;
            }
        }
    }
}



/* Controlled Limiter Function (modified CLIMIT) */

/*
This is a special function created for use with the CLIMIT
controlled limiter model.
*/

void cm_climit_fcn(
    double in,                      /* The input value */
    double in_offset,               /* The input offset */
    double cntl_upper,              /* The upper control input value */
    double cntl_lower,              /* The lower control input value */
    double lower_delta,             /* The delta from control to limit value */
    double upper_delta,             /* The delta from control to limit value */
    double limit_range,             /* The limiting range */
    double gain,                    /* The gain from input to output */
    int    percent,                 /* The fraction vs. absolute range flag */
    double *out_final,              /* The output value */
    double *pout_pin_final,         /* The partial of output wrt input */
    double *pout_pcntl_lower_final, /* The partial of output wrt lower control input */
    double *pout_pcntl_upper_final) /* The partial of output wrt upper control input */
{

/* Define error message string constants */

char *climit_range_error = "\n**** ERROR ****\n* CLIMIT function linear range less than zero. *\n";


double threshold_upper,threshold_lower,linear_range,
           out_lower_limit,out_upper_limit,limited_out,
           out,pout_pin,pout_pcntl_lower,pout_pcntl_upper,junk;
    
    /* Find Upper & Lower Limits */

    out_lower_limit = cntl_lower + lower_delta;
    out_upper_limit = cntl_upper - upper_delta;


    if (percent == TRUE)     /* Set range to absolute value */
        limit_range = limit_range * 
              (out_upper_limit - out_lower_limit);



    threshold_upper = out_upper_limit -   /* Set Upper Threshold */
                         limit_range;
    threshold_lower = out_lower_limit +   /* Set Lower Threshold */
                         limit_range;
    linear_range = threshold_upper - threshold_lower;

    
    /* Test the linear region & make sure there IS one... */
    if (linear_range < 0.0) {
        printf("%s\n",climit_range_error);                   
/*        limited_out = 0.0;
        pout_pin = 0.0;  
        pout_pcntl_lower = 0.0;
        pout_pcntl_upper = 0.0;
        return;
*/    }

    /* Compute Un-Limited Output */
    out = gain * (in_offset + in); 


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


    *out_final = limited_out;
    *pout_pin_final = pout_pin;
    *pout_pcntl_lower_final = pout_pcntl_lower;
    *pout_pcntl_upper_final = pout_pcntl_upper;

}                           

/****  End Controlled Limiter Function  ****/    

/*=============================================================================*/


/* Piecewise Linear Smoothing Function *********************
*      The following is a transfer curve function which    *
*   accepts as input an "x" value, and returns a "y"       *
*   value. The transfer characteristic is a smoothed       *
*   piece-wise linear curve described by *x and *y array   *
*   coordinate pairs.                                      *
*                                                          *
*   Created 8/14/91                                        *
*   Last Modified 8/14/91                   J.P.Murray     *
***********************************************************/

/***********************************************************
*                                                          *
*                           ^    x[4]                      *
*                 x[1]      |    *                         *
*                  |  midpoint  /|\                        *
*                       |   |  /   \                       *
*                  |    V   | /  |  \                      *
*                  *----*----*       \                     *
*        midpoint /|        |    |    \                    *
*              | /          ||         *  <- midpoint      *
*              V/  |        |x[3]       \                  *
*  <-----------*------------O------------\------------->   *
*          |  /             |             \  |     |       *
*            /              |              \               *
*          |/               |               \|     |       *
*          *                |                *-----*--->   *
*         /|                |              x[5]  x[6]      *
*        /                  |                              *
*       / x[0]              |                              *
*      /                    |                              *
*     /                     |                              *
*    /                      |                              *
*                           V                              *
*                                                          *
***********************************************************/

/***********************************************************
*                                                          *
*  Note that for the cm_smooth_pwl function, the arguments *
*  are as listed below:                                    *
*                                                          *
*                                                          *
* double x_input;            input *                       *
* double *x;                 pointer to the x-coordinate   *
*                            array *                       *
* double *y;                 pointer to the y-coordinate   *
*                            array *                       *
* int size;                  size of the arrays            *
*                                                          *
* double input_domain;       smoothing range *             *
* double dout_din;           partial derivative of the     *
*                            output w.r.t. the input *     *
*                                                          *
***********************************************************/


double cm_smooth_pwl(double x_input, double *x, double *y, int size,
               double input_domain, double *dout_din)  
{

    int i;               /* generic loop counter index */

    double lower_seg;      /* x segment below which input resides */
    double upper_seg;      /* x segment above which the input resides */
    double lower_slope;    /* slope of the lower segment */
    double upper_slope;    /* slope of the upper segment */
    double out;            /* output */
    double threshold_lower; /* value below which the output begins smoothing */
    double threshold_upper; /* value above which the output begins smoothing */

                                           
    /*    char *limit_error="\n***ERROR***\nViolation of 50% rule in breakpoints!\n";*/





    /* Determine segment boundaries within which x_input resides */

    if (x_input <= (x[1] + x[0])/2.0) {/*** x_input below lowest midpoint ***/
        *dout_din = (y[1] - y[0])/(x[1] - x[0]);
        out = *y + (x_input - *x) * *dout_din;
        return out;
    }
    else {
        if (x_input >= (x[size-2] + x[size-1])/2.0) { 
                                   /*** x_input above highest midpoint ***/
            *dout_din = (y[size-1] - y[size-2]) /
                          (x[size-1] - x[size-2]);
            out = y[size-1] + (x_input - x[size-1]) * *dout_din;
            return out;
        }
        else { /*** x_input within bounds of end midpoints...     ***/
               /*** must determine position progressively & then  ***/
               /*** calculate required output.                    ***/

            for (i=1; i<size; i++) {

                if (x_input < (x[i] + x[i+1])/2.0) { 
                                   /* approximate position known...          */
             
                    lower_seg = (x[i] - x[i-1]);
                    upper_seg = (x[i+1] - x[i]);


                    /* Calculate input_domain about this region's breakpoint.*/

                    /* Translate input_domain into an absolute....   */
                    if ( lower_seg <= upper_seg )          /* Use lower  */
                                                           /* segment    */
                                                           /* for % calc.*/
                        input_domain = input_domain * lower_seg;
                    else                                   /* Use upper  */
                                                           /* segment    */
                                                           /* for % calc.*/
                        input_domain = input_domain * upper_seg;
                    

                    /* Set up threshold values about breakpoint... */
                    threshold_lower = x[i] - input_domain;
                    threshold_upper = x[i] + input_domain;

                    /* Determine where x_input is within region & determine */
                    /* output and partial values....                        */
                    if (x_input < threshold_lower) { /* Lower linear region */
                        *dout_din = (y[i] - y[i-1])/lower_seg;
                        out = y[i] + (x_input - x[i]) * *dout_din;
                        return out;
                    }
                    else {        
                        if (x_input < threshold_upper) { /* Parabolic region */
                            lower_slope = (y[i] - y[i-1])/lower_seg;
                            upper_slope = (y[i+1] - y[i])/upper_seg;
                            cm_smooth_corner(x_input,x[i],y[i],input_domain,
                                        lower_slope,upper_slope,&out,dout_din);
                            return out;
                        }
                        else {        /* Upper linear region */
                            *dout_din = (y[i+1] - y[i])/upper_seg;
                            out = y[i] + (x_input - x[i]) * *dout_din;
                            return out;
                        }
                    }
                    break;  /* Break search loop...x_input has been found, */
                            /* and out and *dout_din have been assigned.   */
                }
            }
        }
    }
    return NAN;
} 


Complex_t cm_complex_set(double real, double imag)
{
    /* Create a complex number with the real and imaginary */
    /* parts specified in the argument list, and return it */

    Complex_t c;

    c.real = real;
    c.imag = imag;

    return(c);
}

Complex_t cm_complex_add(Complex_t x, Complex_t y)
{
    /* Add the two complex numbers and return the result */

    Complex_t c;

    c.real = x.real + y.real;
    c.imag = x.imag + y.imag;

    return(c);
}

Complex_t cm_complex_subtract(Complex_t x, Complex_t y)
{
    /* Subtract the second arg from the first and return the result */

    Complex_t c;

    c.real = x.real - y.real;
    c.imag = x.imag - y.imag;

    return(c);
}

Complex_t cm_complex_multiply(Complex_t x, Complex_t y)
{
    /* Multiply the two complex numbers and return the result */

    Complex_t c;

    c.real = (x.real * y.real) - (x.imag * y.imag);
    c.imag = (x.real * y.imag) + (x.imag * y.real);

    return(c);
}

Complex_t cm_complex_divide(Complex_t x, Complex_t y)
{
    /* Divide the first number by the second and return the result */

    Complex_t c;
    double    mag_y_squared;

    mag_y_squared = (y.real * y.real) + (y.imag * y.imag);

    if(mag_y_squared < 1e-100) {
        printf("\nWARNING: cm_complex_divide() - divide by zero\n");
        mag_y_squared = 1e-100;
    }

    c.real = ((x.real * y.real) + (x.imag * y.imag)) / mag_y_squared;
    c.imag = ((x.imag * y.real) - (y.imag * x.real)) / mag_y_squared;

    return(c);
}


