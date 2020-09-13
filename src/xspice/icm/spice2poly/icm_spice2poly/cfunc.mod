/* ===========================================================================
FILE    cfunc.mod

MEMBER OF process XSPICE

Public Domain

Georgia Tech Research Corporation
Atlanta, Georgia 30332
PROJECT A-8503


AUTHORS

    9/12/91  Bill Kuhn

MODIFICATIONS

    <date> <person name> <nature of modifications>

SUMMARY

    This file contains the definition of a code model polynomial controlled
    source compatible with SPICE 2G6 poly sources.

INTERFACES

    spice2poly()

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

=========================================================================== */

/*

This code model implements the non-linear polynomial controlled sources
available in SPICE 2G6.  An automatic translator added into the simulator
front end is used to map 2G6 syntax into a call to this model in the
required syntax.

This model may also be called directly as follows:

        a1 [ <input(s)> ] <output> xxx
        .model xxx spice2poly ( coef = [ <list of 2G6 compatible coefficients> ] )

Refer to the 2G6 User Guide for an explanation of the coefficients.


This model is patterned after the FORTRAN code used in the 2G6 simulator.
Function cm_poly() below performs the functions of subroutines NLCSRC and
EVPOLY.  Function evterm() performs the function of subroutine EVTERM,
and function nxtpwr() performs the function of subroutine NXTPWR.

*/




#include <stdlib.h>

/* SPICE 2G6 type utility functions */
static double evterm(double x, int n);
static void   nxtpwr(int *pwrseq, int pdim);

static void
cm_spice2poly_callback(ARGS, Mif_Callback_Reason_t reason)
{
    switch (reason) {
        case MIF_CB_DESTROY: {
            Mif_Inst_Var_Data_t *p = STATIC_VAR_INST(acgains);
            if(p->element) {
                free(p->element);
                p->element = NULL;
            }
            break;
        }
    }
}



void spice2poly (ARGS)
{
    int         num_inputs;   /* Number of inputs to model */
    int         num_coefs;    /* Number of coefficients */
    int         *exp;         /* List of exponents in products */
                              /* One for each input */

    int         i;            /* Counter */
    int         j;            /* Counter */
    int         k;            /* Counter */

    double      *in;          /* Values of inputs to model */
    double      *coef;        /* Values of coefficients    */

    double      sum;          /* Temporary for accumulating sum of terms */
    double      product;      /* Temporary for accumulating product */



    /* Get number of input values */

    num_inputs = PORT_SIZE(in);

    /* If this is the first call to the model, allocate the static variable */
    /* array */

    if(INIT) {
        CALLBACK = cm_spice2poly_callback;
        Mif_Inst_Var_Data_t *p = STATIC_VAR_INST(acgains);
        p -> size    = num_inputs;
        p -> element = (Mif_Value_t *) malloc((size_t) num_inputs * sizeof(Mif_Value_t));
        for(i = 0; i < num_inputs; i++)
            STATIC_VAR(acgains[i]) = 0.0;
    }

    /* If analysis type is AC, use the previously computed DC partials */
    /* for the AC gains */

    if(ANALYSIS == MIF_AC) {
        for(i = 0; i < num_inputs; i++) {
            AC_GAIN(out,in[i]).real = STATIC_VAR(acgains[i]);
            AC_GAIN(out,in[i]).imag = 0.0;
        }
        return;
    }

    /* Get input values and coefficients to local storage for faster access */

    in = (double *) malloc((size_t) num_inputs * sizeof(double));
    for(i = 0; i < num_inputs; i++)
        in[i] = INPUT(in[i]);

    num_coefs = PARAM_SIZE(coef);

    coef = (double *) malloc((size_t) num_coefs * sizeof(double));
    for(i = 0; i < num_coefs; i++)
        coef[i] = PARAM(coef[i]);


    /* Allocate the array of exponents used in computing the poly terms */
    exp = (int *) malloc((size_t) num_inputs * sizeof(int));

    /* Initialize the exponents to zeros */
    for(i = 0; i < num_inputs; i++)
        exp[i] = 0;


    /* Compute the output of the source by summing the required products */
    for(i = 1, sum = coef[0]; i < num_coefs; i++) {

        /* Get the list of powers for the product terms in this term of the sum */
        nxtpwr(exp, num_inputs);

        /* Form the product of the inputs taken to the required powers */
        for(j = 0, product = 1.0; j < num_inputs; j++)
            product *= evterm(in[j], exp[j]);

        /* Add the product times the appropriate coefficient into the sum */
        sum += coef[i] * product;
    }
    OUTPUT(out) = sum;


    /* Compute and output the partials for each input */
    for(i = 0; i < num_inputs; i++) {

        /* Reinitialize the exponent list to zeros */
        for(j = 0; j < num_inputs; j++)
            exp[j] = 0;

        /* Compute the partials by summing the required products */
        for(j = 1, sum = 0.0; j < num_coefs; j++) {

            /* Get the list of powers for the product terms in this term of the sum */
            nxtpwr(exp, num_inputs);

            /* If power for input for which partial is being evaluated */
            /* is zero, the term is a constant, so the partial is zero */
            if(exp[i] == 0)
                continue;

            /* Form the product of the inputs taken to the required powers */
            for(k = 0, product = 1.0; k < num_inputs; k++) {
                /* If input is not the one for which the partial is being taken */
                /* take the term to the specified exponent */
                if(k != i)
                    product *= evterm(in[k], exp[k]);
                /* else, take the derivative of this term as n*x**(n-1) */
                else
                    product *= exp[k] * evterm(in[k], exp[k] - 1);
            }

            /* Add the product times the appropriate coefficient into the sum */
            sum += coef[j] * product;
        }

        PARTIAL(out,in[i]) = sum;

        /* If this is DC analysis, save the partial for use as AC gain */
        /* value in an AC analysis */

        if(ANALYSIS == MIF_DC)
            STATIC_VAR(acgains[i]) = sum;
    }

    /* Free the allocated items and return */
    free(in);
    free(coef);
    free(exp);

    return;
}


/* Function evterm computes the value of x**n */

static double evterm(
    double x,
    int n)
{
    double product;       /* Temporary accumlator for forming the product */

    product = 1.0;
    while(n > 0) {
        product *= x;
        n--;
    }

    return(product);
}



/*

This function is a literal translation of subroutine NXTPWR in SPICE 2G6.
This was done to guarantee compatibility with the ordering of
coefficients used by 2G6.  The 2G6 User Guide does not completely define
the algorithm used and the GOTO loaded FORTRAN code is difficult to unravel.
Therefore, a one-to-one translation was deemed the safest approach.

No attempt is made to document the function statements since no documentaton
is available in the 2G6 code.  However, it can be noted that the code
appears to generate the exponents of the product terms in the sum-of-products
produced by the following expansion for two and three dimensional polynomials:

    2D        (a + b) ** n
    3D        (a + (b + c)) ** n

where n begins at 1 and increments as needed for as many terms as there are
coefficients on the polynomial source SPICE deck card, and where terms that
are identical under the laws of associativity are dropped.  Thus, for example,
the exponents for the following sums are produced:

    2D       a + b + a**2 + ab + b**2 + c**3 + ...
    3D       a + b + c + a**2 + a*b + a*c + b**2 + bc + c**2 + a**3 + ...

*/

/* Define a macro to tranlate between FORTRAN-style array references */
/* and C-style array references */

#define PWRSEQ(x)  pwrseq[x - 1]


static void nxtpwr(
    int *pwrseq,           /* Array of exponents */
    int pdim)
{
    int i;
    int k;
    int km1;
    int psum;

        if(pdim == 1) goto stmt80;
        k = pdim;
stmt10: if(PWRSEQ(k) != 0) goto stmt20;
        k = k - 1;
        if(k != 0) goto stmt10;
        goto stmt80;
stmt20: if(k == pdim) goto stmt30;
        PWRSEQ(k) = PWRSEQ(k) - 1;
        PWRSEQ(k+1) = PWRSEQ(k+1) + 1;
        goto stmt100;
stmt30: km1 = k - 1;
        for(i = 1; i <= km1; i++)
            if(PWRSEQ(i) != 0) goto stmt50;
/*stmt40:*/ PWRSEQ(1) = PWRSEQ(pdim) + 1;
        PWRSEQ(pdim) = 0;
        goto stmt100;
stmt50: psum = 1;
        k = pdim;
stmt60: if(PWRSEQ(k-1) >= 1) goto stmt70;
        psum = psum + PWRSEQ(k);
        PWRSEQ(k) = 0;
        k = k - 1;
        goto stmt60;
stmt70: PWRSEQ(k) = PWRSEQ(k) + psum;
        PWRSEQ(k-1) = PWRSEQ(k-1) - 1;
        goto stmt100;
stmt80: PWRSEQ(1) = PWRSEQ(1) + 1;

stmt100: return;

}

