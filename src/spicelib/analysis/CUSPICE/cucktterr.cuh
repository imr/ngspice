/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

extern "C"
__device__
static
int
cuCKTterr
(
int qcap, double **CKTstates, double *d_CKTdeltaOld,
double d_CKTdelta, int d_CKTorder, int d_CKTintegrateMethod,
double d_CKTabsTol, double d_CKTrelTol, double d_CKTchgTol, double d_CKTtrTol,
//Return Value
double *timeStep
)
{

#define ccap (qcap+1)

/* known integration methods */
#define TRAPEZOIDAL 1
#define GEAR 2

#define MAX(a,b) ((a) > (b) ? (a) : (b))

    double volttol, chargetol, tol, del ;
    double diff [8] ;
    double deltmp [8] ;
    double factor = 0 ;
    int i, j ;
    double gearCoeff [] = {
        .5,
        .2222222222,
        .1363636364,
        .096,
        .07299270073,
        .05830903790
    } ;
    double trapCoeff [] = {
        .5,
        .08333333333
    } ;

    volttol = d_CKTabsTol + d_CKTrelTol * MAX (fabs (CKTstates [0] [ccap]), fabs (CKTstates [1] [ccap])) ;

    chargetol = MAX (fabs (CKTstates [0] [qcap]), fabs (CKTstates [1] [qcap])) ;
    chargetol = d_CKTrelTol * MAX (chargetol, d_CKTchgTol) / d_CKTdelta ;

    tol = MAX (volttol, chargetol) ;

    /* now divided differences */
    for (i = d_CKTorder + 1 ; i >= 0 ; i--)
    {
        diff [i] = CKTstates [i] [qcap] ;
    }
    for (i = 0 ; i <= d_CKTorder ; i++)
    {
        deltmp [i] = d_CKTdeltaOld [i] ;
    }
    j = d_CKTorder ;
    for (;;)
    {
        for (i = 0 ; i <= j ; i++)
        {
            diff [i] = (diff [i] - diff [i + 1]) / deltmp [i] ;
        }

        if (--j < 0)
            break ;

        for (i = 0 ; i <= j ; i++)
        {
            deltmp [i] = deltmp [i + 1] + d_CKTdeltaOld [i] ;
        }
    }
    switch (d_CKTintegrateMethod)
    {
        case GEAR:
            factor = gearCoeff [d_CKTorder - 1] ;
            break ;

        case TRAPEZOIDAL:
            factor = trapCoeff [d_CKTorder - 1] ;
            break ;
    }
    del = d_CKTtrTol * tol / MAX (d_CKTabsTol, factor * fabs (diff [0])) ;
    if (d_CKTorder == 2)
    {
        del = sqrt (del) ;
    } else if (d_CKTorder > 2) {
        del = exp (log (del) / d_CKTorder) ;
    }

    *timeStep = del ;

    return 0 ;
}
