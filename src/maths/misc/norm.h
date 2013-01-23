/**********
 (C) Paolo Nenzi 2001
 **********/ 

/*
 * Bernoulli function
 */
 
#ifndef ngspice_NORM_H
#define ngspice_NORM_H

extern double maxNorm(double *, int);
extern double oneNorm(double *, int);
extern double  l2Norm(double *, int);
extern double     dot(double *, double *, int);

#endif
