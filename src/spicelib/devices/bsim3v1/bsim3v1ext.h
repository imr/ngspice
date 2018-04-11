/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1991 JianHui Huang and Min-Chie Jeng.
Modified: 2002 Paolo Nenzi.
File: bsim3v1ext.h
**********/

extern int BSIM3v1acLoad(GENmodel *, CKTcircuit *);
extern int BSIM3v1ask(CKTcircuit *, GENinstance *, int, IFvalue *, IFvalue *);
extern int BSIM3v1convTest(GENmodel *, CKTcircuit *);
extern int BSIM3v1getic(GENmodel *, CKTcircuit *);
extern int BSIM3v1load(GENmodel *, CKTcircuit *);
extern int BSIM3v1mAsk(CKTcircuit *, GENmodel *, int, IFvalue *);
extern int BSIM3v1mParam(int, IFvalue *, GENmodel *);
extern void BSIM3v1mosCap(CKTcircuit *, double, double, double, double,
        double, double, double, double, double, double, double,
        double, double, double, double, double, double, double*,
        double *, double *, double *, double *, double *, double *, double *,
        double *, double *, double *, double *, double *, double *, double *, 
        double *);
extern int BSIM3v1param(int, IFvalue *, GENinstance *, IFvalue *);
extern int BSIM3v1pzLoad(GENmodel *, CKTcircuit *, SPcomplex *);
extern int BSIM3v1setup(SMPmatrix *, GENmodel *, CKTcircuit *, int *);
extern int BSIM3v1temp(GENmodel *, CKTcircuit *);
extern int BSIM3v1trunc(GENmodel *, CKTcircuit *, double *);
extern int BSIM3v1noise(int, int, GENmodel *, CKTcircuit *, Ndata *, double *);
extern int BSIM3v1unsetup(GENmodel *, CKTcircuit *);

