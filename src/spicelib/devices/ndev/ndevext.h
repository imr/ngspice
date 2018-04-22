/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Karti Mayaram
**********/

#ifndef NDEVEXT_H
#define NDEVEXT_H


extern int NDEVacLoad(GENmodel *, CKTcircuit *);
extern int NDEVask(CKTcircuit *, GENinstance *, int, IFvalue *, IFvalue *);
extern int NDEVgetic(GENmodel *, CKTcircuit *);
extern int NDEVload(GENmodel *, CKTcircuit *);
extern int NDEVaccept(CKTcircuit *, GENmodel *);
extern int NDEVconvTest(GENmodel *, CKTcircuit *);
extern int NDEVmDelete(GENmodel *);
extern int NDEVmParam(int, IFvalue *, GENmodel *);
extern int NDEVparam(int, IFvalue *, GENinstance *, IFvalue *);
extern int NDEVpzLoad(GENmodel *, CKTcircuit *, SPcomplex *);
extern int NDEVsetup(SMPmatrix *, GENmodel *, CKTcircuit *, int *);
extern int NDEVtemp(GENmodel *, CKTcircuit *);
extern int NDEVtrunc(GENmodel *, CKTcircuit *, double *);

extern void NDEV_dump(GENmodel *, CKTcircuit *);
extern void NDEV_acct(GENmodel *, CKTcircuit *, FILE *);

#endif				/* NDEVEXT_H */
