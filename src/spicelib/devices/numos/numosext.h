/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Karti Mayaram
**********/

#ifndef NUMOSEXT_H
#define NUMOSEXT_H


extern int NUMOSacLoad(GENmodel *, CKTcircuit *);
extern int NUMOSask(CKTcircuit *, GENinstance *, int, IFvalue *, IFvalue *);
extern int NUMOSdelete(GENinstance *);
extern int NUMOSmodDelete(GENmodel *);
extern int NUMOSgetic(GENmodel *, CKTcircuit *);
extern int NUMOSload(GENmodel *, CKTcircuit *);
extern int NUMOSmParam(int, IFvalue *, GENmodel *);
extern int NUMOSparam(int, IFvalue *, GENinstance *, IFvalue *);
extern int NUMOSpzLoad(GENmodel *, CKTcircuit *, SPcomplex *);
extern int NUMOSsetup(SMPmatrix *, GENmodel *, CKTcircuit *, int *);
extern int NUMOStemp(GENmodel *, CKTcircuit *);
extern int NUMOStrunc(GENmodel *, CKTcircuit *, double *);

extern void NUMOSdump(GENmodel *, CKTcircuit *);
extern void NUMOSacct(GENmodel *, CKTcircuit *, FILE *);

#endif				/* NUMOSEXT_H */
