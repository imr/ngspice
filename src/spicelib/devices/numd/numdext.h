/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Karti Mayaram
**********/

#ifndef NUMDEXT_H
#define NUMDEXT_H


extern int NUMDacLoad(GENmodel *, CKTcircuit *);
extern int NUMDask(CKTcircuit *, GENinstance *, int, IFvalue *, IFvalue *);
extern int NUMDdelete(GENinstance *);
extern void NUMDdestroy(void);
extern int NUMDgetic(GENmodel *, CKTcircuit *);
extern int NUMDload(GENmodel *, CKTcircuit *);
extern int NUMDmDelete(GENmodel *);
extern int NUMDmParam(int, IFvalue *, GENmodel *);
extern int NUMDparam(int, IFvalue *, GENinstance *, IFvalue *);
extern int NUMDpzLoad(GENmodel *, CKTcircuit *, SPcomplex *);
extern int NUMDsetup(SMPmatrix *, GENmodel *, CKTcircuit *, int *);
extern int NUMDtemp(GENmodel *, CKTcircuit *);
extern int NUMDtrunc(GENmodel *, CKTcircuit *, double *);

extern void NUMDdump(GENmodel *, CKTcircuit *);
extern void NUMDacct(GENmodel *, CKTcircuit *, FILE *);

#ifdef KLU
extern int NUMDbindCSC (GENmodel*, CKTcircuit*) ;
extern int NUMDbindCSCComplex (GENmodel*, CKTcircuit*) ;
extern int NUMDbindCSCComplexToReal (GENmodel*, CKTcircuit*) ;
#endif

#endif				/* NUMDEXT_H */
