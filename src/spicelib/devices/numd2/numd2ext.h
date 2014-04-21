/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Karti Mayaram
**********/

#ifndef NUMD2EXT_H
#define NUMD2EXT_H


extern int NUMD2acLoad(GENmodel *, CKTcircuit *);
extern int NUMD2ask(CKTcircuit *, GENinstance *, int, IFvalue *, IFvalue *);
extern int NUMD2delete(GENinstance *);
extern void NUMD2destroy(void);
extern int NUMD2getic(GENmodel *, CKTcircuit *);
extern int NUMD2load(GENmodel *, CKTcircuit *);
extern int NUMD2mDelete(GENmodel *);
extern int NUMD2mParam(int, IFvalue *, GENmodel *);
extern int NUMD2param(int, IFvalue *, GENinstance *, IFvalue *);
extern int NUMD2pzLoad(GENmodel *, CKTcircuit *, SPcomplex *);
extern int NUMD2setup(SMPmatrix *, GENmodel *, CKTcircuit *, int *);
extern int NUMD2temp(GENmodel *, CKTcircuit *);
extern int NUMD2trunc(GENmodel *, CKTcircuit *, double *);

extern void NUMD2dump(GENmodel *, CKTcircuit *);
extern void NUMD2acct(GENmodel *, CKTcircuit *, FILE *);

#ifdef KLU
extern int NUMD2bindCSC (GENmodel*, CKTcircuit*) ;
extern int NUMD2bindCSCComplex (GENmodel*, CKTcircuit*) ;
extern int NUMD2bindCSCComplexToReal (GENmodel*, CKTcircuit*) ;
#endif

#endif				/* NUMD2EXT_H */
