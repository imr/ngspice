/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Karti Mayaram
**********/

#ifndef NBJT2EXT_H
#define NBJT2EXT_H


extern int NBJT2acLoad(GENmodel *, CKTcircuit *);
extern int NBJT2ask(CKTcircuit *, GENinstance *, int, IFvalue *, IFvalue *);
extern int NBJT2delete(GENinstance *);
extern int NBJT2modDelete(GENmodel *);
extern int NBJT2getic(GENmodel *, CKTcircuit *);
extern int NBJT2load(GENmodel *, CKTcircuit *);
extern int NBJT2mParam(int, IFvalue *, GENmodel *);
extern int NBJT2param(int, IFvalue *, GENinstance *, IFvalue *);
extern int NBJT2pzLoad(GENmodel *, CKTcircuit *, SPcomplex *);
extern int NBJT2setup(SMPmatrix *, GENmodel *, CKTcircuit *, int *);
extern int NBJT2temp(GENmodel *, CKTcircuit *);
extern int NBJT2trunc(GENmodel *, CKTcircuit *, double *);

extern void NBJT2dump(GENmodel *, CKTcircuit *);
extern void NBJT2acct(GENmodel *, CKTcircuit *, FILE *);

#endif				/* NBJT2EXT_H */
