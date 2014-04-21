/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Karti Mayaram
**********/

#ifndef __NBJT_H
#define __NBJT_H

extern int NBJTacLoad(GENmodel *, CKTcircuit *);
extern int NBJTask(CKTcircuit *, GENinstance *, int, IFvalue *, IFvalue *);
extern int NBJTdelete(GENinstance *);
extern void NBJTdestroy(void);
extern int NBJTgetic(GENmodel *, CKTcircuit *);
extern int NBJTload(GENmodel *, CKTcircuit *);
extern int NBJTmDelete(GENmodel *);
extern int NBJTmParam(int, IFvalue *, GENmodel *);
extern int NBJTparam(int, IFvalue *, GENinstance *, IFvalue *);
extern int NBJTpzLoad(GENmodel *, CKTcircuit *, SPcomplex *);
extern int NBJTsetup(SMPmatrix *, GENmodel *, CKTcircuit *, int *);
extern int NBJTtemp(GENmodel *, CKTcircuit *);
extern int NBJTtrunc(GENmodel *, CKTcircuit *, double *);

extern void NBJTdump(GENmodel *, CKTcircuit *);
extern void NBJTacct(GENmodel *, CKTcircuit *, FILE *);

#ifdef KLU
extern int NBJTbindCSC (GENmodel*, CKTcircuit*) ;
extern int NBJTbindCSCComplex (GENmodel*, CKTcircuit*) ;
extern int NBJTbindCSCComplexToReal (GENmodel*, CKTcircuit*) ;
#endif

#endif /* NBJT_H */
