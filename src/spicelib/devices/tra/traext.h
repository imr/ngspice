/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

extern int TRAacLoad(GENmodel*,CKTcircuit*);
extern int TRAaccept(CKTcircuit*,GENmodel*);
extern int TRAask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int TRAdelete(GENinstance*);
extern void TRAdestroy(void);
extern int TRAload(GENmodel*,CKTcircuit*);
extern int TRAmAsk(CKTcircuit*,GENmodel*,int,IFvalue*);
extern int TRAmDelete(GENmodel*);
extern int TRAparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int TRAsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int TRAunsetup(GENmodel*,CKTcircuit*);
extern int TRAtemp(GENmodel*,CKTcircuit*);
extern int TRAtrunc(GENmodel*,CKTcircuit*,double*);

#ifdef KLU
extern int TRAbindCSC (GENmodel*, CKTcircuit*) ;
extern int TRAbindCSCComplex (GENmodel*, CKTcircuit*) ;
extern int TRAbindCSCComplexToReal (GENmodel*, CKTcircuit*) ;
#endif
