/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/


extern int CCCSask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int CCCSdelete(GENinstance*);
extern void CCCSdestroy(void);
extern int CCCSload(GENmodel*,CKTcircuit*);
extern int CCCSmDelete(GENmodel*);
extern int CCCSparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int CCCSpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int CCCSsAcLoad(GENmodel*,CKTcircuit*);
extern int CCCSsLoad(GENmodel*,CKTcircuit*);
extern void CCCSsPrint(GENmodel*,CKTcircuit*);
extern int CCCSsSetup(SENstruct*,GENmodel*);
extern int CCCSsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);

#ifdef KLU
extern int CCCSbindCSC (GENmodel*, CKTcircuit*) ;
extern int CCCSbindCSCComplex (GENmodel*, CKTcircuit*) ;
extern int CCCSbindCSCComplexToReal (GENmodel*, CKTcircuit*) ;
#endif
