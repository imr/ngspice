/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

extern int CAPacLoad(GENmodel*,CKTcircuit*);
extern int CAPask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int CAPgetic(GENmodel*,CKTcircuit*);
extern int CAPload(GENmodel*,CKTcircuit*);
extern int CAPmAsk(CKTcircuit*,GENmodel*,int,IFvalue*);
extern int CAPmParam(int,IFvalue*,GENmodel*);
extern int CAPparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int CAPpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int CAPsAcLoad(GENmodel*,CKTcircuit*);
extern int CAPsLoad(GENmodel*,CKTcircuit*);
extern void CAPsPrint(GENmodel*,CKTcircuit*);
extern int CAPsSetup(SENstruct *,GENmodel*);
extern int CAPsUpdate(GENmodel*,CKTcircuit*);
extern int CAPsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int CAPtemp(GENmodel*,CKTcircuit*);
extern int CAPtrunc(GENmodel*,CKTcircuit*,double*);
extern int CAPsoaCheck(CKTcircuit *, GENmodel *);

