/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifdef __STDC__
extern int CAPacLoad(GENmodel*,CKTcircuit*);
extern int CAPask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int CAPdelete(GENmodel*,IFuid,GENinstance**);
extern void CAPdestroy(GENmodel**);
extern int CAPgetic(GENmodel*,CKTcircuit*);
extern int CAPload(GENmodel*,CKTcircuit*);
extern int CAPmAsk(CKTcircuit*,GENmodel*,int,IFvalue*);
extern int CAPmDelete(GENmodel**,IFuid,GENmodel*);
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
#else /* stdc */
extern int CAPacLoad();
extern int CAPask();
extern int CAPdelete();
extern void CAPdestroy();
extern int CAPgetic();
extern int CAPload();
extern int CAPmAsk();
extern int CAPmDelete();
extern int CAPmParam();
extern int CAPparam();
extern int CAPpzLoad();
extern int CAPsAcLoad();
extern int CAPsLoad();
extern void CAPsPrint();
extern int CAPsSetup();
extern int CAPsUpdate();
extern int CAPsetup();
extern int CAPtemp();
extern int CAPtrunc();
#endif /* stdc */

