/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifdef __STDC__
extern int TRAacLoad(GENmodel*,CKTcircuit*);
extern int TRAaccept(CKTcircuit*,GENmodel*);
extern int TRAask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int TRAdelete(GENmodel*,IFuid,GENinstance**);
extern void TRAdestroy(GENmodel**);
extern int TRAload(GENmodel*,CKTcircuit*);
extern int TRAmAsk(CKTcircuit*,GENmodel*,int,IFvalue*);
extern int TRAmDelete(GENmodel**,IFuid,GENmodel*);
extern int TRAparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int TRAsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int TRAunsetup(GENmodel*,CKTcircuit*);
extern int TRAtemp(GENmodel*,CKTcircuit*);
extern int TRAtrunc(GENmodel*,CKTcircuit*,double*);
#else /* stdc */
extern int TRAacLoad();
extern int TRAaccept();
extern int TRAask();
extern int TRAdelete();
extern void TRAdestroy();
extern int TRAload();
extern int TRAmAsk();
extern int TRAmDelete();
extern int TRAparam();
extern int TRAsetup();
extern int TRAunsetup();
extern int TRAtemp();
extern int TRAtrunc();
#endif /* stdc */
