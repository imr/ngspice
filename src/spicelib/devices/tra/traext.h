/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

extern int TRAacLoad(GENmodel*,CKTcircuit*);
extern int TRAaccept(CKTcircuit*,GENmodel*);
extern int TRAask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int TRAload(GENmodel*,CKTcircuit*);
extern int TRAmAsk(CKTcircuit*,GENmodel*,int,IFvalue*);
extern int TRAparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int TRAsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int TRAunsetup(GENmodel*,CKTcircuit*);
extern int TRAtemp(GENmodel*,CKTcircuit*);
extern int TRAtrunc(GENmodel*,CKTcircuit*,double*);
