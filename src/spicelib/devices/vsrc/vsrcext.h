/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

extern int VSRCacLoad(GENmodel*,CKTcircuit*);
extern int VSRCaccept(CKTcircuit*,GENmodel *);
extern int VSRCask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int VSRCdelete(GENmodel*,IFuid,GENinstance**);
extern void VSRCdestroy(GENmodel**);
extern int VSRCfindBr(CKTcircuit*,GENmodel*,IFuid);
extern int VSRCload(GENmodel*,CKTcircuit*);
extern int VSRCmAsk(CKTcircuit*,GENmodel*,int,IFvalue*);
extern int VSRCmDelete(GENmodel**,IFuid,GENmodel*);
extern int VSRCparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int VSRCpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int VSRCsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int VSRCunsetup(GENmodel*,CKTcircuit*);
extern int VSRCpzSetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int VSRCtemp(GENmodel*,CKTcircuit*);
