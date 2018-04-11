/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

extern int CCVSask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int CCVSfindBr(CKTcircuit*,GENmodel*,IFuid);
extern int CCVSload(GENmodel*,CKTcircuit*);
extern int CCVSparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int CCVSpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int CCVSsAcLoad(GENmodel*,CKTcircuit*);
extern int CCVSsLoad(GENmodel*,CKTcircuit*);
extern void CCVSsPrint(GENmodel*,CKTcircuit*);
extern int CCVSsSetup(SENstruct*,GENmodel*);
extern int CCVSsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int CCVSunsetup(GENmodel*,CKTcircuit*);
