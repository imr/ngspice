/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifdef __STDC__
extern int CCVSask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int CCVSdelete(GENmodel*,IFuid,GENinstance**);
extern void CCVSdestroy(GENmodel**);
extern int CCVSfindBr(CKTcircuit*,GENmodel*,IFuid);
extern int CCVSload(GENmodel*,CKTcircuit*);
extern int CCVSmDelete(GENmodel**,IFuid,GENmodel*);
extern int CCVSparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int CCVSpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int CCVSsAcLoad(GENmodel*,CKTcircuit*);
extern int CCVSsLoad(GENmodel*,CKTcircuit*);
extern void CCVSsPrint(GENmodel*,CKTcircuit*);
extern int CCVSsSetup(SENstruct*,GENmodel*);
extern int CCVSsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int CCVSunsetup(GENmodel*,CKTcircuit*);
#else /* stdc */
extern int CCVSask();
extern int CCVSdelete();
extern void CCVSdestroy();
extern int CCVSfindBr();
extern int CCVSload();
extern int CCVSmDelete();
extern int CCVSparam();
extern int CCVSpzLoad();
extern int CCVSsAcLoad();
extern int CCVSsLoad();
extern void CCVSsPrint();
extern int CCVSsSetup();
extern int CCVSsetup();
extern int CCVSunsetup();
#endif /* stdc */

