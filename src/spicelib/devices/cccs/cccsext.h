/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifdef __STDC__
extern int CCCSask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int CCCSdelete(GENmodel*,IFuid,GENinstance**);
extern void CCCSdestroy(GENmodel**);
extern int CCCSload(GENmodel*,CKTcircuit*);
extern int CCCSmDelete(GENmodel**,IFuid,GENmodel*);
extern int CCCSparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int CCCSpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int CCCSsAcLoad(GENmodel*,CKTcircuit*);
extern int CCCSsLoad(GENmodel*,CKTcircuit*);
extern void CCCSsPrint(GENmodel*,CKTcircuit*);
extern int CCCSsSetup(SENstruct*,GENmodel*);
extern int CCCSsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
#else /* stdc */
extern int CCCSask();
extern int CCCSdelete();
extern void CCCSdestroy();
extern int CCCSload();
extern int CCCSmDelete();
extern int CCCSparam();
extern int CCCSpzLoad();
extern int CCCSsAcLoad();
extern int CCCSsLoad();
extern void CCCSsPrint();
extern int CCCSsSetup();
extern int CCCSsetup();
#endif /* stdc */
