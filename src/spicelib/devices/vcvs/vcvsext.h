/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifdef __STDC__
extern int VCVSask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int VCVSdelete(GENmodel*,IFuid,GENinstance**);
extern void VCVSdestroy(GENmodel**);
extern int VCVSfindBr(CKTcircuit*,GENmodel*,IFuid);
extern int VCVSload(GENmodel*,CKTcircuit*);
extern int VCVSmDelete(GENmodel**,IFuid,GENmodel*);
extern int VCVSparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int VCVSpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int VCVSsAcLoad(GENmodel*,CKTcircuit*);
extern int VCVSsLoad(GENmodel*,CKTcircuit*);
extern int VCVSsSetup(SENstruct*,GENmodel*);
extern void VCVSsPrint(GENmodel*,CKTcircuit*);
extern int VCVSsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int VCVSunsetup(GENmodel*,CKTcircuit*);
#else /* stdc */
extern int VCVSask();
extern int VCVSdelete();
extern void VCVSdestroy();
extern int VCVSfindBr();
extern int VCVSload();
extern int VCVSmDelete();
extern int VCVSparam();
extern int VCVSpzLoad();
extern int VCVSsAcLoad();
extern int VCVSsLoad();
extern int VCVSsSetup();
extern void VCVSsPrint();
extern int VCVSsetup();
extern int VCVSunsetup();
#endif /* stdc */
