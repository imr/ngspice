/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifdef __STDC__

extern int DIOacLoad(GENmodel*,CKTcircuit*);
extern int DIOask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int DIOconvTest(GENmodel *,CKTcircuit*);
extern int DIOdelete(GENmodel*,IFuid,GENinstance**);
extern void DIOdestroy(GENmodel**);
extern int DIOgetic(GENmodel*,CKTcircuit*);
extern int DIOload(GENmodel*,CKTcircuit*);
extern int DIOmAsk(CKTcircuit*,GENmodel*,int,IFvalue*);
extern int DIOmDelete(GENmodel**,IFuid,GENmodel*);
extern int DIOmParam(int,IFvalue*,GENmodel*);
extern int DIOparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int DIOpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int DIOsAcLoad(GENmodel*,CKTcircuit*);
extern int DIOsLoad(GENmodel*,CKTcircuit*);
extern int DIOsSetup(SENstruct*,GENmodel*);
extern void DIOsPrint(GENmodel*,CKTcircuit*);
extern int DIOsUpdate(GENmodel*,CKTcircuit*);
extern int DIOsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int DIOunsetup(GENmodel*,CKTcircuit*);
extern int DIOtemp(GENmodel*,CKTcircuit*);
extern int DIOtrunc(GENmodel*,CKTcircuit*,double*);
extern int DIOdisto(int,GENmodel*,CKTcircuit*);
extern int DIOnoise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);

#else /* stdc */

extern int DIOacLoad();
extern int DIOask();
extern int DIOconvTest();
extern int DIOdelete();
extern void DIOdestroy();
extern int DIOgetic();
extern int DIOload();
extern int DIOmAsk();
extern int DIOmDelete();
extern int DIOmParam();
extern int DIOparam();
extern int DIOpzLoad();
extern int DIOsAcLoad();
extern int DIOsLoad();
extern int DIOsSetup();
extern void DIOsPrint();
extern int DIOsUpdate();
extern int DIOsetup();
extern int DIOunsetup();
extern int DIOtemp();
extern int DIOtrunc();
extern int DIOdisto();
extern int DIOnoise();
#endif /* stdc */

