/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifdef __STDC__
extern int MOS1acLoad(GENmodel *,CKTcircuit*);
extern int MOS1ask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int MOS1delete(GENmodel*,IFuid,GENinstance**);
extern void MOS1destroy(GENmodel**);
extern int MOS1getic(GENmodel*,CKTcircuit*);
extern int MOS1load(GENmodel*,CKTcircuit*);
extern int MOS1mAsk(CKTcircuit *,GENmodel *,int,IFvalue*);
extern int MOS1mDelete(GENmodel**,IFuid,GENmodel*);
extern int MOS1mParam(int,IFvalue*,GENmodel*);
extern int MOS1param(int,IFvalue*,GENinstance*,IFvalue*);
extern int MOS1pzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int MOS1sAcLoad(GENmodel*,CKTcircuit*);
extern int MOS1sLoad(GENmodel*,CKTcircuit*);
extern void MOS1sPrint(GENmodel*,CKTcircuit*);
extern int MOS1sSetup(SENstruct*,GENmodel*);
extern int MOS1sUpdate(GENmodel*,CKTcircuit*);
extern int MOS1setup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int MOS1unsetup(GENmodel*,CKTcircuit*);
extern int MOS1temp(GENmodel*,CKTcircuit*);
extern int MOS1trunc(GENmodel*,CKTcircuit*,double*);
extern int MOS1convTest(GENmodel*,CKTcircuit*);
extern int MOS1disto(int,GENmodel*,CKTcircuit*);
extern int MOS1noise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);

#else /* stdc */
extern int MOS1acLoad();
extern int MOS1ask();
extern int MOS1delete();
extern void MOS1destroy();
extern int MOS1getic();
extern int MOS1load();
extern int MOS1mAsk();
extern int MOS1mDelete();
extern int MOS1mParam();
extern int MOS1param();
extern int MOS1pzLoad();
extern int MOS1sAcLoad();
extern int MOS1sLoad();
extern void MOS1sPrint();
extern int MOS1sSetup();
extern int MOS1sUpdate();
extern int MOS1setup();
extern int MOS1unsetup();
extern int MOS1temp();
extern int MOS1trunc();
extern int MOS1convTest();
extern int MOS1disto();
extern int MOS1noise();
#endif /* stdc */
