/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifdef __STDC__
extern int MOS2acLoad(GENmodel*,CKTcircuit*);
extern int MOS2ask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int MOS2mAsk(CKTcircuit*,GENmodel*,int,IFvalue*);
extern int MOS2convTest(GENmodel*,CKTcircuit*);
extern int MOS2delete(GENmodel*,IFuid,GENinstance**);
extern void MOS2destroy(GENmodel**);
extern int MOS2getic(GENmodel*,CKTcircuit*);
extern int MOS2load(GENmodel*,CKTcircuit*);
extern int MOS2mDelete(GENmodel**,IFuid,GENmodel*);
extern int MOS2mParam(int,IFvalue*,GENmodel*);
extern int MOS2param(int,IFvalue*,GENinstance*,IFvalue*);
extern int MOS2pzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int MOS2sAcLoad(GENmodel*,CKTcircuit*);
extern int MOS2sLoad(GENmodel*,CKTcircuit*);
extern void MOS2sPrint(GENmodel*,CKTcircuit*);
extern int MOS2sSetup(SENstruct*,GENmodel*);
extern int MOS2sUpdate(GENmodel*,CKTcircuit*);
extern int MOS2setup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int MOS2unsetup(GENmodel*,CKTcircuit*);
extern int MOS2temp(GENmodel*,CKTcircuit*);
extern int MOS2trunc(GENmodel*,CKTcircuit*,double*);
extern int MOS2disto(int,GENmodel*,CKTcircuit*);
extern int MOS2noise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
#else /* stdc */
extern int MOS2acLoad();
extern int MOS2ask();
extern int MOS2mAsk();
extern int MOS2convTest();
extern int MOS2delete();
extern void MOS2destroy();
extern int MOS2getic();
extern int MOS2load();
extern int MOS2mDelete();
extern int MOS2mParam();
extern int MOS2param();
extern int MOS2pzLoad();
extern int MOS2sAcLoad();
extern int MOS2sLoad();
extern void MOS2sPrint();
extern int MOS2sSetup();
extern int MOS2sUpdate();
extern int MOS2setup();
extern int MOS2unsetup();
extern int MOS2temp();
extern int MOS2trunc();
extern int MOS2disto();
extern int MOS2noise();
#endif /* stdc */
