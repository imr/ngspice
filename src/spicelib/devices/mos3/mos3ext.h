/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifdef __STDC__
extern int MOS3acLoad(GENmodel*,CKTcircuit*);
extern int MOS3ask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int MOS3convTest(GENmodel *,CKTcircuit *);
extern int MOS3delete(GENmodel*,IFuid,GENinstance**);
extern void MOS3destroy(GENmodel**);
extern int MOS3getic(GENmodel*,CKTcircuit*);
extern int MOS3load(GENmodel*,CKTcircuit*);
extern int MOS3mAsk(CKTcircuit*,GENmodel*,int,IFvalue*);
extern int MOS3mDelete(GENmodel**,IFuid,GENmodel*);
extern int MOS3mParam(int,IFvalue*,GENmodel*);
extern int MOS3param(int,IFvalue*,GENinstance*,IFvalue*);
extern int MOS3pzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int MOS3sAcLoad(GENmodel*,CKTcircuit*);
extern int MOS3sLoad(GENmodel*,CKTcircuit*);
extern void MOS3sPrint(GENmodel*,CKTcircuit*);
extern int MOS3sSetup(SENstruct*,GENmodel*);
extern int MOS3sUpdate(GENmodel*,CKTcircuit*);
extern int MOS3setup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int MOS3unsetup(GENmodel*,CKTcircuit*);
extern int MOS3temp(GENmodel*,CKTcircuit*);
extern int MOS3trunc(GENmodel*,CKTcircuit*,double*);
extern int MOS3disto(int,GENmodel*,CKTcircuit*);
extern int MOS3noise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
#else /* stdc */
extern int MOS3acLoad();
extern int MOS3ask();
extern int MOS3convTest();
extern int MOS3delete();
extern void MOS3destroy();
extern int MOS3getic();
extern int MOS3load();
extern int MOS3mAsk();
extern int MOS3mDelete();
extern int MOS3mParam();
extern int MOS3param();
extern int MOS3pzLoad();
extern int MOS3sAcLoad();
extern int MOS3sLoad();
extern void MOS3sPrint();
extern int MOS3sSetup();
extern int MOS3sUpdate();
extern int MOS3setup();
extern int MOS3unsetup();
extern int MOS3temp();
extern int MOS3trunc();
extern int MOS3disto();
extern int MOS3noise();
#endif /* stdc */
