/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

extern int MOS1acLoad(GENmodel *,CKTcircuit*);
extern int MOS1ask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int MOS1delete(GENinstance*);
extern int MOS1getic(GENmodel*,CKTcircuit*);
extern int MOS1load(GENmodel*,CKTcircuit*);
extern int MOS1mAsk(CKTcircuit *,GENmodel *,int,IFvalue*);
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
extern int MOS1dSetup(GENmodel*,CKTcircuit*);
