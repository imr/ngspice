/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

extern int MOS3acLoad(GENmodel*,CKTcircuit*);
extern int MOS3ask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int MOS3convTest(GENmodel *,CKTcircuit *);
extern int MOS3delete(GENinstance*);
extern int MOS3getic(GENmodel*,CKTcircuit*);
extern int MOS3load(GENmodel*,CKTcircuit*);
extern int MOS3mAsk(CKTcircuit*,GENmodel*,int,IFvalue*);
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
extern int MOS3dSetup(GENmodel*,CKTcircuit*);
