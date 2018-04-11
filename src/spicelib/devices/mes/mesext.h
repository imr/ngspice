/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 S. Hwang
Modified: 2000 AlansFixes
**********/

extern int MESacLoad(GENmodel*,CKTcircuit*);
extern int MESask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int MESgetic(GENmodel*,CKTcircuit*);
extern int MESload(GENmodel*,CKTcircuit*);
extern int MESmAsk(CKTcircuit*,GENmodel*,int,IFvalue*);
extern int MESmParam(int,IFvalue*,GENmodel*);
extern int MESparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int MESpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int MESsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int MESunsetup(GENmodel*,CKTcircuit*);
extern int MEStemp(GENmodel*,CKTcircuit*);
extern int MEStrunc(GENmodel*,CKTcircuit*,double*);
extern int MESdisto(int,GENmodel*,CKTcircuit*);
extern int MESnoise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
extern int MESdSetup(GENmodel*,CKTcircuit*);

