/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

extern int JFETacLoad(GENmodel*,CKTcircuit*);
extern int JFETask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int JFETgetic(GENmodel*,CKTcircuit*);
extern int JFETload(GENmodel*,CKTcircuit*);
extern int JFETmAsk(CKTcircuit*,GENmodel*,int,IFvalue*);
extern int JFETmParam(int,IFvalue*,GENmodel*);
extern int JFETparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int JFETpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int JFETsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int JFETunsetup(GENmodel*,CKTcircuit*);
extern int JFETtemp(GENmodel*,CKTcircuit*);
extern int JFETtrunc(GENmodel*,CKTcircuit*,double*);
extern int JFETdisto(int,GENmodel*,CKTcircuit*);
extern int JFETnoise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
extern int JFETdSetup(GENmodel*,CKTcircuit*);
