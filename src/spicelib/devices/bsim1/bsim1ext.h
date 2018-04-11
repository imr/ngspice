/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985      Hong June Park, Thomas L. Quarles
Modified: 2000 AlansFixes
**********/


extern int B1acLoad(GENmodel *,CKTcircuit*);
extern int B1ask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int B1convTest(GENmodel *,CKTcircuit*);
extern int B1getic(GENmodel*,CKTcircuit*);
extern int B1load(GENmodel*,CKTcircuit*);
extern int B1mAsk(CKTcircuit*,GENmodel *,int, IFvalue*);
extern int B1mParam(int,IFvalue*,GENmodel*);
extern void B1mosCap(CKTcircuit*, double, double, double, double*,
	double, double, double, double, double, double,
	double*, double*, double*, double*, double*, double*, double*, double*,
	double*, double*, double*, double*, double*, double*, double*,
	double*);
extern int B1noise (int, int, GENmodel *, CKTcircuit *, Ndata *, double *);
extern int B1param(int,IFvalue*,GENinstance*,IFvalue*);
extern int B1pzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int B1setup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int B1unsetup(GENmodel*,CKTcircuit*);
extern int B1temp(GENmodel*,CKTcircuit*);
extern int B1trunc(GENmodel*,CKTcircuit*,double*);
extern int B1disto(int,GENmodel*,CKTcircuit*);
extern int B1dSetup(GENmodel*, register CKTcircuit*);
