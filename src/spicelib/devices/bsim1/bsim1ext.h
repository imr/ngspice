/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985      Hong June Park, Thomas L. Quarles
**********/

#ifdef __STDC__
extern int B1acLoad(GENmodel *,CKTcircuit*);
extern int B1ask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int B1convTest(GENmodel *,CKTcircuit*);
extern int B1delete(GENmodel*,IFuid,GENinstance**);
extern void B1destroy(GENmodel**);
extern int B1getic(GENmodel*,CKTcircuit*);
extern int B1load(GENmodel*,CKTcircuit*);
extern int B1mAsk(CKTcircuit*,GENmodel *,int, IFvalue*);
extern int B1mDelete(GENmodel**,IFuid,GENmodel*);
extern int B1mParam(int,IFvalue*,GENmodel*);
extern void B1mosCap(CKTcircuit*, double, double, double, double*,
	double, double, double, double, double, double,
	double*, double*, double*, double*, double*, double*, double*, double*,
	double*, double*, double*, double*, double*, double*, double*,
	double*);
extern int B1param(int,IFvalue*,GENinstance*,IFvalue*);
extern int B1pzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int B1setup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int B1unsetup(GENmodel*,CKTcircuit*);
extern int B1temp(GENmodel*,CKTcircuit*);
extern int B1trunc(GENmodel*,CKTcircuit*,double*);
extern int B1disto(int,GENmodel*,CKTcircuit*);
#else /* stdc */
extern int B1acLoad();
extern int B1ask();
extern int B1convTest();
extern int B1delete();
extern void B1destroy();
extern int B1getic();
extern int B1load();
extern int B1mAsk();
extern int B1mDelete();
extern int B1mParam();
extern void B1mosCap();
extern int B1param();
extern int B1pzLoad();
extern int B1setup();
extern int B1unsetup();
extern int B1temp();
extern int B1trunc();
extern int B1disto();
#endif /* stdc */

