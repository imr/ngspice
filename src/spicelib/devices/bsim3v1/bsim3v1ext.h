/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1991 JianHui Huang and Min-Chie Jeng.
File: bsim3v1ext.h
**********/

#ifdef __STDC__
extern int BSIM3V1acLoad(GENmodel *,CKTcircuit*);
extern int BSIM3V1ask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int BSIM3V1convTest(GENmodel *,CKTcircuit*);
extern int BSIM3V1delete(GENmodel*,IFuid,GENinstance**);
extern void BSIM3V1destroy(GENmodel**);
extern int BSIM3V1getic(GENmodel*,CKTcircuit*);
extern int BSIM3V1load(GENmodel*,CKTcircuit*);
extern int BSIM3V1mAsk(CKTcircuit*,GENmodel *,int, IFvalue*);
extern int BSIM3V1mDelete(GENmodel**,IFuid,GENmodel*);
extern int BSIM3V1mParam(int,IFvalue*,GENmodel*);
extern void BSIM3V1mosCap(CKTcircuit*, double, double, double, double,
        double, double, double, double, double, double, double,
        double, double, double, double, double, double, double*,
        double*, double*, double*, double*, double*, double*, double*,
        double*, double*, double*, double*, double*, double*, double*, 
        double*);
extern int BSIM3V1param(int,IFvalue*,GENinstance*,IFvalue*);
extern int BSIM3V1pzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int BSIM3V1setup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int BSIM3V1temp(GENmodel*,CKTcircuit*);
extern int BSIM3V1trunc(GENmodel*,CKTcircuit*,double*);
extern int BSIM3V1noise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);

#else /* stdc */
extern int BSIM3V1acLoad();
extern int BSIM3V1delete();
extern void BSIM3V1destroy();
extern int BSIM3V1getic();
extern int BSIM3V1load();
extern int BSIM3V1mDelete();
extern int BSIM3V1ask();
extern int BSIM3V1mAsk();
extern int BSIM3V1convTest();
extern int BSIM3V1temp();
extern int BSIM3V1mParam();
extern void BSIM3V1mosCap();
extern int BSIM3V1param();
extern int BSIM3V1pzLoad();
extern int BSIM3V1setup();
extern int BSIM3V1trunc();
extern int BSIM3V1noise();

#endif /* stdc */

