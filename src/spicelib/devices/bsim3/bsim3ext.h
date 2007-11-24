/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1991 JianHui Huang and Min-Chie Jeng.
Modified by Yuhua Cheng to use BSIM3v3 in Spice3f5 (Jan. 1997)
File: bsim3ext.h
**********/

#ifdef __STDC__
extern int BSIM3acLoad(GENmodel *,CKTcircuit*);
extern int BSIM3ask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int BSIM3convTest(GENmodel *,CKTcircuit*);
extern int BSIM3delete(GENmodel*,IFuid,GENinstance**);
extern void BSIM3destroy(GENmodel**);
extern int BSIM3getic(GENmodel*,CKTcircuit*);
extern int BSIM3load(GENmodel*,CKTcircuit*);
extern int BSIM3mAsk(CKTcircuit*,GENmodel *,int, IFvalue*);
extern int BSIM3mDelete(GENmodel**,IFuid,GENmodel*);
extern int BSIM3mParam(int,IFvalue*,GENmodel*);
extern void BSIM3mosCap(CKTcircuit*, double, double, double, double,
        double, double, double, double, double, double, double,
        double, double, double, double, double, double, double*,
        double*, double*, double*, double*, double*, double*, double*,
        double*, double*, double*, double*, double*, double*, double*, 
        double*);
extern int BSIM3param(int,IFvalue*,GENinstance*,IFvalue*);
extern int BSIM3pzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int BSIM3setup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int BSIM3temp(GENmodel*,CKTcircuit*);
extern int BSIM3trunc(GENmodel*,CKTcircuit*,double*);
extern int BSIM3noise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
extern int BSIM3unsetup(GENmodel*,CKTcircuit*);

#else /* stdc */
extern int BSIM3acLoad();
extern int BSIM3delete();
extern void BSIM3destroy();
extern int BSIM3getic();
extern int BSIM3load();
extern int BSIM3mDelete();
extern int BSIM3ask();
extern int BSIM3mAsk();
extern int BSIM3convTest();
extern int BSIM3temp();
extern int BSIM3mParam();
extern void BSIM3mosCap();
extern int BSIM3param();
extern int BSIM3pzLoad();
extern int BSIM3setup();
extern int BSIM3trunc();
extern int BSIM3noise();
extern int BSIM3unsetup();

#endif /* stdc */

