/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1991 JianHui Huang and Min-Chie Jeng.
File: bsim3v2ext.h
**********/

#ifdef __STDC__
extern int BSIM3V2acLoad(GENmodel *,CKTcircuit*);
extern int BSIM3V2ask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int BSIM3V2convTest(GENmodel *,CKTcircuit*);
extern int BSIM3V2delete(GENmodel*,IFuid,GENinstance**);
extern void BSIM3V2destroy(GENmodel**);
extern int BSIM3V2getic(GENmodel*,CKTcircuit*);
extern int BSIM3V2load(GENmodel*,CKTcircuit*);
extern int BSIM3V2mAsk(CKTcircuit*,GENmodel *,int, IFvalue*);
extern int BSIM3V2mDelete(GENmodel**,IFuid,GENmodel*);
extern int BSIM3V2mParam(int,IFvalue*,GENmodel*);
extern void BSIM3V2mosCap(CKTcircuit*, double, double, double, double,
        double, double, double, double, double, double, double,
        double, double, double, double, double, double, double*,
        double*, double*, double*, double*, double*, double*, double*,
        double*, double*, double*, double*, double*, double*, double*, 
        double*);
extern int BSIM3V2param(int,IFvalue*,GENinstance*,IFvalue*);
extern int BSIM3V2pzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int BSIM3V2setup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int BSIM3V2unsetup(GENmodel*,CKTcircuit*);
extern int BSIM3V2temp(GENmodel*,CKTcircuit*);
extern int BSIM3V2trunc(GENmodel*,CKTcircuit*,double*);
extern int BSIM3V2noise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);

#else /* stdc */
extern int BSIM3V2acLoad();
extern int BSIM3V2delete();
extern void BSIM3V2destroy();
extern int BSIM3V2getic();
extern int BSIM3V2load();
extern int BSIM3V2mDelete();
extern int BSIM3V2ask();
extern int BSIM3V2mAsk();
extern int BSIM3V2convTest();
extern int BSIM3V2temp();
extern int BSIM3V2mParam();
extern void BSIM3V2mosCap();
extern int BSIM3V2param();
extern int BSIM3V2pzLoad();
extern int BSIM3V2setup();
extern int BSIM3V2unsetup();
extern int BSIM3V2trunc();
extern int BSIM3V2noise();

#endif /* stdc */

