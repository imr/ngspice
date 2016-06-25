/**********
Copyright 2000 Regents of the University of California.  All rights reserved.
Author: 2000 Weidong Liu.
File: bsim4v0ext.h
**********/

#ifdef __STDC__
extern int BSIM4v0acLoad(GENmodel *,CKTcircuit*);
extern int BSIM4v0ask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int BSIM4v0convTest(GENmodel *,CKTcircuit*);
extern int BSIM4v0delete(GENmodel*,IFuid,GENinstance**);
extern void BSIM4v0destroy(GENmodel**);
extern int BSIM4v0getic(GENmodel*,CKTcircuit*);
extern int BSIM4v0load(GENmodel*,CKTcircuit*);
extern int BSIM4v0mAsk(CKTcircuit*,GENmodel *,int, IFvalue*);
extern int BSIM4v0mDelete(GENmodel**,IFuid,GENmodel*);
extern int BSIM4v0mParam(int,IFvalue*,GENmodel*);
extern void BSIM4v0mosCap(CKTcircuit*, double, double, double, double,
        double, double, double, double, double, double, double,
        double, double, double, double, double, double, double*,
        double*, double*, double*, double*, double*, double*, double*,
        double*, double*, double*, double*, double*, double*, double*, 
        double*);
extern int BSIM4v0param(int,IFvalue*,GENinstance*,IFvalue*);
extern int BSIM4v0pzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int BSIM4v0setup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int BSIM4v0temp(GENmodel*,CKTcircuit*);
extern int BSIM4v0trunc(GENmodel*,CKTcircuit*,double*);
extern int BSIM4v0noise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
extern int BSIM4v0unsetup(GENmodel*,CKTcircuit*);
//extern int BSIM4v0soaCheck(CKTcircuit *, GENmodel *);

#else /* stdc */
extern int BSIM4v0acLoad();
extern int BSIM4v0delete();
extern void BSIM4v0destroy();
extern int BSIM4v0getic();
extern int BSIM4v0load();
extern int BSIM4v0mDelete();
extern int BSIM4v0ask();
extern int BSIM4v0mAsk();
extern int BSIM4v0convTest();
extern int BSIM4v0temp();
extern int BSIM4v0mParam();
extern void BSIM4v0mosCap();
extern int BSIM4v0param();
extern int BSIM4v0pzLoad();
extern int BSIM4v0setup();
extern int BSIM4v0trunc();
extern int BSIM4v0noise();
extern int BSIM4v0unsetup();
//extern int BSIM4v0soaCheck();

#endif /* stdc */

