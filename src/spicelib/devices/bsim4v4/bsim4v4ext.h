/**********
Copyright 2004 Regents of the University of California.  All rights reserved.
Author: 2000 Weidong Liu
Author: 2001- Xuemei Xi
File: bsim4v4ext.h
**********/

#ifdef __STDC__
extern int BSIM4v4acLoad(GENmodel *,CKTcircuit*);
extern int BSIM4v4ask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int BSIM4v4convTest(GENmodel *,CKTcircuit*);
extern int BSIM4v4delete(GENmodel*,IFuid,GENinstance**);
extern void BSIM4v4destroy(GENmodel**);
extern int BSIM4v4getic(GENmodel*,CKTcircuit*);
extern int BSIM4v4load(GENmodel*,CKTcircuit*);
extern int BSIM4v4mAsk(CKTcircuit*,GENmodel *,int, IFvalue*);
extern int BSIM4v4mDelete(GENmodel**,IFuid,GENmodel*);
extern int BSIM4v4mParam(int,IFvalue*,GENmodel*);
extern void BSIM4v4mosCap(CKTcircuit*, double, double, double, double,
        double, double, double, double, double, double, double,
        double, double, double, double, double, double, double*,
        double*, double*, double*, double*, double*, double*, double*,
        double*, double*, double*, double*, double*, double*, double*,
        double*);
extern int BSIM4v4param(int,IFvalue*,GENinstance*,IFvalue*);
extern int BSIM4v4pzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int BSIM4v4setup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int BSIM4v4temp(GENmodel*,CKTcircuit*);
extern int BSIM4v4trunc(GENmodel*,CKTcircuit*,double*);
extern int BSIM4v4noise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
extern int BSIM4v4unsetup(GENmodel*,CKTcircuit*);
//extern int BSIM4v4soaCheck(CKTcircuit *, GENmodel *);

#else /* stdc */
extern int BSIM4v4acLoad();
extern int BSIM4v4delete();
extern void BSIM4v4destroy();
extern int BSIM4v4getic();
extern int BSIM4v4load();
extern int BSIM4v4mDelete();
extern int BSIM4v4ask();
extern int BSIM4v4mAsk();
extern int BSIM4v4convTest();
extern int BSIM4v4temp();
extern int BSIM4v4mParam();
extern void BSIM4v4mosCap();
extern int BSIM4v4param();
extern int BSIM4v4pzLoad();
extern int BSIM4v4setup();
extern int BSIM4v4trunc();
extern int BSIM4v4noise();
extern int BSIM4v4unsetup();
//extern int BSIM4v4soaCheck();

#endif /* stdc */

