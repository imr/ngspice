/**********
Copyright 2004 Regents of the University of California.  All rights reserved.
Author: 2000 Weidong Liu
Author: 2001- Xuemei Xi
File: bsim4v6ext.h
**********/

extern int BSIM4v6acLoad(GENmodel *,CKTcircuit*);
extern int BSIM4v6ask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int BSIM4v6convTest(GENmodel *,CKTcircuit*);
extern int BSIM4v6getic(GENmodel*,CKTcircuit*);
extern int BSIM4v6load(GENmodel*,CKTcircuit*);
extern int BSIM4v6mAsk(CKTcircuit*,GENmodel *,int, IFvalue*);
extern int BSIM4v6mDelete(GENmodel*);
extern int BSIM4v6mParam(int,IFvalue*,GENmodel*);
extern void BSIM4v6mosCap(CKTcircuit*, double, double, double, double,
        double, double, double, double, double, double, double,
        double, double, double, double, double, double, double*,
        double*, double*, double*, double*, double*, double*, double*,
        double*, double*, double*, double*, double*, double*, double*, 
        double*);
extern int BSIM4v6param(int,IFvalue*,GENinstance*,IFvalue*);
extern int BSIM4v6pzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int BSIM4v6setup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int BSIM4v6temp(GENmodel*,CKTcircuit*);
extern int BSIM4v6trunc(GENmodel*,CKTcircuit*,double*);
extern int BSIM4v6noise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
extern int BSIM4v6unsetup(GENmodel*,CKTcircuit*);
extern int BSIM4v6soaCheck(CKTcircuit *, GENmodel *);
