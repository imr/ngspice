/**********
Copyright 2004 Regents of the University of California.  All rights reserved.
Author: 2000 Weidong Liu
Author: 2001- Xuemei Xi
File: bsim4v5ext.h
**********/

extern int BSIM4v5acLoad(GENmodel *,CKTcircuit*);
extern int BSIM4v5ask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int BSIM4v5convTest(GENmodel *,CKTcircuit*);
extern int BSIM4v5getic(GENmodel*,CKTcircuit*);
extern int BSIM4v5load(GENmodel*,CKTcircuit*);
extern int BSIM4v5mAsk(CKTcircuit*,GENmodel *,int, IFvalue*);
extern int BSIM4v5mDelete(GENmodel*);
extern int BSIM4v5mParam(int,IFvalue*,GENmodel*);
extern void BSIM4v5mosCap(CKTcircuit*, double, double, double, double,
        double, double, double, double, double, double, double,
        double, double, double, double, double, double, double*,
        double*, double*, double*, double*, double*, double*, double*,
        double*, double*, double*, double*, double*, double*, double*, 
        double*);
extern int BSIM4v5param(int,IFvalue*,GENinstance*,IFvalue*);
extern int BSIM4v5pzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int BSIM4v5setup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int BSIM4v5temp(GENmodel*,CKTcircuit*);
extern int BSIM4v5trunc(GENmodel*,CKTcircuit*,double*);
extern int BSIM4v5noise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
extern int BSIM4v5unsetup(GENmodel*,CKTcircuit*);
extern int BSIM4v5soaCheck(CKTcircuit *, GENmodel *);
