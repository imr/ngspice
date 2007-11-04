/**********
Copyright 2004 Regents of the University of California.  All rights reserved.
Author: 2000 Weidong Liu
Author: 2001- Xuemei Xi
File: bsim4v5ext.h
**********/

#ifdef __STDC__
extern int BSIM4v5acLoad(GENmodel *,CKTcircuit*);
extern int BSIM4v5ask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int BSIM4v5convTest(GENmodel *,CKTcircuit*);
extern int BSIM4v5delete(GENmodel*,IFuid,GENinstance**);
extern void BSIM4v5destroy(GENmodel**);
extern int BSIM4v5getic(GENmodel*,CKTcircuit*);
extern int BSIM4v5load(GENmodel*,CKTcircuit*);
extern int BSIM4v5mAsk(CKTcircuit*,GENmodel *,int, IFvalue*);
extern int BSIM4v5mDelete(GENmodel**,IFuid,GENmodel*);
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

#else /* stdc */
extern int BSIM4v5acLoad();
extern int BSIM4v5delete();
extern void BSIM4v5destroy();
extern int BSIM4v5getic();
extern int BSIM4v5load();
extern int BSIM4v5mDelete();
extern int BSIM4v5ask();
extern int BSIM4v5mAsk();
extern int BSIM4v5convTest();
extern int BSIM4v5temp();
extern int BSIM4v5mParam();
extern void BSIM4v5mosCap();
extern int BSIM4v5param();
extern int BSIM4v5pzLoad();
extern int BSIM4v5setup();
extern int BSIM4v5trunc();
extern int BSIM4v5noise();
extern int BSIM4v5unsetup();

#endif /* stdc */

