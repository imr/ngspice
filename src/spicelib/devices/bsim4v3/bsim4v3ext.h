/**********
Copyright 2003 Regents of the University of California.  All rights reserved.
Author: 2000 Weidong Liu
Author: 2001- Xuemei Xi
File: bsim4v3ext.h
**********/

#ifdef __STDC__
extern int BSIM4v3acLoad(GENmodel *,CKTcircuit*);
extern int BSIM4v3ask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int BSIM4v3convTest(GENmodel *,CKTcircuit*);
extern int BSIM4v3delete(GENmodel*,IFuid,GENinstance**);
extern void BSIM4v3destroy(GENmodel**);
extern int BSIM4v3getic(GENmodel*,CKTcircuit*);
extern int BSIM4v3load(GENmodel*,CKTcircuit*);
extern int BSIM4v3mAsk(CKTcircuit*,GENmodel *,int, IFvalue*);
extern int BSIM4v3mDelete(GENmodel**,IFuid,GENmodel*);
extern int BSIM4v3mParam(int,IFvalue*,GENmodel*);
extern void BSIM4v3mosCap(CKTcircuit*, double, double, double, double,
        double, double, double, double, double, double, double,
        double, double, double, double, double, double, double*,
        double*, double*, double*, double*, double*, double*, double*,
        double*, double*, double*, double*, double*, double*, double*, 
        double*);
extern int BSIM4v3param(int,IFvalue*,GENinstance*,IFvalue*);
extern int BSIM4v3pzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int BSIM4v3setup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int BSIM4v3temp(GENmodel*,CKTcircuit*);
extern int BSIM4v3trunc(GENmodel*,CKTcircuit*,double*);
extern int BSIM4v3noise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
extern int BSIM4v3unsetup(GENmodel*,CKTcircuit*);

#else /* stdc */
extern int BSIM4v3acLoad();
extern int BSIM4v3delete();
extern void BSIM4v3destroy();
extern int BSIM4v3getic();
extern int BSIM4v3load();
extern int BSIM4v3mDelete();
extern int BSIM4v3ask();
extern int BSIM4v3mAsk();
extern int BSIM4v3convTest();
extern int BSIM4v3temp();
extern int BSIM4v3mParam();
extern void BSIM4v3mosCap();
extern int BSIM4v3param();
extern int BSIM4v3pzLoad();
extern int BSIM4v3setup();
extern int BSIM4v3trunc();
extern int BSIM4v3noise();
extern int BSIM4v3unsetup();

#endif /* stdc */

