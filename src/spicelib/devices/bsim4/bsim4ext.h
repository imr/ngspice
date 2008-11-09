/**********
Copyright 2004 Regents of the University of California.  All rights reserved.
Author: 2000 Weidong Liu
Author: 2001- Xuemei Xi
File: bsim4ext.h
**********/

#ifdef __STDC__
extern int BSIM4acLoad(GENmodel *,CKTcircuit*);
extern int BSIM4ask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int BSIM4convTest(GENmodel *,CKTcircuit*);
extern int BSIM4delete(GENmodel*,IFuid,GENinstance**);
extern void BSIM4destroy(GENmodel**);
extern int BSIM4getic(GENmodel*,CKTcircuit*);
extern int BSIM4load(GENmodel*,CKTcircuit*);
extern int BSIM4mAsk(CKTcircuit*,GENmodel *,int, IFvalue*);
extern int BSIM4mDelete(GENmodel**,IFuid,GENmodel*);
extern int BSIM4mParam(int,IFvalue*,GENmodel*);
extern void BSIM4mosCap(CKTcircuit*, double, double, double, double,
        double, double, double, double, double, double, double,
        double, double, double, double, double, double, double*,
        double*, double*, double*, double*, double*, double*, double*,
        double*, double*, double*, double*, double*, double*, double*, 
        double*);
extern int BSIM4param(int,IFvalue*,GENinstance*,IFvalue*);
extern int BSIM4pzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int BSIM4setup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int BSIM4temp(GENmodel*,CKTcircuit*);
extern int BSIM4trunc(GENmodel*,CKTcircuit*,double*);
extern int BSIM4noise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
extern int BSIM4unsetup(GENmodel*,CKTcircuit*);

#else /* stdc */
extern int BSIM4acLoad();
extern int BSIM4delete();
extern void BSIM4destroy();
extern int BSIM4getic();
extern int BSIM4load();
extern int BSIM4mDelete();
extern int BSIM4ask();
extern int BSIM4mAsk();
extern int BSIM4convTest();
extern int BSIM4temp();
extern int BSIM4mParam();
extern void BSIM4mosCap();
extern int BSIM4param();
extern int BSIM4pzLoad();
extern int BSIM4setup();
extern int BSIM4trunc();
extern int BSIM4noise();
extern int BSIM4unsetup();

#endif /* stdc */

