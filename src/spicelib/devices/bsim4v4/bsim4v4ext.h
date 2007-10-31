/**********
Copyright 2004 Regents of the University of California.  All rights reserved.
Author: 2000 Weidong Liu
Author: 2001- Xuemei Xi
File: bsim4ext.h
**********/


extern int BSIM4V4acLoad(GENmodel *,CKTcircuit*);
extern int BSIM4V4ask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int BSIM4V4convTest(GENmodel *,CKTcircuit*);
extern int BSIM4V4delete(GENmodel*,IFuid,GENinstance**);
extern void BSIM4V4destroy(GENmodel**);
extern int BSIM4V4getic(GENmodel*,CKTcircuit*);
extern int BSIM4V4load(GENmodel*,CKTcircuit*);
extern int BSIM4V4mAsk(CKTcircuit*,GENmodel *,int, IFvalue*);
extern int BSIM4V4mDelete(GENmodel**,IFuid,GENmodel*);
extern int BSIM4V4mParam(int,IFvalue*,GENmodel*);
extern void BSIM4V4mosCap(CKTcircuit*, double, double, double, double,
        double, double, double, double, double, double, double,
        double, double, double, double, double, double, double*,
        double*, double*, double*, double*, double*, double*, double*,
        double*, double*, double*, double*, double*, double*, double*, 
        double*);
extern int BSIM4V4param(int,IFvalue*,GENinstance*,IFvalue*);
extern int BSIM4V4pzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int BSIM4V4setup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int BSIM4V4temp(GENmodel*,CKTcircuit*);
extern int BSIM4V4trunc(GENmodel*,CKTcircuit*,double*);
extern int BSIM4V4noise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
extern int BSIM4V4unsetup(GENmodel*,CKTcircuit*);


