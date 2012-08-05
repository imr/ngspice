/**********
Copyright 2004 Regents of the University of California.  All rights reserved.
Author: 2000 Weidong Liu
Author: 2001- Xuemei Xi
File: bsim4ext.h
**********/


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



#if defined(KLU) || defined(SuperLU) || defined(UMFPACK)
extern int BSIM4v4bindCSC (GENmodel*, CKTcircuit*) ;
extern int BSIM4v4bindCSCComplex (GENmodel*, CKTcircuit*) ;
extern int BSIM4v4bindCSCComplexToReal (GENmodel*, CKTcircuit*) ;
#endif
