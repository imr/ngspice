/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Min-Chie Jeng, Hong June Park, Thomas L. Quarles
**********/

#ifdef __STDC__
extern int B2acLoad(GENmodel *,CKTcircuit*);
extern int B2ask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int B2convTest(GENmodel *,CKTcircuit*);
extern int B2delete(GENmodel*,IFuid,GENinstance**);
extern void B2destroy(GENmodel**);
extern int B2getic(GENmodel*,CKTcircuit*);
extern int B2load(GENmodel*,CKTcircuit*);
extern int B2mAsk(CKTcircuit*,GENmodel *,int, IFvalue*);
extern int B2mDelete(GENmodel**,IFuid,GENmodel*);
extern int B2mParam(int,IFvalue*,GENmodel*);
extern void B2mosCap(CKTcircuit*, double, double, double, double*,
        double, double, double, double, double, double,
	double*, double*, double*, double*, double*, double*, double*, double*,
        double*, double*, double*, double*, double*, double*, double*, 
        double*);
extern int B2param(int,IFvalue*,GENinstance*,IFvalue*);
extern int B2pzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int B2setup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int B2unsetup(GENmodel*,CKTcircuit*);
extern int B2temp(GENmodel*,CKTcircuit*);
extern int B2trunc(GENmodel*,CKTcircuit*,double*);
#else /* stdc */
extern int B2acLoad();
extern int B2delete();
extern void B2destroy();
extern int B2getic();
extern int B2load();
extern int B2mDelete();
extern int B2ask();
extern int B2mAsk();
extern int B2convTest();
extern int B2temp();
extern int B2mParam();
extern void B2mosCap();
extern int B2param();
extern int B2pzLoad();
extern int B2setup();
extern int B2unsetup();
extern int B2trunc();

#endif /* stdc */
