/**********
Copyright 1993: T. Ytterdal, K. Lee, M. Shur and T. A. Fjeldly. All rights reserved.
Author: Trond Ytterdal
**********/

#ifdef __STDC__
extern int  HFET2acLoad(GENmodel*,CKTcircuit*);
extern int  HFET2ask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int  HFET2delete(GENmodel*,IFuid,GENinstance**);
extern void HFET2destroy(GENmodel**);
extern int  HFET2getic(GENmodel*,CKTcircuit*);
extern int  HFET2load(GENmodel*,CKTcircuit*);
extern int  HFET2mAsk(CKTcircuit*,GENmodel*,int,IFvalue*);
extern int  HFET2mDelete(GENmodel**,IFuid,GENmodel*);
extern int  HFET2mParam(int,IFvalue*,GENmodel*);
extern int  HFET2param(int,IFvalue*,GENinstance*,IFvalue*);
extern int  HFET2setup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int  HFET2temp(GENmodel*,CKTcircuit*);
extern int  HFET2trunc(GENmodel*,CKTcircuit*,double*);
extern int  HFET2unsetup( GENmodel*,CKTcircuit*);

#else  /* stdc */
extern int  HFET2acLoad();
extern int  HFET2ask();
extern int  HFET2delete();
extern void HFET2destroy();
extern int  HFET2getic();
extern int  HFET2load();
extern int HFETAmAsk();
extern int HFETAmDelete();
extern int  HFET2mParam();
extern int  HFET2param();
extern int  HFET2setup();
extern int  HFET2temp();
extern int  HFET2trunc();
extern int  HFET2unsetup();
#endif

