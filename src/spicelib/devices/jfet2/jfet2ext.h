/**********
Based on jfetext.h
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

Modified to add PS model and new parameter definitions ( Anthony E. Parker )
   Copyright 1994  Macquarie University, Sydney Australia.
**********/

#ifdef __STDC__
extern int JFET2acLoad(GENmodel*,CKTcircuit*);
extern int JFET2ask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int JFET2delete(GENmodel*,IFuid,GENinstance**);
extern void JFET2destroy(GENmodel**);
extern int JFET2getic(GENmodel*,CKTcircuit*);
extern int JFET2load(GENmodel*,CKTcircuit*);
extern int JFET2mAsk(CKTcircuit*,GENmodel*,int,IFvalue*);
extern int JFET2mDelete(GENmodel**,IFuid,GENmodel*);
extern int JFET2mParam(int,IFvalue*,GENmodel*);
extern int JFET2param(int,IFvalue*,GENinstance*,IFvalue*);
extern int JFET2setup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int JFET2unsetup(GENmodel*,CKTcircuit*);
extern int JFET2temp(GENmodel*,CKTcircuit*);
extern int JFET2trunc(GENmodel*,CKTcircuit*,double*);
extern int JFET2noise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);

#else /* stdc */
extern int JFET2acLoad();
extern int JFET2ask();
extern int JFET2delete();
extern void JFET2destroy();
extern int JFET2getic();
extern int JFET2load();
extern int JFET2mAsk();
extern int JFET2mDelete();
extern int JFET2mParam();
extern int JFET2param();
extern int JFET2setup();
extern int JFET2unsetup();
extern int JFET2temp();
extern int JFET2trunc();
extern int JFET2noise();
#endif /* stdc */
