/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung
File: b3soiddext.h
**********/


#ifdef __STDC__
extern int B3SOIDDacLoad(GENmodel *,CKTcircuit*);
extern int B3SOIDDask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int B3SOIDDconvTest(GENmodel *,CKTcircuit*);
extern int B3SOIDDdelete(GENmodel*,IFuid,GENinstance**);
extern void B3SOIDDdestroy(GENmodel**);
extern int B3SOIDDgetic(GENmodel*,CKTcircuit*);
extern int B3SOIDDload(GENmodel*,CKTcircuit*);
extern int B3SOIDDmAsk(CKTcircuit*,GENmodel *,int, IFvalue*);
extern int B3SOIDDmDelete(GENmodel**,IFuid,GENmodel*);
extern int B3SOIDDmParam(int,IFvalue*,GENmodel*);
extern void B3SOIDDmosCap(CKTcircuit*, double, double, double, double,
        double, double, double, double, double, double, double,
        double, double, double, double, double, double, double*,
        double*, double*, double*, double*, double*, double*, double*,
        double*, double*, double*, double*, double*, double*, double*, 
        double*);
extern int B3SOIDDparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int B3SOIDDpzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int B3SOIDDsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int B3SOIDDtemp(GENmodel*,CKTcircuit*);
extern int B3SOIDDtrunc(GENmodel*,CKTcircuit*,double*);
extern int B3SOIDDnoise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
extern int B3SOIDDunsetup(GENmodel*,CKTcircuit*);

#else /* stdc */
extern int B3SOIDDacLoad();
extern int B3SOIDDdelete();
extern void B3SOIDDdestroy();
extern int B3SOIDDgetic();
extern int B3SOIDDload();
extern int B3SOIDDmDelete();
extern int B3SOIDDask();
extern int B3SOIDDmAsk();
extern int B3SOIDDconvTest();
extern int B3SOIDDtemp();
extern int B3SOIDDmParam();
extern void B3SOIDDmosCap();
extern int B3SOIDDparam();
extern int B3SOIDDpzLoad();
extern int B3SOIDDsetup();
extern int B3SOIDDtrunc();
extern int B3SOIDDnoise();
extern int B3SOIDDunsetup();

#endif /* stdc */

