/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1990 Jaijeet S. Roychowdhury
**********/

#ifdef __STDC__
extern int LTRAaccept(CKTcircuit*,GENmodel*);
extern int LTRAask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int LTRAacLoad(GENmodel*,CKTcircuit*);
extern int LTRAdelete(GENmodel*,IFuid,GENinstance**);
extern void LTRAdestroy(GENmodel**);
extern int LTRAload(GENmodel*,CKTcircuit*);
extern int LTRAmAsk(CKTcircuit*,GENmodel*,int,IFvalue*);
extern int LTRAmDelete(GENmodel**,IFuid,GENmodel*);
extern int LTRAparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int LTRAmParam(int,IFvalue*,GENmodel*);
extern int LTRAsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int LTRAunsetup(GENmodel*,CKTcircuit*);
extern int LTRAtemp(GENmodel*,CKTcircuit*);
extern int LTRAtrunc(GENmodel*,CKTcircuit*,double*);

extern int LTRAquadInterp(double,double,double,double,double*,double*,double*);
/*
extern double LTRAcoeffSetup(double*,int,double,double,double*,double,double*,int,int*);
extern double LTRAtCoeffSetup(double*,int,double,double*,double,double*,double,double*,int,int*);
extern double LTRAdivDiffs(double*,double*,double,double,double*,int);
*/
extern double LTRArlcH1dashFunc(double,double,double,double);
extern double LTRArlcH2Func(double,double,double,double);
extern double LTRArlcH3dashFunc(double,double,double,double);
extern double LTRArlcH1dashTwiceIntFunc(double,double);
extern double LTRArlcH3dashIntFunc(double,double,double);
extern double LTRArcH1dashTwiceIntFunc(double,double);
extern double LTRArcH2TwiceIntFunc(double,double);
extern double LTRArcH3dashTwiceIntFunc(double,double,double);
extern double LTRAlteCalculate(CKTcircuit*,GENmodel*,GENinstance*,double);
/*
extern double LTRAh1dashCoeffSetup(double*,int,double,double,double*,int,int*);
extern double LTRAh3dashCoeffSetup(double*,int,double,double,double,double*,int,int*);
*/
extern void LTRArcCoeffsSetup(double*,double*,double*,double*,double*,double*,int,double,double,double,double*,int,double);
extern void LTRArlcCoeffsSetup(double*,double*,double*,double*,double*,double*,int,double,double,double,double,double*,int,double,int*);
extern int LTRAstraightLineCheck(double,double,double,double,double,double,double,double);
#else /* stdc */
extern int LTRAaccept();
extern int LTRAask();
extern int LTRAacLoad();
extern int LTRAdelete();
extern void LTRAdestroy();
extern int LTRAload();
extern int LTRAmAsk();
extern int LTRAmDelete();
extern int LTRAparam();
extern int LTRAmParam();
extern int LTRAsetup();
extern int LTRAunsetup();
extern int LTRAtemp();
extern int LTRAtrunc();

extern int LTRAquadInterp();
/*
extern double LTRAcoeffSetup();
extern double LTRAtCoeffSetup();
extern double LTRAdivDiffs();
*/
extern double LTRArlcH1dashFunc();
extern double LTRArlcH2Func();
extern double LTRArlcH3dashFunc();
extern double LTRArlcH1dashTwiceIntFunc();
extern double LTRArlcH3dashIntFunc();
extern double LTRArcH1dashTwiceIntFunc();
extern double LTRArcH2TwiceIntFunc();
extern double LTRArcH3dashTwiceIntFunc();
extern double LTRAlteCalculate();
/*
extern double LTRAh1dashCoeffSetup();
extern double LTRAh3dashCoeffSetup();
*/
extern void LTRArcCoeffsSetup();
extern void LTRArlcCoeffsSetup();
extern int LTRAstraightLineCheck();
#endif /* stdc */

