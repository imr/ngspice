#ifdef __STDC__
extern int HFETAacLoad(GENmodel*,CKTcircuit*);
extern int HFETAask(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);
extern int HFETAdelete(GENmodel*,IFuid,GENinstance**);
extern void HFETAdestroy(GENmodel**);
extern int HFETAgetic(GENmodel*,CKTcircuit*);
extern int HFETAload(GENmodel*,CKTcircuit*);
extern int HFETAmAsk(CKTcircuit*,GENmodel*,int,IFvalue*);
extern int HFETAmDelete(GENmodel**,IFuid,GENmodel*);
extern int HFETAmParam(int,IFvalue*,GENmodel*);
extern int HFETAparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int HFETAsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int HFETAtemp(GENmodel*,CKTcircuit*);
extern int HFETAtrunc(GENmodel*,CKTcircuit*,double*);
extern int HFETAunsetup(GENmodel*,CKTcircuit*);

#else /*stdc*/
extern int HFETAacLoad();
extern int HFETAask();
extern int HFETAdelete();
extern void HFETAdestroy();
extern int HFETAgetic();
extern int HFETAload();
extern int HFETAmAsk();
extern int HFETAmDelete();
extern int HFETAmParam();
extern int HFETAparam();
extern int HFETAsetup();
extern int HFETAtemp();
extern int HFETAtrunc();
extern int HFETAunsetup();


#endif