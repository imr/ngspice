#ifdef __STDC__
/* extern int TXLaccept(CKTcircuit*,GENmodel*); */
extern int TXLdelete(GENmodel*,IFuid,GENinstance**);
extern void TXLdestroy(GENmodel**);
extern int TXLload(GENmodel*,CKTcircuit*);
extern int TXLmDelete(GENmodel**,IFuid,GENmodel*);
extern int TXLmParam(int,IFvalue*,GENmodel*);
extern int TXLparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int TXLsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
#else /* stdc */
/* extern int TXLaccept(); */
extern int TXLdelete();
extern void TXLdestroy();
extern int TXLload();
extern int TXLmDelete();
extern int TXLmParam();
extern int TXLparam();
extern int TXLsetup();
#endif /* stdc */
