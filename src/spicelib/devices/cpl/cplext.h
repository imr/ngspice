#ifdef __STDC__
/* extern int CPLaccept(CKTcircuit*,GENmodel*); */
extern int CPLdelete(GENmodel*,IFuid,GENinstance**);
extern void CPLdestroy(GENmodel**);
extern int CPLload(GENmodel*,CKTcircuit*);
extern int CPLmDelete(GENmodel**,IFuid,GENmodel*);
extern int CPLmParam(int,IFvalue*,GENmodel*);
extern int CPLparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int CPLsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
#else /* stdc */
/* extern int CPLaccept(); */
extern int CPLdelete();
extern void CPLdestroy();
extern int CPLload();
extern int CPLmDelete();
extern int CPLmParam();
extern int CPLparam();
extern int CPLsetup();
#endif /* stdc */
