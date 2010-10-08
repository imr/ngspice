/*
 * Copyright (c) 1985 Thomas L. Quarles
 * Modified 1999 Paolo Nenzi - Removed non STDC definitions
 * Kept only prototypes (structs defined in struct.h) ER
 */


#ifndef CKT_H_INCLUDED
#define CKT_H_INCLUDED


/* function prototypes */

int ACan( CKTcircuit *, int );
int ACaskQuest( CKTcircuit *, JOB *, int , IFvalue *);
int ACsetParm( CKTcircuit *, JOB *, int , IFvalue *);
int CKTacDump( CKTcircuit *, double , void *);
int CKTacLoad( CKTcircuit *);
int CKTaccept( CKTcircuit *);
int CKTacct( CKTcircuit *, JOB *, int , IFvalue *);
int CKTask( CKTcircuit *, GENinstance *, int , IFvalue *, IFvalue *);
int CKTaskAnalQ( CKTcircuit *, JOB *, int , IFvalue *, IFvalue *);
int CKTaskNodQst( CKTcircuit *, CKTnode *, int , IFvalue *, IFvalue *);
int CKTbindNode( CKTcircuit *, GENinstance *, int , CKTnode *);
void CKTbreakDump( CKTcircuit *);
int CKTclrBreak( CKTcircuit *);
int CKTconvTest( CKTcircuit *);
int CKTcrtElt( CKTcircuit *, GENmodel *, GENinstance **, IFuid );
int CKTdelTask( CKTcircuit *, TSKtask *);
int CKTdestroy( CKTcircuit *);
int CKTdltAnal( void *, void *, void *);
int CKTdltInst( CKTcircuit *, void *);
int CKTdltMod( CKTcircuit *, GENmodel *);
int CKTdltNod( CKTcircuit *, CKTnode *);
int CKTdoJob( CKTcircuit *, int , TSKtask *);
void CKTdump( CKTcircuit *, double, void *);
int CKTfndAnal( CKTcircuit *, int *, JOB **, IFuid , TSKtask *, IFuid );
int CKTfndBranch( CKTcircuit *, IFuid);
int CKTfndDev( CKTcircuit *, int *, GENinstance **, IFuid , GENmodel *, IFuid );
int CKTfndMod( CKTcircuit *, int *, GENmodel **, IFuid );
int CKTfndNode( CKTcircuit *, CKTnode **, IFuid );
int CKTfndTask( CKTcircuit *, TSKtask **, IFuid  );
int CKTground( CKTcircuit *, CKTnode **, IFuid );
int CKTic( CKTcircuit *);
int CKTinit( CKTcircuit **);
int CKTinst2Node( CKTcircuit *, void *, int , CKTnode **, IFuid *);
int CKTlinkEq(CKTcircuit*,CKTnode*);
int CKTload( CKTcircuit *);
int CKTmapNode( CKTcircuit *, CKTnode **, IFuid );
int CKTmkCur( CKTcircuit  *, CKTnode **, IFuid , char *);
int CKTmkNode(CKTcircuit*,CKTnode**);
int CKTmkVolt( CKTcircuit  *, CKTnode **, IFuid , char *);
int CKTmodAsk( CKTcircuit *, GENmodel *, int , IFvalue *, IFvalue *);
int CKTmodCrt( CKTcircuit *, int , GENmodel **, IFuid );
int CKTmodParam( CKTcircuit *, GENmodel *, int , IFvalue *, IFvalue *);
int CKTnames(CKTcircuit *, int *, IFuid **);
int CKTnewAnal( CKTcircuit *, int , IFuid , JOB **, TSKtask *);
int CKTnewEq( CKTcircuit *, CKTnode **, IFuid );
int CKTnewNode( CKTcircuit *, CKTnode **, IFuid );
int CKTnewTask( CKTcircuit *, TSKtask **, IFuid, TSKtask **);
IFuid CKTnodName( CKTcircuit *, int );
void CKTnodOut( CKTcircuit *);
CKTnode * CKTnum2nod( CKTcircuit *, int );
int CKTop(CKTcircuit *, long, long, int );
int CKTpModName( char *, IFvalue *, CKTcircuit *, int , IFuid , GENmodel **);
int CKTpName( char *, IFvalue *, CKTcircuit *, int , char *, GENinstance **);
int CKTparam( CKTcircuit *, GENinstance *, int , IFvalue *, IFvalue *);
int CKTpzFindZeros( CKTcircuit *, PZtrial **, int * );
int CKTpzLoad( CKTcircuit *, SPcomplex * );
int CKTpzSetup( CKTcircuit *, int);
int CKTsenAC( CKTcircuit *);
int CKTsenComp( CKTcircuit *);
int CKTsenDCtran( CKTcircuit *);
int CKTsenLoad( CKTcircuit *);
void CKTsenPrint( CKTcircuit *);
int CKTsenSetup( CKTcircuit *);
int CKTsenUpdate( CKTcircuit *);
int CKTsetAnalPm( CKTcircuit *, JOB *, int , IFvalue *, IFvalue *);
int CKTsetBreak( CKTcircuit *, double );
int CKTsetNodPm( CKTcircuit *, CKTnode *, int , IFvalue *, IFvalue *);
int CKTsetOpt( CKTcircuit *, JOB *, int , IFvalue *);
int CKTsetup( CKTcircuit *);
int CKTunsetup(CKTcircuit *ckt); 
int CKTtemp( CKTcircuit *);
char *CKTtrouble(CKTcircuit *, char *);
void CKTterr( int , CKTcircuit *, double *);
int CKTtrunc( CKTcircuit *, double *);
int CKTtypelook( char *);
int DCOaskQuest( CKTcircuit *, JOB *, int , IFvalue *);
int DCOsetParm( CKTcircuit  *, JOB *, int , IFvalue *);
int DCTaskQuest( CKTcircuit *, JOB *, int , IFvalue *);
int DCTsetParm( CKTcircuit  *, JOB *, int , IFvalue *);
int DCop( CKTcircuit *, int );
int DCtrCurv( CKTcircuit *, int );
int DCtran( CKTcircuit *, int );
int DISTOan(CKTcircuit *, int);
int NOISEan(CKTcircuit *, int);
int PZan( CKTcircuit *, int );
int PZinit( CKTcircuit * );
int PZpost( CKTcircuit * );
int PZaskQuest( CKTcircuit *, JOB *, int , IFvalue *);
int PZsetParm( CKTcircuit *, JOB *, int , IFvalue *);
int SENaskQuest( CKTcircuit *, JOB *, int , IFvalue *);
void SENdestroy( SENstruct *);
int SENsetParm( CKTcircuit *, JOB *, int , IFvalue *);
int SENstartup( CKTcircuit *);
int SPIinit( IFfrontEnd *, IFsimulator **);
char * SPerror( int );
int TFanal( CKTcircuit *, int );
int TFaskQuest( CKTcircuit *, JOB *, int , IFvalue *);
int TFsetParm( CKTcircuit *, JOB *, int , IFvalue *);
int TRANaskQuest( CKTcircuit *, JOB *, int , IFvalue *);
int TRANsetParm( CKTcircuit *, JOB *, int , IFvalue *);
int TRANinit(CKTcircuit *, JOB *);
int NIacIter( CKTcircuit * );
int NIcomCof( CKTcircuit * ); 
int NIconvTest(CKTcircuit * );
void NIdestroy(CKTcircuit * );
int NIinit( CKTcircuit  * );
int NIintegrate( CKTcircuit *, double *, double *, double , int );
int NIiter( CKTcircuit * , int );
int NIpzMuller(PZtrial **, PZtrial *);
int NIpzComplex(PZtrial **, PZtrial *);
int NIpzSym(PZtrial **, PZtrial *);
int NIpzSym2(PZtrial **, PZtrial *);
int NIreinit( CKTcircuit *);
int NIsenReinit( CKTcircuit *);
extern IFfrontEnd *SPfrontEnd;

#endif /*CKT*/
