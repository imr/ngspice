/*
 * Copyright (c) 1985 Thomas L. Quarles
 * Modified 1999 Paolo Nenzi - Removed non STDC definitions
 * Kept only prototypes (structs defined in struct.h) ER
 */


#ifndef CKT_H_INCLUDED
#define CKT_H_INCLUDED


/* function prototypes */

int ACan( CKTcircuit *, int );
int ACaskQuest( CKTcircuit *, void *, int , IFvalue *);
int ACsetParm( CKTcircuit *, void *, int , IFvalue *);
int CKTacDump( CKTcircuit *, double , void *);
int CKTacLoad( CKTcircuit *);
int CKTaccept( CKTcircuit *);
int CKTacct( CKTcircuit *, void *, int , IFvalue *);
int CKTask( CKTcircuit *, GENinstance *, int , IFvalue *, IFvalue *);
int CKTaskAnalQ( CKTcircuit *, void *, int , IFvalue *, IFvalue *);
int CKTaskNodQst( CKTcircuit *, void *, int , IFvalue *, IFvalue *);
int CKTbindNode( CKTcircuit *, void *, int , void *);
void CKTbreakDump( CKTcircuit *);
int CKTclrBreak( CKTcircuit *);
int CKTconvTest( CKTcircuit *);
int CKTcrtElt( CKTcircuit *, GENmodel *, GENinstance **, IFuid );
int CKTdelTask( CKTcircuit *, void *);
int CKTdestroy( CKTcircuit *);
int CKTdltAnal( void *, void *, void *);
int CKTdltInst( CKTcircuit *, void *);
int CKTdltMod( CKTcircuit *, GENmodel *);
int CKTdltNod( CKTcircuit *, void *);
int CKTdoJob( CKTcircuit *, int , void *);
void CKTdump( CKTcircuit *, double, void *);
int CKTfndAnal( CKTcircuit *, int *, void **, IFuid , void *, IFuid );
int CKTfndBranch( CKTcircuit *, IFuid);
int CKTfndDev( CKTcircuit *, int *, GENinstance **, IFuid , GENmodel *, IFuid );
int CKTfndMod( CKTcircuit *, int *, GENmodel **, IFuid );
int CKTfndNode( CKTcircuit *, void **, IFuid );
int CKTfndTask( CKTcircuit *, void **, IFuid  );
int CKTground( CKTcircuit *, void **, IFuid );
int CKTic( CKTcircuit *);
int CKTinit( CKTcircuit **);
int CKTinst2Node( CKTcircuit *, void *, int , CKTnode **, IFuid *);
int CKTlinkEq(CKTcircuit*,CKTnode*);
int CKTload( CKTcircuit *);
int CKTmapNode( CKTcircuit *, void **, IFuid );
int CKTmkCur( CKTcircuit  *, CKTnode **, IFuid , char *);
int CKTmkNode(CKTcircuit*,CKTnode**);
int CKTmkVolt( CKTcircuit  *, CKTnode **, IFuid , char *);
int CKTmodAsk( CKTcircuit *, GENmodel *, int , IFvalue *, IFvalue *);
int CKTmodCrt( CKTcircuit *, int , GENmodel **, IFuid );
int CKTmodParam( CKTcircuit *, GENmodel *, int , IFvalue *, IFvalue *);
int CKTnames(CKTcircuit *, int *, IFuid **);
int CKTnewAnal( CKTcircuit *, int , IFuid , void **, void *);
int CKTnewEq( CKTcircuit *, void **, IFuid );
int CKTnewNode( CKTcircuit *, void **, IFuid );
int CKTnewTask( CKTcircuit *, void **, IFuid );
IFuid CKTnodName( CKTcircuit *, int );
void CKTnodOut( CKTcircuit *);
CKTnode * CKTnum2nod( CKTcircuit *, int );
int CKTop(CKTcircuit *, long, long, int );
int CKTpModName( char *, IFvalue *, CKTcircuit *, int , IFuid , GENmodel **);
int CKTpName( char *, IFvalue *, CKTcircuit *, int , char *, GENinstance **);
int CKTparam( CKTcircuit *, void *, int , IFvalue *, IFvalue *);
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
int CKTsetAnalPm( CKTcircuit *, void *, int , IFvalue *, IFvalue *);
int CKTsetBreak( CKTcircuit *, double );
int CKTsetNodPm( CKTcircuit *, void *, int , IFvalue *, IFvalue *);
int CKTsetOpt( CKTcircuit *, void *, int , IFvalue *);
int CKTsetup( CKTcircuit *);
int CKTunsetup(CKTcircuit *ckt); 
int CKTtemp( CKTcircuit *);
char *CKTtrouble(CKTcircuit *, char *);
void CKTterr( int , CKTcircuit *, double *);
int CKTtrunc( CKTcircuit *, double *);
int CKTtypelook( char *);
int DCOaskQuest( CKTcircuit *, void *, int , IFvalue *);
int DCOsetParm( CKTcircuit  *, void *, int , IFvalue *);
int DCTaskQuest( CKTcircuit *, void *, int , IFvalue *);
int DCTsetParm( CKTcircuit  *, void *, int , IFvalue *);
int DCop( CKTcircuit *, int );
int DCtrCurv( CKTcircuit *, int );
int DCtran( CKTcircuit *, int );
int DISTOan(CKTcircuit *, int);
int NOISEan(CKTcircuit *, int);
int PZan( CKTcircuit *, int );
int PZinit( CKTcircuit * );
int PZpost( CKTcircuit * );
int PZaskQuest( CKTcircuit *, void *, int , IFvalue *);
int PZsetParm( CKTcircuit *, void *, int , IFvalue *);
int SENaskQuest( CKTcircuit *, void *, int , IFvalue *);
void SENdestroy( SENstruct *);
int SENsetParm( CKTcircuit *, void *, int , IFvalue *);
int SENstartup( CKTcircuit *);
int SPIinit( IFfrontEnd *, IFsimulator **);
char * SPerror( int );
int TFanal( CKTcircuit *, int );
int TFaskQuest( CKTcircuit *, void *, int , IFvalue *);
int TFsetParm( CKTcircuit *, void *, int , IFvalue *);
int TRANaskQuest( CKTcircuit *, void *, int , IFvalue *);
int TRANsetParm( CKTcircuit *, void *, int , IFvalue *);
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
IFfrontEnd *SPfrontEnd;

#endif /*CKT*/
