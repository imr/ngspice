/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000  AlansFixes
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "sperror.h"
#include "ifsim.h"


int
DCop(CKTcircuit *ckt, int notused)
{
    int CKTload(CKTcircuit *ckt);
    int converged;
    int error;
    IFuid *nameList; /* va: tmalloc'ed list */
    int numNames;
    void *plot = NULL;
    
    error = CKTnames(ckt,&numNames,&nameList);
    if(error) return(error);
    error = (*(SPfrontEnd->OUTpBeginPlot))((void *)ckt,
	(void*)ckt->CKTcurJob, ckt->CKTcurJob->JOBname,
	(IFuid)NULL,IF_REAL,numNames,nameList, IF_REAL,&plot);
    tfree(nameList); /* va: nameList not used any longer, it was a memory leak */
    if(error) return(error);

    converged = CKTop(ckt,
            (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITJCT,
            (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITFLOAT,
            ckt->CKTdcMaxIter);
     if(converged != 0) {
     	fprintf(stdout,"\nDC solution failed -\n");
     	CKTncDump(ckt);
/*
           CKTnode *node;
           double new, old, tol;
           int i=1;

           fprintf(stdout,"\nDC solution failed -\n\n");
           fprintf(stdout,"Last Node Voltages\n");
           fprintf(stdout,"------------------\n\n");
           fprintf(stdout,"%-30s %20s %20s\n", "Node", "Last Voltage",
                                                              "Previous Iter");
           fprintf(stdout,"%-30s %20s %20s\n", "----", "------------",
                                                              "-------------");
           for(node=ckt->CKTnodes->next;node;node=node->next) {
             if (strstr(node->name, "#branch") || !strstr(node->name, "#")) {
               new =  *((ckt->CKTrhsOld) + i ) ;
               old =  *((ckt->CKTrhs) + i ) ;
               fprintf(stdout,"%-30s %20g %20g", node->name, new, old);
               if(node->type == 3) {
                   tol =  ckt->CKTreltol * (MAX(fabs(old),fabs(new))) +
                           ckt->CKTvoltTol;
               } else {
                   tol =  ckt->CKTreltol * (MAX(fabs(old),fabs(new))) +
                           ckt->CKTabstol;
               }
               if (fabs(new-old) >tol ) {
                    fprintf(stdout," *");
               }
               fprintf(stdout,"\n");
             };
             i++;
           };
           fprintf(stdout,"\n");
	   (*(SPfrontEnd->OUTendPlot))(plot); */
	   
	   return(converged);
	 };

    ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITSMSIG;


#ifdef WANT_SENSE2
    if(ckt->CKTsenInfo && ((ckt->CKTsenInfo->SENmode&DCSEN) || 
            (ckt->CKTsenInfo->SENmode&ACSEN)) ){
#ifdef SENSDEBUG
         printf("\nDC Operating Point Sensitivity Results\n\n");
         CKTsenPrint(ckt);
#endif /* SENSDEBUG */
         senmode = ckt->CKTsenInfo->SENmode;
         save = ckt->CKTmode;
         ckt->CKTsenInfo->SENmode = DCSEN;
         size = SMPmatSize(ckt->CKTmatrix);
         for(i = 1; i<=size ; i++){
             *(ckt->CKTrhsOp + i) = *(ckt->CKTrhsOld + i);
         }
         if(error = CKTsenDCtran(ckt)) return(error);
         ckt->CKTmode = save;
         ckt->CKTsenInfo->SENmode = senmode;

    }
#endif
    converged = CKTload(ckt);
    if(converged == 0) {
	   CKTdump(ckt,(double)0,plot);
         } else {
           fprintf(stderr,"error: circuit reload failed.\n");
         };
    (*(SPfrontEnd->OUTendPlot))(plot);
    return(converged);
}
