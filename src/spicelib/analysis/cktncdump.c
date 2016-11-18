/**********
Copyright 1999 AG inc.  All rights reserved.
Author: 1999 Alan Gillespie
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/suffix.h"

void
CKTncDump(
    CKTcircuit *ckt)
{
  CKTnode *node;
  double new, old, tol;
  int i=1;

  fprintf(stdout,"\n");
  fprintf(stdout,"Last Node Voltages\n");
  fprintf(stdout,"------------------\n\n");
  fprintf(stdout,"%-30s %20s %20s\n", "Node", "Last Voltage", "Previous Iter");
  fprintf(stdout,"%-30s %20s %20s\n", "----", "------------", "-------------");
  for(node=ckt->CKTnodes->next;node;node=node->next) {
    if (strstr(node->name, "#branch") || !strchr(node->name, '#')) {
      new =  ckt->CKTrhsOld [i] ;
      old =  ckt->CKTrhs [i] ;
      fprintf(stdout,"%-30s %20g %20g", node->name, new, old);
      if(node->type == SP_VOLTAGE) {
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
    }
    i++;
  }
  fprintf(stdout,"\n");
}
