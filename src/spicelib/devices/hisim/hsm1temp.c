/***********************************************************************
 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2003 STARC

 VERSION : HiSIM 1.2.0
 FILE : hsm1temp.c of HiSIM 1.2.0

 April 9, 2003 : released by STARC Physical Design Group
***********************************************************************/

#include "ngspice.h"
#include "smpdefs.h"
#include "cktdefs.h"
#include "hsm1def.h"
#include "const.h"
#include "sperror.h"
#include "suffix.h"

int 
HSM1temp(GENmodel *inModel, CKTcircuit *ckt)
{
  /* "ckt->CKTtemp" dependence of HiSIM parameters is treated all in
   * HSM1evaluate102/112/120(). So there is no task in HSM1temp().
   */
   
   /* PN:
    * Hope the temp dependence treated in the evaluate function does
    * not break the parallel code. Parallel code structure here suggests:
     
   if (here->HSM1owner != ARCHme)
                      continue;
   */ 
   
  return(OK);
}
