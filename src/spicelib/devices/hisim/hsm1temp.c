/***********************************************************************
 HiSIM v1.1.0
 File: hsm1temp.c of HiSIM v1.1.0

 Copyright (C) 2002 STARC

 June 30, 2002: developed by Hiroshima University and STARC
 June 30, 2002: posted by Keiichi MORIKAWA, STARC Physical Design Group
***********************************************************************/

/*
 * Modified by Paolo Nenzi 2002
 * ngspice integration
 */

#include "ngspice.h"
#include "smpdefs.h"
#include "cktdefs.h"
#include "hsm1def.h"
#include "const.h"
#include "sperror.h"
#include "suffix.h"

int HSM1temp(GENmodel *inModel, CKTcircuit *ckt)
{
  /* "ckt->CKTtemp" dependence of HiSIM parameters is treated all in
   * HSM1evaluate1_0/1_1(). So there is no task in HSM1temp().
   */
  
  /* 
   if (here->HSM1owner != ARCHme)
                      continue;
   */
  return(OK);
}
