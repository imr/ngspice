/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine sets model parameters for NDEVs in the circuit.
 */

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "ndevdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
NDEVmParam(int param, IFvalue * value, GENmodel * inModel)
{
  
  NDEVmodel *model = (NDEVmodel *) inModel;
  switch (param) {
  case NDEV_REMOTE:
    model->host = value->sValue;
    break;
  case NDEV_PORT:
    model->port = value->iValue;
    break;      
  case NDEV_MOD_NDEV:
    /* no action , but this */
    /* makes life easier for spice-2 like parsers */
    break;
  default:
    return (E_BADPARM);
  }
      
  return (OK);
}
