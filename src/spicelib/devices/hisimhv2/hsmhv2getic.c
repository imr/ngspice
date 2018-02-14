/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2014 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 2  SUBVERSION : 2  REVISION : 0 ) 
 Model Parameter 'VERSION' : 2.20
 FILE : hsmhvgetic.c

 DATE : 2014.6.11

 released by 
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

/**********************************************************************

The following source code, and all copyrights, trade secrets or other
intellectual property rights in and to the source code in its entirety,
is owned by the Hiroshima University and the STARC organization.

All users need to follow the "HISIM_HV Distribution Statement and
Copyright Notice" attached to HiSIM_HV model.

-----HISIM_HV Distribution Statement and Copyright Notice--------------

Software is distributed as is, completely without warranty or service
support. Hiroshima University or STARC and its employees are not liable
for the condition or performance of the software.

Hiroshima University and STARC own the copyright and grant users a perpetual,
irrevocable, worldwide, non-exclusive, royalty-free license with respect 
to the software as set forth below.   

Hiroshima University and STARC hereby disclaims all implied warranties.

Hiroshima University and STARC grant the users the right to modify, copy,
and redistribute the software and documentation, both within the user's
organization and externally, subject to the following restrictions

1. The users agree not to charge for Hiroshima University and STARC code
itself but may charge for additions, extensions, or support.

2. In any product based on the software, the users agree to acknowledge
Hiroshima University and STARC that developed the software. This
acknowledgment shall appear in the product documentation.

3. The users agree to reproduce any copyright notice which appears on
the software on any copy or modification of such made available
to others."

Toshimasa Asahara, President, Hiroshima University
Mitiko Miura-Mattausch, Professor, Hiroshima University
Katsuhiro Shimohigashi, President&CEO, STARC
June 2008 (revised October 2011) 
*************************************************************************/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "hsmhv2def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int HSMHV2getic(
     GENmodel *inModel,
     CKTcircuit *ckt)
{
  HSMHV2model *model = (HSMHV2model*)inModel;
  HSMHV2instance *here;
  /*
   * grab initial conditions out of rhs array.   User specified, so use
   * external nodes to get values
   */

  for ( ;model ;model = HSMHV2nextModel(model)) {
    for ( here = HSMHV2instances(model); here ;here = HSMHV2nextInstance(here)) {
      if (!here->HSMHV2_icVBS_Given) {
	here->HSMHV2_icVBS = 
	  *(ckt->CKTrhs + here->HSMHV2bNode) - 
	  *(ckt->CKTrhs + here->HSMHV2sNode);
      }
      if (!here->HSMHV2_icVDS_Given) {
	here->HSMHV2_icVDS = 
	  *(ckt->CKTrhs + here->HSMHV2dNode) - 
	  *(ckt->CKTrhs + here->HSMHV2sNode);
      }
      if (!here->HSMHV2_icVGS_Given) {
	here->HSMHV2_icVGS = 
	  *(ckt->CKTrhs + here->HSMHV2gNode) - 
	  *(ckt->CKTrhs + here->HSMHV2sNode);
      }
    }
  }
  return(OK);
}

