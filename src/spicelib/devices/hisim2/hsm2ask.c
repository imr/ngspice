/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2014 Hiroshima University & STARC

 MODEL NAME : HiSIM
 ( VERSION : 2  SUBVERSION : 8  REVISION : 0 )
 
 FILE : hsm2ask.c

 Date : 2014.6.5

 released by 
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

/**********************************************************************

The following source code, and all copyrights, trade secrets or other
intellectual property rights in and to the source code in its entirety,
is owned by the Hiroshima University and the STARC organization.

All users need to follow the "HiSIM2 Distribution Statement and
Copyright Notice" attached to HiSIM2 model.

-----HiSIM2 Distribution Statement and Copyright Notice--------------

Software is distributed as is, completely without warranty or service
support. Hiroshima University or STARC and its employees are not liable
for the condition or performance of the software.

Hiroshima University and STARC own the copyright and grant users a perpetual,
irrevocable, worldwide, non-exclusive, royalty-free license with respect 
to the software as set forth below.   

Hiroshima University and STARC hereby disclaim all implied warranties.

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


*************************************************************************/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "hsm2def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int HSM2ask(
     CKTcircuit *ckt,
     GENinstance *inst,
     int which,
     IFvalue *value,
     IFvalue *select)
{
  HSM2instance *here = (HSM2instance*)inst;

  NG_IGNORE(select);

  switch (which) {
  case HSM2_L:
    value->rValue = here->HSM2_l;
    return(OK);
  case HSM2_W:
    value->rValue = here->HSM2_w;
    return(OK);
  case HSM2_AS:
    value->rValue = here->HSM2_as;
    return(OK);
  case HSM2_AD:
    value->rValue = here->HSM2_ad;
    return(OK);
  case HSM2_PS:
    value->rValue = here->HSM2_ps;
    return(OK);
  case HSM2_PD:
    value->rValue = here->HSM2_pd;
    return(OK);
  case HSM2_NRS:
    value->rValue = here->HSM2_nrs;
    return(OK);
  case HSM2_NRD:
    value->rValue = here->HSM2_nrd;
    return(OK);
  case HSM2_TEMP:
    value->rValue = here->HSM2_temp;
    return(OK);
  case HSM2_DTEMP:
    value->rValue = here->HSM2_dtemp;
    return(OK);
  case HSM2_OFF:
    value->iValue = here->HSM2_off;
    return(OK);
  case HSM2_IC_VBS:
    value->rValue = here->HSM2_icVBS;
    return(OK);
  case HSM2_IC_VDS:
    value->rValue = here->HSM2_icVDS;
    return(OK);
  case HSM2_IC_VGS:
    value->rValue = here->HSM2_icVGS;
    return(OK);
  case HSM2_DNODE:
    value->iValue = here->HSM2dNode;
    return(OK);
  case HSM2_GNODE:
    value->iValue = here->HSM2gNode;
    return(OK);
  case HSM2_SNODE:
    value->iValue = here->HSM2sNode;
    return(OK);
  case HSM2_BNODE:
    value->iValue = here->HSM2bNode;
    return(OK);
  case HSM2_DNODEPRIME:
    value->iValue = here->HSM2dNodePrime;
    return(OK);
  case HSM2_SNODEPRIME:
    value->iValue = here->HSM2sNodePrime;
    return(OK);
  case HSM2_SOURCECONDUCT:
    value->rValue = here->HSM2sourceConductance;
    return(OK);
  case HSM2_DRAINCONDUCT:
    value->rValue = here->HSM2drainConductance;
    return(OK);
  case HSM2_VBD:
    value->rValue = *(ckt->CKTstate0 + here->HSM2vbd);
    return(OK);
  case HSM2_VBS:
    value->rValue = *(ckt->CKTstate0 + here->HSM2vbs);
    return(OK);
  case HSM2_VGS:
    value->rValue = *(ckt->CKTstate0 + here->HSM2vgs);
    return(OK);
  case HSM2_VDS:
    value->rValue = *(ckt->CKTstate0 + here->HSM2vds);
    return(OK);
  case HSM2_CD:
    value->rValue = here->HSM2_ids;
    return(OK);
  case HSM2_ISUB:
    value->rValue = here->HSM2_isub;
    return(OK);
  case HSM2_IGIDL:
    value->rValue = here->HSM2_igidl;
    return(OK);
  case HSM2_IGISL:
    value->rValue = here->HSM2_igisl;
    return(OK);
  case HSM2_IGD:
    value->rValue = here->HSM2_igd;
    return(OK);
  case HSM2_IGS:
    value->rValue = here->HSM2_igs;
    return(OK);
  case HSM2_IGB:
    value->rValue = here->HSM2_igb;
    return(OK);
  case HSM2_CBS:
    value->rValue = here->HSM2_ibs;
    return(OK);
  case HSM2_CBD:
    value->rValue = here->HSM2_ibd;
    return(OK);
  case HSM2_GM:
    value->rValue = here->HSM2_gm;
    return(OK);
  case HSM2_GDS:
    value->rValue = here->HSM2_gds;
    return(OK);
  case HSM2_GMBS:
    value->rValue = here->HSM2_gmbs;
    return(OK);
  case HSM2_GBD:
    value->rValue = here->HSM2_gbd;
    return(OK);
  case HSM2_GBS:
    value->rValue = here->HSM2_gbs;
    return(OK);
  case HSM2_QB:
    value->rValue = *(ckt->CKTstate0 + here->HSM2qb); 
    return(OK);
  case HSM2_CQB:
    value->rValue = *(ckt->CKTstate0 + here->HSM2cqb); 
    return(OK);
  case HSM2_QG:
    value->rValue = *(ckt->CKTstate0 + here->HSM2qg); 
    return(OK);
  case HSM2_CQG:
    value->rValue = *(ckt->CKTstate0 + here->HSM2cqg); 
    return(OK);
  case HSM2_QD:
    value->rValue = *(ckt->CKTstate0 + here->HSM2qd); 
    return(OK);
  case HSM2_CQD:
    value->rValue = *(ckt->CKTstate0 + here->HSM2cqd); 
    return(OK);
  case HSM2_CGG:
    value->rValue = here->HSM2_cggb; 
    return(OK);
  case HSM2_CGD:
    value->rValue = here->HSM2_cgdb;
    return(OK);
  case HSM2_CGS:
    value->rValue = here->HSM2_cgsb;
    return(OK);
  case HSM2_CDG:
    value->rValue = here->HSM2_cdgb; 
    return(OK);
  case HSM2_CDD:
    value->rValue = here->HSM2_cddb; 
    return(OK);
  case HSM2_CDS:
    value->rValue = here->HSM2_cdsb; 
    return(OK);
  case HSM2_CBG:
    value->rValue = here->HSM2_cbgb;
    return(OK);
  case HSM2_CBDB:
    value->rValue = here->HSM2_cbdb;
    return(OK);
  case HSM2_CBSB:
    value->rValue = here->HSM2_cbsb;
    return(OK);
  case HSM2_CGDO:
    value->rValue = here->HSM2_cgdo;
    return(OK);
  case HSM2_CGSO:
    value->rValue = here->HSM2_cgso;
    return(OK);
  case HSM2_CGBO:
    value->rValue = here->HSM2_cgbo;
    return(OK);
  case HSM2_CAPBD:
    value->rValue = here->HSM2_capbd; 
    return(OK);
  case HSM2_CAPBS:
    value->rValue = here->HSM2_capbs;
    return(OK);
  case HSM2_VON:
    value->rValue = here->HSM2_von; 
    return(OK);
  case HSM2_VDSAT:
    value->rValue = here->HSM2_vdsat; 
    return(OK);
  case HSM2_QBS:
    value->rValue = *(ckt->CKTstate0 + here->HSM2qbs); 
    return(OK);
  case HSM2_QBD:
    value->rValue = *(ckt->CKTstate0 + here->HSM2qbd); 
    return(OK);
  case  HSM2_CORBNET: 
    value->iValue = here->HSM2_corbnet;
    return(OK);
  case  HSM2_RBPB:
    value->rValue = here->HSM2_rbpb;
    return (OK);
  case  HSM2_RBPD:
    value->rValue = here->HSM2_rbpd;
    return(OK);
  case  HSM2_RBPS:
    value->rValue = here->HSM2_rbps;
    return(OK);
  case  HSM2_RBDB:
    value->rValue = here->HSM2_rbdb;
    return(OK);
  case  HSM2_RBSB:
    value->rValue = here->HSM2_rbsb;
    return(OK);
  case  HSM2_CORG: 
    value->iValue = here->HSM2_corg;
    return(OK);
/*   case  HSM2_RSHG: */
/*     value->rValue = here->HSM2_rshg; */
/*     return(OK); */
  case  HSM2_NGCON:
    value->rValue = here->HSM2_ngcon;
    return(OK);
  case  HSM2_XGW:
    value->rValue = here->HSM2_xgw;
    return(OK);
  case  HSM2_XGL:
    value->rValue = here->HSM2_xgl;
    return(OK);
  case  HSM2_NF:
    value->rValue = here->HSM2_nf;
    return(OK);
  case  HSM2_SA:
    value->rValue = here->HSM2_sa;
    return(OK);
  case  HSM2_SB:
    value->rValue = here->HSM2_sb;
    return(OK);
  case  HSM2_SD:
    value->rValue = here->HSM2_sd;
    return(OK);
  case  HSM2_NSUBCDFM:
    value->rValue = here->HSM2_nsubcdfm;
    return(OK);
  case  HSM2_MPHDFM:
    value->rValue = here->HSM2_mphdfm;
    return(OK);
  case  HSM2_M:
    value->rValue = here->HSM2_m;
    return(OK);
/* WPE */
  case HSM2_SCA:
	value->rValue = here->HSM2_sca;
	return(OK);
  case HSM2_SCB:
    value->rValue = here->HSM2_scb;
	return(OK);
  case HSM2_SCC:
    value->rValue = here->HSM2_scc;
	return(OK);
  default:
    return(E_BADPARM);
  }
  /* NOTREACHED */
}
