/**** BSIM4.6.2 Released by Wenwei Yang 07/31/2008 ****/

/**********
 * Copyright 2006 Regents of the University of California. All rights reserved.
 * File: b4mask.c of BSIM4.6.2.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Authors: 2006- Mohan Dunga, Ali Niknejad, Chenming Hu
 * Authors: 2007- Mohan Dunga, Wenwei Yang, Ali Niknejad, Chenming Hu
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 * Modified by Xuemei Xi, 11/15/2002.
 * Modified by Xuemei Xi, 05/09/2003.
 * Modified by Xuemei Xi, Mohan Dunga, 07/29/2005.
 * Modified by Mohan Dunga, 12/13/2006
 * Modified by Mohan Dunga, Wenwei Yang, 07/31/2008.
 **********/


#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "bsim4v6def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
BSIM4v6mAsk(
CKTcircuit *ckt,
GENmodel *inst,
int which,
IFvalue *value)
{
    BSIM4v6model *model = (BSIM4v6model *)inst;

    NG_IGNORE(ckt);

    switch(which) 
    {   case BSIM4v6_MOD_MOBMOD :
            value->iValue = model->BSIM4v6mobMod; 
            return(OK);
        case BSIM4v6_MOD_PARAMCHK :
            value->iValue = model->BSIM4v6paramChk; 
            return(OK);
        case BSIM4v6_MOD_BINUNIT :
            value->iValue = model->BSIM4v6binUnit; 
            return(OK);
        case BSIM4v6_MOD_CVCHARGEMOD :
            value->iValue = model->BSIM4v6cvchargeMod; 
            return(OK);
        case BSIM4v6_MOD_CAPMOD :
            value->iValue = model->BSIM4v6capMod; 
            return(OK);
        case BSIM4v6_MOD_DIOMOD :
            value->iValue = model->BSIM4v6dioMod;
            return(OK);
        case BSIM4v6_MOD_TRNQSMOD :
            value->iValue = model->BSIM4v6trnqsMod;
            return(OK);
        case BSIM4v6_MOD_ACNQSMOD :
            value->iValue = model->BSIM4v6acnqsMod;
            return(OK);
        case BSIM4v6_MOD_FNOIMOD :
            value->iValue = model->BSIM4v6fnoiMod; 
            return(OK);
        case BSIM4v6_MOD_TNOIMOD :
            value->iValue = model->BSIM4v6tnoiMod;
            return(OK);
        case BSIM4v6_MOD_RDSMOD :
            value->iValue = model->BSIM4v6rdsMod;
            return(OK);
        case BSIM4v6_MOD_RBODYMOD :
            value->iValue = model->BSIM4v6rbodyMod;
            return(OK);
        case BSIM4v6_MOD_RGATEMOD :
            value->iValue = model->BSIM4v6rgateMod;
            return(OK);
        case BSIM4v6_MOD_PERMOD :
            value->iValue = model->BSIM4v6perMod;
            return(OK);
        case BSIM4v6_MOD_GEOMOD :
            value->iValue = model->BSIM4v6geoMod;
            return(OK);
        case BSIM4v6_MOD_RGEOMOD :
            value->iValue = model->BSIM4v6rgeoMod;
            return(OK);
        case BSIM4v6_MOD_MTRLMOD :
            value->iValue = model->BSIM4v6mtrlMod;
            return(OK);

        case BSIM4v6_MOD_IGCMOD :
            value->iValue = model->BSIM4v6igcMod;
            return(OK);
        case BSIM4v6_MOD_IGBMOD :
            value->iValue = model->BSIM4v6igbMod;
            return(OK);
        case  BSIM4v6_MOD_TEMPMOD :
            value->iValue = model->BSIM4v6tempMod;
            return(OK);

        case  BSIM4v6_MOD_VERSION :
          value->sValue = model->BSIM4v6version;
            return(OK);
        case  BSIM4v6_MOD_TOXREF :
          value->rValue = model->BSIM4v6toxref;
          return(OK);
        case  BSIM4v6_MOD_EOT :
          value->rValue = model->BSIM4v6eot;
            return(OK);
        case  BSIM4v6_MOD_VDDEOT :
          value->rValue = model->BSIM4v6vddeot;
            return(OK);
        case  BSIM4v6_MOD_TEMPEOT :
          value->rValue = model->BSIM4v6tempeot;
            return(OK);
        case  BSIM4v6_MOD_LEFFEOT :
          value->rValue = model->BSIM4v6leffeot;
            return(OK);
        case  BSIM4v6_MOD_WEFFEOT :
          value->rValue = model->BSIM4v6weffeot;
            return(OK);
        case  BSIM4v6_MOD_ADOS :
          value->rValue = model->BSIM4v6ados;
            return(OK);
        case  BSIM4v6_MOD_BDOS :
          value->rValue = model->BSIM4v6bdos;
            return(OK);
        case  BSIM4v6_MOD_TOXE :
          value->rValue = model->BSIM4v6toxe;
            return(OK);
        case  BSIM4v6_MOD_TOXP :
          value->rValue = model->BSIM4v6toxp;
            return(OK);
        case  BSIM4v6_MOD_TOXM :
          value->rValue = model->BSIM4v6toxm;
            return(OK);
        case  BSIM4v6_MOD_DTOX :
          value->rValue = model->BSIM4v6dtox;
            return(OK);
        case  BSIM4v6_MOD_EPSROX :
          value->rValue = model->BSIM4v6epsrox;
            return(OK);
        case  BSIM4v6_MOD_CDSC :
          value->rValue = model->BSIM4v6cdsc;
            return(OK);
        case  BSIM4v6_MOD_CDSCB :
          value->rValue = model->BSIM4v6cdscb;
            return(OK);

        case  BSIM4v6_MOD_CDSCD :
          value->rValue = model->BSIM4v6cdscd;
            return(OK);

        case  BSIM4v6_MOD_CIT :
          value->rValue = model->BSIM4v6cit;
            return(OK);
        case  BSIM4v6_MOD_NFACTOR :
          value->rValue = model->BSIM4v6nfactor;
            return(OK);
        case BSIM4v6_MOD_XJ:
            value->rValue = model->BSIM4v6xj;
            return(OK);
        case BSIM4v6_MOD_VSAT:
            value->rValue = model->BSIM4v6vsat;
            return(OK);
        case BSIM4v6_MOD_VTL:
            value->rValue = model->BSIM4v6vtl;
            return(OK);
        case BSIM4v6_MOD_XN:
            value->rValue = model->BSIM4v6xn;
            return(OK);
        case BSIM4v6_MOD_LC:
            value->rValue = model->BSIM4v6lc;
            return(OK);
        case BSIM4v6_MOD_LAMBDA:
            value->rValue = model->BSIM4v6lambda;
            return(OK);
        case BSIM4v6_MOD_AT:
            value->rValue = model->BSIM4v6at;
            return(OK);
        case BSIM4v6_MOD_A0:
            value->rValue = model->BSIM4v6a0;
            return(OK);

        case BSIM4v6_MOD_AGS:
            value->rValue = model->BSIM4v6ags;
            return(OK);

        case BSIM4v6_MOD_A1:
            value->rValue = model->BSIM4v6a1;
            return(OK);
        case BSIM4v6_MOD_A2:
            value->rValue = model->BSIM4v6a2;
            return(OK);
        case BSIM4v6_MOD_KETA:
            value->rValue = model->BSIM4v6keta;
            return(OK);   
        case BSIM4v6_MOD_NSUB:
            value->rValue = model->BSIM4v6nsub;
            return(OK);
        case BSIM4v6_MOD_PHIG:
	    value->rValue = model->BSIM4v6phig;
	    return(OK);
        case BSIM4v6_MOD_EPSRGATE:
	    value->rValue = model->BSIM4v6epsrgate;
	    return(OK);
        case BSIM4v6_MOD_EASUB:
            value->rValue = model->BSIM4v6easub;
            return(OK);
        case BSIM4v6_MOD_EPSRSUB:
            value->rValue = model->BSIM4v6epsrsub;
            return(OK);
        case BSIM4v6_MOD_NI0SUB:
            value->rValue = model->BSIM4v6ni0sub;
            return(OK);
        case BSIM4v6_MOD_BG0SUB:
            value->rValue = model->BSIM4v6bg0sub;
            return(OK);
        case BSIM4v6_MOD_TBGASUB:
            value->rValue = model->BSIM4v6tbgasub;
            return(OK);
        case BSIM4v6_MOD_TBGBSUB:
            value->rValue = model->BSIM4v6tbgbsub;
            return(OK);
        case BSIM4v6_MOD_NDEP:
            value->rValue = model->BSIM4v6ndep;
            return(OK);
        case BSIM4v6_MOD_NSD:
            value->rValue = model->BSIM4v6nsd;
            return(OK);
        case BSIM4v6_MOD_NGATE:
            value->rValue = model->BSIM4v6ngate;
            return(OK);
        case BSIM4v6_MOD_GAMMA1:
            value->rValue = model->BSIM4v6gamma1;
            return(OK);
        case BSIM4v6_MOD_GAMMA2:
            value->rValue = model->BSIM4v6gamma2;
            return(OK);
        case BSIM4v6_MOD_VBX:
            value->rValue = model->BSIM4v6vbx;
            return(OK);
        case BSIM4v6_MOD_VBM:
            value->rValue = model->BSIM4v6vbm;
            return(OK);
        case BSIM4v6_MOD_XT:
            value->rValue = model->BSIM4v6xt;
            return(OK);
        case  BSIM4v6_MOD_K1:
          value->rValue = model->BSIM4v6k1;
            return(OK);
        case  BSIM4v6_MOD_KT1:
          value->rValue = model->BSIM4v6kt1;
            return(OK);
        case  BSIM4v6_MOD_KT1L:
          value->rValue = model->BSIM4v6kt1l;
            return(OK);
        case  BSIM4v6_MOD_KT2 :
          value->rValue = model->BSIM4v6kt2;
            return(OK);
        case  BSIM4v6_MOD_K2 :
          value->rValue = model->BSIM4v6k2;
            return(OK);
        case  BSIM4v6_MOD_K3:
          value->rValue = model->BSIM4v6k3;
            return(OK);
        case  BSIM4v6_MOD_K3B:
          value->rValue = model->BSIM4v6k3b;
            return(OK);
        case  BSIM4v6_MOD_W0:
          value->rValue = model->BSIM4v6w0;
            return(OK);
        case  BSIM4v6_MOD_LPE0:
          value->rValue = model->BSIM4v6lpe0;
            return(OK);
        case  BSIM4v6_MOD_LPEB:
          value->rValue = model->BSIM4v6lpeb;
            return(OK);
        case  BSIM4v6_MOD_DVTP0:
          value->rValue = model->BSIM4v6dvtp0;
            return(OK);
        case  BSIM4v6_MOD_DVTP1:
          value->rValue = model->BSIM4v6dvtp1;
            return(OK);
        case  BSIM4v6_MOD_DVT0 :                
          value->rValue = model->BSIM4v6dvt0;
            return(OK);
        case  BSIM4v6_MOD_DVT1 :             
          value->rValue = model->BSIM4v6dvt1;
            return(OK);
        case  BSIM4v6_MOD_DVT2 :             
          value->rValue = model->BSIM4v6dvt2;
            return(OK);
        case  BSIM4v6_MOD_DVT0W :                
          value->rValue = model->BSIM4v6dvt0w;
            return(OK);
        case  BSIM4v6_MOD_DVT1W :             
          value->rValue = model->BSIM4v6dvt1w;
            return(OK);
        case  BSIM4v6_MOD_DVT2W :             
          value->rValue = model->BSIM4v6dvt2w;
            return(OK);
        case  BSIM4v6_MOD_DROUT :           
          value->rValue = model->BSIM4v6drout;
            return(OK);
        case  BSIM4v6_MOD_DSUB :           
          value->rValue = model->BSIM4v6dsub;
            return(OK);
        case BSIM4v6_MOD_VTH0:
            value->rValue = model->BSIM4v6vth0; 
            return(OK);
        case BSIM4v6_MOD_EU:
            value->rValue = model->BSIM4v6eu;
            return(OK);
        case BSIM4v6_MOD_UCS:
            value->rValue = model->BSIM4v6ucs;
            return(OK);
        case BSIM4v6_MOD_UA:
            value->rValue = model->BSIM4v6ua; 
            return(OK);
        case BSIM4v6_MOD_UA1:
            value->rValue = model->BSIM4v6ua1; 
            return(OK);
        case BSIM4v6_MOD_UB:
            value->rValue = model->BSIM4v6ub;  
            return(OK);
        case BSIM4v6_MOD_UB1:
            value->rValue = model->BSIM4v6ub1;  
            return(OK);
        case BSIM4v6_MOD_UC:
            value->rValue = model->BSIM4v6uc; 
            return(OK);
        case BSIM4v6_MOD_UC1:
            value->rValue = model->BSIM4v6uc1; 
            return(OK);
        case BSIM4v6_MOD_UD:
            value->rValue = model->BSIM4v6ud; 
            return(OK);
        case BSIM4v6_MOD_UD1:
            value->rValue = model->BSIM4v6ud1; 
            return(OK);
        case BSIM4v6_MOD_UP:
            value->rValue = model->BSIM4v6up; 
            return(OK);
        case BSIM4v6_MOD_LP:
            value->rValue = model->BSIM4v6lp; 
            return(OK);
        case BSIM4v6_MOD_U0:
            value->rValue = model->BSIM4v6u0;
            return(OK);
        case BSIM4v6_MOD_UTE:
            value->rValue = model->BSIM4v6ute;
            return(OK);
        case BSIM4v6_MOD_UCSTE:
            value->rValue = model->BSIM4v6ucste;
            return(OK);
        case BSIM4v6_MOD_VOFF:
            value->rValue = model->BSIM4v6voff;
            return(OK);
        case BSIM4v6_MOD_TVOFF:
            value->rValue = model->BSIM4v6tvoff;
            return(OK);
        case BSIM4v6_MOD_VFBSDOFF:
            value->rValue = model->BSIM4v6vfbsdoff;
            return(OK);
        case BSIM4v6_MOD_TVFBSDOFF:
            value->rValue = model->BSIM4v6tvfbsdoff;
            return(OK);
        case BSIM4v6_MOD_VOFFL:
            value->rValue = model->BSIM4v6voffl;
            return(OK);
        case BSIM4v6_MOD_VOFFCVL:
            value->rValue = model->BSIM4v6voffcvl;
            return(OK);
        case BSIM4v6_MOD_MINV:
            value->rValue = model->BSIM4v6minv;
            return(OK);
        case BSIM4v6_MOD_MINVCV:
            value->rValue = model->BSIM4v6minvcv;
            return(OK);
        case BSIM4v6_MOD_FPROUT:
            value->rValue = model->BSIM4v6fprout;
            return(OK);
        case BSIM4v6_MOD_PDITS:
            value->rValue = model->BSIM4v6pdits;
            return(OK);
        case BSIM4v6_MOD_PDITSD:
            value->rValue = model->BSIM4v6pditsd;
            return(OK);
        case BSIM4v6_MOD_PDITSL:
            value->rValue = model->BSIM4v6pditsl;
            return(OK);
        case BSIM4v6_MOD_DELTA:
            value->rValue = model->BSIM4v6delta;
            return(OK);
        case BSIM4v6_MOD_RDSW:
            value->rValue = model->BSIM4v6rdsw; 
            return(OK);
        case BSIM4v6_MOD_RDSWMIN:
            value->rValue = model->BSIM4v6rdswmin;
            return(OK);
        case BSIM4v6_MOD_RDWMIN:
            value->rValue = model->BSIM4v6rdwmin;
            return(OK);
        case BSIM4v6_MOD_RSWMIN:
            value->rValue = model->BSIM4v6rswmin;
            return(OK);
        case BSIM4v6_MOD_RDW:
            value->rValue = model->BSIM4v6rdw;
            return(OK);
        case BSIM4v6_MOD_RSW:
            value->rValue = model->BSIM4v6rsw;
            return(OK);
        case BSIM4v6_MOD_PRWG:
            value->rValue = model->BSIM4v6prwg; 
            return(OK);             
        case BSIM4v6_MOD_PRWB:
            value->rValue = model->BSIM4v6prwb; 
            return(OK);             
        case BSIM4v6_MOD_PRT:
            value->rValue = model->BSIM4v6prt; 
            return(OK);              
        case BSIM4v6_MOD_ETA0:
            value->rValue = model->BSIM4v6eta0; 
            return(OK);               
        case BSIM4v6_MOD_ETAB:
            value->rValue = model->BSIM4v6etab; 
            return(OK);               
        case BSIM4v6_MOD_PCLM:
            value->rValue = model->BSIM4v6pclm; 
            return(OK);               
        case BSIM4v6_MOD_PDIBL1:
            value->rValue = model->BSIM4v6pdibl1; 
            return(OK);               
        case BSIM4v6_MOD_PDIBL2:
            value->rValue = model->BSIM4v6pdibl2; 
            return(OK);               
        case BSIM4v6_MOD_PDIBLB:
            value->rValue = model->BSIM4v6pdiblb; 
            return(OK);               
        case BSIM4v6_MOD_PSCBE1:
            value->rValue = model->BSIM4v6pscbe1; 
            return(OK);               
        case BSIM4v6_MOD_PSCBE2:
            value->rValue = model->BSIM4v6pscbe2; 
            return(OK);               
        case BSIM4v6_MOD_PVAG:
            value->rValue = model->BSIM4v6pvag; 
            return(OK);               
        case BSIM4v6_MOD_WR:
            value->rValue = model->BSIM4v6wr;
            return(OK);
        case BSIM4v6_MOD_DWG:
            value->rValue = model->BSIM4v6dwg;
            return(OK);
        case BSIM4v6_MOD_DWB:
            value->rValue = model->BSIM4v6dwb;
            return(OK);
        case BSIM4v6_MOD_B0:
            value->rValue = model->BSIM4v6b0;
            return(OK);
        case BSIM4v6_MOD_B1:
            value->rValue = model->BSIM4v6b1;
            return(OK);
        case BSIM4v6_MOD_ALPHA0:
            value->rValue = model->BSIM4v6alpha0;
            return(OK);
        case BSIM4v6_MOD_ALPHA1:
            value->rValue = model->BSIM4v6alpha1;
            return(OK);
        case BSIM4v6_MOD_BETA0:
            value->rValue = model->BSIM4v6beta0;
            return(OK);
        case BSIM4v6_MOD_AGIDL:
            value->rValue = model->BSIM4v6agidl;
            return(OK);
        case BSIM4v6_MOD_BGIDL:
            value->rValue = model->BSIM4v6bgidl;
            return(OK);
        case BSIM4v6_MOD_CGIDL:
            value->rValue = model->BSIM4v6cgidl;
            return(OK);
        case BSIM4v6_MOD_EGIDL:
            value->rValue = model->BSIM4v6egidl;
            return(OK);
        case BSIM4v6_MOD_AGISL:
            value->rValue = model->BSIM4v6agisl;
            return(OK);
        case BSIM4v6_MOD_BGISL:
            value->rValue = model->BSIM4v6bgisl;
            return(OK);
        case BSIM4v6_MOD_CGISL:
            value->rValue = model->BSIM4v6cgisl;
            return(OK);
        case BSIM4v6_MOD_EGISL:
            value->rValue = model->BSIM4v6egisl;
            return(OK);
        case BSIM4v6_MOD_AIGC:
            value->rValue = model->BSIM4v6aigc;
            return(OK);
        case BSIM4v6_MOD_BIGC:
            value->rValue = model->BSIM4v6bigc;
            return(OK);
        case BSIM4v6_MOD_CIGC:
            value->rValue = model->BSIM4v6cigc;
            return(OK);
        case BSIM4v6_MOD_AIGSD:
            value->rValue = model->BSIM4v6aigsd;
            return(OK);
        case BSIM4v6_MOD_BIGSD:
            value->rValue = model->BSIM4v6bigsd;
            return(OK);
        case BSIM4v6_MOD_CIGSD:
            value->rValue = model->BSIM4v6cigsd;
            return(OK);
        case BSIM4v6_MOD_AIGS:
            value->rValue = model->BSIM4v6aigs;
            return(OK);
        case BSIM4v6_MOD_BIGS:
            value->rValue = model->BSIM4v6bigs;
            return(OK);
        case BSIM4v6_MOD_CIGS:
            value->rValue = model->BSIM4v6cigs;
            return(OK);
        case BSIM4v6_MOD_AIGD:
            value->rValue = model->BSIM4v6aigd;
            return(OK);
        case BSIM4v6_MOD_BIGD:
            value->rValue = model->BSIM4v6bigd;
            return(OK);
        case BSIM4v6_MOD_CIGD:
            value->rValue = model->BSIM4v6cigd;
            return(OK);
        case BSIM4v6_MOD_AIGBACC:
            value->rValue = model->BSIM4v6aigbacc;
            return(OK);
        case BSIM4v6_MOD_BIGBACC:
            value->rValue = model->BSIM4v6bigbacc;
            return(OK);
        case BSIM4v6_MOD_CIGBACC:
            value->rValue = model->BSIM4v6cigbacc;
            return(OK);
        case BSIM4v6_MOD_AIGBINV:
            value->rValue = model->BSIM4v6aigbinv;
            return(OK);
        case BSIM4v6_MOD_BIGBINV:
            value->rValue = model->BSIM4v6bigbinv;
            return(OK);
        case BSIM4v6_MOD_CIGBINV:
            value->rValue = model->BSIM4v6cigbinv;
            return(OK);
        case BSIM4v6_MOD_NIGC:
            value->rValue = model->BSIM4v6nigc;
            return(OK);
        case BSIM4v6_MOD_NIGBACC:
            value->rValue = model->BSIM4v6nigbacc;
            return(OK);
        case BSIM4v6_MOD_NIGBINV:
            value->rValue = model->BSIM4v6nigbinv;
            return(OK);
        case BSIM4v6_MOD_NTOX:
            value->rValue = model->BSIM4v6ntox;
            return(OK);
        case BSIM4v6_MOD_EIGBINV:
            value->rValue = model->BSIM4v6eigbinv;
            return(OK);
        case BSIM4v6_MOD_PIGCD:
            value->rValue = model->BSIM4v6pigcd;
            return(OK);
        case BSIM4v6_MOD_POXEDGE:
            value->rValue = model->BSIM4v6poxedge;
            return(OK);
        case BSIM4v6_MOD_PHIN:
            value->rValue = model->BSIM4v6phin;
            return(OK);
        case BSIM4v6_MOD_XRCRG1:
            value->rValue = model->BSIM4v6xrcrg1;
            return(OK);
        case BSIM4v6_MOD_XRCRG2:
            value->rValue = model->BSIM4v6xrcrg2;
            return(OK);
        case BSIM4v6_MOD_TNOIA:
            value->rValue = model->BSIM4v6tnoia;
            return(OK);
        case BSIM4v6_MOD_TNOIB:
            value->rValue = model->BSIM4v6tnoib;
            return(OK);
        case BSIM4v6_MOD_RNOIA:
            value->rValue = model->BSIM4v6rnoia;
            return(OK);
        case BSIM4v6_MOD_RNOIB:
            value->rValue = model->BSIM4v6rnoib;
            return(OK);
        case BSIM4v6_MOD_NTNOI:
            value->rValue = model->BSIM4v6ntnoi;
            return(OK);
        case BSIM4v6_MOD_IJTHDFWD:
            value->rValue = model->BSIM4v6ijthdfwd;
            return(OK);
        case BSIM4v6_MOD_IJTHSFWD:
            value->rValue = model->BSIM4v6ijthsfwd;
            return(OK);
        case BSIM4v6_MOD_IJTHDREV:
            value->rValue = model->BSIM4v6ijthdrev;
            return(OK);
        case BSIM4v6_MOD_IJTHSREV:
            value->rValue = model->BSIM4v6ijthsrev;
            return(OK);
        case BSIM4v6_MOD_XJBVD:
            value->rValue = model->BSIM4v6xjbvd;
            return(OK);
        case BSIM4v6_MOD_XJBVS:
            value->rValue = model->BSIM4v6xjbvs;
            return(OK);
        case BSIM4v6_MOD_BVD:
            value->rValue = model->BSIM4v6bvd;
            return(OK);
        case BSIM4v6_MOD_BVS:
            value->rValue = model->BSIM4v6bvs;
            return(OK);
        case BSIM4v6_MOD_VFB:
            value->rValue = model->BSIM4v6vfb;
            return(OK);

        case BSIM4v6_MOD_JTSS:
            value->rValue = model->BSIM4v6jtss;
            return(OK);
        case BSIM4v6_MOD_JTSD:
            value->rValue = model->BSIM4v6jtsd;
            return(OK);
        case BSIM4v6_MOD_JTSSWS:
            value->rValue = model->BSIM4v6jtssws;
            return(OK);
        case BSIM4v6_MOD_JTSSWD:
            value->rValue = model->BSIM4v6jtsswd;
            return(OK);
        case BSIM4v6_MOD_JTSSWGS:
            value->rValue = model->BSIM4v6jtsswgs;
            return(OK);
        case BSIM4v6_MOD_JTSSWGD:
            value->rValue = model->BSIM4v6jtsswgd;
            return(OK);
        case BSIM4v6_MOD_JTWEFF:
            value->rValue = model->BSIM4v6jtweff;
            return(OK);
        case BSIM4v6_MOD_NJTS:
            value->rValue = model->BSIM4v6njts;
            return(OK);
        case BSIM4v6_MOD_NJTSSW:
            value->rValue = model->BSIM4v6njtssw;
            return(OK);
        case BSIM4v6_MOD_NJTSSWG:
            value->rValue = model->BSIM4v6njtsswg;
            return(OK);
        case BSIM4v6_MOD_NJTSD:
            value->rValue = model->BSIM4v6njtsd;
            return(OK);
        case BSIM4v6_MOD_NJTSSWD:
            value->rValue = model->BSIM4v6njtsswd;
            return(OK);
        case BSIM4v6_MOD_NJTSSWGD:
            value->rValue = model->BSIM4v6njtsswgd;
            return(OK);
        case BSIM4v6_MOD_XTSS:
            value->rValue = model->BSIM4v6xtss;
            return(OK);
        case BSIM4v6_MOD_XTSD:
            value->rValue = model->BSIM4v6xtsd;
            return(OK);
        case BSIM4v6_MOD_XTSSWS:
            value->rValue = model->BSIM4v6xtssws;
            return(OK);
        case BSIM4v6_MOD_XTSSWD:
            value->rValue = model->BSIM4v6xtsswd;
            return(OK);
        case BSIM4v6_MOD_XTSSWGS:
            value->rValue = model->BSIM4v6xtsswgs;
            return(OK);
        case BSIM4v6_MOD_XTSSWGD:
            value->rValue = model->BSIM4v6xtsswgd;
            return(OK);
        case BSIM4v6_MOD_TNJTS:
            value->rValue = model->BSIM4v6tnjts;
            return(OK);
        case BSIM4v6_MOD_TNJTSSW:
            value->rValue = model->BSIM4v6tnjtssw;
            return(OK);
        case BSIM4v6_MOD_TNJTSSWG:
            value->rValue = model->BSIM4v6tnjtsswg;
            return(OK);
        case BSIM4v6_MOD_TNJTSD:
            value->rValue = model->BSIM4v6tnjtsd;
            return(OK);
        case BSIM4v6_MOD_TNJTSSWD:
            value->rValue = model->BSIM4v6tnjtsswd;
            return(OK);
        case BSIM4v6_MOD_TNJTSSWGD:
            value->rValue = model->BSIM4v6tnjtsswgd;
            return(OK);
        case BSIM4v6_MOD_VTSS:
            value->rValue = model->BSIM4v6vtss;
            return(OK);
        case BSIM4v6_MOD_VTSD:
            value->rValue = model->BSIM4v6vtsd;
            return(OK);
        case BSIM4v6_MOD_VTSSWS:
            value->rValue = model->BSIM4v6vtssws;
            return(OK);
        case BSIM4v6_MOD_VTSSWD:
            value->rValue = model->BSIM4v6vtsswd;
            return(OK);
        case BSIM4v6_MOD_VTSSWGS:
            value->rValue = model->BSIM4v6vtsswgs;
            return(OK);
        case BSIM4v6_MOD_VTSSWGD:
            value->rValue = model->BSIM4v6vtsswgd;
            return(OK);

        case BSIM4v6_MOD_GBMIN:
            value->rValue = model->BSIM4v6gbmin;
            return(OK);
        case BSIM4v6_MOD_RBDB:
            value->rValue = model->BSIM4v6rbdb;
            return(OK);
        case BSIM4v6_MOD_RBPB:
            value->rValue = model->BSIM4v6rbpb;
            return(OK);
        case BSIM4v6_MOD_RBSB:
            value->rValue = model->BSIM4v6rbsb;
            return(OK);
        case BSIM4v6_MOD_RBPS:
            value->rValue = model->BSIM4v6rbps;
            return(OK);
        case BSIM4v6_MOD_RBPD:
            value->rValue = model->BSIM4v6rbpd;
            return(OK);

        case BSIM4v6_MOD_RBPS0:
            value->rValue = model->BSIM4v6rbps0;
            return(OK);
        case BSIM4v6_MOD_RBPSL:
            value->rValue = model->BSIM4v6rbpsl;
            return(OK);
        case BSIM4v6_MOD_RBPSW:
            value->rValue = model->BSIM4v6rbpsw;
            return(OK);
        case BSIM4v6_MOD_RBPSNF:
            value->rValue = model->BSIM4v6rbpsnf;
            return(OK);
        case BSIM4v6_MOD_RBPD0:
            value->rValue = model->BSIM4v6rbpd0;
            return(OK);
        case BSIM4v6_MOD_RBPDL:
            value->rValue = model->BSIM4v6rbpdl;
            return(OK);
        case BSIM4v6_MOD_RBPDW:
            value->rValue = model->BSIM4v6rbpdw;
            return(OK);
        case BSIM4v6_MOD_RBPDNF:
            value->rValue = model->BSIM4v6rbpdnf;
            return(OK);
        case BSIM4v6_MOD_RBPBX0:
            value->rValue = model->BSIM4v6rbpbx0;
            return(OK);
        case BSIM4v6_MOD_RBPBXL:
            value->rValue = model->BSIM4v6rbpbxl;
            return(OK);
        case BSIM4v6_MOD_RBPBXW:
            value->rValue = model->BSIM4v6rbpbxw;
            return(OK);
        case BSIM4v6_MOD_RBPBXNF:
            value->rValue = model->BSIM4v6rbpbxnf;
            return(OK);
        case BSIM4v6_MOD_RBPBY0:
            value->rValue = model->BSIM4v6rbpby0;
            return(OK);
        case BSIM4v6_MOD_RBPBYL:
            value->rValue = model->BSIM4v6rbpbyl;
            return(OK);
        case BSIM4v6_MOD_RBPBYW:
            value->rValue = model->BSIM4v6rbpbyw;
            return(OK);
        case BSIM4v6_MOD_RBPBYNF:
            value->rValue = model->BSIM4v6rbpbynf;
            return(OK);

        case BSIM4v6_MOD_RBSBX0:
            value->rValue = model->BSIM4v6rbsbx0;
            return(OK);
        case BSIM4v6_MOD_RBSBY0:
            value->rValue = model->BSIM4v6rbsby0;
            return(OK);
        case BSIM4v6_MOD_RBDBX0:
            value->rValue = model->BSIM4v6rbdbx0;
            return(OK);
        case BSIM4v6_MOD_RBDBY0:
            value->rValue = model->BSIM4v6rbdby0;
            return(OK);
        case BSIM4v6_MOD_RBSDBXL:
            value->rValue = model->BSIM4v6rbsdbxl;
            return(OK);
        case BSIM4v6_MOD_RBSDBXW:
            value->rValue = model->BSIM4v6rbsdbxw;
            return(OK);
        case BSIM4v6_MOD_RBSDBXNF:
            value->rValue = model->BSIM4v6rbsdbxnf;
            return(OK);
        case BSIM4v6_MOD_RBSDBYL:
            value->rValue = model->BSIM4v6rbsdbyl;
            return(OK);
        case BSIM4v6_MOD_RBSDBYW:
            value->rValue = model->BSIM4v6rbsdbyw;
            return(OK);
        case BSIM4v6_MOD_RBSDBYNF:
            value->rValue = model->BSIM4v6rbsdbynf;
            return(OK);


        case BSIM4v6_MOD_CGSL:
            value->rValue = model->BSIM4v6cgsl;
            return(OK);
        case BSIM4v6_MOD_CGDL:
            value->rValue = model->BSIM4v6cgdl;
            return(OK);
        case BSIM4v6_MOD_CKAPPAS:
            value->rValue = model->BSIM4v6ckappas;
            return(OK);
        case BSIM4v6_MOD_CKAPPAD:
            value->rValue = model->BSIM4v6ckappad;
            return(OK);
        case BSIM4v6_MOD_CF:
            value->rValue = model->BSIM4v6cf;
            return(OK);
        case BSIM4v6_MOD_CLC:
            value->rValue = model->BSIM4v6clc;
            return(OK);
        case BSIM4v6_MOD_CLE:
            value->rValue = model->BSIM4v6cle;
            return(OK);
        case BSIM4v6_MOD_DWC:
            value->rValue = model->BSIM4v6dwc;
            return(OK);
        case BSIM4v6_MOD_DLC:
            value->rValue = model->BSIM4v6dlc;
            return(OK);
        case BSIM4v6_MOD_XW:
            value->rValue = model->BSIM4v6xw;
            return(OK);
        case BSIM4v6_MOD_XL:
            value->rValue = model->BSIM4v6xl;
            return(OK);
        case BSIM4v6_MOD_DLCIG:
            value->rValue = model->BSIM4v6dlcig;
            return(OK);
        case BSIM4v6_MOD_DLCIGD:
            value->rValue = model->BSIM4v6dlcigd;
            return(OK);
        case BSIM4v6_MOD_DWJ:
            value->rValue = model->BSIM4v6dwj;
            return(OK);
        case BSIM4v6_MOD_VFBCV:
            value->rValue = model->BSIM4v6vfbcv; 
            return(OK);
        case BSIM4v6_MOD_ACDE:
            value->rValue = model->BSIM4v6acde;
            return(OK);
        case BSIM4v6_MOD_MOIN:
            value->rValue = model->BSIM4v6moin;
            return(OK);
        case BSIM4v6_MOD_NOFF:
            value->rValue = model->BSIM4v6noff;
            return(OK);
        case BSIM4v6_MOD_VOFFCV:
            value->rValue = model->BSIM4v6voffcv;
            return(OK);
        case BSIM4v6_MOD_DMCG:
            value->rValue = model->BSIM4v6dmcg;
            return(OK);
        case BSIM4v6_MOD_DMCI:
            value->rValue = model->BSIM4v6dmci;
            return(OK);
        case BSIM4v6_MOD_DMDG:
            value->rValue = model->BSIM4v6dmdg;
            return(OK);
        case BSIM4v6_MOD_DMCGT:
            value->rValue = model->BSIM4v6dmcgt;
            return(OK);
        case BSIM4v6_MOD_XGW:
            value->rValue = model->BSIM4v6xgw;
            return(OK);
        case BSIM4v6_MOD_XGL:
            value->rValue = model->BSIM4v6xgl;
            return(OK);
        case BSIM4v6_MOD_RSHG:
            value->rValue = model->BSIM4v6rshg;
            return(OK);
        case BSIM4v6_MOD_TCJ:
            value->rValue = model->BSIM4v6tcj;
            return(OK);
        case BSIM4v6_MOD_TPB:
            value->rValue = model->BSIM4v6tpb;
            return(OK);
        case BSIM4v6_MOD_TCJSW:
            value->rValue = model->BSIM4v6tcjsw;
            return(OK);
        case BSIM4v6_MOD_TPBSW:
            value->rValue = model->BSIM4v6tpbsw;
            return(OK);
        case BSIM4v6_MOD_TCJSWG:
            value->rValue = model->BSIM4v6tcjswg;
            return(OK);
        case BSIM4v6_MOD_TPBSWG:
            value->rValue = model->BSIM4v6tpbswg;
            return(OK);

	/* Length dependence */
        case  BSIM4v6_MOD_LCDSC :
          value->rValue = model->BSIM4v6lcdsc;
            return(OK);
        case  BSIM4v6_MOD_LCDSCB :
          value->rValue = model->BSIM4v6lcdscb;
            return(OK);
        case  BSIM4v6_MOD_LCDSCD :
          value->rValue = model->BSIM4v6lcdscd;
            return(OK);
        case  BSIM4v6_MOD_LCIT :
          value->rValue = model->BSIM4v6lcit;
            return(OK);
        case  BSIM4v6_MOD_LNFACTOR :
          value->rValue = model->BSIM4v6lnfactor;
            return(OK);
        case BSIM4v6_MOD_LXJ:
            value->rValue = model->BSIM4v6lxj;
            return(OK);
        case BSIM4v6_MOD_LVSAT:
            value->rValue = model->BSIM4v6lvsat;
            return(OK);
        case BSIM4v6_MOD_LAT:
            value->rValue = model->BSIM4v6lat;
            return(OK);
        case BSIM4v6_MOD_LA0:
            value->rValue = model->BSIM4v6la0;
            return(OK);
        case BSIM4v6_MOD_LAGS:
            value->rValue = model->BSIM4v6lags;
            return(OK);
        case BSIM4v6_MOD_LA1:
            value->rValue = model->BSIM4v6la1;
            return(OK);
        case BSIM4v6_MOD_LA2:
            value->rValue = model->BSIM4v6la2;
            return(OK);
        case BSIM4v6_MOD_LKETA:
            value->rValue = model->BSIM4v6lketa;
            return(OK);   
        case BSIM4v6_MOD_LNSUB:
            value->rValue = model->BSIM4v6lnsub;
            return(OK);
        case BSIM4v6_MOD_LNDEP:
            value->rValue = model->BSIM4v6lndep;
            return(OK);
        case BSIM4v6_MOD_LNSD:
            value->rValue = model->BSIM4v6lnsd;
            return(OK);
        case BSIM4v6_MOD_LNGATE:
            value->rValue = model->BSIM4v6lngate;
            return(OK);
        case BSIM4v6_MOD_LGAMMA1:
            value->rValue = model->BSIM4v6lgamma1;
            return(OK);
        case BSIM4v6_MOD_LGAMMA2:
            value->rValue = model->BSIM4v6lgamma2;
            return(OK);
        case BSIM4v6_MOD_LVBX:
            value->rValue = model->BSIM4v6lvbx;
            return(OK);
        case BSIM4v6_MOD_LVBM:
            value->rValue = model->BSIM4v6lvbm;
            return(OK);
        case BSIM4v6_MOD_LXT:
            value->rValue = model->BSIM4v6lxt;
            return(OK);
        case  BSIM4v6_MOD_LK1:
          value->rValue = model->BSIM4v6lk1;
            return(OK);
        case  BSIM4v6_MOD_LKT1:
          value->rValue = model->BSIM4v6lkt1;
            return(OK);
        case  BSIM4v6_MOD_LKT1L:
          value->rValue = model->BSIM4v6lkt1l;
            return(OK);
        case  BSIM4v6_MOD_LKT2 :
          value->rValue = model->BSIM4v6lkt2;
            return(OK);
        case  BSIM4v6_MOD_LK2 :
          value->rValue = model->BSIM4v6lk2;
            return(OK);
        case  BSIM4v6_MOD_LK3:
          value->rValue = model->BSIM4v6lk3;
            return(OK);
        case  BSIM4v6_MOD_LK3B:
          value->rValue = model->BSIM4v6lk3b;
            return(OK);
        case  BSIM4v6_MOD_LW0:
          value->rValue = model->BSIM4v6lw0;
            return(OK);
        case  BSIM4v6_MOD_LLPE0:
          value->rValue = model->BSIM4v6llpe0;
            return(OK);
        case  BSIM4v6_MOD_LLPEB:
          value->rValue = model->BSIM4v6llpeb;
            return(OK);
        case  BSIM4v6_MOD_LDVTP0:
          value->rValue = model->BSIM4v6ldvtp0;
            return(OK);
        case  BSIM4v6_MOD_LDVTP1:
          value->rValue = model->BSIM4v6ldvtp1;
            return(OK);
        case  BSIM4v6_MOD_LDVT0:                
          value->rValue = model->BSIM4v6ldvt0;
            return(OK);
        case  BSIM4v6_MOD_LDVT1 :             
          value->rValue = model->BSIM4v6ldvt1;
            return(OK);
        case  BSIM4v6_MOD_LDVT2 :             
          value->rValue = model->BSIM4v6ldvt2;
            return(OK);
        case  BSIM4v6_MOD_LDVT0W :                
          value->rValue = model->BSIM4v6ldvt0w;
            return(OK);
        case  BSIM4v6_MOD_LDVT1W :             
          value->rValue = model->BSIM4v6ldvt1w;
            return(OK);
        case  BSIM4v6_MOD_LDVT2W :             
          value->rValue = model->BSIM4v6ldvt2w;
            return(OK);
        case  BSIM4v6_MOD_LDROUT :           
          value->rValue = model->BSIM4v6ldrout;
            return(OK);
        case  BSIM4v6_MOD_LDSUB :           
          value->rValue = model->BSIM4v6ldsub;
            return(OK);
        case BSIM4v6_MOD_LVTH0:
            value->rValue = model->BSIM4v6lvth0; 
            return(OK);
        case BSIM4v6_MOD_LUA:
            value->rValue = model->BSIM4v6lua; 
            return(OK);
        case BSIM4v6_MOD_LUA1:
            value->rValue = model->BSIM4v6lua1; 
            return(OK);
        case BSIM4v6_MOD_LUB:
            value->rValue = model->BSIM4v6lub;  
            return(OK);
        case BSIM4v6_MOD_LUB1:
            value->rValue = model->BSIM4v6lub1;  
            return(OK);
        case BSIM4v6_MOD_LUC:
            value->rValue = model->BSIM4v6luc; 
            return(OK);
        case BSIM4v6_MOD_LUC1:
            value->rValue = model->BSIM4v6luc1; 
            return(OK);
        case BSIM4v6_MOD_LUD:
            value->rValue = model->BSIM4v6lud; 
            return(OK);
        case BSIM4v6_MOD_LUD1:
            value->rValue = model->BSIM4v6lud1; 
            return(OK);
        case BSIM4v6_MOD_LUP:
            value->rValue = model->BSIM4v6lup; 
            return(OK);
        case BSIM4v6_MOD_LLP:
            value->rValue = model->BSIM4v6llp; 
            return(OK);
        case BSIM4v6_MOD_LU0:
            value->rValue = model->BSIM4v6lu0;
            return(OK);
        case BSIM4v6_MOD_LUTE:
            value->rValue = model->BSIM4v6lute;
            return(OK);
        case BSIM4v6_MOD_LUCSTE:
            value->rValue = model->BSIM4v6lucste;
            return(OK);
        case BSIM4v6_MOD_LVOFF:
            value->rValue = model->BSIM4v6lvoff;
            return(OK);
        case BSIM4v6_MOD_LTVOFF:
            value->rValue = model->BSIM4v6ltvoff;
            return(OK);
        case BSIM4v6_MOD_LMINV:
            value->rValue = model->BSIM4v6lminv;
            return(OK);
        case BSIM4v6_MOD_LMINVCV:
            value->rValue = model->BSIM4v6lminvcv;
            return(OK);
        case BSIM4v6_MOD_LFPROUT:
            value->rValue = model->BSIM4v6lfprout;
            return(OK);
        case BSIM4v6_MOD_LPDITS:
            value->rValue = model->BSIM4v6lpdits;
            return(OK);
        case BSIM4v6_MOD_LPDITSD:
            value->rValue = model->BSIM4v6lpditsd;
            return(OK);
        case BSIM4v6_MOD_LDELTA:
            value->rValue = model->BSIM4v6ldelta;
            return(OK);
        case BSIM4v6_MOD_LRDSW:
            value->rValue = model->BSIM4v6lrdsw; 
            return(OK);             
        case BSIM4v6_MOD_LRDW:
            value->rValue = model->BSIM4v6lrdw;
            return(OK);
        case BSIM4v6_MOD_LRSW:
            value->rValue = model->BSIM4v6lrsw;
            return(OK);
        case BSIM4v6_MOD_LPRWB:
            value->rValue = model->BSIM4v6lprwb; 
            return(OK);             
        case BSIM4v6_MOD_LPRWG:
            value->rValue = model->BSIM4v6lprwg; 
            return(OK);             
        case BSIM4v6_MOD_LPRT:
            value->rValue = model->BSIM4v6lprt; 
            return(OK);              
        case BSIM4v6_MOD_LETA0:
            value->rValue = model->BSIM4v6leta0; 
            return(OK);               
        case BSIM4v6_MOD_LETAB:
            value->rValue = model->BSIM4v6letab; 
            return(OK);               
        case BSIM4v6_MOD_LPCLM:
            value->rValue = model->BSIM4v6lpclm; 
            return(OK);               
        case BSIM4v6_MOD_LPDIBL1:
            value->rValue = model->BSIM4v6lpdibl1; 
            return(OK);               
        case BSIM4v6_MOD_LPDIBL2:
            value->rValue = model->BSIM4v6lpdibl2; 
            return(OK);               
        case BSIM4v6_MOD_LPDIBLB:
            value->rValue = model->BSIM4v6lpdiblb; 
            return(OK);               
        case BSIM4v6_MOD_LPSCBE1:
            value->rValue = model->BSIM4v6lpscbe1; 
            return(OK);               
        case BSIM4v6_MOD_LPSCBE2:
            value->rValue = model->BSIM4v6lpscbe2; 
            return(OK);               
        case BSIM4v6_MOD_LPVAG:
            value->rValue = model->BSIM4v6lpvag; 
            return(OK);               
        case BSIM4v6_MOD_LWR:
            value->rValue = model->BSIM4v6lwr;
            return(OK);
        case BSIM4v6_MOD_LDWG:
            value->rValue = model->BSIM4v6ldwg;
            return(OK);
        case BSIM4v6_MOD_LDWB:
            value->rValue = model->BSIM4v6ldwb;
            return(OK);
        case BSIM4v6_MOD_LB0:
            value->rValue = model->BSIM4v6lb0;
            return(OK);
        case BSIM4v6_MOD_LB1:
            value->rValue = model->BSIM4v6lb1;
            return(OK);
        case BSIM4v6_MOD_LALPHA0:
            value->rValue = model->BSIM4v6lalpha0;
            return(OK);
        case BSIM4v6_MOD_LALPHA1:
            value->rValue = model->BSIM4v6lalpha1;
            return(OK);
        case BSIM4v6_MOD_LBETA0:
            value->rValue = model->BSIM4v6lbeta0;
            return(OK);
        case BSIM4v6_MOD_LAGIDL:
            value->rValue = model->BSIM4v6lagidl;
            return(OK);
        case BSIM4v6_MOD_LBGIDL:
            value->rValue = model->BSIM4v6lbgidl;
            return(OK);
        case BSIM4v6_MOD_LCGIDL:
            value->rValue = model->BSIM4v6lcgidl;
            return(OK);
        case BSIM4v6_MOD_LEGIDL:
            value->rValue = model->BSIM4v6legidl;
            return(OK);
        case BSIM4v6_MOD_LAGISL:
            value->rValue = model->BSIM4v6lagisl;
            return(OK);
        case BSIM4v6_MOD_LBGISL:
            value->rValue = model->BSIM4v6lbgisl;
            return(OK);
        case BSIM4v6_MOD_LCGISL:
            value->rValue = model->BSIM4v6lcgisl;
            return(OK);
        case BSIM4v6_MOD_LEGISL:
            value->rValue = model->BSIM4v6legisl;
            return(OK);
        case BSIM4v6_MOD_LAIGC:
            value->rValue = model->BSIM4v6laigc;
            return(OK);
        case BSIM4v6_MOD_LBIGC:
            value->rValue = model->BSIM4v6lbigc;
            return(OK);
        case BSIM4v6_MOD_LCIGC:
            value->rValue = model->BSIM4v6lcigc;
            return(OK);
        case BSIM4v6_MOD_LAIGSD:
            value->rValue = model->BSIM4v6laigsd;
            return(OK);
        case BSIM4v6_MOD_LBIGSD:
            value->rValue = model->BSIM4v6lbigsd;
            return(OK);
        case BSIM4v6_MOD_LCIGSD:
            value->rValue = model->BSIM4v6lcigsd;
            return(OK);
        case BSIM4v6_MOD_LAIGS:
            value->rValue = model->BSIM4v6laigs;
            return(OK);
        case BSIM4v6_MOD_LBIGS:
            value->rValue = model->BSIM4v6lbigs;
            return(OK);
        case BSIM4v6_MOD_LCIGS:
            value->rValue = model->BSIM4v6lcigs;
            return(OK);
        case BSIM4v6_MOD_LAIGD:
            value->rValue = model->BSIM4v6laigd;
            return(OK);
        case BSIM4v6_MOD_LBIGD:
            value->rValue = model->BSIM4v6lbigd;
            return(OK);
        case BSIM4v6_MOD_LCIGD:
            value->rValue = model->BSIM4v6lcigd;
            return(OK);
        case BSIM4v6_MOD_LAIGBACC:
            value->rValue = model->BSIM4v6laigbacc;
            return(OK);
        case BSIM4v6_MOD_LBIGBACC:
            value->rValue = model->BSIM4v6lbigbacc;
            return(OK);
        case BSIM4v6_MOD_LCIGBACC:
            value->rValue = model->BSIM4v6lcigbacc;
            return(OK);
        case BSIM4v6_MOD_LAIGBINV:
            value->rValue = model->BSIM4v6laigbinv;
            return(OK);
        case BSIM4v6_MOD_LBIGBINV:
            value->rValue = model->BSIM4v6lbigbinv;
            return(OK);
        case BSIM4v6_MOD_LCIGBINV:
            value->rValue = model->BSIM4v6lcigbinv;
            return(OK);
        case BSIM4v6_MOD_LNIGC:
            value->rValue = model->BSIM4v6lnigc;
            return(OK);
        case BSIM4v6_MOD_LNIGBACC:
            value->rValue = model->BSIM4v6lnigbacc;
            return(OK);
        case BSIM4v6_MOD_LNIGBINV:
            value->rValue = model->BSIM4v6lnigbinv;
            return(OK);
        case BSIM4v6_MOD_LNTOX:
            value->rValue = model->BSIM4v6lntox;
            return(OK);
        case BSIM4v6_MOD_LEIGBINV:
            value->rValue = model->BSIM4v6leigbinv;
            return(OK);
        case BSIM4v6_MOD_LPIGCD:
            value->rValue = model->BSIM4v6lpigcd;
            return(OK);
        case BSIM4v6_MOD_LPOXEDGE:
            value->rValue = model->BSIM4v6lpoxedge;
            return(OK);
        case BSIM4v6_MOD_LPHIN:
            value->rValue = model->BSIM4v6lphin;
            return(OK);
        case BSIM4v6_MOD_LXRCRG1:
            value->rValue = model->BSIM4v6lxrcrg1;
            return(OK);
        case BSIM4v6_MOD_LXRCRG2:
            value->rValue = model->BSIM4v6lxrcrg2;
            return(OK);
        case BSIM4v6_MOD_LEU:
            value->rValue = model->BSIM4v6leu;
            return(OK);
        case BSIM4v6_MOD_LUCS:
            value->rValue = model->BSIM4v6lucs;
            return(OK);
        case BSIM4v6_MOD_LVFB:
            value->rValue = model->BSIM4v6lvfb;
            return(OK);

        case BSIM4v6_MOD_LCGSL:
            value->rValue = model->BSIM4v6lcgsl;
            return(OK);
        case BSIM4v6_MOD_LCGDL:
            value->rValue = model->BSIM4v6lcgdl;
            return(OK);
        case BSIM4v6_MOD_LCKAPPAS:
            value->rValue = model->BSIM4v6lckappas;
            return(OK);
        case BSIM4v6_MOD_LCKAPPAD:
            value->rValue = model->BSIM4v6lckappad;
            return(OK);
        case BSIM4v6_MOD_LCF:
            value->rValue = model->BSIM4v6lcf;
            return(OK);
        case BSIM4v6_MOD_LCLC:
            value->rValue = model->BSIM4v6lclc;
            return(OK);
        case BSIM4v6_MOD_LCLE:
            value->rValue = model->BSIM4v6lcle;
            return(OK);
        case BSIM4v6_MOD_LVFBCV:
            value->rValue = model->BSIM4v6lvfbcv;
            return(OK);
        case BSIM4v6_MOD_LACDE:
            value->rValue = model->BSIM4v6lacde;
            return(OK);
        case BSIM4v6_MOD_LMOIN:
            value->rValue = model->BSIM4v6lmoin;
            return(OK);
        case BSIM4v6_MOD_LNOFF:
            value->rValue = model->BSIM4v6lnoff;
            return(OK);
        case BSIM4v6_MOD_LVOFFCV:
            value->rValue = model->BSIM4v6lvoffcv;
            return(OK);
        case BSIM4v6_MOD_LVFBSDOFF:
            value->rValue = model->BSIM4v6lvfbsdoff;
            return(OK);
        case BSIM4v6_MOD_LTVFBSDOFF:
            value->rValue = model->BSIM4v6ltvfbsdoff;
            return(OK);

	/* Width dependence */
        case  BSIM4v6_MOD_WCDSC :
          value->rValue = model->BSIM4v6wcdsc;
            return(OK);
        case  BSIM4v6_MOD_WCDSCB :
          value->rValue = model->BSIM4v6wcdscb;
            return(OK);
        case  BSIM4v6_MOD_WCDSCD :
          value->rValue = model->BSIM4v6wcdscd;
            return(OK);
        case  BSIM4v6_MOD_WCIT :
          value->rValue = model->BSIM4v6wcit;
            return(OK);
        case  BSIM4v6_MOD_WNFACTOR :
          value->rValue = model->BSIM4v6wnfactor;
            return(OK);
        case BSIM4v6_MOD_WXJ:
            value->rValue = model->BSIM4v6wxj;
            return(OK);
        case BSIM4v6_MOD_WVSAT:
            value->rValue = model->BSIM4v6wvsat;
            return(OK);
        case BSIM4v6_MOD_WAT:
            value->rValue = model->BSIM4v6wat;
            return(OK);
        case BSIM4v6_MOD_WA0:
            value->rValue = model->BSIM4v6wa0;
            return(OK);
        case BSIM4v6_MOD_WAGS:
            value->rValue = model->BSIM4v6wags;
            return(OK);
        case BSIM4v6_MOD_WA1:
            value->rValue = model->BSIM4v6wa1;
            return(OK);
        case BSIM4v6_MOD_WA2:
            value->rValue = model->BSIM4v6wa2;
            return(OK);
        case BSIM4v6_MOD_WKETA:
            value->rValue = model->BSIM4v6wketa;
            return(OK);   
        case BSIM4v6_MOD_WNSUB:
            value->rValue = model->BSIM4v6wnsub;
            return(OK);
        case BSIM4v6_MOD_WNDEP:
            value->rValue = model->BSIM4v6wndep;
            return(OK);
        case BSIM4v6_MOD_WNSD:
            value->rValue = model->BSIM4v6wnsd;
            return(OK);
        case BSIM4v6_MOD_WNGATE:
            value->rValue = model->BSIM4v6wngate;
            return(OK);
        case BSIM4v6_MOD_WGAMMA1:
            value->rValue = model->BSIM4v6wgamma1;
            return(OK);
        case BSIM4v6_MOD_WGAMMA2:
            value->rValue = model->BSIM4v6wgamma2;
            return(OK);
        case BSIM4v6_MOD_WVBX:
            value->rValue = model->BSIM4v6wvbx;
            return(OK);
        case BSIM4v6_MOD_WVBM:
            value->rValue = model->BSIM4v6wvbm;
            return(OK);
        case BSIM4v6_MOD_WXT:
            value->rValue = model->BSIM4v6wxt;
            return(OK);
        case  BSIM4v6_MOD_WK1:
          value->rValue = model->BSIM4v6wk1;
            return(OK);
        case  BSIM4v6_MOD_WKT1:
          value->rValue = model->BSIM4v6wkt1;
            return(OK);
        case  BSIM4v6_MOD_WKT1L:
          value->rValue = model->BSIM4v6wkt1l;
            return(OK);
        case  BSIM4v6_MOD_WKT2 :
          value->rValue = model->BSIM4v6wkt2;
            return(OK);
        case  BSIM4v6_MOD_WK2 :
          value->rValue = model->BSIM4v6wk2;
            return(OK);
        case  BSIM4v6_MOD_WK3:
          value->rValue = model->BSIM4v6wk3;
            return(OK);
        case  BSIM4v6_MOD_WK3B:
          value->rValue = model->BSIM4v6wk3b;
            return(OK);
        case  BSIM4v6_MOD_WW0:
          value->rValue = model->BSIM4v6ww0;
            return(OK);
        case  BSIM4v6_MOD_WLPE0:
          value->rValue = model->BSIM4v6wlpe0;
            return(OK);
        case  BSIM4v6_MOD_WDVTP0:
          value->rValue = model->BSIM4v6wdvtp0;
            return(OK);
        case  BSIM4v6_MOD_WDVTP1:
          value->rValue = model->BSIM4v6wdvtp1;
            return(OK);
        case  BSIM4v6_MOD_WLPEB:
          value->rValue = model->BSIM4v6wlpeb;
            return(OK);
        case  BSIM4v6_MOD_WDVT0:                
          value->rValue = model->BSIM4v6wdvt0;
            return(OK);
        case  BSIM4v6_MOD_WDVT1 :             
          value->rValue = model->BSIM4v6wdvt1;
            return(OK);
        case  BSIM4v6_MOD_WDVT2 :             
          value->rValue = model->BSIM4v6wdvt2;
            return(OK);
        case  BSIM4v6_MOD_WDVT0W :                
          value->rValue = model->BSIM4v6wdvt0w;
            return(OK);
        case  BSIM4v6_MOD_WDVT1W :             
          value->rValue = model->BSIM4v6wdvt1w;
            return(OK);
        case  BSIM4v6_MOD_WDVT2W :             
          value->rValue = model->BSIM4v6wdvt2w;
            return(OK);
        case  BSIM4v6_MOD_WDROUT :           
          value->rValue = model->BSIM4v6wdrout;
            return(OK);
        case  BSIM4v6_MOD_WDSUB :           
          value->rValue = model->BSIM4v6wdsub;
            return(OK);
        case BSIM4v6_MOD_WVTH0:
            value->rValue = model->BSIM4v6wvth0; 
            return(OK);
        case BSIM4v6_MOD_WUA:
            value->rValue = model->BSIM4v6wua; 
            return(OK);
        case BSIM4v6_MOD_WUA1:
            value->rValue = model->BSIM4v6wua1; 
            return(OK);
        case BSIM4v6_MOD_WUB:
            value->rValue = model->BSIM4v6wub;  
            return(OK);
        case BSIM4v6_MOD_WUB1:
            value->rValue = model->BSIM4v6wub1;  
            return(OK);
        case BSIM4v6_MOD_WUC:
            value->rValue = model->BSIM4v6wuc; 
            return(OK);
        case BSIM4v6_MOD_WUC1:
            value->rValue = model->BSIM4v6wuc1; 
            return(OK);
        case BSIM4v6_MOD_WUD:
            value->rValue = model->BSIM4v6wud; 
            return(OK);
        case BSIM4v6_MOD_WUD1:
            value->rValue = model->BSIM4v6wud1; 
            return(OK);
        case BSIM4v6_MOD_WUP:
            value->rValue = model->BSIM4v6wup; 
            return(OK);
        case BSIM4v6_MOD_WLP:
            value->rValue = model->BSIM4v6wlp; 
            return(OK);
        case BSIM4v6_MOD_WU0:
            value->rValue = model->BSIM4v6wu0;
            return(OK);
        case BSIM4v6_MOD_WUTE:
            value->rValue = model->BSIM4v6wute;
            return(OK);
        case BSIM4v6_MOD_WUCSTE:
            value->rValue = model->BSIM4v6wucste;
            return(OK);
        case BSIM4v6_MOD_WVOFF:
            value->rValue = model->BSIM4v6wvoff;
            return(OK);
        case BSIM4v6_MOD_WTVOFF:
            value->rValue = model->BSIM4v6wtvoff;
            return(OK);
        case BSIM4v6_MOD_WMINV:
            value->rValue = model->BSIM4v6wminv;
            return(OK);
        case BSIM4v6_MOD_WMINVCV:
            value->rValue = model->BSIM4v6wminvcv;
            return(OK);
        case BSIM4v6_MOD_WFPROUT:
            value->rValue = model->BSIM4v6wfprout;
            return(OK);
        case BSIM4v6_MOD_WPDITS:
            value->rValue = model->BSIM4v6wpdits;
            return(OK);
        case BSIM4v6_MOD_WPDITSD:
            value->rValue = model->BSIM4v6wpditsd;
            return(OK);
        case BSIM4v6_MOD_WDELTA:
            value->rValue = model->BSIM4v6wdelta;
            return(OK);
        case BSIM4v6_MOD_WRDSW:
            value->rValue = model->BSIM4v6wrdsw; 
            return(OK);             
        case BSIM4v6_MOD_WRDW:
            value->rValue = model->BSIM4v6wrdw;
            return(OK);
        case BSIM4v6_MOD_WRSW:
            value->rValue = model->BSIM4v6wrsw;
            return(OK);
        case BSIM4v6_MOD_WPRWB:
            value->rValue = model->BSIM4v6wprwb; 
            return(OK);             
        case BSIM4v6_MOD_WPRWG:
            value->rValue = model->BSIM4v6wprwg; 
            return(OK);             
        case BSIM4v6_MOD_WPRT:
            value->rValue = model->BSIM4v6wprt; 
            return(OK);              
        case BSIM4v6_MOD_WETA0:
            value->rValue = model->BSIM4v6weta0; 
            return(OK);               
        case BSIM4v6_MOD_WETAB:
            value->rValue = model->BSIM4v6wetab; 
            return(OK);               
        case BSIM4v6_MOD_WPCLM:
            value->rValue = model->BSIM4v6wpclm; 
            return(OK);               
        case BSIM4v6_MOD_WPDIBL1:
            value->rValue = model->BSIM4v6wpdibl1; 
            return(OK);               
        case BSIM4v6_MOD_WPDIBL2:
            value->rValue = model->BSIM4v6wpdibl2; 
            return(OK);               
        case BSIM4v6_MOD_WPDIBLB:
            value->rValue = model->BSIM4v6wpdiblb; 
            return(OK);               
        case BSIM4v6_MOD_WPSCBE1:
            value->rValue = model->BSIM4v6wpscbe1; 
            return(OK);               
        case BSIM4v6_MOD_WPSCBE2:
            value->rValue = model->BSIM4v6wpscbe2; 
            return(OK);               
        case BSIM4v6_MOD_WPVAG:
            value->rValue = model->BSIM4v6wpvag; 
            return(OK);               
        case BSIM4v6_MOD_WWR:
            value->rValue = model->BSIM4v6wwr;
            return(OK);
        case BSIM4v6_MOD_WDWG:
            value->rValue = model->BSIM4v6wdwg;
            return(OK);
        case BSIM4v6_MOD_WDWB:
            value->rValue = model->BSIM4v6wdwb;
            return(OK);
        case BSIM4v6_MOD_WB0:
            value->rValue = model->BSIM4v6wb0;
            return(OK);
        case BSIM4v6_MOD_WB1:
            value->rValue = model->BSIM4v6wb1;
            return(OK);
        case BSIM4v6_MOD_WALPHA0:
            value->rValue = model->BSIM4v6walpha0;
            return(OK);
        case BSIM4v6_MOD_WALPHA1:
            value->rValue = model->BSIM4v6walpha1;
            return(OK);
        case BSIM4v6_MOD_WBETA0:
            value->rValue = model->BSIM4v6wbeta0;
            return(OK);
        case BSIM4v6_MOD_WAGIDL:
            value->rValue = model->BSIM4v6wagidl;
            return(OK);
        case BSIM4v6_MOD_WBGIDL:
            value->rValue = model->BSIM4v6wbgidl;
            return(OK);
        case BSIM4v6_MOD_WCGIDL:
            value->rValue = model->BSIM4v6wcgidl;
            return(OK);
        case BSIM4v6_MOD_WEGIDL:
            value->rValue = model->BSIM4v6wegidl;
            return(OK);
        case BSIM4v6_MOD_WAGISL:
            value->rValue = model->BSIM4v6wagisl;
            return(OK);
        case BSIM4v6_MOD_WBGISL:
            value->rValue = model->BSIM4v6wbgisl;
            return(OK);
        case BSIM4v6_MOD_WCGISL:
            value->rValue = model->BSIM4v6wcgisl;
            return(OK);
        case BSIM4v6_MOD_WEGISL:
            value->rValue = model->BSIM4v6wegisl;
            return(OK);
        case BSIM4v6_MOD_WAIGC:
            value->rValue = model->BSIM4v6waigc;
            return(OK);
        case BSIM4v6_MOD_WBIGC:
            value->rValue = model->BSIM4v6wbigc;
            return(OK);
        case BSIM4v6_MOD_WCIGC:
            value->rValue = model->BSIM4v6wcigc;
            return(OK);
        case BSIM4v6_MOD_WAIGSD:
            value->rValue = model->BSIM4v6waigsd;
            return(OK);
        case BSIM4v6_MOD_WBIGSD:
            value->rValue = model->BSIM4v6wbigsd;
            return(OK);
        case BSIM4v6_MOD_WCIGSD:
            value->rValue = model->BSIM4v6wcigsd;
            return(OK);
        case BSIM4v6_MOD_WAIGS:
            value->rValue = model->BSIM4v6waigs;
            return(OK);
        case BSIM4v6_MOD_WBIGS:
            value->rValue = model->BSIM4v6wbigs;
            return(OK);
        case BSIM4v6_MOD_WCIGS:
            value->rValue = model->BSIM4v6wcigs;
            return(OK);
        case BSIM4v6_MOD_WAIGD:
            value->rValue = model->BSIM4v6waigd;
            return(OK);
        case BSIM4v6_MOD_WBIGD:
            value->rValue = model->BSIM4v6wbigd;
            return(OK);
        case BSIM4v6_MOD_WCIGD:
            value->rValue = model->BSIM4v6wcigd;
            return(OK);
        case BSIM4v6_MOD_WAIGBACC:
            value->rValue = model->BSIM4v6waigbacc;
            return(OK);
        case BSIM4v6_MOD_WBIGBACC:
            value->rValue = model->BSIM4v6wbigbacc;
            return(OK);
        case BSIM4v6_MOD_WCIGBACC:
            value->rValue = model->BSIM4v6wcigbacc;
            return(OK);
        case BSIM4v6_MOD_WAIGBINV:
            value->rValue = model->BSIM4v6waigbinv;
            return(OK);
        case BSIM4v6_MOD_WBIGBINV:
            value->rValue = model->BSIM4v6wbigbinv;
            return(OK);
        case BSIM4v6_MOD_WCIGBINV:
            value->rValue = model->BSIM4v6wcigbinv;
            return(OK);
        case BSIM4v6_MOD_WNIGC:
            value->rValue = model->BSIM4v6wnigc;
            return(OK);
        case BSIM4v6_MOD_WNIGBACC:
            value->rValue = model->BSIM4v6wnigbacc;
            return(OK);
        case BSIM4v6_MOD_WNIGBINV:
            value->rValue = model->BSIM4v6wnigbinv;
            return(OK);
        case BSIM4v6_MOD_WNTOX:
            value->rValue = model->BSIM4v6wntox;
            return(OK);
        case BSIM4v6_MOD_WEIGBINV:
            value->rValue = model->BSIM4v6weigbinv;
            return(OK);
        case BSIM4v6_MOD_WPIGCD:
            value->rValue = model->BSIM4v6wpigcd;
            return(OK);
        case BSIM4v6_MOD_WPOXEDGE:
            value->rValue = model->BSIM4v6wpoxedge;
            return(OK);
        case BSIM4v6_MOD_WPHIN:
            value->rValue = model->BSIM4v6wphin;
            return(OK);
        case BSIM4v6_MOD_WXRCRG1:
            value->rValue = model->BSIM4v6wxrcrg1;
            return(OK);
        case BSIM4v6_MOD_WXRCRG2:
            value->rValue = model->BSIM4v6wxrcrg2;
            return(OK);
        case BSIM4v6_MOD_WEU:
            value->rValue = model->BSIM4v6weu;
            return(OK);
        case BSIM4v6_MOD_WUCS:
            value->rValue = model->BSIM4v6wucs;
            return(OK);
        case BSIM4v6_MOD_WVFB:
            value->rValue = model->BSIM4v6wvfb;
            return(OK);

        case BSIM4v6_MOD_WCGSL:
            value->rValue = model->BSIM4v6wcgsl;
            return(OK);
        case BSIM4v6_MOD_WCGDL:
            value->rValue = model->BSIM4v6wcgdl;
            return(OK);
        case BSIM4v6_MOD_WCKAPPAS:
            value->rValue = model->BSIM4v6wckappas;
            return(OK);
        case BSIM4v6_MOD_WCKAPPAD:
            value->rValue = model->BSIM4v6wckappad;
            return(OK);
        case BSIM4v6_MOD_WCF:
            value->rValue = model->BSIM4v6wcf;
            return(OK);
        case BSIM4v6_MOD_WCLC:
            value->rValue = model->BSIM4v6wclc;
            return(OK);
        case BSIM4v6_MOD_WCLE:
            value->rValue = model->BSIM4v6wcle;
            return(OK);
        case BSIM4v6_MOD_WVFBCV:
            value->rValue = model->BSIM4v6wvfbcv;
            return(OK);
        case BSIM4v6_MOD_WACDE:
            value->rValue = model->BSIM4v6wacde;
            return(OK);
        case BSIM4v6_MOD_WMOIN:
            value->rValue = model->BSIM4v6wmoin;
            return(OK);
        case BSIM4v6_MOD_WNOFF:
            value->rValue = model->BSIM4v6wnoff;
            return(OK);
        case BSIM4v6_MOD_WVOFFCV:
            value->rValue = model->BSIM4v6wvoffcv;
            return(OK);
        case BSIM4v6_MOD_WVFBSDOFF:
            value->rValue = model->BSIM4v6wvfbsdoff;
            return(OK);
        case BSIM4v6_MOD_WTVFBSDOFF:
            value->rValue = model->BSIM4v6wtvfbsdoff;
            return(OK);

	/* Cross-term dependence */
        case  BSIM4v6_MOD_PCDSC :
          value->rValue = model->BSIM4v6pcdsc;
            return(OK);
        case  BSIM4v6_MOD_PCDSCB :
          value->rValue = model->BSIM4v6pcdscb;
            return(OK);
        case  BSIM4v6_MOD_PCDSCD :
          value->rValue = model->BSIM4v6pcdscd;
            return(OK);
         case  BSIM4v6_MOD_PCIT :
          value->rValue = model->BSIM4v6pcit;
            return(OK);
        case  BSIM4v6_MOD_PNFACTOR :
          value->rValue = model->BSIM4v6pnfactor;
            return(OK);
        case BSIM4v6_MOD_PXJ:
            value->rValue = model->BSIM4v6pxj;
            return(OK);
        case BSIM4v6_MOD_PVSAT:
            value->rValue = model->BSIM4v6pvsat;
            return(OK);
        case BSIM4v6_MOD_PAT:
            value->rValue = model->BSIM4v6pat;
            return(OK);
        case BSIM4v6_MOD_PA0:
            value->rValue = model->BSIM4v6pa0;
            return(OK);
        case BSIM4v6_MOD_PAGS:
            value->rValue = model->BSIM4v6pags;
            return(OK);
        case BSIM4v6_MOD_PA1:
            value->rValue = model->BSIM4v6pa1;
            return(OK);
        case BSIM4v6_MOD_PA2:
            value->rValue = model->BSIM4v6pa2;
            return(OK);
        case BSIM4v6_MOD_PKETA:
            value->rValue = model->BSIM4v6pketa;
            return(OK);   
        case BSIM4v6_MOD_PNSUB:
            value->rValue = model->BSIM4v6pnsub;
            return(OK);
        case BSIM4v6_MOD_PNDEP:
            value->rValue = model->BSIM4v6pndep;
            return(OK);
        case BSIM4v6_MOD_PNSD:
            value->rValue = model->BSIM4v6pnsd;
            return(OK);
        case BSIM4v6_MOD_PNGATE:
            value->rValue = model->BSIM4v6pngate;
            return(OK);
        case BSIM4v6_MOD_PGAMMA1:
            value->rValue = model->BSIM4v6pgamma1;
            return(OK);
        case BSIM4v6_MOD_PGAMMA2:
            value->rValue = model->BSIM4v6pgamma2;
            return(OK);
        case BSIM4v6_MOD_PVBX:
            value->rValue = model->BSIM4v6pvbx;
            return(OK);
        case BSIM4v6_MOD_PVBM:
            value->rValue = model->BSIM4v6pvbm;
            return(OK);
        case BSIM4v6_MOD_PXT:
            value->rValue = model->BSIM4v6pxt;
            return(OK);
        case  BSIM4v6_MOD_PK1:
          value->rValue = model->BSIM4v6pk1;
            return(OK);
        case  BSIM4v6_MOD_PKT1:
          value->rValue = model->BSIM4v6pkt1;
            return(OK);
        case  BSIM4v6_MOD_PKT1L:
          value->rValue = model->BSIM4v6pkt1l;
            return(OK);
        case  BSIM4v6_MOD_PKT2 :
          value->rValue = model->BSIM4v6pkt2;
            return(OK);
        case  BSIM4v6_MOD_PK2 :
          value->rValue = model->BSIM4v6pk2;
            return(OK);
        case  BSIM4v6_MOD_PK3:
          value->rValue = model->BSIM4v6pk3;
            return(OK);
        case  BSIM4v6_MOD_PK3B:
          value->rValue = model->BSIM4v6pk3b;
            return(OK);
        case  BSIM4v6_MOD_PW0:
          value->rValue = model->BSIM4v6pw0;
            return(OK);
        case  BSIM4v6_MOD_PLPE0:
          value->rValue = model->BSIM4v6plpe0;
            return(OK);
        case  BSIM4v6_MOD_PLPEB:
          value->rValue = model->BSIM4v6plpeb;
            return(OK);
        case  BSIM4v6_MOD_PDVTP0:
          value->rValue = model->BSIM4v6pdvtp0;
            return(OK);
        case  BSIM4v6_MOD_PDVTP1:
          value->rValue = model->BSIM4v6pdvtp1;
            return(OK);
        case  BSIM4v6_MOD_PDVT0 :                
          value->rValue = model->BSIM4v6pdvt0;
            return(OK);
        case  BSIM4v6_MOD_PDVT1 :             
          value->rValue = model->BSIM4v6pdvt1;
            return(OK);
        case  BSIM4v6_MOD_PDVT2 :             
          value->rValue = model->BSIM4v6pdvt2;
            return(OK);
        case  BSIM4v6_MOD_PDVT0W :                
          value->rValue = model->BSIM4v6pdvt0w;
            return(OK);
        case  BSIM4v6_MOD_PDVT1W :             
          value->rValue = model->BSIM4v6pdvt1w;
            return(OK);
        case  BSIM4v6_MOD_PDVT2W :             
          value->rValue = model->BSIM4v6pdvt2w;
            return(OK);
        case  BSIM4v6_MOD_PDROUT :           
          value->rValue = model->BSIM4v6pdrout;
            return(OK);
        case  BSIM4v6_MOD_PDSUB :           
          value->rValue = model->BSIM4v6pdsub;
            return(OK);
        case BSIM4v6_MOD_PVTH0:
            value->rValue = model->BSIM4v6pvth0; 
            return(OK);
        case BSIM4v6_MOD_PUA:
            value->rValue = model->BSIM4v6pua; 
            return(OK);
        case BSIM4v6_MOD_PUA1:
            value->rValue = model->BSIM4v6pua1; 
            return(OK);
        case BSIM4v6_MOD_PUB:
            value->rValue = model->BSIM4v6pub;  
            return(OK);
        case BSIM4v6_MOD_PUB1:
            value->rValue = model->BSIM4v6pub1;  
            return(OK);
        case BSIM4v6_MOD_PUC:
            value->rValue = model->BSIM4v6puc; 
            return(OK);
        case BSIM4v6_MOD_PUC1:
            value->rValue = model->BSIM4v6puc1; 
            return(OK);
        case BSIM4v6_MOD_PUD:
            value->rValue = model->BSIM4v6pud; 
            return(OK);
        case BSIM4v6_MOD_PUD1:
            value->rValue = model->BSIM4v6pud1; 
            return(OK);
        case BSIM4v6_MOD_PUP:
            value->rValue = model->BSIM4v6pup; 
            return(OK);
        case BSIM4v6_MOD_PLP:
            value->rValue = model->BSIM4v6plp; 
            return(OK);
        case BSIM4v6_MOD_PU0:
            value->rValue = model->BSIM4v6pu0;
            return(OK);
        case BSIM4v6_MOD_PUTE:
            value->rValue = model->BSIM4v6pute;
            return(OK);
        case BSIM4v6_MOD_PUCSTE:
            value->rValue = model->BSIM4v6pucste;
            return(OK);
        case BSIM4v6_MOD_PVOFF:
            value->rValue = model->BSIM4v6pvoff;
            return(OK);
        case BSIM4v6_MOD_PTVOFF:
            value->rValue = model->BSIM4v6ptvoff;
            return(OK);
        case BSIM4v6_MOD_PMINV:
            value->rValue = model->BSIM4v6pminv;
            return(OK);
        case BSIM4v6_MOD_PMINVCV:
            value->rValue = model->BSIM4v6pminvcv;
            return(OK);
        case BSIM4v6_MOD_PFPROUT:
            value->rValue = model->BSIM4v6pfprout;
            return(OK);
        case BSIM4v6_MOD_PPDITS:
            value->rValue = model->BSIM4v6ppdits;
            return(OK);
        case BSIM4v6_MOD_PPDITSD:
            value->rValue = model->BSIM4v6ppditsd;
            return(OK);
        case BSIM4v6_MOD_PDELTA:
            value->rValue = model->BSIM4v6pdelta;
            return(OK);
        case BSIM4v6_MOD_PRDSW:
            value->rValue = model->BSIM4v6prdsw; 
            return(OK);             
        case BSIM4v6_MOD_PRDW:
            value->rValue = model->BSIM4v6prdw;
            return(OK);
        case BSIM4v6_MOD_PRSW:
            value->rValue = model->BSIM4v6prsw;
            return(OK);
        case BSIM4v6_MOD_PPRWB:
            value->rValue = model->BSIM4v6pprwb; 
            return(OK);             
        case BSIM4v6_MOD_PPRWG:
            value->rValue = model->BSIM4v6pprwg; 
            return(OK);             
        case BSIM4v6_MOD_PPRT:
            value->rValue = model->BSIM4v6pprt; 
            return(OK);              
        case BSIM4v6_MOD_PETA0:
            value->rValue = model->BSIM4v6peta0; 
            return(OK);               
        case BSIM4v6_MOD_PETAB:
            value->rValue = model->BSIM4v6petab; 
            return(OK);               
        case BSIM4v6_MOD_PPCLM:
            value->rValue = model->BSIM4v6ppclm; 
            return(OK);               
        case BSIM4v6_MOD_PPDIBL1:
            value->rValue = model->BSIM4v6ppdibl1; 
            return(OK);               
        case BSIM4v6_MOD_PPDIBL2:
            value->rValue = model->BSIM4v6ppdibl2; 
            return(OK);               
        case BSIM4v6_MOD_PPDIBLB:
            value->rValue = model->BSIM4v6ppdiblb; 
            return(OK);               
        case BSIM4v6_MOD_PPSCBE1:
            value->rValue = model->BSIM4v6ppscbe1; 
            return(OK);               
        case BSIM4v6_MOD_PPSCBE2:
            value->rValue = model->BSIM4v6ppscbe2; 
            return(OK);               
        case BSIM4v6_MOD_PPVAG:
            value->rValue = model->BSIM4v6ppvag; 
            return(OK);               
        case BSIM4v6_MOD_PWR:
            value->rValue = model->BSIM4v6pwr;
            return(OK);
        case BSIM4v6_MOD_PDWG:
            value->rValue = model->BSIM4v6pdwg;
            return(OK);
        case BSIM4v6_MOD_PDWB:
            value->rValue = model->BSIM4v6pdwb;
            return(OK);
        case BSIM4v6_MOD_PB0:
            value->rValue = model->BSIM4v6pb0;
            return(OK);
        case BSIM4v6_MOD_PB1:
            value->rValue = model->BSIM4v6pb1;
            return(OK);
        case BSIM4v6_MOD_PALPHA0:
            value->rValue = model->BSIM4v6palpha0;
            return(OK);
        case BSIM4v6_MOD_PALPHA1:
            value->rValue = model->BSIM4v6palpha1;
            return(OK);
        case BSIM4v6_MOD_PBETA0:
            value->rValue = model->BSIM4v6pbeta0;
            return(OK);
        case BSIM4v6_MOD_PAGIDL:
            value->rValue = model->BSIM4v6pagidl;
            return(OK);
        case BSIM4v6_MOD_PBGIDL:
            value->rValue = model->BSIM4v6pbgidl;
            return(OK);
        case BSIM4v6_MOD_PCGIDL:
            value->rValue = model->BSIM4v6pcgidl;
            return(OK);
        case BSIM4v6_MOD_PEGIDL:
            value->rValue = model->BSIM4v6pegidl;
            return(OK);
        case BSIM4v6_MOD_PAGISL:
            value->rValue = model->BSIM4v6pagisl;
            return(OK);
        case BSIM4v6_MOD_PBGISL:
            value->rValue = model->BSIM4v6pbgisl;
            return(OK);
        case BSIM4v6_MOD_PCGISL:
            value->rValue = model->BSIM4v6pcgisl;
            return(OK);
        case BSIM4v6_MOD_PEGISL:
            value->rValue = model->BSIM4v6pegisl;
            return(OK);
        case BSIM4v6_MOD_PAIGC:
            value->rValue = model->BSIM4v6paigc;
            return(OK);
        case BSIM4v6_MOD_PBIGC:
            value->rValue = model->BSIM4v6pbigc;
            return(OK);
        case BSIM4v6_MOD_PCIGC:
            value->rValue = model->BSIM4v6pcigc;
            return(OK);
        case BSIM4v6_MOD_PAIGSD:
            value->rValue = model->BSIM4v6paigsd;
            return(OK);
        case BSIM4v6_MOD_PBIGSD:
            value->rValue = model->BSIM4v6pbigsd;
            return(OK);
        case BSIM4v6_MOD_PCIGSD:
            value->rValue = model->BSIM4v6pcigsd;
            return(OK);
        case BSIM4v6_MOD_PAIGS:
            value->rValue = model->BSIM4v6paigs;
            return(OK);
        case BSIM4v6_MOD_PBIGS:
            value->rValue = model->BSIM4v6pbigs;
            return(OK);
        case BSIM4v6_MOD_PCIGS:
            value->rValue = model->BSIM4v6pcigs;
            return(OK);
        case BSIM4v6_MOD_PAIGD:
            value->rValue = model->BSIM4v6paigd;
            return(OK);
        case BSIM4v6_MOD_PBIGD:
            value->rValue = model->BSIM4v6pbigd;
            return(OK);
        case BSIM4v6_MOD_PCIGD:
            value->rValue = model->BSIM4v6pcigd;
            return(OK);
        case BSIM4v6_MOD_PAIGBACC:
            value->rValue = model->BSIM4v6paigbacc;
            return(OK);
        case BSIM4v6_MOD_PBIGBACC:
            value->rValue = model->BSIM4v6pbigbacc;
            return(OK);
        case BSIM4v6_MOD_PCIGBACC:
            value->rValue = model->BSIM4v6pcigbacc;
            return(OK);
        case BSIM4v6_MOD_PAIGBINV:
            value->rValue = model->BSIM4v6paigbinv;
            return(OK);
        case BSIM4v6_MOD_PBIGBINV:
            value->rValue = model->BSIM4v6pbigbinv;
            return(OK);
        case BSIM4v6_MOD_PCIGBINV:
            value->rValue = model->BSIM4v6pcigbinv;
            return(OK);
        case BSIM4v6_MOD_PNIGC:
            value->rValue = model->BSIM4v6pnigc;
            return(OK);
        case BSIM4v6_MOD_PNIGBACC:
            value->rValue = model->BSIM4v6pnigbacc;
            return(OK);
        case BSIM4v6_MOD_PNIGBINV:
            value->rValue = model->BSIM4v6pnigbinv;
            return(OK);
        case BSIM4v6_MOD_PNTOX:
            value->rValue = model->BSIM4v6pntox;
            return(OK);
        case BSIM4v6_MOD_PEIGBINV:
            value->rValue = model->BSIM4v6peigbinv;
            return(OK);
        case BSIM4v6_MOD_PPIGCD:
            value->rValue = model->BSIM4v6ppigcd;
            return(OK);
        case BSIM4v6_MOD_PPOXEDGE:
            value->rValue = model->BSIM4v6ppoxedge;
            return(OK);
        case BSIM4v6_MOD_PPHIN:
            value->rValue = model->BSIM4v6pphin;
            return(OK);
        case BSIM4v6_MOD_PXRCRG1:
            value->rValue = model->BSIM4v6pxrcrg1;
            return(OK);
        case BSIM4v6_MOD_PXRCRG2:
            value->rValue = model->BSIM4v6pxrcrg2;
            return(OK);
        case BSIM4v6_MOD_PEU:
            value->rValue = model->BSIM4v6peu;
            return(OK);
        case BSIM4v6_MOD_PUCS:
            value->rValue = model->BSIM4v6pucs;
            return(OK);
        case BSIM4v6_MOD_PVFB:
            value->rValue = model->BSIM4v6pvfb;
            return(OK);

        case BSIM4v6_MOD_PCGSL:
            value->rValue = model->BSIM4v6pcgsl;
            return(OK);
        case BSIM4v6_MOD_PCGDL:
            value->rValue = model->BSIM4v6pcgdl;
            return(OK);
        case BSIM4v6_MOD_PCKAPPAS:
            value->rValue = model->BSIM4v6pckappas;
            return(OK);
        case BSIM4v6_MOD_PCKAPPAD:
            value->rValue = model->BSIM4v6pckappad;
            return(OK);
        case BSIM4v6_MOD_PCF:
            value->rValue = model->BSIM4v6pcf;
            return(OK);
        case BSIM4v6_MOD_PCLC:
            value->rValue = model->BSIM4v6pclc;
            return(OK);
        case BSIM4v6_MOD_PCLE:
            value->rValue = model->BSIM4v6pcle;
            return(OK);
        case BSIM4v6_MOD_PVFBCV:
            value->rValue = model->BSIM4v6pvfbcv;
            return(OK);
        case BSIM4v6_MOD_PACDE:
            value->rValue = model->BSIM4v6pacde;
            return(OK);
        case BSIM4v6_MOD_PMOIN:
            value->rValue = model->BSIM4v6pmoin;
            return(OK);
        case BSIM4v6_MOD_PNOFF:
            value->rValue = model->BSIM4v6pnoff;
            return(OK);
        case BSIM4v6_MOD_PVOFFCV:
            value->rValue = model->BSIM4v6pvoffcv;
            return(OK);
        case BSIM4v6_MOD_PVFBSDOFF:
            value->rValue = model->BSIM4v6pvfbsdoff;
            return(OK);
        case BSIM4v6_MOD_PTVFBSDOFF:
            value->rValue = model->BSIM4v6ptvfbsdoff;
            return(OK);

        case  BSIM4v6_MOD_TNOM :
          value->rValue = model->BSIM4v6tnom;
            return(OK);
        case BSIM4v6_MOD_CGSO:
            value->rValue = model->BSIM4v6cgso; 
            return(OK);
        case BSIM4v6_MOD_CGDO:
            value->rValue = model->BSIM4v6cgdo; 
            return(OK);
        case BSIM4v6_MOD_CGBO:
            value->rValue = model->BSIM4v6cgbo; 
            return(OK);
        case BSIM4v6_MOD_XPART:
            value->rValue = model->BSIM4v6xpart; 
            return(OK);
        case BSIM4v6_MOD_RSH:
            value->rValue = model->BSIM4v6sheetResistance; 
            return(OK);
        case BSIM4v6_MOD_JSS:
            value->rValue = model->BSIM4v6SjctSatCurDensity; 
            return(OK);
        case BSIM4v6_MOD_JSWS:
            value->rValue = model->BSIM4v6SjctSidewallSatCurDensity; 
            return(OK);
        case BSIM4v6_MOD_JSWGS:
            value->rValue = model->BSIM4v6SjctGateSidewallSatCurDensity;
            return(OK);
        case BSIM4v6_MOD_PBS:
            value->rValue = model->BSIM4v6SbulkJctPotential; 
            return(OK);
        case BSIM4v6_MOD_MJS:
            value->rValue = model->BSIM4v6SbulkJctBotGradingCoeff; 
            return(OK);
        case BSIM4v6_MOD_PBSWS:
            value->rValue = model->BSIM4v6SsidewallJctPotential; 
            return(OK);
        case BSIM4v6_MOD_MJSWS:
            value->rValue = model->BSIM4v6SbulkJctSideGradingCoeff; 
            return(OK);
        case BSIM4v6_MOD_CJS:
            value->rValue = model->BSIM4v6SunitAreaJctCap; 
            return(OK);
        case BSIM4v6_MOD_CJSWS:
            value->rValue = model->BSIM4v6SunitLengthSidewallJctCap; 
            return(OK);
        case BSIM4v6_MOD_PBSWGS:
            value->rValue = model->BSIM4v6SGatesidewallJctPotential; 
            return(OK);
        case BSIM4v6_MOD_MJSWGS:
            value->rValue = model->BSIM4v6SbulkJctGateSideGradingCoeff; 
            return(OK);
        case BSIM4v6_MOD_CJSWGS:
            value->rValue = model->BSIM4v6SunitLengthGateSidewallJctCap; 
            return(OK);
        case BSIM4v6_MOD_NJS:
            value->rValue = model->BSIM4v6SjctEmissionCoeff; 
            return(OK);
        case BSIM4v6_MOD_XTIS:
            value->rValue = model->BSIM4v6SjctTempExponent; 
            return(OK);
        case BSIM4v6_MOD_JSD:
            value->rValue = model->BSIM4v6DjctSatCurDensity;
            return(OK);
        case BSIM4v6_MOD_JSWD:
            value->rValue = model->BSIM4v6DjctSidewallSatCurDensity;
            return(OK);
        case BSIM4v6_MOD_JSWGD:
            value->rValue = model->BSIM4v6DjctGateSidewallSatCurDensity;
            return(OK);
        case BSIM4v6_MOD_PBD:
            value->rValue = model->BSIM4v6DbulkJctPotential;
            return(OK);
        case BSIM4v6_MOD_MJD:
            value->rValue = model->BSIM4v6DbulkJctBotGradingCoeff;
            return(OK);
        case BSIM4v6_MOD_PBSWD:
            value->rValue = model->BSIM4v6DsidewallJctPotential;
            return(OK);
        case BSIM4v6_MOD_MJSWD:
            value->rValue = model->BSIM4v6DbulkJctSideGradingCoeff;
            return(OK);
        case BSIM4v6_MOD_CJD:
            value->rValue = model->BSIM4v6DunitAreaJctCap;
            return(OK);
        case BSIM4v6_MOD_CJSWD:
            value->rValue = model->BSIM4v6DunitLengthSidewallJctCap;
            return(OK);
        case BSIM4v6_MOD_PBSWGD:
            value->rValue = model->BSIM4v6DGatesidewallJctPotential;
            return(OK);
        case BSIM4v6_MOD_MJSWGD:
            value->rValue = model->BSIM4v6DbulkJctGateSideGradingCoeff;
            return(OK);
        case BSIM4v6_MOD_CJSWGD:
            value->rValue = model->BSIM4v6DunitLengthGateSidewallJctCap;
            return(OK);
        case BSIM4v6_MOD_NJD:
            value->rValue = model->BSIM4v6DjctEmissionCoeff;
            return(OK);
        case BSIM4v6_MOD_XTID:
            value->rValue = model->BSIM4v6DjctTempExponent;
            return(OK);
        case BSIM4v6_MOD_LINT:
            value->rValue = model->BSIM4v6Lint; 
            return(OK);
        case BSIM4v6_MOD_LL:
            value->rValue = model->BSIM4v6Ll;
            return(OK);
        case BSIM4v6_MOD_LLC:
            value->rValue = model->BSIM4v6Llc;
            return(OK);
        case BSIM4v6_MOD_LLN:
            value->rValue = model->BSIM4v6Lln;
            return(OK);
        case BSIM4v6_MOD_LW:
            value->rValue = model->BSIM4v6Lw;
            return(OK);
        case BSIM4v6_MOD_LWC:
            value->rValue = model->BSIM4v6Lwc;
            return(OK);
        case BSIM4v6_MOD_LWN:
            value->rValue = model->BSIM4v6Lwn;
            return(OK);
        case BSIM4v6_MOD_LWL:
            value->rValue = model->BSIM4v6Lwl;
            return(OK);
        case BSIM4v6_MOD_LWLC:
            value->rValue = model->BSIM4v6Lwlc;
            return(OK);
        case BSIM4v6_MOD_LMIN:
            value->rValue = model->BSIM4v6Lmin;
            return(OK);
        case BSIM4v6_MOD_LMAX:
            value->rValue = model->BSIM4v6Lmax;
            return(OK);
        case BSIM4v6_MOD_WINT:
            value->rValue = model->BSIM4v6Wint;
            return(OK);
        case BSIM4v6_MOD_WL:
            value->rValue = model->BSIM4v6Wl;
            return(OK);
        case BSIM4v6_MOD_WLC:
            value->rValue = model->BSIM4v6Wlc;
            return(OK);
        case BSIM4v6_MOD_WLN:
            value->rValue = model->BSIM4v6Wln;
            return(OK);
        case BSIM4v6_MOD_WW:
            value->rValue = model->BSIM4v6Ww;
            return(OK);
        case BSIM4v6_MOD_WWC:
            value->rValue = model->BSIM4v6Wwc;
            return(OK);
        case BSIM4v6_MOD_WWN:
            value->rValue = model->BSIM4v6Wwn;
            return(OK);
        case BSIM4v6_MOD_WWL:
            value->rValue = model->BSIM4v6Wwl;
            return(OK);
        case BSIM4v6_MOD_WWLC:
            value->rValue = model->BSIM4v6Wwlc;
            return(OK);
        case BSIM4v6_MOD_WMIN:
            value->rValue = model->BSIM4v6Wmin;
            return(OK);
        case BSIM4v6_MOD_WMAX:
            value->rValue = model->BSIM4v6Wmax;
            return(OK);

        /* stress effect */
        case BSIM4v6_MOD_SAREF:
            value->rValue = model->BSIM4v6saref;
            return(OK);
        case BSIM4v6_MOD_SBREF:
            value->rValue = model->BSIM4v6sbref;
            return(OK);
	case BSIM4v6_MOD_WLOD:
            value->rValue = model->BSIM4v6wlod;
            return(OK);
        case BSIM4v6_MOD_KU0:
            value->rValue = model->BSIM4v6ku0;
            return(OK);
        case BSIM4v6_MOD_KVSAT:
            value->rValue = model->BSIM4v6kvsat;
            return(OK);
        case BSIM4v6_MOD_KVTH0:
            value->rValue = model->BSIM4v6kvth0;
            return(OK);
        case BSIM4v6_MOD_TKU0:
            value->rValue = model->BSIM4v6tku0;
            return(OK);
        case BSIM4v6_MOD_LLODKU0:
            value->rValue = model->BSIM4v6llodku0;
            return(OK);
        case BSIM4v6_MOD_WLODKU0:
            value->rValue = model->BSIM4v6wlodku0;
            return(OK);
        case BSIM4v6_MOD_LLODVTH:
            value->rValue = model->BSIM4v6llodvth;
            return(OK);
        case BSIM4v6_MOD_WLODVTH:
            value->rValue = model->BSIM4v6wlodvth;
            return(OK);
        case BSIM4v6_MOD_LKU0:
            value->rValue = model->BSIM4v6lku0;
            return(OK);
        case BSIM4v6_MOD_WKU0:
            value->rValue = model->BSIM4v6wku0;
            return(OK);
        case BSIM4v6_MOD_PKU0:
            value->rValue = model->BSIM4v6pku0;
            return(OK);
        case BSIM4v6_MOD_LKVTH0:
            value->rValue = model->BSIM4v6lkvth0;
            return(OK);
        case BSIM4v6_MOD_WKVTH0:
            value->rValue = model->BSIM4v6wkvth0;
            return(OK);
        case BSIM4v6_MOD_PKVTH0:
            value->rValue = model->BSIM4v6pkvth0;
            return(OK);
        case BSIM4v6_MOD_STK2:
            value->rValue = model->BSIM4v6stk2;
            return(OK);
        case BSIM4v6_MOD_LODK2:
            value->rValue = model->BSIM4v6lodk2;
            return(OK);
        case BSIM4v6_MOD_STETA0:
            value->rValue = model->BSIM4v6steta0;
            return(OK);
        case BSIM4v6_MOD_LODETA0:
            value->rValue = model->BSIM4v6lodeta0;
            return(OK);

        /* Well Proximity Effect  */
        case BSIM4v6_MOD_WEB:
            value->rValue = model->BSIM4v6web;
            return(OK);
        case BSIM4v6_MOD_WEC:
            value->rValue = model->BSIM4v6wec;
            return(OK);
        case BSIM4v6_MOD_KVTH0WE:
            value->rValue = model->BSIM4v6kvth0we;
            return(OK);
        case BSIM4v6_MOD_K2WE:
            value->rValue = model->BSIM4v6k2we;
            return(OK);
        case BSIM4v6_MOD_KU0WE:
            value->rValue = model->BSIM4v6ku0we;
            return(OK);
        case BSIM4v6_MOD_SCREF:
            value->rValue = model->BSIM4v6scref;
            return(OK);
        case BSIM4v6_MOD_WPEMOD:
            value->rValue = model->BSIM4v6wpemod;
            return(OK);
        case BSIM4v6_MOD_LKVTH0WE:
            value->rValue = model->BSIM4v6lkvth0we;
            return(OK);
        case BSIM4v6_MOD_LK2WE:
            value->rValue = model->BSIM4v6lk2we;
            return(OK);
        case BSIM4v6_MOD_LKU0WE:
            value->rValue = model->BSIM4v6lku0we;
            return(OK);
        case BSIM4v6_MOD_WKVTH0WE:
            value->rValue = model->BSIM4v6wkvth0we;
            return(OK);
        case BSIM4v6_MOD_WK2WE:
            value->rValue = model->BSIM4v6wk2we;
            return(OK);
        case BSIM4v6_MOD_WKU0WE:
            value->rValue = model->BSIM4v6wku0we;
            return(OK);
        case BSIM4v6_MOD_PKVTH0WE:
            value->rValue = model->BSIM4v6pkvth0we;
            return(OK);
        case BSIM4v6_MOD_PK2WE:
            value->rValue = model->BSIM4v6pk2we;
            return(OK);
        case BSIM4v6_MOD_PKU0WE:
            value->rValue = model->BSIM4v6pku0we;
            return(OK);

        case BSIM4v6_MOD_NOIA:
            value->rValue = model->BSIM4v6oxideTrapDensityA;
            return(OK);
        case BSIM4v6_MOD_NOIB:
            value->rValue = model->BSIM4v6oxideTrapDensityB;
            return(OK);
        case BSIM4v6_MOD_NOIC:
            value->rValue = model->BSIM4v6oxideTrapDensityC;
            return(OK);
        case BSIM4v6_MOD_EM:
            value->rValue = model->BSIM4v6em;
            return(OK);
        case BSIM4v6_MOD_EF:
            value->rValue = model->BSIM4v6ef;
            return(OK);
        case BSIM4v6_MOD_AF:
            value->rValue = model->BSIM4v6af;
            return(OK);
        case BSIM4v6_MOD_KF:
            value->rValue = model->BSIM4v6kf;
            return(OK);

        case BSIM4v6_MOD_VGS_MAX:
            value->rValue = model->BSIM4v6vgsMax;
            return(OK);
        case BSIM4v6_MOD_VGD_MAX:
            value->rValue = model->BSIM4v6vgdMax;
            return(OK);
        case BSIM4v6_MOD_VGB_MAX:
            value->rValue = model->BSIM4v6vgbMax;
            return(OK);
        case BSIM4v6_MOD_VDS_MAX:
            value->rValue = model->BSIM4v6vdsMax;
            return(OK);
        case BSIM4v6_MOD_VBS_MAX:
            value->rValue = model->BSIM4v6vbsMax;
            return(OK);
        case BSIM4v6_MOD_VBD_MAX:
            value->rValue = model->BSIM4v6vbdMax;
            return(OK);
        case BSIM4v6_MOD_VGSR_MAX:
            value->rValue = model->BSIM4v6vgsrMax;
            return(OK);
        case BSIM4v6_MOD_VGDR_MAX:
            value->rValue = model->BSIM4v6vgdrMax;
            return(OK);
        case BSIM4v6_MOD_VGBR_MAX:
            value->rValue = model->BSIM4v6vgbrMax;
            return(OK);
        case BSIM4v6_MOD_VBSR_MAX:
            value->rValue = model->BSIM4v6vbsrMax;
            return(OK);
        case BSIM4v6_MOD_VBDR_MAX:
            value->rValue = model->BSIM4v6vbdrMax;
            return(OK);

        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}



