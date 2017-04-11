/**** BSIM4.5.0 Released by Xuemei (Jane) Xi 07/29/2005 ****/

/**********
 * Copyright 2005 Regents of the University of California. All rights reserved.
 * File: b4mask.c of BSIM4.5.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 * Modified by Xuemei Xi, 11/15/2002.
 * Modified by Xuemei Xi, 05/09/2003.
 * Modified by Xuemei Xi, Mohan Dunga, 07/29/2005.
 **********/


#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "bsim4v5def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
BSIM4v5mAsk(
CKTcircuit *ckt,
GENmodel *inst,
int which,
IFvalue *value)
{
    BSIM4v5model *model = (BSIM4v5model *)inst;

    NG_IGNORE(ckt);

    switch(which) 
    {   case BSIM4v5_MOD_MOBMOD :
            value->iValue = model->BSIM4v5mobMod; 
            return(OK);
        case BSIM4v5_MOD_PARAMCHK :
            value->iValue = model->BSIM4v5paramChk; 
            return(OK);
        case BSIM4v5_MOD_BINUNIT :
            value->iValue = model->BSIM4v5binUnit; 
            return(OK);
        case BSIM4v5_MOD_CAPMOD :
            value->iValue = model->BSIM4v5capMod; 
            return(OK);
        case BSIM4v5_MOD_DIOMOD :
            value->iValue = model->BSIM4v5dioMod;
            return(OK);
        case BSIM4v5_MOD_TRNQSMOD :
            value->iValue = model->BSIM4v5trnqsMod;
            return(OK);
        case BSIM4v5_MOD_ACNQSMOD :
            value->iValue = model->BSIM4v5acnqsMod;
            return(OK);
        case BSIM4v5_MOD_FNOIMOD :
            value->iValue = model->BSIM4v5fnoiMod; 
            return(OK);
        case BSIM4v5_MOD_TNOIMOD :
            value->iValue = model->BSIM4v5tnoiMod;
            return(OK);
        case BSIM4v5_MOD_RDSMOD :
            value->iValue = model->BSIM4v5rdsMod;
            return(OK);
        case BSIM4v5_MOD_RBODYMOD :
            value->iValue = model->BSIM4v5rbodyMod;
            return(OK);
        case BSIM4v5_MOD_RGATEMOD :
            value->iValue = model->BSIM4v5rgateMod;
            return(OK);
        case BSIM4v5_MOD_PERMOD :
            value->iValue = model->BSIM4v5perMod;
            return(OK);
        case BSIM4v5_MOD_GEOMOD :
            value->iValue = model->BSIM4v5geoMod;
            return(OK);
        case BSIM4v5_MOD_RGEOMOD :
            value->iValue = model->BSIM4v5rgeoMod;
            return(OK);
        case BSIM4v5_MOD_IGCMOD :
            value->iValue = model->BSIM4v5igcMod;
            return(OK);
        case BSIM4v5_MOD_IGBMOD :
            value->iValue = model->BSIM4v5igbMod;
            return(OK);
        case  BSIM4v5_MOD_TEMPMOD :
            value->iValue = model->BSIM4v5tempMod;
            return(OK);

        case  BSIM4v5_MOD_VERSION :
          value->sValue = model->BSIM4v5version;
            return(OK);
        case  BSIM4v5_MOD_TOXREF :
          value->rValue = model->BSIM4v5toxref;
          return(OK);
        case  BSIM4v5_MOD_TOXE :
          value->rValue = model->BSIM4v5toxe;
            return(OK);
        case  BSIM4v5_MOD_TOXP :
          value->rValue = model->BSIM4v5toxp;
            return(OK);
        case  BSIM4v5_MOD_TOXM :
          value->rValue = model->BSIM4v5toxm;
            return(OK);
        case  BSIM4v5_MOD_DTOX :
          value->rValue = model->BSIM4v5dtox;
            return(OK);
        case  BSIM4v5_MOD_EPSROX :
          value->rValue = model->BSIM4v5epsrox;
            return(OK);
        case  BSIM4v5_MOD_CDSC :
          value->rValue = model->BSIM4v5cdsc;
            return(OK);
        case  BSIM4v5_MOD_CDSCB :
          value->rValue = model->BSIM4v5cdscb;
            return(OK);

        case  BSIM4v5_MOD_CDSCD :
          value->rValue = model->BSIM4v5cdscd;
            return(OK);

        case  BSIM4v5_MOD_CIT :
          value->rValue = model->BSIM4v5cit;
            return(OK);
        case  BSIM4v5_MOD_NFACTOR :
          value->rValue = model->BSIM4v5nfactor;
            return(OK);
        case BSIM4v5_MOD_XJ:
            value->rValue = model->BSIM4v5xj;
            return(OK);
        case BSIM4v5_MOD_VSAT:
            value->rValue = model->BSIM4v5vsat;
            return(OK);
        case BSIM4v5_MOD_VTL:
            value->rValue = model->BSIM4v5vtl;
            return(OK);
        case BSIM4v5_MOD_XN:
            value->rValue = model->BSIM4v5xn;
            return(OK);
        case BSIM4v5_MOD_LC:
            value->rValue = model->BSIM4v5lc;
            return(OK);
        case BSIM4v5_MOD_LAMBDA:
            value->rValue = model->BSIM4v5lambda;
            return(OK);
        case BSIM4v5_MOD_AT:
            value->rValue = model->BSIM4v5at;
            return(OK);
        case BSIM4v5_MOD_A0:
            value->rValue = model->BSIM4v5a0;
            return(OK);

        case BSIM4v5_MOD_AGS:
            value->rValue = model->BSIM4v5ags;
            return(OK);

        case BSIM4v5_MOD_A1:
            value->rValue = model->BSIM4v5a1;
            return(OK);
        case BSIM4v5_MOD_A2:
            value->rValue = model->BSIM4v5a2;
            return(OK);
        case BSIM4v5_MOD_KETA:
            value->rValue = model->BSIM4v5keta;
            return(OK);   
        case BSIM4v5_MOD_NSUB:
            value->rValue = model->BSIM4v5nsub;
            return(OK);
        case BSIM4v5_MOD_NDEP:
            value->rValue = model->BSIM4v5ndep;
            return(OK);
        case BSIM4v5_MOD_NSD:
            value->rValue = model->BSIM4v5nsd;
            return(OK);
        case BSIM4v5_MOD_NGATE:
            value->rValue = model->BSIM4v5ngate;
            return(OK);
        case BSIM4v5_MOD_GAMMA1:
            value->rValue = model->BSIM4v5gamma1;
            return(OK);
        case BSIM4v5_MOD_GAMMA2:
            value->rValue = model->BSIM4v5gamma2;
            return(OK);
        case BSIM4v5_MOD_VBX:
            value->rValue = model->BSIM4v5vbx;
            return(OK);
        case BSIM4v5_MOD_VBM:
            value->rValue = model->BSIM4v5vbm;
            return(OK);
        case BSIM4v5_MOD_XT:
            value->rValue = model->BSIM4v5xt;
            return(OK);
        case  BSIM4v5_MOD_K1:
          value->rValue = model->BSIM4v5k1;
            return(OK);
        case  BSIM4v5_MOD_KT1:
          value->rValue = model->BSIM4v5kt1;
            return(OK);
        case  BSIM4v5_MOD_KT1L:
          value->rValue = model->BSIM4v5kt1l;
            return(OK);
        case  BSIM4v5_MOD_KT2 :
          value->rValue = model->BSIM4v5kt2;
            return(OK);
        case  BSIM4v5_MOD_K2 :
          value->rValue = model->BSIM4v5k2;
            return(OK);
        case  BSIM4v5_MOD_K3:
          value->rValue = model->BSIM4v5k3;
            return(OK);
        case  BSIM4v5_MOD_K3B:
          value->rValue = model->BSIM4v5k3b;
            return(OK);
        case  BSIM4v5_MOD_W0:
          value->rValue = model->BSIM4v5w0;
            return(OK);
        case  BSIM4v5_MOD_LPE0:
          value->rValue = model->BSIM4v5lpe0;
            return(OK);
        case  BSIM4v5_MOD_LPEB:
          value->rValue = model->BSIM4v5lpeb;
            return(OK);
        case  BSIM4v5_MOD_DVTP0:
          value->rValue = model->BSIM4v5dvtp0;
            return(OK);
        case  BSIM4v5_MOD_DVTP1:
          value->rValue = model->BSIM4v5dvtp1;
            return(OK);
        case  BSIM4v5_MOD_DVT0 :                
          value->rValue = model->BSIM4v5dvt0;
            return(OK);
        case  BSIM4v5_MOD_DVT1 :             
          value->rValue = model->BSIM4v5dvt1;
            return(OK);
        case  BSIM4v5_MOD_DVT2 :             
          value->rValue = model->BSIM4v5dvt2;
            return(OK);
        case  BSIM4v5_MOD_DVT0W :                
          value->rValue = model->BSIM4v5dvt0w;
            return(OK);
        case  BSIM4v5_MOD_DVT1W :             
          value->rValue = model->BSIM4v5dvt1w;
            return(OK);
        case  BSIM4v5_MOD_DVT2W :             
          value->rValue = model->BSIM4v5dvt2w;
            return(OK);
        case  BSIM4v5_MOD_DROUT :           
          value->rValue = model->BSIM4v5drout;
            return(OK);
        case  BSIM4v5_MOD_DSUB :           
          value->rValue = model->BSIM4v5dsub;
            return(OK);
        case BSIM4v5_MOD_VTH0:
            value->rValue = model->BSIM4v5vth0; 
            return(OK);
        case BSIM4v5_MOD_EU:
            value->rValue = model->BSIM4v5eu;
            return(OK);
        case BSIM4v5_MOD_UA:
            value->rValue = model->BSIM4v5ua; 
            return(OK);
        case BSIM4v5_MOD_UA1:
            value->rValue = model->BSIM4v5ua1; 
            return(OK);
        case BSIM4v5_MOD_UB:
            value->rValue = model->BSIM4v5ub;  
            return(OK);
        case BSIM4v5_MOD_UB1:
            value->rValue = model->BSIM4v5ub1;  
            return(OK);
        case BSIM4v5_MOD_UC:
            value->rValue = model->BSIM4v5uc; 
            return(OK);
        case BSIM4v5_MOD_UC1:
            value->rValue = model->BSIM4v5uc1; 
            return(OK);
        case BSIM4v5_MOD_UD:
            value->rValue = model->BSIM4v5ud; 
            return(OK);
        case BSIM4v5_MOD_UD1:
            value->rValue = model->BSIM4v5ud1; 
            return(OK);
        case BSIM4v5_MOD_UP:
            value->rValue = model->BSIM4v5up; 
            return(OK);
        case BSIM4v5_MOD_LP:
            value->rValue = model->BSIM4v5lp; 
            return(OK);
        case BSIM4v5_MOD_U0:
            value->rValue = model->BSIM4v5u0;
            return(OK);
        case BSIM4v5_MOD_UTE:
            value->rValue = model->BSIM4v5ute;
            return(OK);
        case BSIM4v5_MOD_VOFF:
            value->rValue = model->BSIM4v5voff;
            return(OK);
        case BSIM4v5_MOD_TVOFF:
            value->rValue = model->BSIM4v5tvoff;
            return(OK);
        case BSIM4v5_MOD_VFBSDOFF:
            value->rValue = model->BSIM4v5vfbsdoff;
            return(OK);
        case BSIM4v5_MOD_TVFBSDOFF:
            value->rValue = model->BSIM4v5tvfbsdoff;
            return(OK);
        case BSIM4v5_MOD_VOFFL:
            value->rValue = model->BSIM4v5voffl;
            return(OK);
        case BSIM4v5_MOD_MINV:
            value->rValue = model->BSIM4v5minv;
            return(OK);
        case BSIM4v5_MOD_FPROUT:
            value->rValue = model->BSIM4v5fprout;
            return(OK);
        case BSIM4v5_MOD_PDITS:
            value->rValue = model->BSIM4v5pdits;
            return(OK);
        case BSIM4v5_MOD_PDITSD:
            value->rValue = model->BSIM4v5pditsd;
            return(OK);
        case BSIM4v5_MOD_PDITSL:
            value->rValue = model->BSIM4v5pditsl;
            return(OK);
        case BSIM4v5_MOD_DELTA:
            value->rValue = model->BSIM4v5delta;
            return(OK);
        case BSIM4v5_MOD_RDSW:
            value->rValue = model->BSIM4v5rdsw; 
            return(OK);
        case BSIM4v5_MOD_RDSWMIN:
            value->rValue = model->BSIM4v5rdswmin;
            return(OK);
        case BSIM4v5_MOD_RDWMIN:
            value->rValue = model->BSIM4v5rdwmin;
            return(OK);
        case BSIM4v5_MOD_RSWMIN:
            value->rValue = model->BSIM4v5rswmin;
            return(OK);
        case BSIM4v5_MOD_RDW:
            value->rValue = model->BSIM4v5rdw;
            return(OK);
        case BSIM4v5_MOD_RSW:
            value->rValue = model->BSIM4v5rsw;
            return(OK);
        case BSIM4v5_MOD_PRWG:
            value->rValue = model->BSIM4v5prwg; 
            return(OK);             
        case BSIM4v5_MOD_PRWB:
            value->rValue = model->BSIM4v5prwb; 
            return(OK);             
        case BSIM4v5_MOD_PRT:
            value->rValue = model->BSIM4v5prt; 
            return(OK);              
        case BSIM4v5_MOD_ETA0:
            value->rValue = model->BSIM4v5eta0; 
            return(OK);               
        case BSIM4v5_MOD_ETAB:
            value->rValue = model->BSIM4v5etab; 
            return(OK);               
        case BSIM4v5_MOD_PCLM:
            value->rValue = model->BSIM4v5pclm; 
            return(OK);               
        case BSIM4v5_MOD_PDIBL1:
            value->rValue = model->BSIM4v5pdibl1; 
            return(OK);               
        case BSIM4v5_MOD_PDIBL2:
            value->rValue = model->BSIM4v5pdibl2; 
            return(OK);               
        case BSIM4v5_MOD_PDIBLB:
            value->rValue = model->BSIM4v5pdiblb; 
            return(OK);               
        case BSIM4v5_MOD_PSCBE1:
            value->rValue = model->BSIM4v5pscbe1; 
            return(OK);               
        case BSIM4v5_MOD_PSCBE2:
            value->rValue = model->BSIM4v5pscbe2; 
            return(OK);               
        case BSIM4v5_MOD_PVAG:
            value->rValue = model->BSIM4v5pvag; 
            return(OK);               
        case BSIM4v5_MOD_WR:
            value->rValue = model->BSIM4v5wr;
            return(OK);
        case BSIM4v5_MOD_DWG:
            value->rValue = model->BSIM4v5dwg;
            return(OK);
        case BSIM4v5_MOD_DWB:
            value->rValue = model->BSIM4v5dwb;
            return(OK);
        case BSIM4v5_MOD_B0:
            value->rValue = model->BSIM4v5b0;
            return(OK);
        case BSIM4v5_MOD_B1:
            value->rValue = model->BSIM4v5b1;
            return(OK);
        case BSIM4v5_MOD_ALPHA0:
            value->rValue = model->BSIM4v5alpha0;
            return(OK);
        case BSIM4v5_MOD_ALPHA1:
            value->rValue = model->BSIM4v5alpha1;
            return(OK);
        case BSIM4v5_MOD_BETA0:
            value->rValue = model->BSIM4v5beta0;
            return(OK);
        case BSIM4v5_MOD_AGIDL:
            value->rValue = model->BSIM4v5agidl;
            return(OK);
        case BSIM4v5_MOD_BGIDL:
            value->rValue = model->BSIM4v5bgidl;
            return(OK);
        case BSIM4v5_MOD_CGIDL:
            value->rValue = model->BSIM4v5cgidl;
            return(OK);
        case BSIM4v5_MOD_EGIDL:
            value->rValue = model->BSIM4v5egidl;
            return(OK);
        case BSIM4v5_MOD_AIGC:
            value->rValue = model->BSIM4v5aigc;
            return(OK);
        case BSIM4v5_MOD_BIGC:
            value->rValue = model->BSIM4v5bigc;
            return(OK);
        case BSIM4v5_MOD_CIGC:
            value->rValue = model->BSIM4v5cigc;
            return(OK);
        case BSIM4v5_MOD_AIGSD:
            value->rValue = model->BSIM4v5aigsd;
            return(OK);
        case BSIM4v5_MOD_BIGSD:
            value->rValue = model->BSIM4v5bigsd;
            return(OK);
        case BSIM4v5_MOD_CIGSD:
            value->rValue = model->BSIM4v5cigsd;
            return(OK);
        case BSIM4v5_MOD_AIGBACC:
            value->rValue = model->BSIM4v5aigbacc;
            return(OK);
        case BSIM4v5_MOD_BIGBACC:
            value->rValue = model->BSIM4v5bigbacc;
            return(OK);
        case BSIM4v5_MOD_CIGBACC:
            value->rValue = model->BSIM4v5cigbacc;
            return(OK);
        case BSIM4v5_MOD_AIGBINV:
            value->rValue = model->BSIM4v5aigbinv;
            return(OK);
        case BSIM4v5_MOD_BIGBINV:
            value->rValue = model->BSIM4v5bigbinv;
            return(OK);
        case BSIM4v5_MOD_CIGBINV:
            value->rValue = model->BSIM4v5cigbinv;
            return(OK);
        case BSIM4v5_MOD_NIGC:
            value->rValue = model->BSIM4v5nigc;
            return(OK);
        case BSIM4v5_MOD_NIGBACC:
            value->rValue = model->BSIM4v5nigbacc;
            return(OK);
        case BSIM4v5_MOD_NIGBINV:
            value->rValue = model->BSIM4v5nigbinv;
            return(OK);
        case BSIM4v5_MOD_NTOX:
            value->rValue = model->BSIM4v5ntox;
            return(OK);
        case BSIM4v5_MOD_EIGBINV:
            value->rValue = model->BSIM4v5eigbinv;
            return(OK);
        case BSIM4v5_MOD_PIGCD:
            value->rValue = model->BSIM4v5pigcd;
            return(OK);
        case BSIM4v5_MOD_POXEDGE:
            value->rValue = model->BSIM4v5poxedge;
            return(OK);
        case BSIM4v5_MOD_PHIN:
            value->rValue = model->BSIM4v5phin;
            return(OK);
        case BSIM4v5_MOD_XRCRG1:
            value->rValue = model->BSIM4v5xrcrg1;
            return(OK);
        case BSIM4v5_MOD_XRCRG2:
            value->rValue = model->BSIM4v5xrcrg2;
            return(OK);
        case BSIM4v5_MOD_TNOIA:
            value->rValue = model->BSIM4v5tnoia;
            return(OK);
        case BSIM4v5_MOD_TNOIB:
            value->rValue = model->BSIM4v5tnoib;
            return(OK);
        case BSIM4v5_MOD_RNOIA:
            value->rValue = model->BSIM4v5rnoia;
            return(OK);
        case BSIM4v5_MOD_RNOIB:
            value->rValue = model->BSIM4v5rnoib;
            return(OK);
        case BSIM4v5_MOD_NTNOI:
            value->rValue = model->BSIM4v5ntnoi;
            return(OK);
        case BSIM4v5_MOD_IJTHDFWD:
            value->rValue = model->BSIM4v5ijthdfwd;
            return(OK);
        case BSIM4v5_MOD_IJTHSFWD:
            value->rValue = model->BSIM4v5ijthsfwd;
            return(OK);
        case BSIM4v5_MOD_IJTHDREV:
            value->rValue = model->BSIM4v5ijthdrev;
            return(OK);
        case BSIM4v5_MOD_IJTHSREV:
            value->rValue = model->BSIM4v5ijthsrev;
            return(OK);
        case BSIM4v5_MOD_XJBVD:
            value->rValue = model->BSIM4v5xjbvd;
            return(OK);
        case BSIM4v5_MOD_XJBVS:
            value->rValue = model->BSIM4v5xjbvs;
            return(OK);
        case BSIM4v5_MOD_BVD:
            value->rValue = model->BSIM4v5bvd;
            return(OK);
        case BSIM4v5_MOD_BVS:
            value->rValue = model->BSIM4v5bvs;
            return(OK);
        case BSIM4v5_MOD_VFB:
            value->rValue = model->BSIM4v5vfb;
            return(OK);

        case BSIM4v5_MOD_JTSS:
            value->rValue = model->BSIM4v5jtss;
            return(OK);
        case BSIM4v5_MOD_JTSD:
            value->rValue = model->BSIM4v5jtsd;
            return(OK);
        case BSIM4v5_MOD_JTSSWS:
            value->rValue = model->BSIM4v5jtssws;
            return(OK);
        case BSIM4v5_MOD_JTSSWD:
            value->rValue = model->BSIM4v5jtsswd;
            return(OK);
        case BSIM4v5_MOD_JTSSWGS:
            value->rValue = model->BSIM4v5jtsswgs;
            return(OK);
        case BSIM4v5_MOD_JTSSWGD:
            value->rValue = model->BSIM4v5jtsswgd;
            return(OK);
        case BSIM4v5_MOD_NJTS:
            value->rValue = model->BSIM4v5njts;
            return(OK);
        case BSIM4v5_MOD_NJTSSW:
            value->rValue = model->BSIM4v5njtssw;
            return(OK);
        case BSIM4v5_MOD_NJTSSWG:
            value->rValue = model->BSIM4v5njtsswg;
            return(OK);
        case BSIM4v5_MOD_XTSS:
            value->rValue = model->BSIM4v5xtss;
            return(OK);
        case BSIM4v5_MOD_XTSD:
            value->rValue = model->BSIM4v5xtsd;
            return(OK);
        case BSIM4v5_MOD_XTSSWS:
            value->rValue = model->BSIM4v5xtssws;
            return(OK);
        case BSIM4v5_MOD_XTSSWD:
            value->rValue = model->BSIM4v5xtsswd;
            return(OK);
        case BSIM4v5_MOD_XTSSWGS:
            value->rValue = model->BSIM4v5xtsswgs;
            return(OK);
        case BSIM4v5_MOD_XTSSWGD:
            value->rValue = model->BSIM4v5xtsswgd;
            return(OK);
        case BSIM4v5_MOD_TNJTS:
            value->rValue = model->BSIM4v5tnjts;
            return(OK);
        case BSIM4v5_MOD_TNJTSSW:
            value->rValue = model->BSIM4v5tnjtssw;
            return(OK);
        case BSIM4v5_MOD_TNJTSSWG:
            value->rValue = model->BSIM4v5tnjtsswg;
            return(OK);
        case BSIM4v5_MOD_VTSS:
            value->rValue = model->BSIM4v5vtss;
            return(OK);
        case BSIM4v5_MOD_VTSD:
            value->rValue = model->BSIM4v5vtsd;
            return(OK);
        case BSIM4v5_MOD_VTSSWS:
            value->rValue = model->BSIM4v5vtssws;
            return(OK);
        case BSIM4v5_MOD_VTSSWD:
            value->rValue = model->BSIM4v5vtsswd;
            return(OK);
        case BSIM4v5_MOD_VTSSWGS:
            value->rValue = model->BSIM4v5vtsswgs;
            return(OK);
        case BSIM4v5_MOD_VTSSWGD:
            value->rValue = model->BSIM4v5vtsswgd;
            return(OK);

        case BSIM4v5_MOD_GBMIN:
            value->rValue = model->BSIM4v5gbmin;
            return(OK);
        case BSIM4v5_MOD_RBDB:
            value->rValue = model->BSIM4v5rbdb;
            return(OK);
        case BSIM4v5_MOD_RBPB:
            value->rValue = model->BSIM4v5rbpb;
            return(OK);
        case BSIM4v5_MOD_RBSB:
            value->rValue = model->BSIM4v5rbsb;
            return(OK);
        case BSIM4v5_MOD_RBPS:
            value->rValue = model->BSIM4v5rbps;
            return(OK);
        case BSIM4v5_MOD_RBPD:
            value->rValue = model->BSIM4v5rbpd;
            return(OK);

        case BSIM4v5_MOD_RBPS0:
            value->rValue = model->BSIM4v5rbps0;
            return(OK);
        case BSIM4v5_MOD_RBPSL:
            value->rValue = model->BSIM4v5rbpsl;
            return(OK);
        case BSIM4v5_MOD_RBPSW:
            value->rValue = model->BSIM4v5rbpsw;
            return(OK);
        case BSIM4v5_MOD_RBPSNF:
            value->rValue = model->BSIM4v5rbpsnf;
            return(OK);
        case BSIM4v5_MOD_RBPD0:
            value->rValue = model->BSIM4v5rbpd0;
            return(OK);
        case BSIM4v5_MOD_RBPDL:
            value->rValue = model->BSIM4v5rbpdl;
            return(OK);
        case BSIM4v5_MOD_RBPDW:
            value->rValue = model->BSIM4v5rbpdw;
            return(OK);
        case BSIM4v5_MOD_RBPDNF:
            value->rValue = model->BSIM4v5rbpdnf;
            return(OK);
        case BSIM4v5_MOD_RBPBX0:
            value->rValue = model->BSIM4v5rbpbx0;
            return(OK);
        case BSIM4v5_MOD_RBPBXL:
            value->rValue = model->BSIM4v5rbpbxl;
            return(OK);
        case BSIM4v5_MOD_RBPBXW:
            value->rValue = model->BSIM4v5rbpbxw;
            return(OK);
        case BSIM4v5_MOD_RBPBXNF:
            value->rValue = model->BSIM4v5rbpbxnf;
            return(OK);
        case BSIM4v5_MOD_RBPBY0:
            value->rValue = model->BSIM4v5rbpby0;
            return(OK);
        case BSIM4v5_MOD_RBPBYL:
            value->rValue = model->BSIM4v5rbpbyl;
            return(OK);
        case BSIM4v5_MOD_RBPBYW:
            value->rValue = model->BSIM4v5rbpbyw;
            return(OK);
        case BSIM4v5_MOD_RBPBYNF:
            value->rValue = model->BSIM4v5rbpbynf;
            return(OK);

        case BSIM4v5_MOD_RBSBX0:
            value->rValue = model->BSIM4v5rbsbx0;
            return(OK);
        case BSIM4v5_MOD_RBSBY0:
            value->rValue = model->BSIM4v5rbsby0;
            return(OK);
        case BSIM4v5_MOD_RBDBX0:
            value->rValue = model->BSIM4v5rbdbx0;
            return(OK);
        case BSIM4v5_MOD_RBDBY0:
            value->rValue = model->BSIM4v5rbdby0;
            return(OK);
        case BSIM4v5_MOD_RBSDBXL:
            value->rValue = model->BSIM4v5rbsdbxl;
            return(OK);
        case BSIM4v5_MOD_RBSDBXW:
            value->rValue = model->BSIM4v5rbsdbxw;
            return(OK);
        case BSIM4v5_MOD_RBSDBXNF:
            value->rValue = model->BSIM4v5rbsdbxnf;
            return(OK);
        case BSIM4v5_MOD_RBSDBYL:
            value->rValue = model->BSIM4v5rbsdbyl;
            return(OK);
        case BSIM4v5_MOD_RBSDBYW:
            value->rValue = model->BSIM4v5rbsdbyw;
            return(OK);
        case BSIM4v5_MOD_RBSDBYNF:
            value->rValue = model->BSIM4v5rbsdbynf;
            return(OK);


        case BSIM4v5_MOD_CGSL:
            value->rValue = model->BSIM4v5cgsl;
            return(OK);
        case BSIM4v5_MOD_CGDL:
            value->rValue = model->BSIM4v5cgdl;
            return(OK);
        case BSIM4v5_MOD_CKAPPAS:
            value->rValue = model->BSIM4v5ckappas;
            return(OK);
        case BSIM4v5_MOD_CKAPPAD:
            value->rValue = model->BSIM4v5ckappad;
            return(OK);
        case BSIM4v5_MOD_CF:
            value->rValue = model->BSIM4v5cf;
            return(OK);
        case BSIM4v5_MOD_CLC:
            value->rValue = model->BSIM4v5clc;
            return(OK);
        case BSIM4v5_MOD_CLE:
            value->rValue = model->BSIM4v5cle;
            return(OK);
        case BSIM4v5_MOD_DWC:
            value->rValue = model->BSIM4v5dwc;
            return(OK);
        case BSIM4v5_MOD_DLC:
            value->rValue = model->BSIM4v5dlc;
            return(OK);
        case BSIM4v5_MOD_XW:
            value->rValue = model->BSIM4v5xw;
            return(OK);
        case BSIM4v5_MOD_XL:
            value->rValue = model->BSIM4v5xl;
            return(OK);
        case BSIM4v5_MOD_DLCIG:
            value->rValue = model->BSIM4v5dlcig;
            return(OK);
        case BSIM4v5_MOD_DWJ:
            value->rValue = model->BSIM4v5dwj;
            return(OK);
        case BSIM4v5_MOD_VFBCV:
            value->rValue = model->BSIM4v5vfbcv; 
            return(OK);
        case BSIM4v5_MOD_ACDE:
            value->rValue = model->BSIM4v5acde;
            return(OK);
        case BSIM4v5_MOD_MOIN:
            value->rValue = model->BSIM4v5moin;
            return(OK);
        case BSIM4v5_MOD_NOFF:
            value->rValue = model->BSIM4v5noff;
            return(OK);
        case BSIM4v5_MOD_VOFFCV:
            value->rValue = model->BSIM4v5voffcv;
            return(OK);
        case BSIM4v5_MOD_DMCG:
            value->rValue = model->BSIM4v5dmcg;
            return(OK);
        case BSIM4v5_MOD_DMCI:
            value->rValue = model->BSIM4v5dmci;
            return(OK);
        case BSIM4v5_MOD_DMDG:
            value->rValue = model->BSIM4v5dmdg;
            return(OK);
        case BSIM4v5_MOD_DMCGT:
            value->rValue = model->BSIM4v5dmcgt;
            return(OK);
        case BSIM4v5_MOD_XGW:
            value->rValue = model->BSIM4v5xgw;
            return(OK);
        case BSIM4v5_MOD_XGL:
            value->rValue = model->BSIM4v5xgl;
            return(OK);
        case BSIM4v5_MOD_RSHG:
            value->rValue = model->BSIM4v5rshg;
            return(OK);
        case BSIM4v5_MOD_NGCON:
            value->rValue = model->BSIM4v5ngcon;
            return(OK);
        case BSIM4v5_MOD_TCJ:
            value->rValue = model->BSIM4v5tcj;
            return(OK);
        case BSIM4v5_MOD_TPB:
            value->rValue = model->BSIM4v5tpb;
            return(OK);
        case BSIM4v5_MOD_TCJSW:
            value->rValue = model->BSIM4v5tcjsw;
            return(OK);
        case BSIM4v5_MOD_TPBSW:
            value->rValue = model->BSIM4v5tpbsw;
            return(OK);
        case BSIM4v5_MOD_TCJSWG:
            value->rValue = model->BSIM4v5tcjswg;
            return(OK);
        case BSIM4v5_MOD_TPBSWG:
            value->rValue = model->BSIM4v5tpbswg;
            return(OK);

	/* Length dependence */
        case  BSIM4v5_MOD_LCDSC :
          value->rValue = model->BSIM4v5lcdsc;
            return(OK);
        case  BSIM4v5_MOD_LCDSCB :
          value->rValue = model->BSIM4v5lcdscb;
            return(OK);
        case  BSIM4v5_MOD_LCDSCD :
          value->rValue = model->BSIM4v5lcdscd;
            return(OK);
        case  BSIM4v5_MOD_LCIT :
          value->rValue = model->BSIM4v5lcit;
            return(OK);
        case  BSIM4v5_MOD_LNFACTOR :
          value->rValue = model->BSIM4v5lnfactor;
            return(OK);
        case BSIM4v5_MOD_LXJ:
            value->rValue = model->BSIM4v5lxj;
            return(OK);
        case BSIM4v5_MOD_LVSAT:
            value->rValue = model->BSIM4v5lvsat;
            return(OK);
        case BSIM4v5_MOD_LAT:
            value->rValue = model->BSIM4v5lat;
            return(OK);
        case BSIM4v5_MOD_LA0:
            value->rValue = model->BSIM4v5la0;
            return(OK);
        case BSIM4v5_MOD_LAGS:
            value->rValue = model->BSIM4v5lags;
            return(OK);
        case BSIM4v5_MOD_LA1:
            value->rValue = model->BSIM4v5la1;
            return(OK);
        case BSIM4v5_MOD_LA2:
            value->rValue = model->BSIM4v5la2;
            return(OK);
        case BSIM4v5_MOD_LKETA:
            value->rValue = model->BSIM4v5lketa;
            return(OK);   
        case BSIM4v5_MOD_LNSUB:
            value->rValue = model->BSIM4v5lnsub;
            return(OK);
        case BSIM4v5_MOD_LNDEP:
            value->rValue = model->BSIM4v5lndep;
            return(OK);
        case BSIM4v5_MOD_LNSD:
            value->rValue = model->BSIM4v5lnsd;
            return(OK);
        case BSIM4v5_MOD_LNGATE:
            value->rValue = model->BSIM4v5lngate;
            return(OK);
        case BSIM4v5_MOD_LGAMMA1:
            value->rValue = model->BSIM4v5lgamma1;
            return(OK);
        case BSIM4v5_MOD_LGAMMA2:
            value->rValue = model->BSIM4v5lgamma2;
            return(OK);
        case BSIM4v5_MOD_LVBX:
            value->rValue = model->BSIM4v5lvbx;
            return(OK);
        case BSIM4v5_MOD_LVBM:
            value->rValue = model->BSIM4v5lvbm;
            return(OK);
        case BSIM4v5_MOD_LXT:
            value->rValue = model->BSIM4v5lxt;
            return(OK);
        case  BSIM4v5_MOD_LK1:
          value->rValue = model->BSIM4v5lk1;
            return(OK);
        case  BSIM4v5_MOD_LKT1:
          value->rValue = model->BSIM4v5lkt1;
            return(OK);
        case  BSIM4v5_MOD_LKT1L:
          value->rValue = model->BSIM4v5lkt1l;
            return(OK);
        case  BSIM4v5_MOD_LKT2 :
          value->rValue = model->BSIM4v5lkt2;
            return(OK);
        case  BSIM4v5_MOD_LK2 :
          value->rValue = model->BSIM4v5lk2;
            return(OK);
        case  BSIM4v5_MOD_LK3:
          value->rValue = model->BSIM4v5lk3;
            return(OK);
        case  BSIM4v5_MOD_LK3B:
          value->rValue = model->BSIM4v5lk3b;
            return(OK);
        case  BSIM4v5_MOD_LW0:
          value->rValue = model->BSIM4v5lw0;
            return(OK);
        case  BSIM4v5_MOD_LLPE0:
          value->rValue = model->BSIM4v5llpe0;
            return(OK);
        case  BSIM4v5_MOD_LLPEB:
          value->rValue = model->BSIM4v5llpeb;
            return(OK);
        case  BSIM4v5_MOD_LDVTP0:
          value->rValue = model->BSIM4v5ldvtp0;
            return(OK);
        case  BSIM4v5_MOD_LDVTP1:
          value->rValue = model->BSIM4v5ldvtp1;
            return(OK);
        case  BSIM4v5_MOD_LDVT0:                
          value->rValue = model->BSIM4v5ldvt0;
            return(OK);
        case  BSIM4v5_MOD_LDVT1 :             
          value->rValue = model->BSIM4v5ldvt1;
            return(OK);
        case  BSIM4v5_MOD_LDVT2 :             
          value->rValue = model->BSIM4v5ldvt2;
            return(OK);
        case  BSIM4v5_MOD_LDVT0W :                
          value->rValue = model->BSIM4v5ldvt0w;
            return(OK);
        case  BSIM4v5_MOD_LDVT1W :             
          value->rValue = model->BSIM4v5ldvt1w;
            return(OK);
        case  BSIM4v5_MOD_LDVT2W :             
          value->rValue = model->BSIM4v5ldvt2w;
            return(OK);
        case  BSIM4v5_MOD_LDROUT :           
          value->rValue = model->BSIM4v5ldrout;
            return(OK);
        case  BSIM4v5_MOD_LDSUB :           
          value->rValue = model->BSIM4v5ldsub;
            return(OK);
        case BSIM4v5_MOD_LVTH0:
            value->rValue = model->BSIM4v5lvth0; 
            return(OK);
        case BSIM4v5_MOD_LUA:
            value->rValue = model->BSIM4v5lua; 
            return(OK);
        case BSIM4v5_MOD_LUA1:
            value->rValue = model->BSIM4v5lua1; 
            return(OK);
        case BSIM4v5_MOD_LUB:
            value->rValue = model->BSIM4v5lub;  
            return(OK);
        case BSIM4v5_MOD_LUB1:
            value->rValue = model->BSIM4v5lub1;  
            return(OK);
        case BSIM4v5_MOD_LUC:
            value->rValue = model->BSIM4v5luc; 
            return(OK);
        case BSIM4v5_MOD_LUC1:
            value->rValue = model->BSIM4v5luc1; 
            return(OK);
        case BSIM4v5_MOD_LUD:
            value->rValue = model->BSIM4v5lud; 
            return(OK);
        case BSIM4v5_MOD_LUD1:
            value->rValue = model->BSIM4v5lud1; 
            return(OK);
        case BSIM4v5_MOD_LUP:
            value->rValue = model->BSIM4v5lup; 
            return(OK);
        case BSIM4v5_MOD_LLP:
            value->rValue = model->BSIM4v5llp; 
            return(OK);
        case BSIM4v5_MOD_LU0:
            value->rValue = model->BSIM4v5lu0;
            return(OK);
        case BSIM4v5_MOD_LUTE:
            value->rValue = model->BSIM4v5lute;
            return(OK);
        case BSIM4v5_MOD_LVOFF:
            value->rValue = model->BSIM4v5lvoff;
            return(OK);
        case BSIM4v5_MOD_LTVOFF:
            value->rValue = model->BSIM4v5ltvoff;
            return(OK);
        case BSIM4v5_MOD_LMINV:
            value->rValue = model->BSIM4v5lminv;
            return(OK);
        case BSIM4v5_MOD_LFPROUT:
            value->rValue = model->BSIM4v5lfprout;
            return(OK);
        case BSIM4v5_MOD_LPDITS:
            value->rValue = model->BSIM4v5lpdits;
            return(OK);
        case BSIM4v5_MOD_LPDITSD:
            value->rValue = model->BSIM4v5lpditsd;
            return(OK);
        case BSIM4v5_MOD_LDELTA:
            value->rValue = model->BSIM4v5ldelta;
            return(OK);
        case BSIM4v5_MOD_LRDSW:
            value->rValue = model->BSIM4v5lrdsw; 
            return(OK);             
        case BSIM4v5_MOD_LRDW:
            value->rValue = model->BSIM4v5lrdw;
            return(OK);
        case BSIM4v5_MOD_LRSW:
            value->rValue = model->BSIM4v5lrsw;
            return(OK);
        case BSIM4v5_MOD_LPRWB:
            value->rValue = model->BSIM4v5lprwb; 
            return(OK);             
        case BSIM4v5_MOD_LPRWG:
            value->rValue = model->BSIM4v5lprwg; 
            return(OK);             
        case BSIM4v5_MOD_LPRT:
            value->rValue = model->BSIM4v5lprt; 
            return(OK);              
        case BSIM4v5_MOD_LETA0:
            value->rValue = model->BSIM4v5leta0; 
            return(OK);               
        case BSIM4v5_MOD_LETAB:
            value->rValue = model->BSIM4v5letab; 
            return(OK);               
        case BSIM4v5_MOD_LPCLM:
            value->rValue = model->BSIM4v5lpclm; 
            return(OK);               
        case BSIM4v5_MOD_LPDIBL1:
            value->rValue = model->BSIM4v5lpdibl1; 
            return(OK);               
        case BSIM4v5_MOD_LPDIBL2:
            value->rValue = model->BSIM4v5lpdibl2; 
            return(OK);               
        case BSIM4v5_MOD_LPDIBLB:
            value->rValue = model->BSIM4v5lpdiblb; 
            return(OK);               
        case BSIM4v5_MOD_LPSCBE1:
            value->rValue = model->BSIM4v5lpscbe1; 
            return(OK);               
        case BSIM4v5_MOD_LPSCBE2:
            value->rValue = model->BSIM4v5lpscbe2; 
            return(OK);               
        case BSIM4v5_MOD_LPVAG:
            value->rValue = model->BSIM4v5lpvag; 
            return(OK);               
        case BSIM4v5_MOD_LWR:
            value->rValue = model->BSIM4v5lwr;
            return(OK);
        case BSIM4v5_MOD_LDWG:
            value->rValue = model->BSIM4v5ldwg;
            return(OK);
        case BSIM4v5_MOD_LDWB:
            value->rValue = model->BSIM4v5ldwb;
            return(OK);
        case BSIM4v5_MOD_LB0:
            value->rValue = model->BSIM4v5lb0;
            return(OK);
        case BSIM4v5_MOD_LB1:
            value->rValue = model->BSIM4v5lb1;
            return(OK);
        case BSIM4v5_MOD_LALPHA0:
            value->rValue = model->BSIM4v5lalpha0;
            return(OK);
        case BSIM4v5_MOD_LALPHA1:
            value->rValue = model->BSIM4v5lalpha1;
            return(OK);
        case BSIM4v5_MOD_LBETA0:
            value->rValue = model->BSIM4v5lbeta0;
            return(OK);
        case BSIM4v5_MOD_LAGIDL:
            value->rValue = model->BSIM4v5lagidl;
            return(OK);
        case BSIM4v5_MOD_LBGIDL:
            value->rValue = model->BSIM4v5lbgidl;
            return(OK);
        case BSIM4v5_MOD_LCGIDL:
            value->rValue = model->BSIM4v5lcgidl;
            return(OK);
        case BSIM4v5_MOD_LEGIDL:
            value->rValue = model->BSIM4v5legidl;
            return(OK);
        case BSIM4v5_MOD_LAIGC:
            value->rValue = model->BSIM4v5laigc;
            return(OK);
        case BSIM4v5_MOD_LBIGC:
            value->rValue = model->BSIM4v5lbigc;
            return(OK);
        case BSIM4v5_MOD_LCIGC:
            value->rValue = model->BSIM4v5lcigc;
            return(OK);
        case BSIM4v5_MOD_LAIGSD:
            value->rValue = model->BSIM4v5laigsd;
            return(OK);
        case BSIM4v5_MOD_LBIGSD:
            value->rValue = model->BSIM4v5lbigsd;
            return(OK);
        case BSIM4v5_MOD_LCIGSD:
            value->rValue = model->BSIM4v5lcigsd;
            return(OK);
        case BSIM4v5_MOD_LAIGBACC:
            value->rValue = model->BSIM4v5laigbacc;
            return(OK);
        case BSIM4v5_MOD_LBIGBACC:
            value->rValue = model->BSIM4v5lbigbacc;
            return(OK);
        case BSIM4v5_MOD_LCIGBACC:
            value->rValue = model->BSIM4v5lcigbacc;
            return(OK);
        case BSIM4v5_MOD_LAIGBINV:
            value->rValue = model->BSIM4v5laigbinv;
            return(OK);
        case BSIM4v5_MOD_LBIGBINV:
            value->rValue = model->BSIM4v5lbigbinv;
            return(OK);
        case BSIM4v5_MOD_LCIGBINV:
            value->rValue = model->BSIM4v5lcigbinv;
            return(OK);
        case BSIM4v5_MOD_LNIGC:
            value->rValue = model->BSIM4v5lnigc;
            return(OK);
        case BSIM4v5_MOD_LNIGBACC:
            value->rValue = model->BSIM4v5lnigbacc;
            return(OK);
        case BSIM4v5_MOD_LNIGBINV:
            value->rValue = model->BSIM4v5lnigbinv;
            return(OK);
        case BSIM4v5_MOD_LNTOX:
            value->rValue = model->BSIM4v5lntox;
            return(OK);
        case BSIM4v5_MOD_LEIGBINV:
            value->rValue = model->BSIM4v5leigbinv;
            return(OK);
        case BSIM4v5_MOD_LPIGCD:
            value->rValue = model->BSIM4v5lpigcd;
            return(OK);
        case BSIM4v5_MOD_LPOXEDGE:
            value->rValue = model->BSIM4v5lpoxedge;
            return(OK);
        case BSIM4v5_MOD_LPHIN:
            value->rValue = model->BSIM4v5lphin;
            return(OK);
        case BSIM4v5_MOD_LXRCRG1:
            value->rValue = model->BSIM4v5lxrcrg1;
            return(OK);
        case BSIM4v5_MOD_LXRCRG2:
            value->rValue = model->BSIM4v5lxrcrg2;
            return(OK);
        case BSIM4v5_MOD_LEU:
            value->rValue = model->BSIM4v5leu;
            return(OK);
        case BSIM4v5_MOD_LVFB:
            value->rValue = model->BSIM4v5lvfb;
            return(OK);

        case BSIM4v5_MOD_LCGSL:
            value->rValue = model->BSIM4v5lcgsl;
            return(OK);
        case BSIM4v5_MOD_LCGDL:
            value->rValue = model->BSIM4v5lcgdl;
            return(OK);
        case BSIM4v5_MOD_LCKAPPAS:
            value->rValue = model->BSIM4v5lckappas;
            return(OK);
        case BSIM4v5_MOD_LCKAPPAD:
            value->rValue = model->BSIM4v5lckappad;
            return(OK);
        case BSIM4v5_MOD_LCF:
            value->rValue = model->BSIM4v5lcf;
            return(OK);
        case BSIM4v5_MOD_LCLC:
            value->rValue = model->BSIM4v5lclc;
            return(OK);
        case BSIM4v5_MOD_LCLE:
            value->rValue = model->BSIM4v5lcle;
            return(OK);
        case BSIM4v5_MOD_LVFBCV:
            value->rValue = model->BSIM4v5lvfbcv;
            return(OK);
        case BSIM4v5_MOD_LACDE:
            value->rValue = model->BSIM4v5lacde;
            return(OK);
        case BSIM4v5_MOD_LMOIN:
            value->rValue = model->BSIM4v5lmoin;
            return(OK);
        case BSIM4v5_MOD_LNOFF:
            value->rValue = model->BSIM4v5lnoff;
            return(OK);
        case BSIM4v5_MOD_LVOFFCV:
            value->rValue = model->BSIM4v5lvoffcv;
            return(OK);
        case BSIM4v5_MOD_LVFBSDOFF:
            value->rValue = model->BSIM4v5lvfbsdoff;
            return(OK);
        case BSIM4v5_MOD_LTVFBSDOFF:
            value->rValue = model->BSIM4v5ltvfbsdoff;
            return(OK);

	/* Width dependence */
        case  BSIM4v5_MOD_WCDSC :
          value->rValue = model->BSIM4v5wcdsc;
            return(OK);
        case  BSIM4v5_MOD_WCDSCB :
          value->rValue = model->BSIM4v5wcdscb;
            return(OK);
        case  BSIM4v5_MOD_WCDSCD :
          value->rValue = model->BSIM4v5wcdscd;
            return(OK);
        case  BSIM4v5_MOD_WCIT :
          value->rValue = model->BSIM4v5wcit;
            return(OK);
        case  BSIM4v5_MOD_WNFACTOR :
          value->rValue = model->BSIM4v5wnfactor;
            return(OK);
        case BSIM4v5_MOD_WXJ:
            value->rValue = model->BSIM4v5wxj;
            return(OK);
        case BSIM4v5_MOD_WVSAT:
            value->rValue = model->BSIM4v5wvsat;
            return(OK);
        case BSIM4v5_MOD_WAT:
            value->rValue = model->BSIM4v5wat;
            return(OK);
        case BSIM4v5_MOD_WA0:
            value->rValue = model->BSIM4v5wa0;
            return(OK);
        case BSIM4v5_MOD_WAGS:
            value->rValue = model->BSIM4v5wags;
            return(OK);
        case BSIM4v5_MOD_WA1:
            value->rValue = model->BSIM4v5wa1;
            return(OK);
        case BSIM4v5_MOD_WA2:
            value->rValue = model->BSIM4v5wa2;
            return(OK);
        case BSIM4v5_MOD_WKETA:
            value->rValue = model->BSIM4v5wketa;
            return(OK);   
        case BSIM4v5_MOD_WNSUB:
            value->rValue = model->BSIM4v5wnsub;
            return(OK);
        case BSIM4v5_MOD_WNDEP:
            value->rValue = model->BSIM4v5wndep;
            return(OK);
        case BSIM4v5_MOD_WNSD:
            value->rValue = model->BSIM4v5wnsd;
            return(OK);
        case BSIM4v5_MOD_WNGATE:
            value->rValue = model->BSIM4v5wngate;
            return(OK);
        case BSIM4v5_MOD_WGAMMA1:
            value->rValue = model->BSIM4v5wgamma1;
            return(OK);
        case BSIM4v5_MOD_WGAMMA2:
            value->rValue = model->BSIM4v5wgamma2;
            return(OK);
        case BSIM4v5_MOD_WVBX:
            value->rValue = model->BSIM4v5wvbx;
            return(OK);
        case BSIM4v5_MOD_WVBM:
            value->rValue = model->BSIM4v5wvbm;
            return(OK);
        case BSIM4v5_MOD_WXT:
            value->rValue = model->BSIM4v5wxt;
            return(OK);
        case  BSIM4v5_MOD_WK1:
          value->rValue = model->BSIM4v5wk1;
            return(OK);
        case  BSIM4v5_MOD_WKT1:
          value->rValue = model->BSIM4v5wkt1;
            return(OK);
        case  BSIM4v5_MOD_WKT1L:
          value->rValue = model->BSIM4v5wkt1l;
            return(OK);
        case  BSIM4v5_MOD_WKT2 :
          value->rValue = model->BSIM4v5wkt2;
            return(OK);
        case  BSIM4v5_MOD_WK2 :
          value->rValue = model->BSIM4v5wk2;
            return(OK);
        case  BSIM4v5_MOD_WK3:
          value->rValue = model->BSIM4v5wk3;
            return(OK);
        case  BSIM4v5_MOD_WK3B:
          value->rValue = model->BSIM4v5wk3b;
            return(OK);
        case  BSIM4v5_MOD_WW0:
          value->rValue = model->BSIM4v5ww0;
            return(OK);
        case  BSIM4v5_MOD_WLPE0:
          value->rValue = model->BSIM4v5wlpe0;
            return(OK);
        case  BSIM4v5_MOD_WDVTP0:
          value->rValue = model->BSIM4v5wdvtp0;
            return(OK);
        case  BSIM4v5_MOD_WDVTP1:
          value->rValue = model->BSIM4v5wdvtp1;
            return(OK);
        case  BSIM4v5_MOD_WLPEB:
          value->rValue = model->BSIM4v5wlpeb;
            return(OK);
        case  BSIM4v5_MOD_WDVT0:                
          value->rValue = model->BSIM4v5wdvt0;
            return(OK);
        case  BSIM4v5_MOD_WDVT1 :             
          value->rValue = model->BSIM4v5wdvt1;
            return(OK);
        case  BSIM4v5_MOD_WDVT2 :             
          value->rValue = model->BSIM4v5wdvt2;
            return(OK);
        case  BSIM4v5_MOD_WDVT0W :                
          value->rValue = model->BSIM4v5wdvt0w;
            return(OK);
        case  BSIM4v5_MOD_WDVT1W :             
          value->rValue = model->BSIM4v5wdvt1w;
            return(OK);
        case  BSIM4v5_MOD_WDVT2W :             
          value->rValue = model->BSIM4v5wdvt2w;
            return(OK);
        case  BSIM4v5_MOD_WDROUT :           
          value->rValue = model->BSIM4v5wdrout;
            return(OK);
        case  BSIM4v5_MOD_WDSUB :           
          value->rValue = model->BSIM4v5wdsub;
            return(OK);
        case BSIM4v5_MOD_WVTH0:
            value->rValue = model->BSIM4v5wvth0; 
            return(OK);
        case BSIM4v5_MOD_WUA:
            value->rValue = model->BSIM4v5wua; 
            return(OK);
        case BSIM4v5_MOD_WUA1:
            value->rValue = model->BSIM4v5wua1; 
            return(OK);
        case BSIM4v5_MOD_WUB:
            value->rValue = model->BSIM4v5wub;  
            return(OK);
        case BSIM4v5_MOD_WUB1:
            value->rValue = model->BSIM4v5wub1;  
            return(OK);
        case BSIM4v5_MOD_WUC:
            value->rValue = model->BSIM4v5wuc; 
            return(OK);
        case BSIM4v5_MOD_WUC1:
            value->rValue = model->BSIM4v5wuc1; 
            return(OK);
        case BSIM4v5_MOD_WUD:
            value->rValue = model->BSIM4v5wud; 
            return(OK);
        case BSIM4v5_MOD_WUD1:
            value->rValue = model->BSIM4v5wud1; 
            return(OK);
        case BSIM4v5_MOD_WUP:
            value->rValue = model->BSIM4v5wup; 
            return(OK);
        case BSIM4v5_MOD_WLP:
            value->rValue = model->BSIM4v5wlp; 
            return(OK);
        case BSIM4v5_MOD_WU0:
            value->rValue = model->BSIM4v5wu0;
            return(OK);
        case BSIM4v5_MOD_WUTE:
            value->rValue = model->BSIM4v5wute;
            return(OK);
        case BSIM4v5_MOD_WVOFF:
            value->rValue = model->BSIM4v5wvoff;
            return(OK);
        case BSIM4v5_MOD_WTVOFF:
            value->rValue = model->BSIM4v5wtvoff;
            return(OK);
        case BSIM4v5_MOD_WMINV:
            value->rValue = model->BSIM4v5wminv;
            return(OK);
        case BSIM4v5_MOD_WFPROUT:
            value->rValue = model->BSIM4v5wfprout;
            return(OK);
        case BSIM4v5_MOD_WPDITS:
            value->rValue = model->BSIM4v5wpdits;
            return(OK);
        case BSIM4v5_MOD_WPDITSD:
            value->rValue = model->BSIM4v5wpditsd;
            return(OK);
        case BSIM4v5_MOD_WDELTA:
            value->rValue = model->BSIM4v5wdelta;
            return(OK);
        case BSIM4v5_MOD_WRDSW:
            value->rValue = model->BSIM4v5wrdsw; 
            return(OK);             
        case BSIM4v5_MOD_WRDW:
            value->rValue = model->BSIM4v5wrdw;
            return(OK);
        case BSIM4v5_MOD_WRSW:
            value->rValue = model->BSIM4v5wrsw;
            return(OK);
        case BSIM4v5_MOD_WPRWB:
            value->rValue = model->BSIM4v5wprwb; 
            return(OK);             
        case BSIM4v5_MOD_WPRWG:
            value->rValue = model->BSIM4v5wprwg; 
            return(OK);             
        case BSIM4v5_MOD_WPRT:
            value->rValue = model->BSIM4v5wprt; 
            return(OK);              
        case BSIM4v5_MOD_WETA0:
            value->rValue = model->BSIM4v5weta0; 
            return(OK);               
        case BSIM4v5_MOD_WETAB:
            value->rValue = model->BSIM4v5wetab; 
            return(OK);               
        case BSIM4v5_MOD_WPCLM:
            value->rValue = model->BSIM4v5wpclm; 
            return(OK);               
        case BSIM4v5_MOD_WPDIBL1:
            value->rValue = model->BSIM4v5wpdibl1; 
            return(OK);               
        case BSIM4v5_MOD_WPDIBL2:
            value->rValue = model->BSIM4v5wpdibl2; 
            return(OK);               
        case BSIM4v5_MOD_WPDIBLB:
            value->rValue = model->BSIM4v5wpdiblb; 
            return(OK);               
        case BSIM4v5_MOD_WPSCBE1:
            value->rValue = model->BSIM4v5wpscbe1; 
            return(OK);               
        case BSIM4v5_MOD_WPSCBE2:
            value->rValue = model->BSIM4v5wpscbe2; 
            return(OK);               
        case BSIM4v5_MOD_WPVAG:
            value->rValue = model->BSIM4v5wpvag; 
            return(OK);               
        case BSIM4v5_MOD_WWR:
            value->rValue = model->BSIM4v5wwr;
            return(OK);
        case BSIM4v5_MOD_WDWG:
            value->rValue = model->BSIM4v5wdwg;
            return(OK);
        case BSIM4v5_MOD_WDWB:
            value->rValue = model->BSIM4v5wdwb;
            return(OK);
        case BSIM4v5_MOD_WB0:
            value->rValue = model->BSIM4v5wb0;
            return(OK);
        case BSIM4v5_MOD_WB1:
            value->rValue = model->BSIM4v5wb1;
            return(OK);
        case BSIM4v5_MOD_WALPHA0:
            value->rValue = model->BSIM4v5walpha0;
            return(OK);
        case BSIM4v5_MOD_WALPHA1:
            value->rValue = model->BSIM4v5walpha1;
            return(OK);
        case BSIM4v5_MOD_WBETA0:
            value->rValue = model->BSIM4v5wbeta0;
            return(OK);
        case BSIM4v5_MOD_WAGIDL:
            value->rValue = model->BSIM4v5wagidl;
            return(OK);
        case BSIM4v5_MOD_WBGIDL:
            value->rValue = model->BSIM4v5wbgidl;
            return(OK);
        case BSIM4v5_MOD_WCGIDL:
            value->rValue = model->BSIM4v5wcgidl;
            return(OK);
        case BSIM4v5_MOD_WEGIDL:
            value->rValue = model->BSIM4v5wegidl;
            return(OK);
        case BSIM4v5_MOD_WAIGC:
            value->rValue = model->BSIM4v5waigc;
            return(OK);
        case BSIM4v5_MOD_WBIGC:
            value->rValue = model->BSIM4v5wbigc;
            return(OK);
        case BSIM4v5_MOD_WCIGC:
            value->rValue = model->BSIM4v5wcigc;
            return(OK);
        case BSIM4v5_MOD_WAIGSD:
            value->rValue = model->BSIM4v5waigsd;
            return(OK);
        case BSIM4v5_MOD_WBIGSD:
            value->rValue = model->BSIM4v5wbigsd;
            return(OK);
        case BSIM4v5_MOD_WCIGSD:
            value->rValue = model->BSIM4v5wcigsd;
            return(OK);
        case BSIM4v5_MOD_WAIGBACC:
            value->rValue = model->BSIM4v5waigbacc;
            return(OK);
        case BSIM4v5_MOD_WBIGBACC:
            value->rValue = model->BSIM4v5wbigbacc;
            return(OK);
        case BSIM4v5_MOD_WCIGBACC:
            value->rValue = model->BSIM4v5wcigbacc;
            return(OK);
        case BSIM4v5_MOD_WAIGBINV:
            value->rValue = model->BSIM4v5waigbinv;
            return(OK);
        case BSIM4v5_MOD_WBIGBINV:
            value->rValue = model->BSIM4v5wbigbinv;
            return(OK);
        case BSIM4v5_MOD_WCIGBINV:
            value->rValue = model->BSIM4v5wcigbinv;
            return(OK);
        case BSIM4v5_MOD_WNIGC:
            value->rValue = model->BSIM4v5wnigc;
            return(OK);
        case BSIM4v5_MOD_WNIGBACC:
            value->rValue = model->BSIM4v5wnigbacc;
            return(OK);
        case BSIM4v5_MOD_WNIGBINV:
            value->rValue = model->BSIM4v5wnigbinv;
            return(OK);
        case BSIM4v5_MOD_WNTOX:
            value->rValue = model->BSIM4v5wntox;
            return(OK);
        case BSIM4v5_MOD_WEIGBINV:
            value->rValue = model->BSIM4v5weigbinv;
            return(OK);
        case BSIM4v5_MOD_WPIGCD:
            value->rValue = model->BSIM4v5wpigcd;
            return(OK);
        case BSIM4v5_MOD_WPOXEDGE:
            value->rValue = model->BSIM4v5wpoxedge;
            return(OK);
        case BSIM4v5_MOD_WPHIN:
            value->rValue = model->BSIM4v5wphin;
            return(OK);
        case BSIM4v5_MOD_WXRCRG1:
            value->rValue = model->BSIM4v5wxrcrg1;
            return(OK);
        case BSIM4v5_MOD_WXRCRG2:
            value->rValue = model->BSIM4v5wxrcrg2;
            return(OK);
        case BSIM4v5_MOD_WEU:
            value->rValue = model->BSIM4v5weu;
            return(OK);
        case BSIM4v5_MOD_WVFB:
            value->rValue = model->BSIM4v5wvfb;
            return(OK);

        case BSIM4v5_MOD_WCGSL:
            value->rValue = model->BSIM4v5wcgsl;
            return(OK);
        case BSIM4v5_MOD_WCGDL:
            value->rValue = model->BSIM4v5wcgdl;
            return(OK);
        case BSIM4v5_MOD_WCKAPPAS:
            value->rValue = model->BSIM4v5wckappas;
            return(OK);
        case BSIM4v5_MOD_WCKAPPAD:
            value->rValue = model->BSIM4v5wckappad;
            return(OK);
        case BSIM4v5_MOD_WCF:
            value->rValue = model->BSIM4v5wcf;
            return(OK);
        case BSIM4v5_MOD_WCLC:
            value->rValue = model->BSIM4v5wclc;
            return(OK);
        case BSIM4v5_MOD_WCLE:
            value->rValue = model->BSIM4v5wcle;
            return(OK);
        case BSIM4v5_MOD_WVFBCV:
            value->rValue = model->BSIM4v5wvfbcv;
            return(OK);
        case BSIM4v5_MOD_WACDE:
            value->rValue = model->BSIM4v5wacde;
            return(OK);
        case BSIM4v5_MOD_WMOIN:
            value->rValue = model->BSIM4v5wmoin;
            return(OK);
        case BSIM4v5_MOD_WNOFF:
            value->rValue = model->BSIM4v5wnoff;
            return(OK);
        case BSIM4v5_MOD_WVOFFCV:
            value->rValue = model->BSIM4v5wvoffcv;
            return(OK);
        case BSIM4v5_MOD_WVFBSDOFF:
            value->rValue = model->BSIM4v5wvfbsdoff;
            return(OK);
        case BSIM4v5_MOD_WTVFBSDOFF:
            value->rValue = model->BSIM4v5wtvfbsdoff;
            return(OK);

	/* Cross-term dependence */
        case  BSIM4v5_MOD_PCDSC :
          value->rValue = model->BSIM4v5pcdsc;
            return(OK);
        case  BSIM4v5_MOD_PCDSCB :
          value->rValue = model->BSIM4v5pcdscb;
            return(OK);
        case  BSIM4v5_MOD_PCDSCD :
          value->rValue = model->BSIM4v5pcdscd;
            return(OK);
         case  BSIM4v5_MOD_PCIT :
          value->rValue = model->BSIM4v5pcit;
            return(OK);
        case  BSIM4v5_MOD_PNFACTOR :
          value->rValue = model->BSIM4v5pnfactor;
            return(OK);
        case BSIM4v5_MOD_PXJ:
            value->rValue = model->BSIM4v5pxj;
            return(OK);
        case BSIM4v5_MOD_PVSAT:
            value->rValue = model->BSIM4v5pvsat;
            return(OK);
        case BSIM4v5_MOD_PAT:
            value->rValue = model->BSIM4v5pat;
            return(OK);
        case BSIM4v5_MOD_PA0:
            value->rValue = model->BSIM4v5pa0;
            return(OK);
        case BSIM4v5_MOD_PAGS:
            value->rValue = model->BSIM4v5pags;
            return(OK);
        case BSIM4v5_MOD_PA1:
            value->rValue = model->BSIM4v5pa1;
            return(OK);
        case BSIM4v5_MOD_PA2:
            value->rValue = model->BSIM4v5pa2;
            return(OK);
        case BSIM4v5_MOD_PKETA:
            value->rValue = model->BSIM4v5pketa;
            return(OK);   
        case BSIM4v5_MOD_PNSUB:
            value->rValue = model->BSIM4v5pnsub;
            return(OK);
        case BSIM4v5_MOD_PNDEP:
            value->rValue = model->BSIM4v5pndep;
            return(OK);
        case BSIM4v5_MOD_PNSD:
            value->rValue = model->BSIM4v5pnsd;
            return(OK);
        case BSIM4v5_MOD_PNGATE:
            value->rValue = model->BSIM4v5pngate;
            return(OK);
        case BSIM4v5_MOD_PGAMMA1:
            value->rValue = model->BSIM4v5pgamma1;
            return(OK);
        case BSIM4v5_MOD_PGAMMA2:
            value->rValue = model->BSIM4v5pgamma2;
            return(OK);
        case BSIM4v5_MOD_PVBX:
            value->rValue = model->BSIM4v5pvbx;
            return(OK);
        case BSIM4v5_MOD_PVBM:
            value->rValue = model->BSIM4v5pvbm;
            return(OK);
        case BSIM4v5_MOD_PXT:
            value->rValue = model->BSIM4v5pxt;
            return(OK);
        case  BSIM4v5_MOD_PK1:
          value->rValue = model->BSIM4v5pk1;
            return(OK);
        case  BSIM4v5_MOD_PKT1:
          value->rValue = model->BSIM4v5pkt1;
            return(OK);
        case  BSIM4v5_MOD_PKT1L:
          value->rValue = model->BSIM4v5pkt1l;
            return(OK);
        case  BSIM4v5_MOD_PKT2 :
          value->rValue = model->BSIM4v5pkt2;
            return(OK);
        case  BSIM4v5_MOD_PK2 :
          value->rValue = model->BSIM4v5pk2;
            return(OK);
        case  BSIM4v5_MOD_PK3:
          value->rValue = model->BSIM4v5pk3;
            return(OK);
        case  BSIM4v5_MOD_PK3B:
          value->rValue = model->BSIM4v5pk3b;
            return(OK);
        case  BSIM4v5_MOD_PW0:
          value->rValue = model->BSIM4v5pw0;
            return(OK);
        case  BSIM4v5_MOD_PLPE0:
          value->rValue = model->BSIM4v5plpe0;
            return(OK);
        case  BSIM4v5_MOD_PLPEB:
          value->rValue = model->BSIM4v5plpeb;
            return(OK);
        case  BSIM4v5_MOD_PDVTP0:
          value->rValue = model->BSIM4v5pdvtp0;
            return(OK);
        case  BSIM4v5_MOD_PDVTP1:
          value->rValue = model->BSIM4v5pdvtp1;
            return(OK);
        case  BSIM4v5_MOD_PDVT0 :                
          value->rValue = model->BSIM4v5pdvt0;
            return(OK);
        case  BSIM4v5_MOD_PDVT1 :             
          value->rValue = model->BSIM4v5pdvt1;
            return(OK);
        case  BSIM4v5_MOD_PDVT2 :             
          value->rValue = model->BSIM4v5pdvt2;
            return(OK);
        case  BSIM4v5_MOD_PDVT0W :                
          value->rValue = model->BSIM4v5pdvt0w;
            return(OK);
        case  BSIM4v5_MOD_PDVT1W :             
          value->rValue = model->BSIM4v5pdvt1w;
            return(OK);
        case  BSIM4v5_MOD_PDVT2W :             
          value->rValue = model->BSIM4v5pdvt2w;
            return(OK);
        case  BSIM4v5_MOD_PDROUT :           
          value->rValue = model->BSIM4v5pdrout;
            return(OK);
        case  BSIM4v5_MOD_PDSUB :           
          value->rValue = model->BSIM4v5pdsub;
            return(OK);
        case BSIM4v5_MOD_PVTH0:
            value->rValue = model->BSIM4v5pvth0; 
            return(OK);
        case BSIM4v5_MOD_PUA:
            value->rValue = model->BSIM4v5pua; 
            return(OK);
        case BSIM4v5_MOD_PUA1:
            value->rValue = model->BSIM4v5pua1; 
            return(OK);
        case BSIM4v5_MOD_PUB:
            value->rValue = model->BSIM4v5pub;  
            return(OK);
        case BSIM4v5_MOD_PUB1:
            value->rValue = model->BSIM4v5pub1;  
            return(OK);
        case BSIM4v5_MOD_PUC:
            value->rValue = model->BSIM4v5puc; 
            return(OK);
        case BSIM4v5_MOD_PUC1:
            value->rValue = model->BSIM4v5puc1; 
            return(OK);
        case BSIM4v5_MOD_PUD:
            value->rValue = model->BSIM4v5pud; 
            return(OK);
        case BSIM4v5_MOD_PUD1:
            value->rValue = model->BSIM4v5pud1; 
            return(OK);
        case BSIM4v5_MOD_PUP:
            value->rValue = model->BSIM4v5pup; 
            return(OK);
        case BSIM4v5_MOD_PLP:
            value->rValue = model->BSIM4v5plp; 
            return(OK);
        case BSIM4v5_MOD_PU0:
            value->rValue = model->BSIM4v5pu0;
            return(OK);
        case BSIM4v5_MOD_PUTE:
            value->rValue = model->BSIM4v5pute;
            return(OK);
        case BSIM4v5_MOD_PVOFF:
            value->rValue = model->BSIM4v5pvoff;
            return(OK);
        case BSIM4v5_MOD_PTVOFF:
            value->rValue = model->BSIM4v5ptvoff;
            return(OK);
        case BSIM4v5_MOD_PMINV:
            value->rValue = model->BSIM4v5pminv;
            return(OK);
        case BSIM4v5_MOD_PFPROUT:
            value->rValue = model->BSIM4v5pfprout;
            return(OK);
        case BSIM4v5_MOD_PPDITS:
            value->rValue = model->BSIM4v5ppdits;
            return(OK);
        case BSIM4v5_MOD_PPDITSD:
            value->rValue = model->BSIM4v5ppditsd;
            return(OK);
        case BSIM4v5_MOD_PDELTA:
            value->rValue = model->BSIM4v5pdelta;
            return(OK);
        case BSIM4v5_MOD_PRDSW:
            value->rValue = model->BSIM4v5prdsw; 
            return(OK);             
        case BSIM4v5_MOD_PRDW:
            value->rValue = model->BSIM4v5prdw;
            return(OK);
        case BSIM4v5_MOD_PRSW:
            value->rValue = model->BSIM4v5prsw;
            return(OK);
        case BSIM4v5_MOD_PPRWB:
            value->rValue = model->BSIM4v5pprwb; 
            return(OK);             
        case BSIM4v5_MOD_PPRWG:
            value->rValue = model->BSIM4v5pprwg; 
            return(OK);             
        case BSIM4v5_MOD_PPRT:
            value->rValue = model->BSIM4v5pprt; 
            return(OK);              
        case BSIM4v5_MOD_PETA0:
            value->rValue = model->BSIM4v5peta0; 
            return(OK);               
        case BSIM4v5_MOD_PETAB:
            value->rValue = model->BSIM4v5petab; 
            return(OK);               
        case BSIM4v5_MOD_PPCLM:
            value->rValue = model->BSIM4v5ppclm; 
            return(OK);               
        case BSIM4v5_MOD_PPDIBL1:
            value->rValue = model->BSIM4v5ppdibl1; 
            return(OK);               
        case BSIM4v5_MOD_PPDIBL2:
            value->rValue = model->BSIM4v5ppdibl2; 
            return(OK);               
        case BSIM4v5_MOD_PPDIBLB:
            value->rValue = model->BSIM4v5ppdiblb; 
            return(OK);               
        case BSIM4v5_MOD_PPSCBE1:
            value->rValue = model->BSIM4v5ppscbe1; 
            return(OK);               
        case BSIM4v5_MOD_PPSCBE2:
            value->rValue = model->BSIM4v5ppscbe2; 
            return(OK);               
        case BSIM4v5_MOD_PPVAG:
            value->rValue = model->BSIM4v5ppvag; 
            return(OK);               
        case BSIM4v5_MOD_PWR:
            value->rValue = model->BSIM4v5pwr;
            return(OK);
        case BSIM4v5_MOD_PDWG:
            value->rValue = model->BSIM4v5pdwg;
            return(OK);
        case BSIM4v5_MOD_PDWB:
            value->rValue = model->BSIM4v5pdwb;
            return(OK);
        case BSIM4v5_MOD_PB0:
            value->rValue = model->BSIM4v5pb0;
            return(OK);
        case BSIM4v5_MOD_PB1:
            value->rValue = model->BSIM4v5pb1;
            return(OK);
        case BSIM4v5_MOD_PALPHA0:
            value->rValue = model->BSIM4v5palpha0;
            return(OK);
        case BSIM4v5_MOD_PALPHA1:
            value->rValue = model->BSIM4v5palpha1;
            return(OK);
        case BSIM4v5_MOD_PBETA0:
            value->rValue = model->BSIM4v5pbeta0;
            return(OK);
        case BSIM4v5_MOD_PAGIDL:
            value->rValue = model->BSIM4v5pagidl;
            return(OK);
        case BSIM4v5_MOD_PBGIDL:
            value->rValue = model->BSIM4v5pbgidl;
            return(OK);
        case BSIM4v5_MOD_PCGIDL:
            value->rValue = model->BSIM4v5pcgidl;
            return(OK);
        case BSIM4v5_MOD_PEGIDL:
            value->rValue = model->BSIM4v5pegidl;
            return(OK);
        case BSIM4v5_MOD_PAIGC:
            value->rValue = model->BSIM4v5paigc;
            return(OK);
        case BSIM4v5_MOD_PBIGC:
            value->rValue = model->BSIM4v5pbigc;
            return(OK);
        case BSIM4v5_MOD_PCIGC:
            value->rValue = model->BSIM4v5pcigc;
            return(OK);
        case BSIM4v5_MOD_PAIGSD:
            value->rValue = model->BSIM4v5paigsd;
            return(OK);
        case BSIM4v5_MOD_PBIGSD:
            value->rValue = model->BSIM4v5pbigsd;
            return(OK);
        case BSIM4v5_MOD_PCIGSD:
            value->rValue = model->BSIM4v5pcigsd;
            return(OK);
        case BSIM4v5_MOD_PAIGBACC:
            value->rValue = model->BSIM4v5paigbacc;
            return(OK);
        case BSIM4v5_MOD_PBIGBACC:
            value->rValue = model->BSIM4v5pbigbacc;
            return(OK);
        case BSIM4v5_MOD_PCIGBACC:
            value->rValue = model->BSIM4v5pcigbacc;
            return(OK);
        case BSIM4v5_MOD_PAIGBINV:
            value->rValue = model->BSIM4v5paigbinv;
            return(OK);
        case BSIM4v5_MOD_PBIGBINV:
            value->rValue = model->BSIM4v5pbigbinv;
            return(OK);
        case BSIM4v5_MOD_PCIGBINV:
            value->rValue = model->BSIM4v5pcigbinv;
            return(OK);
        case BSIM4v5_MOD_PNIGC:
            value->rValue = model->BSIM4v5pnigc;
            return(OK);
        case BSIM4v5_MOD_PNIGBACC:
            value->rValue = model->BSIM4v5pnigbacc;
            return(OK);
        case BSIM4v5_MOD_PNIGBINV:
            value->rValue = model->BSIM4v5pnigbinv;
            return(OK);
        case BSIM4v5_MOD_PNTOX:
            value->rValue = model->BSIM4v5pntox;
            return(OK);
        case BSIM4v5_MOD_PEIGBINV:
            value->rValue = model->BSIM4v5peigbinv;
            return(OK);
        case BSIM4v5_MOD_PPIGCD:
            value->rValue = model->BSIM4v5ppigcd;
            return(OK);
        case BSIM4v5_MOD_PPOXEDGE:
            value->rValue = model->BSIM4v5ppoxedge;
            return(OK);
        case BSIM4v5_MOD_PPHIN:
            value->rValue = model->BSIM4v5pphin;
            return(OK);
        case BSIM4v5_MOD_PXRCRG1:
            value->rValue = model->BSIM4v5pxrcrg1;
            return(OK);
        case BSIM4v5_MOD_PXRCRG2:
            value->rValue = model->BSIM4v5pxrcrg2;
            return(OK);
        case BSIM4v5_MOD_PEU:
            value->rValue = model->BSIM4v5peu;
            return(OK);
        case BSIM4v5_MOD_PVFB:
            value->rValue = model->BSIM4v5pvfb;
            return(OK);

        case BSIM4v5_MOD_PCGSL:
            value->rValue = model->BSIM4v5pcgsl;
            return(OK);
        case BSIM4v5_MOD_PCGDL:
            value->rValue = model->BSIM4v5pcgdl;
            return(OK);
        case BSIM4v5_MOD_PCKAPPAS:
            value->rValue = model->BSIM4v5pckappas;
            return(OK);
        case BSIM4v5_MOD_PCKAPPAD:
            value->rValue = model->BSIM4v5pckappad;
            return(OK);
        case BSIM4v5_MOD_PCF:
            value->rValue = model->BSIM4v5pcf;
            return(OK);
        case BSIM4v5_MOD_PCLC:
            value->rValue = model->BSIM4v5pclc;
            return(OK);
        case BSIM4v5_MOD_PCLE:
            value->rValue = model->BSIM4v5pcle;
            return(OK);
        case BSIM4v5_MOD_PVFBCV:
            value->rValue = model->BSIM4v5pvfbcv;
            return(OK);
        case BSIM4v5_MOD_PACDE:
            value->rValue = model->BSIM4v5pacde;
            return(OK);
        case BSIM4v5_MOD_PMOIN:
            value->rValue = model->BSIM4v5pmoin;
            return(OK);
        case BSIM4v5_MOD_PNOFF:
            value->rValue = model->BSIM4v5pnoff;
            return(OK);
        case BSIM4v5_MOD_PVOFFCV:
            value->rValue = model->BSIM4v5pvoffcv;
            return(OK);
        case BSIM4v5_MOD_PVFBSDOFF:
            value->rValue = model->BSIM4v5pvfbsdoff;
            return(OK);
        case BSIM4v5_MOD_PTVFBSDOFF:
            value->rValue = model->BSIM4v5ptvfbsdoff;
            return(OK);

        case  BSIM4v5_MOD_TNOM :
          value->rValue = model->BSIM4v5tnom;
            return(OK);
        case BSIM4v5_MOD_CGSO:
            value->rValue = model->BSIM4v5cgso; 
            return(OK);
        case BSIM4v5_MOD_CGDO:
            value->rValue = model->BSIM4v5cgdo; 
            return(OK);
        case BSIM4v5_MOD_CGBO:
            value->rValue = model->BSIM4v5cgbo; 
            return(OK);
        case BSIM4v5_MOD_XPART:
            value->rValue = model->BSIM4v5xpart; 
            return(OK);
        case BSIM4v5_MOD_RSH:
            value->rValue = model->BSIM4v5sheetResistance; 
            return(OK);
        case BSIM4v5_MOD_JSS:
            value->rValue = model->BSIM4v5SjctSatCurDensity; 
            return(OK);
        case BSIM4v5_MOD_JSWS:
            value->rValue = model->BSIM4v5SjctSidewallSatCurDensity; 
            return(OK);
        case BSIM4v5_MOD_JSWGS:
            value->rValue = model->BSIM4v5SjctGateSidewallSatCurDensity;
            return(OK);
        case BSIM4v5_MOD_PBS:
            value->rValue = model->BSIM4v5SbulkJctPotential; 
            return(OK);
        case BSIM4v5_MOD_MJS:
            value->rValue = model->BSIM4v5SbulkJctBotGradingCoeff; 
            return(OK);
        case BSIM4v5_MOD_PBSWS:
            value->rValue = model->BSIM4v5SsidewallJctPotential; 
            return(OK);
        case BSIM4v5_MOD_MJSWS:
            value->rValue = model->BSIM4v5SbulkJctSideGradingCoeff; 
            return(OK);
        case BSIM4v5_MOD_CJS:
            value->rValue = model->BSIM4v5SunitAreaJctCap; 
            return(OK);
        case BSIM4v5_MOD_CJSWS:
            value->rValue = model->BSIM4v5SunitLengthSidewallJctCap; 
            return(OK);
        case BSIM4v5_MOD_PBSWGS:
            value->rValue = model->BSIM4v5SGatesidewallJctPotential; 
            return(OK);
        case BSIM4v5_MOD_MJSWGS:
            value->rValue = model->BSIM4v5SbulkJctGateSideGradingCoeff; 
            return(OK);
        case BSIM4v5_MOD_CJSWGS:
            value->rValue = model->BSIM4v5SunitLengthGateSidewallJctCap; 
            return(OK);
        case BSIM4v5_MOD_NJS:
            value->rValue = model->BSIM4v5SjctEmissionCoeff; 
            return(OK);
        case BSIM4v5_MOD_XTIS:
            value->rValue = model->BSIM4v5SjctTempExponent; 
            return(OK);
        case BSIM4v5_MOD_JSD:
            value->rValue = model->BSIM4v5DjctSatCurDensity;
            return(OK);
        case BSIM4v5_MOD_JSWD:
            value->rValue = model->BSIM4v5DjctSidewallSatCurDensity;
            return(OK);
        case BSIM4v5_MOD_JSWGD:
            value->rValue = model->BSIM4v5DjctGateSidewallSatCurDensity;
            return(OK);
        case BSIM4v5_MOD_PBD:
            value->rValue = model->BSIM4v5DbulkJctPotential;
            return(OK);
        case BSIM4v5_MOD_MJD:
            value->rValue = model->BSIM4v5DbulkJctBotGradingCoeff;
            return(OK);
        case BSIM4v5_MOD_PBSWD:
            value->rValue = model->BSIM4v5DsidewallJctPotential;
            return(OK);
        case BSIM4v5_MOD_MJSWD:
            value->rValue = model->BSIM4v5DbulkJctSideGradingCoeff;
            return(OK);
        case BSIM4v5_MOD_CJD:
            value->rValue = model->BSIM4v5DunitAreaJctCap;
            return(OK);
        case BSIM4v5_MOD_CJSWD:
            value->rValue = model->BSIM4v5DunitLengthSidewallJctCap;
            return(OK);
        case BSIM4v5_MOD_PBSWGD:
            value->rValue = model->BSIM4v5DGatesidewallJctPotential;
            return(OK);
        case BSIM4v5_MOD_MJSWGD:
            value->rValue = model->BSIM4v5DbulkJctGateSideGradingCoeff;
            return(OK);
        case BSIM4v5_MOD_CJSWGD:
            value->rValue = model->BSIM4v5DunitLengthGateSidewallJctCap;
            return(OK);
        case BSIM4v5_MOD_NJD:
            value->rValue = model->BSIM4v5DjctEmissionCoeff;
            return(OK);
        case BSIM4v5_MOD_XTID:
            value->rValue = model->BSIM4v5DjctTempExponent;
            return(OK);
        case BSIM4v5_MOD_LINT:
            value->rValue = model->BSIM4v5Lint; 
            return(OK);
        case BSIM4v5_MOD_LL:
            value->rValue = model->BSIM4v5Ll;
            return(OK);
        case BSIM4v5_MOD_LLC:
            value->rValue = model->BSIM4v5Llc;
            return(OK);
        case BSIM4v5_MOD_LLN:
            value->rValue = model->BSIM4v5Lln;
            return(OK);
        case BSIM4v5_MOD_LW:
            value->rValue = model->BSIM4v5Lw;
            return(OK);
        case BSIM4v5_MOD_LWC:
            value->rValue = model->BSIM4v5Lwc;
            return(OK);
        case BSIM4v5_MOD_LWN:
            value->rValue = model->BSIM4v5Lwn;
            return(OK);
        case BSIM4v5_MOD_LWL:
            value->rValue = model->BSIM4v5Lwl;
            return(OK);
        case BSIM4v5_MOD_LWLC:
            value->rValue = model->BSIM4v5Lwlc;
            return(OK);
        case BSIM4v5_MOD_LMIN:
            value->rValue = model->BSIM4v5Lmin;
            return(OK);
        case BSIM4v5_MOD_LMAX:
            value->rValue = model->BSIM4v5Lmax;
            return(OK);
        case BSIM4v5_MOD_WINT:
            value->rValue = model->BSIM4v5Wint;
            return(OK);
        case BSIM4v5_MOD_WL:
            value->rValue = model->BSIM4v5Wl;
            return(OK);
        case BSIM4v5_MOD_WLC:
            value->rValue = model->BSIM4v5Wlc;
            return(OK);
        case BSIM4v5_MOD_WLN:
            value->rValue = model->BSIM4v5Wln;
            return(OK);
        case BSIM4v5_MOD_WW:
            value->rValue = model->BSIM4v5Ww;
            return(OK);
        case BSIM4v5_MOD_WWC:
            value->rValue = model->BSIM4v5Wwc;
            return(OK);
        case BSIM4v5_MOD_WWN:
            value->rValue = model->BSIM4v5Wwn;
            return(OK);
        case BSIM4v5_MOD_WWL:
            value->rValue = model->BSIM4v5Wwl;
            return(OK);
        case BSIM4v5_MOD_WWLC:
            value->rValue = model->BSIM4v5Wwlc;
            return(OK);
        case BSIM4v5_MOD_WMIN:
            value->rValue = model->BSIM4v5Wmin;
            return(OK);
        case BSIM4v5_MOD_WMAX:
            value->rValue = model->BSIM4v5Wmax;
            return(OK);

        /* stress effect */
        case BSIM4v5_MOD_SAREF:
            value->rValue = model->BSIM4v5saref;
            return(OK);
        case BSIM4v5_MOD_SBREF:
            value->rValue = model->BSIM4v5sbref;
            return(OK);
	case BSIM4v5_MOD_WLOD:
            value->rValue = model->BSIM4v5wlod;
            return(OK);
        case BSIM4v5_MOD_KU0:
            value->rValue = model->BSIM4v5ku0;
            return(OK);
        case BSIM4v5_MOD_KVSAT:
            value->rValue = model->BSIM4v5kvsat;
            return(OK);
        case BSIM4v5_MOD_KVTH0:
            value->rValue = model->BSIM4v5kvth0;
            return(OK);
        case BSIM4v5_MOD_TKU0:
            value->rValue = model->BSIM4v5tku0;
            return(OK);
        case BSIM4v5_MOD_LLODKU0:
            value->rValue = model->BSIM4v5llodku0;
            return(OK);
        case BSIM4v5_MOD_WLODKU0:
            value->rValue = model->BSIM4v5wlodku0;
            return(OK);
        case BSIM4v5_MOD_LLODVTH:
            value->rValue = model->BSIM4v5llodvth;
            return(OK);
        case BSIM4v5_MOD_WLODVTH:
            value->rValue = model->BSIM4v5wlodvth;
            return(OK);
        case BSIM4v5_MOD_LKU0:
            value->rValue = model->BSIM4v5lku0;
            return(OK);
        case BSIM4v5_MOD_WKU0:
            value->rValue = model->BSIM4v5wku0;
            return(OK);
        case BSIM4v5_MOD_PKU0:
            value->rValue = model->BSIM4v5pku0;
            return(OK);
        case BSIM4v5_MOD_LKVTH0:
            value->rValue = model->BSIM4v5lkvth0;
            return(OK);
        case BSIM4v5_MOD_WKVTH0:
            value->rValue = model->BSIM4v5wkvth0;
            return(OK);
        case BSIM4v5_MOD_PKVTH0:
            value->rValue = model->BSIM4v5pkvth0;
            return(OK);
        case BSIM4v5_MOD_STK2:
            value->rValue = model->BSIM4v5stk2;
            return(OK);
        case BSIM4v5_MOD_LODK2:
            value->rValue = model->BSIM4v5lodk2;
            return(OK);
        case BSIM4v5_MOD_STETA0:
            value->rValue = model->BSIM4v5steta0;
            return(OK);
        case BSIM4v5_MOD_LODETA0:
            value->rValue = model->BSIM4v5lodeta0;
            return(OK);

        /* Well Proximity Effect  */
        case BSIM4v5_MOD_WEB:
            value->rValue = model->BSIM4v5web;
            return(OK);
        case BSIM4v5_MOD_WEC:
            value->rValue = model->BSIM4v5wec;
            return(OK);
        case BSIM4v5_MOD_KVTH0WE:
            value->rValue = model->BSIM4v5kvth0we;
            return(OK);
        case BSIM4v5_MOD_K2WE:
            value->rValue = model->BSIM4v5k2we;
            return(OK);
        case BSIM4v5_MOD_KU0WE:
            value->rValue = model->BSIM4v5ku0we;
            return(OK);
        case BSIM4v5_MOD_SCREF:
            value->rValue = model->BSIM4v5scref;
            return(OK);
        case BSIM4v5_MOD_WPEMOD:
            value->rValue = model->BSIM4v5wpemod;
            return(OK);
        case BSIM4v5_MOD_LKVTH0WE:
            value->rValue = model->BSIM4v5lkvth0we;
            return(OK);
        case BSIM4v5_MOD_LK2WE:
            value->rValue = model->BSIM4v5lk2we;
            return(OK);
        case BSIM4v5_MOD_LKU0WE:
            value->rValue = model->BSIM4v5lku0we;
            return(OK);
        case BSIM4v5_MOD_WKVTH0WE:
            value->rValue = model->BSIM4v5wkvth0we;
            return(OK);
        case BSIM4v5_MOD_WK2WE:
            value->rValue = model->BSIM4v5wk2we;
            return(OK);
        case BSIM4v5_MOD_WKU0WE:
            value->rValue = model->BSIM4v5wku0we;
            return(OK);
        case BSIM4v5_MOD_PKVTH0WE:
            value->rValue = model->BSIM4v5pkvth0we;
            return(OK);
        case BSIM4v5_MOD_PK2WE:
            value->rValue = model->BSIM4v5pk2we;
            return(OK);
        case BSIM4v5_MOD_PKU0WE:
            value->rValue = model->BSIM4v5pku0we;
            return(OK);

        case BSIM4v5_MOD_NOIA:
            value->rValue = model->BSIM4v5oxideTrapDensityA;
            return(OK);
        case BSIM4v5_MOD_NOIB:
            value->rValue = model->BSIM4v5oxideTrapDensityB;
            return(OK);
        case BSIM4v5_MOD_NOIC:
            value->rValue = model->BSIM4v5oxideTrapDensityC;
            return(OK);
        case BSIM4v5_MOD_EM:
            value->rValue = model->BSIM4v5em;
            return(OK);
        case BSIM4v5_MOD_EF:
            value->rValue = model->BSIM4v5ef;
            return(OK);
        case BSIM4v5_MOD_AF:
            value->rValue = model->BSIM4v5af;
            return(OK);
        case BSIM4v5_MOD_KF:
            value->rValue = model->BSIM4v5kf;
            return(OK);

        case BSIM4v5_MOD_VGS_MAX:
            value->rValue = model->BSIM4v5vgsMax;
            return(OK);
        case BSIM4v5_MOD_VGD_MAX:
            value->rValue = model->BSIM4v5vgdMax;
            return(OK);
        case BSIM4v5_MOD_VGB_MAX:
            value->rValue = model->BSIM4v5vgbMax;
            return(OK);
        case BSIM4v5_MOD_VDS_MAX:
            value->rValue = model->BSIM4v5vdsMax;
            return(OK);
        case BSIM4v5_MOD_VBS_MAX:
            value->rValue = model->BSIM4v5vbsMax;
            return(OK);
        case BSIM4v5_MOD_VBD_MAX:
            value->rValue = model->BSIM4v5vbdMax;
            return(OK);
        case BSIM4v5_MOD_VGSR_MAX:
            value->rValue = model->BSIM4v5vgsrMax;
            return(OK);
        case BSIM4v5_MOD_VGDR_MAX:
            value->rValue = model->BSIM4v5vgdrMax;
            return(OK);
        case BSIM4v5_MOD_VGBR_MAX:
            value->rValue = model->BSIM4v5vgbrMax;
            return(OK);
        case BSIM4v5_MOD_VBSR_MAX:
            value->rValue = model->BSIM4v5vbsrMax;
            return(OK);
        case BSIM4v5_MOD_VBDR_MAX:
            value->rValue = model->BSIM4v5vbdrMax;
            return(OK);

        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}



