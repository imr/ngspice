/**** BSIM4.0.0, Released by Weidong Liu 3/24/2000 ****/

/**********
 * Copyright 2000 Regents of the University of California. All rights reserved.
 * File: b4mask.c of BSIM4.0.0.
 * Authors: Weidong Liu, Xiaodong Jin, Kanyu M. Cao, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "bsim4v0def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
BSIM4v0mAsk(ckt,inst,which,value)
CKTcircuit *ckt;
GENmodel *inst;
int which;
IFvalue *value;
{
    BSIM4v0model *model = (BSIM4v0model *)inst;
    switch(which) 
    {   case BSIM4v0_MOD_MOBMOD :
            value->iValue = model->BSIM4v0mobMod; 
            return(OK);
        case BSIM4v0_MOD_PARAMCHK :
            value->iValue = model->BSIM4v0paramChk; 
            return(OK);
        case BSIM4v0_MOD_BINUNIT :
            value->iValue = model->BSIM4v0binUnit; 
            return(OK);
        case BSIM4v0_MOD_CAPMOD :
            value->iValue = model->BSIM4v0capMod; 
            return(OK);
        case BSIM4v0_MOD_DIOMOD :
            value->iValue = model->BSIM4v0dioMod;
            return(OK);
        case BSIM4v0_MOD_TRNQSMOD :
            value->iValue = model->BSIM4v0trnqsMod;
            return(OK);
        case BSIM4v0_MOD_ACNQSMOD :
            value->iValue = model->BSIM4v0acnqsMod;
            return(OK);
        case BSIM4v0_MOD_FNOIMOD :
            value->iValue = model->BSIM4v0fnoiMod; 
            return(OK);
        case BSIM4v0_MOD_TNOIMOD :
            value->iValue = model->BSIM4v0tnoiMod;
            return(OK);
        case BSIM4v0_MOD_RDSMOD :
            value->iValue = model->BSIM4v0rdsMod;
            return(OK);
        case BSIM4v0_MOD_RBODYMOD :
            value->iValue = model->BSIM4v0rbodyMod;
            return(OK);
        case BSIM4v0_MOD_RGATEMOD :
            value->iValue = model->BSIM4v0rgateMod;
            return(OK);
        case BSIM4v0_MOD_PERMOD :
            value->iValue = model->BSIM4v0perMod;
            return(OK);
        case BSIM4v0_MOD_GEOMOD :
            value->iValue = model->BSIM4v0geoMod;
            return(OK);
        case BSIM4v0_MOD_IGCMOD :
            value->iValue = model->BSIM4v0igcMod;
            return(OK);
        case BSIM4v0_MOD_IGBMOD :
            value->iValue = model->BSIM4v0igbMod;
            return(OK);
        case  BSIM4v0_MOD_VERSION :
          value->sValue = model->BSIM4v0version;
            return(OK);
        case  BSIM4v0_MOD_TOXREF :
          value->rValue = model->BSIM4v0toxref;
          return(OK);
        case  BSIM4v0_MOD_TOXE :
          value->rValue = model->BSIM4v0toxe;
            return(OK);
        case  BSIM4v0_MOD_TOXP :
          value->rValue = model->BSIM4v0toxp;
            return(OK);
        case  BSIM4v0_MOD_TOXM :
          value->rValue = model->BSIM4v0toxm;
            return(OK);
        case  BSIM4v0_MOD_DTOX :
          value->rValue = model->BSIM4v0dtox;
            return(OK);
        case  BSIM4v0_MOD_EPSROX :
          value->rValue = model->BSIM4v0epsrox;
            return(OK);
        case  BSIM4v0_MOD_CDSC :
          value->rValue = model->BSIM4v0cdsc;
            return(OK);
        case  BSIM4v0_MOD_CDSCB :
          value->rValue = model->BSIM4v0cdscb;
            return(OK);

        case  BSIM4v0_MOD_CDSCD :
          value->rValue = model->BSIM4v0cdscd;
            return(OK);

        case  BSIM4v0_MOD_CIT :
          value->rValue = model->BSIM4v0cit;
            return(OK);
        case  BSIM4v0_MOD_NFACTOR :
          value->rValue = model->BSIM4v0nfactor;
            return(OK);
        case BSIM4v0_MOD_XJ:
            value->rValue = model->BSIM4v0xj;
            return(OK);
        case BSIM4v0_MOD_VSAT:
            value->rValue = model->BSIM4v0vsat;
            return(OK);
        case BSIM4v0_MOD_AT:
            value->rValue = model->BSIM4v0at;
            return(OK);
        case BSIM4v0_MOD_A0:
            value->rValue = model->BSIM4v0a0;
            return(OK);

        case BSIM4v0_MOD_AGS:
            value->rValue = model->BSIM4v0ags;
            return(OK);

        case BSIM4v0_MOD_A1:
            value->rValue = model->BSIM4v0a1;
            return(OK);
        case BSIM4v0_MOD_A2:
            value->rValue = model->BSIM4v0a2;
            return(OK);
        case BSIM4v0_MOD_KETA:
            value->rValue = model->BSIM4v0keta;
            return(OK);   
        case BSIM4v0_MOD_NSUB:
            value->rValue = model->BSIM4v0nsub;
            return(OK);
        case BSIM4v0_MOD_NDEP:
            value->rValue = model->BSIM4v0ndep;
            return(OK);
        case BSIM4v0_MOD_NSD:
            value->rValue = model->BSIM4v0nsd;
            return(OK);
        case BSIM4v0_MOD_NGATE:
            value->rValue = model->BSIM4v0ngate;
            return(OK);
        case BSIM4v0_MOD_GAMMA1:
            value->rValue = model->BSIM4v0gamma1;
            return(OK);
        case BSIM4v0_MOD_GAMMA2:
            value->rValue = model->BSIM4v0gamma2;
            return(OK);
        case BSIM4v0_MOD_VBX:
            value->rValue = model->BSIM4v0vbx;
            return(OK);
        case BSIM4v0_MOD_VBM:
            value->rValue = model->BSIM4v0vbm;
            return(OK);
        case BSIM4v0_MOD_XT:
            value->rValue = model->BSIM4v0xt;
            return(OK);
        case  BSIM4v0_MOD_K1:
          value->rValue = model->BSIM4v0k1;
            return(OK);
        case  BSIM4v0_MOD_KT1:
          value->rValue = model->BSIM4v0kt1;
            return(OK);
        case  BSIM4v0_MOD_KT1L:
          value->rValue = model->BSIM4v0kt1l;
            return(OK);
        case  BSIM4v0_MOD_KT2 :
          value->rValue = model->BSIM4v0kt2;
            return(OK);
        case  BSIM4v0_MOD_K2 :
          value->rValue = model->BSIM4v0k2;
            return(OK);
        case  BSIM4v0_MOD_K3:
          value->rValue = model->BSIM4v0k3;
            return(OK);
        case  BSIM4v0_MOD_K3B:
          value->rValue = model->BSIM4v0k3b;
            return(OK);
        case  BSIM4v0_MOD_W0:
          value->rValue = model->BSIM4v0w0;
            return(OK);
        case  BSIM4v0_MOD_LPE0:
          value->rValue = model->BSIM4v0lpe0;
            return(OK);
        case  BSIM4v0_MOD_LPEB:
          value->rValue = model->BSIM4v0lpeb;
            return(OK);
        case  BSIM4v0_MOD_DVTP0:
          value->rValue = model->BSIM4v0dvtp0;
            return(OK);
        case  BSIM4v0_MOD_DVTP1:
          value->rValue = model->BSIM4v0dvtp1;
            return(OK);
        case  BSIM4v0_MOD_DVT0 :                
          value->rValue = model->BSIM4v0dvt0;
            return(OK);
        case  BSIM4v0_MOD_DVT1 :             
          value->rValue = model->BSIM4v0dvt1;
            return(OK);
        case  BSIM4v0_MOD_DVT2 :             
          value->rValue = model->BSIM4v0dvt2;
            return(OK);
        case  BSIM4v0_MOD_DVT0W :                
          value->rValue = model->BSIM4v0dvt0w;
            return(OK);
        case  BSIM4v0_MOD_DVT1W :             
          value->rValue = model->BSIM4v0dvt1w;
            return(OK);
        case  BSIM4v0_MOD_DVT2W :             
          value->rValue = model->BSIM4v0dvt2w;
            return(OK);
        case  BSIM4v0_MOD_DROUT :           
          value->rValue = model->BSIM4v0drout;
            return(OK);
        case  BSIM4v0_MOD_DSUB :           
          value->rValue = model->BSIM4v0dsub;
            return(OK);
        case BSIM4v0_MOD_VTH0:
            value->rValue = model->BSIM4v0vth0; 
            return(OK);
        case BSIM4v0_MOD_EU:
            value->rValue = model->BSIM4v0eu;
            return(OK);
        case BSIM4v0_MOD_UA:
            value->rValue = model->BSIM4v0ua; 
            return(OK);
        case BSIM4v0_MOD_UA1:
            value->rValue = model->BSIM4v0ua1; 
            return(OK);
        case BSIM4v0_MOD_UB:
            value->rValue = model->BSIM4v0ub;  
            return(OK);
        case BSIM4v0_MOD_UB1:
            value->rValue = model->BSIM4v0ub1;  
            return(OK);
        case BSIM4v0_MOD_UC:
            value->rValue = model->BSIM4v0uc; 
            return(OK);
        case BSIM4v0_MOD_UC1:
            value->rValue = model->BSIM4v0uc1; 
            return(OK);
        case BSIM4v0_MOD_U0:
            value->rValue = model->BSIM4v0u0;
            return(OK);
        case BSIM4v0_MOD_UTE:
            value->rValue = model->BSIM4v0ute;
            return(OK);
        case BSIM4v0_MOD_VOFF:
            value->rValue = model->BSIM4v0voff;
            return(OK);
        case BSIM4v0_MOD_VOFFL:
            value->rValue = model->BSIM4v0voffl;
            return(OK);
        case BSIM4v0_MOD_MINV:
            value->rValue = model->BSIM4v0minv;
            return(OK);
        case BSIM4v0_MOD_FPROUT:
            value->rValue = model->BSIM4v0fprout;
            return(OK);
        case BSIM4v0_MOD_PDITS:
            value->rValue = model->BSIM4v0pdits;
            return(OK);
        case BSIM4v0_MOD_PDITSD:
            value->rValue = model->BSIM4v0pditsd;
            return(OK);
        case BSIM4v0_MOD_PDITSL:
            value->rValue = model->BSIM4v0pditsl;
            return(OK);
        case BSIM4v0_MOD_DELTA:
            value->rValue = model->BSIM4v0delta;
            return(OK);
        case BSIM4v0_MOD_RDSW:
            value->rValue = model->BSIM4v0rdsw; 
            return(OK);
        case BSIM4v0_MOD_RDSWMIN:
            value->rValue = model->BSIM4v0rdswmin;
            return(OK);
        case BSIM4v0_MOD_RDWMIN:
            value->rValue = model->BSIM4v0rdwmin;
            return(OK);
        case BSIM4v0_MOD_RSWMIN:
            value->rValue = model->BSIM4v0rswmin;
            return(OK);
        case BSIM4v0_MOD_RDW:
            value->rValue = model->BSIM4v0rdw;
            return(OK);
        case BSIM4v0_MOD_RSW:
            value->rValue = model->BSIM4v0rsw;
            return(OK);
        case BSIM4v0_MOD_PRWG:
            value->rValue = model->BSIM4v0prwg; 
            return(OK);             
        case BSIM4v0_MOD_PRWB:
            value->rValue = model->BSIM4v0prwb; 
            return(OK);             
        case BSIM4v0_MOD_PRT:
            value->rValue = model->BSIM4v0prt; 
            return(OK);              
        case BSIM4v0_MOD_ETA0:
            value->rValue = model->BSIM4v0eta0; 
            return(OK);               
        case BSIM4v0_MOD_ETAB:
            value->rValue = model->BSIM4v0etab; 
            return(OK);               
        case BSIM4v0_MOD_PCLM:
            value->rValue = model->BSIM4v0pclm; 
            return(OK);               
        case BSIM4v0_MOD_PDIBL1:
            value->rValue = model->BSIM4v0pdibl1; 
            return(OK);               
        case BSIM4v0_MOD_PDIBL2:
            value->rValue = model->BSIM4v0pdibl2; 
            return(OK);               
        case BSIM4v0_MOD_PDIBLB:
            value->rValue = model->BSIM4v0pdiblb; 
            return(OK);               
        case BSIM4v0_MOD_PSCBE1:
            value->rValue = model->BSIM4v0pscbe1; 
            return(OK);               
        case BSIM4v0_MOD_PSCBE2:
            value->rValue = model->BSIM4v0pscbe2; 
            return(OK);               
        case BSIM4v0_MOD_PVAG:
            value->rValue = model->BSIM4v0pvag; 
            return(OK);               
        case BSIM4v0_MOD_WR:
            value->rValue = model->BSIM4v0wr;
            return(OK);
        case BSIM4v0_MOD_DWG:
            value->rValue = model->BSIM4v0dwg;
            return(OK);
        case BSIM4v0_MOD_DWB:
            value->rValue = model->BSIM4v0dwb;
            return(OK);
        case BSIM4v0_MOD_B0:
            value->rValue = model->BSIM4v0b0;
            return(OK);
        case BSIM4v0_MOD_B1:
            value->rValue = model->BSIM4v0b1;
            return(OK);
        case BSIM4v0_MOD_ALPHA0:
            value->rValue = model->BSIM4v0alpha0;
            return(OK);
        case BSIM4v0_MOD_ALPHA1:
            value->rValue = model->BSIM4v0alpha1;
            return(OK);
        case BSIM4v0_MOD_BETA0:
            value->rValue = model->BSIM4v0beta0;
            return(OK);
        case BSIM4v0_MOD_AGIDL:
            value->rValue = model->BSIM4v0agidl;
            return(OK);
        case BSIM4v0_MOD_BGIDL:
            value->rValue = model->BSIM4v0bgidl;
            return(OK);
        case BSIM4v0_MOD_CGIDL:
            value->rValue = model->BSIM4v0cgidl;
            return(OK);
        case BSIM4v0_MOD_EGIDL:
            value->rValue = model->BSIM4v0egidl;
            return(OK);
        case BSIM4v0_MOD_AIGC:
            value->rValue = model->BSIM4v0aigc;
            return(OK);
        case BSIM4v0_MOD_BIGC:
            value->rValue = model->BSIM4v0bigc;
            return(OK);
        case BSIM4v0_MOD_CIGC:
            value->rValue = model->BSIM4v0cigc;
            return(OK);
        case BSIM4v0_MOD_AIGSD:
            value->rValue = model->BSIM4v0aigsd;
            return(OK);
        case BSIM4v0_MOD_BIGSD:
            value->rValue = model->BSIM4v0bigsd;
            return(OK);
        case BSIM4v0_MOD_CIGSD:
            value->rValue = model->BSIM4v0cigsd;
            return(OK);
        case BSIM4v0_MOD_AIGBACC:
            value->rValue = model->BSIM4v0aigbacc;
            return(OK);
        case BSIM4v0_MOD_BIGBACC:
            value->rValue = model->BSIM4v0bigbacc;
            return(OK);
        case BSIM4v0_MOD_CIGBACC:
            value->rValue = model->BSIM4v0cigbacc;
            return(OK);
        case BSIM4v0_MOD_AIGBINV:
            value->rValue = model->BSIM4v0aigbinv;
            return(OK);
        case BSIM4v0_MOD_BIGBINV:
            value->rValue = model->BSIM4v0bigbinv;
            return(OK);
        case BSIM4v0_MOD_CIGBINV:
            value->rValue = model->BSIM4v0cigbinv;
            return(OK);
        case BSIM4v0_MOD_NIGC:
            value->rValue = model->BSIM4v0nigc;
            return(OK);
        case BSIM4v0_MOD_NIGBACC:
            value->rValue = model->BSIM4v0nigbacc;
            return(OK);
        case BSIM4v0_MOD_NIGBINV:
            value->rValue = model->BSIM4v0nigbinv;
            return(OK);
        case BSIM4v0_MOD_NTOX:
            value->rValue = model->BSIM4v0ntox;
            return(OK);
        case BSIM4v0_MOD_EIGBINV:
            value->rValue = model->BSIM4v0eigbinv;
            return(OK);
        case BSIM4v0_MOD_PIGCD:
            value->rValue = model->BSIM4v0pigcd;
            return(OK);
        case BSIM4v0_MOD_POXEDGE:
            value->rValue = model->BSIM4v0poxedge;
            return(OK);
        case BSIM4v0_MOD_PHIN:
            value->rValue = model->BSIM4v0phin;
            return(OK);
        case BSIM4v0_MOD_XRCRG1:
            value->rValue = model->BSIM4v0xrcrg1;
            return(OK);
        case BSIM4v0_MOD_XRCRG2:
            value->rValue = model->BSIM4v0xrcrg2;
            return(OK);
        case BSIM4v0_MOD_TNOIA:
            value->rValue = model->BSIM4v0tnoia;
            return(OK);
        case BSIM4v0_MOD_TNOIB:
            value->rValue = model->BSIM4v0tnoib;
            return(OK);
        case BSIM4v0_MOD_NTNOI:
            value->rValue = model->BSIM4v0ntnoi;
            return(OK);
        case BSIM4v0_MOD_IJTHDFWD:
            value->rValue = model->BSIM4v0ijthdfwd;
            return(OK);
        case BSIM4v0_MOD_IJTHSFWD:
            value->rValue = model->BSIM4v0ijthsfwd;
            return(OK);
        case BSIM4v0_MOD_IJTHDREV:
            value->rValue = model->BSIM4v0ijthdrev;
            return(OK);
        case BSIM4v0_MOD_IJTHSREV:
            value->rValue = model->BSIM4v0ijthsrev;
            return(OK);
        case BSIM4v0_MOD_XJBVD:
            value->rValue = model->BSIM4v0xjbvd;
            return(OK);
        case BSIM4v0_MOD_XJBVS:
            value->rValue = model->BSIM4v0xjbvs;
            return(OK);
        case BSIM4v0_MOD_BVD:
            value->rValue = model->BSIM4v0bvd;
            return(OK);
        case BSIM4v0_MOD_BVS:
            value->rValue = model->BSIM4v0bvs;
            return(OK);
        case BSIM4v0_MOD_VFB:
            value->rValue = model->BSIM4v0vfb;
            return(OK);

        case BSIM4v0_MOD_GBMIN:
            value->rValue = model->BSIM4v0gbmin;
            return(OK);
        case BSIM4v0_MOD_RBDB:
            value->rValue = model->BSIM4v0rbdb;
            return(OK);
        case BSIM4v0_MOD_RBPB:
            value->rValue = model->BSIM4v0rbpb;
            return(OK);
        case BSIM4v0_MOD_RBSB:
            value->rValue = model->BSIM4v0rbsb;
            return(OK);
        case BSIM4v0_MOD_RBPS:
            value->rValue = model->BSIM4v0rbps;
            return(OK);
        case BSIM4v0_MOD_RBPD:
            value->rValue = model->BSIM4v0rbpd;
            return(OK);

        case BSIM4v0_MOD_CGSL:
            value->rValue = model->BSIM4v0cgsl;
            return(OK);
        case BSIM4v0_MOD_CGDL:
            value->rValue = model->BSIM4v0cgdl;
            return(OK);
        case BSIM4v0_MOD_CKAPPAS:
            value->rValue = model->BSIM4v0ckappas;
            return(OK);
        case BSIM4v0_MOD_CKAPPAD:
            value->rValue = model->BSIM4v0ckappad;
            return(OK);
        case BSIM4v0_MOD_CF:
            value->rValue = model->BSIM4v0cf;
            return(OK);
        case BSIM4v0_MOD_CLC:
            value->rValue = model->BSIM4v0clc;
            return(OK);
        case BSIM4v0_MOD_CLE:
            value->rValue = model->BSIM4v0cle;
            return(OK);
        case BSIM4v0_MOD_DWC:
            value->rValue = model->BSIM4v0dwc;
            return(OK);
        case BSIM4v0_MOD_DLC:
            value->rValue = model->BSIM4v0dlc;
            return(OK);
        case BSIM4v0_MOD_DLCIG:
            value->rValue = model->BSIM4v0dlcig;
            return(OK);
        case BSIM4v0_MOD_DWJ:
            value->rValue = model->BSIM4v0dwj;
            return(OK);
        case BSIM4v0_MOD_VFBCV:
            value->rValue = model->BSIM4v0vfbcv; 
            return(OK);
        case BSIM4v0_MOD_ACDE:
            value->rValue = model->BSIM4v0acde;
            return(OK);
        case BSIM4v0_MOD_MOIN:
            value->rValue = model->BSIM4v0moin;
            return(OK);
        case BSIM4v0_MOD_NOFF:
            value->rValue = model->BSIM4v0noff;
            return(OK);
        case BSIM4v0_MOD_VOFFCV:
            value->rValue = model->BSIM4v0voffcv;
            return(OK);
        case BSIM4v0_MOD_DMCG:
            value->rValue = model->BSIM4v0dmcg;
            return(OK);
        case BSIM4v0_MOD_DMCI:
            value->rValue = model->BSIM4v0dmci;
            return(OK);
        case BSIM4v0_MOD_DMDG:
            value->rValue = model->BSIM4v0dmdg;
            return(OK);
        case BSIM4v0_MOD_DMCGT:
            value->rValue = model->BSIM4v0dmcgt;
            return(OK);
        case BSIM4v0_MOD_XGW:
            value->rValue = model->BSIM4v0xgw;
            return(OK);
        case BSIM4v0_MOD_XGL:
            value->rValue = model->BSIM4v0xgl;
            return(OK);
        case BSIM4v0_MOD_RSHG:
            value->rValue = model->BSIM4v0rshg;
            return(OK);
        case BSIM4v0_MOD_NGCON:
            value->rValue = model->BSIM4v0ngcon;
            return(OK);
        case BSIM4v0_MOD_TCJ:
            value->rValue = model->BSIM4v0tcj;
            return(OK);
        case BSIM4v0_MOD_TPB:
            value->rValue = model->BSIM4v0tpb;
            return(OK);
        case BSIM4v0_MOD_TCJSW:
            value->rValue = model->BSIM4v0tcjsw;
            return(OK);
        case BSIM4v0_MOD_TPBSW:
            value->rValue = model->BSIM4v0tpbsw;
            return(OK);
        case BSIM4v0_MOD_TCJSWG:
            value->rValue = model->BSIM4v0tcjswg;
            return(OK);
        case BSIM4v0_MOD_TPBSWG:
            value->rValue = model->BSIM4v0tpbswg;
            return(OK);

	/* Length dependence */
        case  BSIM4v0_MOD_LCDSC :
          value->rValue = model->BSIM4v0lcdsc;
            return(OK);
        case  BSIM4v0_MOD_LCDSCB :
          value->rValue = model->BSIM4v0lcdscb;
            return(OK);
        case  BSIM4v0_MOD_LCDSCD :
          value->rValue = model->BSIM4v0lcdscd;
            return(OK);
        case  BSIM4v0_MOD_LCIT :
          value->rValue = model->BSIM4v0lcit;
            return(OK);
        case  BSIM4v0_MOD_LNFACTOR :
          value->rValue = model->BSIM4v0lnfactor;
            return(OK);
        case BSIM4v0_MOD_LXJ:
            value->rValue = model->BSIM4v0lxj;
            return(OK);
        case BSIM4v0_MOD_LVSAT:
            value->rValue = model->BSIM4v0lvsat;
            return(OK);
        case BSIM4v0_MOD_LAT:
            value->rValue = model->BSIM4v0lat;
            return(OK);
        case BSIM4v0_MOD_LA0:
            value->rValue = model->BSIM4v0la0;
            return(OK);
        case BSIM4v0_MOD_LAGS:
            value->rValue = model->BSIM4v0lags;
            return(OK);
        case BSIM4v0_MOD_LA1:
            value->rValue = model->BSIM4v0la1;
            return(OK);
        case BSIM4v0_MOD_LA2:
            value->rValue = model->BSIM4v0la2;
            return(OK);
        case BSIM4v0_MOD_LKETA:
            value->rValue = model->BSIM4v0lketa;
            return(OK);   
        case BSIM4v0_MOD_LNSUB:
            value->rValue = model->BSIM4v0lnsub;
            return(OK);
        case BSIM4v0_MOD_LNDEP:
            value->rValue = model->BSIM4v0lndep;
            return(OK);
        case BSIM4v0_MOD_LNSD:
            value->rValue = model->BSIM4v0lnsd;
            return(OK);
        case BSIM4v0_MOD_LNGATE:
            value->rValue = model->BSIM4v0lngate;
            return(OK);
        case BSIM4v0_MOD_LGAMMA1:
            value->rValue = model->BSIM4v0lgamma1;
            return(OK);
        case BSIM4v0_MOD_LGAMMA2:
            value->rValue = model->BSIM4v0lgamma2;
            return(OK);
        case BSIM4v0_MOD_LVBX:
            value->rValue = model->BSIM4v0lvbx;
            return(OK);
        case BSIM4v0_MOD_LVBM:
            value->rValue = model->BSIM4v0lvbm;
            return(OK);
        case BSIM4v0_MOD_LXT:
            value->rValue = model->BSIM4v0lxt;
            return(OK);
        case  BSIM4v0_MOD_LK1:
          value->rValue = model->BSIM4v0lk1;
            return(OK);
        case  BSIM4v0_MOD_LKT1:
          value->rValue = model->BSIM4v0lkt1;
            return(OK);
        case  BSIM4v0_MOD_LKT1L:
          value->rValue = model->BSIM4v0lkt1l;
            return(OK);
        case  BSIM4v0_MOD_LKT2 :
          value->rValue = model->BSIM4v0lkt2;
            return(OK);
        case  BSIM4v0_MOD_LK2 :
          value->rValue = model->BSIM4v0lk2;
            return(OK);
        case  BSIM4v0_MOD_LK3:
          value->rValue = model->BSIM4v0lk3;
            return(OK);
        case  BSIM4v0_MOD_LK3B:
          value->rValue = model->BSIM4v0lk3b;
            return(OK);
        case  BSIM4v0_MOD_LW0:
          value->rValue = model->BSIM4v0lw0;
            return(OK);
        case  BSIM4v0_MOD_LLPE0:
          value->rValue = model->BSIM4v0llpe0;
            return(OK);
        case  BSIM4v0_MOD_LLPEB:
          value->rValue = model->BSIM4v0llpeb;
            return(OK);
        case  BSIM4v0_MOD_LDVTP0:
          value->rValue = model->BSIM4v0ldvtp0;
            return(OK);
        case  BSIM4v0_MOD_LDVTP1:
          value->rValue = model->BSIM4v0ldvtp1;
            return(OK);
        case  BSIM4v0_MOD_LDVT0:                
          value->rValue = model->BSIM4v0ldvt0;
            return(OK);
        case  BSIM4v0_MOD_LDVT1 :             
          value->rValue = model->BSIM4v0ldvt1;
            return(OK);
        case  BSIM4v0_MOD_LDVT2 :             
          value->rValue = model->BSIM4v0ldvt2;
            return(OK);
        case  BSIM4v0_MOD_LDVT0W :                
          value->rValue = model->BSIM4v0ldvt0w;
            return(OK);
        case  BSIM4v0_MOD_LDVT1W :             
          value->rValue = model->BSIM4v0ldvt1w;
            return(OK);
        case  BSIM4v0_MOD_LDVT2W :             
          value->rValue = model->BSIM4v0ldvt2w;
            return(OK);
        case  BSIM4v0_MOD_LDROUT :           
          value->rValue = model->BSIM4v0ldrout;
            return(OK);
        case  BSIM4v0_MOD_LDSUB :           
          value->rValue = model->BSIM4v0ldsub;
            return(OK);
        case BSIM4v0_MOD_LVTH0:
            value->rValue = model->BSIM4v0lvth0; 
            return(OK);
        case BSIM4v0_MOD_LUA:
            value->rValue = model->BSIM4v0lua; 
            return(OK);
        case BSIM4v0_MOD_LUA1:
            value->rValue = model->BSIM4v0lua1; 
            return(OK);
        case BSIM4v0_MOD_LUB:
            value->rValue = model->BSIM4v0lub;  
            return(OK);
        case BSIM4v0_MOD_LUB1:
            value->rValue = model->BSIM4v0lub1;  
            return(OK);
        case BSIM4v0_MOD_LUC:
            value->rValue = model->BSIM4v0luc; 
            return(OK);
        case BSIM4v0_MOD_LUC1:
            value->rValue = model->BSIM4v0luc1; 
            return(OK);
        case BSIM4v0_MOD_LU0:
            value->rValue = model->BSIM4v0lu0;
            return(OK);
        case BSIM4v0_MOD_LUTE:
            value->rValue = model->BSIM4v0lute;
            return(OK);
        case BSIM4v0_MOD_LVOFF:
            value->rValue = model->BSIM4v0lvoff;
            return(OK);
        case BSIM4v0_MOD_LMINV:
            value->rValue = model->BSIM4v0lminv;
            return(OK);
        case BSIM4v0_MOD_LFPROUT:
            value->rValue = model->BSIM4v0lfprout;
            return(OK);
        case BSIM4v0_MOD_LPDITS:
            value->rValue = model->BSIM4v0lpdits;
            return(OK);
        case BSIM4v0_MOD_LPDITSD:
            value->rValue = model->BSIM4v0lpditsd;
            return(OK);
        case BSIM4v0_MOD_LDELTA:
            value->rValue = model->BSIM4v0ldelta;
            return(OK);
        case BSIM4v0_MOD_LRDSW:
            value->rValue = model->BSIM4v0lrdsw; 
            return(OK);             
        case BSIM4v0_MOD_LRDW:
            value->rValue = model->BSIM4v0lrdw;
            return(OK);
        case BSIM4v0_MOD_LRSW:
            value->rValue = model->BSIM4v0lrsw;
            return(OK);
        case BSIM4v0_MOD_LPRWB:
            value->rValue = model->BSIM4v0lprwb; 
            return(OK);             
        case BSIM4v0_MOD_LPRWG:
            value->rValue = model->BSIM4v0lprwg; 
            return(OK);             
        case BSIM4v0_MOD_LPRT:
            value->rValue = model->BSIM4v0lprt; 
            return(OK);              
        case BSIM4v0_MOD_LETA0:
            value->rValue = model->BSIM4v0leta0; 
            return(OK);               
        case BSIM4v0_MOD_LETAB:
            value->rValue = model->BSIM4v0letab; 
            return(OK);               
        case BSIM4v0_MOD_LPCLM:
            value->rValue = model->BSIM4v0lpclm; 
            return(OK);               
        case BSIM4v0_MOD_LPDIBL1:
            value->rValue = model->BSIM4v0lpdibl1; 
            return(OK);               
        case BSIM4v0_MOD_LPDIBL2:
            value->rValue = model->BSIM4v0lpdibl2; 
            return(OK);               
        case BSIM4v0_MOD_LPDIBLB:
            value->rValue = model->BSIM4v0lpdiblb; 
            return(OK);               
        case BSIM4v0_MOD_LPSCBE1:
            value->rValue = model->BSIM4v0lpscbe1; 
            return(OK);               
        case BSIM4v0_MOD_LPSCBE2:
            value->rValue = model->BSIM4v0lpscbe2; 
            return(OK);               
        case BSIM4v0_MOD_LPVAG:
            value->rValue = model->BSIM4v0lpvag; 
            return(OK);               
        case BSIM4v0_MOD_LWR:
            value->rValue = model->BSIM4v0lwr;
            return(OK);
        case BSIM4v0_MOD_LDWG:
            value->rValue = model->BSIM4v0ldwg;
            return(OK);
        case BSIM4v0_MOD_LDWB:
            value->rValue = model->BSIM4v0ldwb;
            return(OK);
        case BSIM4v0_MOD_LB0:
            value->rValue = model->BSIM4v0lb0;
            return(OK);
        case BSIM4v0_MOD_LB1:
            value->rValue = model->BSIM4v0lb1;
            return(OK);
        case BSIM4v0_MOD_LALPHA0:
            value->rValue = model->BSIM4v0lalpha0;
            return(OK);
        case BSIM4v0_MOD_LALPHA1:
            value->rValue = model->BSIM4v0lalpha1;
            return(OK);
        case BSIM4v0_MOD_LBETA0:
            value->rValue = model->BSIM4v0lbeta0;
            return(OK);
        case BSIM4v0_MOD_LAGIDL:
            value->rValue = model->BSIM4v0lagidl;
            return(OK);
        case BSIM4v0_MOD_LBGIDL:
            value->rValue = model->BSIM4v0lbgidl;
            return(OK);
        case BSIM4v0_MOD_LCGIDL:
            value->rValue = model->BSIM4v0lcgidl;
            return(OK);
        case BSIM4v0_MOD_LEGIDL:
            value->rValue = model->BSIM4v0legidl;
            return(OK);
        case BSIM4v0_MOD_LAIGC:
            value->rValue = model->BSIM4v0laigc;
            return(OK);
        case BSIM4v0_MOD_LBIGC:
            value->rValue = model->BSIM4v0lbigc;
            return(OK);
        case BSIM4v0_MOD_LCIGC:
            value->rValue = model->BSIM4v0lcigc;
            return(OK);
        case BSIM4v0_MOD_LAIGSD:
            value->rValue = model->BSIM4v0laigsd;
            return(OK);
        case BSIM4v0_MOD_LBIGSD:
            value->rValue = model->BSIM4v0lbigsd;
            return(OK);
        case BSIM4v0_MOD_LCIGSD:
            value->rValue = model->BSIM4v0lcigsd;
            return(OK);
        case BSIM4v0_MOD_LAIGBACC:
            value->rValue = model->BSIM4v0laigbacc;
            return(OK);
        case BSIM4v0_MOD_LBIGBACC:
            value->rValue = model->BSIM4v0lbigbacc;
            return(OK);
        case BSIM4v0_MOD_LCIGBACC:
            value->rValue = model->BSIM4v0lcigbacc;
            return(OK);
        case BSIM4v0_MOD_LAIGBINV:
            value->rValue = model->BSIM4v0laigbinv;
            return(OK);
        case BSIM4v0_MOD_LBIGBINV:
            value->rValue = model->BSIM4v0lbigbinv;
            return(OK);
        case BSIM4v0_MOD_LCIGBINV:
            value->rValue = model->BSIM4v0lcigbinv;
            return(OK);
        case BSIM4v0_MOD_LNIGC:
            value->rValue = model->BSIM4v0lnigc;
            return(OK);
        case BSIM4v0_MOD_LNIGBACC:
            value->rValue = model->BSIM4v0lnigbacc;
            return(OK);
        case BSIM4v0_MOD_LNIGBINV:
            value->rValue = model->BSIM4v0lnigbinv;
            return(OK);
        case BSIM4v0_MOD_LNTOX:
            value->rValue = model->BSIM4v0lntox;
            return(OK);
        case BSIM4v0_MOD_LEIGBINV:
            value->rValue = model->BSIM4v0leigbinv;
            return(OK);
        case BSIM4v0_MOD_LPIGCD:
            value->rValue = model->BSIM4v0lpigcd;
            return(OK);
        case BSIM4v0_MOD_LPOXEDGE:
            value->rValue = model->BSIM4v0lpoxedge;
            return(OK);
        case BSIM4v0_MOD_LPHIN:
            value->rValue = model->BSIM4v0lphin;
            return(OK);
        case BSIM4v0_MOD_LXRCRG1:
            value->rValue = model->BSIM4v0lxrcrg1;
            return(OK);
        case BSIM4v0_MOD_LXRCRG2:
            value->rValue = model->BSIM4v0lxrcrg2;
            return(OK);
        case BSIM4v0_MOD_LEU:
            value->rValue = model->BSIM4v0leu;
            return(OK);
        case BSIM4v0_MOD_LVFB:
            value->rValue = model->BSIM4v0lvfb;
            return(OK);

        case BSIM4v0_MOD_LCGSL:
            value->rValue = model->BSIM4v0lcgsl;
            return(OK);
        case BSIM4v0_MOD_LCGDL:
            value->rValue = model->BSIM4v0lcgdl;
            return(OK);
        case BSIM4v0_MOD_LCKAPPAS:
            value->rValue = model->BSIM4v0lckappas;
            return(OK);
        case BSIM4v0_MOD_LCKAPPAD:
            value->rValue = model->BSIM4v0lckappad;
            return(OK);
        case BSIM4v0_MOD_LCF:
            value->rValue = model->BSIM4v0lcf;
            return(OK);
        case BSIM4v0_MOD_LCLC:
            value->rValue = model->BSIM4v0lclc;
            return(OK);
        case BSIM4v0_MOD_LCLE:
            value->rValue = model->BSIM4v0lcle;
            return(OK);
        case BSIM4v0_MOD_LVFBCV:
            value->rValue = model->BSIM4v0lvfbcv;
            return(OK);
        case BSIM4v0_MOD_LACDE:
            value->rValue = model->BSIM4v0lacde;
            return(OK);
        case BSIM4v0_MOD_LMOIN:
            value->rValue = model->BSIM4v0lmoin;
            return(OK);
        case BSIM4v0_MOD_LNOFF:
            value->rValue = model->BSIM4v0lnoff;
            return(OK);
        case BSIM4v0_MOD_LVOFFCV:
            value->rValue = model->BSIM4v0lvoffcv;
            return(OK);

	/* Width dependence */
        case  BSIM4v0_MOD_WCDSC :
          value->rValue = model->BSIM4v0wcdsc;
            return(OK);
        case  BSIM4v0_MOD_WCDSCB :
          value->rValue = model->BSIM4v0wcdscb;
            return(OK);
        case  BSIM4v0_MOD_WCDSCD :
          value->rValue = model->BSIM4v0wcdscd;
            return(OK);
        case  BSIM4v0_MOD_WCIT :
          value->rValue = model->BSIM4v0wcit;
            return(OK);
        case  BSIM4v0_MOD_WNFACTOR :
          value->rValue = model->BSIM4v0wnfactor;
            return(OK);
        case BSIM4v0_MOD_WXJ:
            value->rValue = model->BSIM4v0wxj;
            return(OK);
        case BSIM4v0_MOD_WVSAT:
            value->rValue = model->BSIM4v0wvsat;
            return(OK);
        case BSIM4v0_MOD_WAT:
            value->rValue = model->BSIM4v0wat;
            return(OK);
        case BSIM4v0_MOD_WA0:
            value->rValue = model->BSIM4v0wa0;
            return(OK);
        case BSIM4v0_MOD_WAGS:
            value->rValue = model->BSIM4v0wags;
            return(OK);
        case BSIM4v0_MOD_WA1:
            value->rValue = model->BSIM4v0wa1;
            return(OK);
        case BSIM4v0_MOD_WA2:
            value->rValue = model->BSIM4v0wa2;
            return(OK);
        case BSIM4v0_MOD_WKETA:
            value->rValue = model->BSIM4v0wketa;
            return(OK);   
        case BSIM4v0_MOD_WNSUB:
            value->rValue = model->BSIM4v0wnsub;
            return(OK);
        case BSIM4v0_MOD_WNDEP:
            value->rValue = model->BSIM4v0wndep;
            return(OK);
        case BSIM4v0_MOD_WNSD:
            value->rValue = model->BSIM4v0wnsd;
            return(OK);
        case BSIM4v0_MOD_WNGATE:
            value->rValue = model->BSIM4v0wngate;
            return(OK);
        case BSIM4v0_MOD_WGAMMA1:
            value->rValue = model->BSIM4v0wgamma1;
            return(OK);
        case BSIM4v0_MOD_WGAMMA2:
            value->rValue = model->BSIM4v0wgamma2;
            return(OK);
        case BSIM4v0_MOD_WVBX:
            value->rValue = model->BSIM4v0wvbx;
            return(OK);
        case BSIM4v0_MOD_WVBM:
            value->rValue = model->BSIM4v0wvbm;
            return(OK);
        case BSIM4v0_MOD_WXT:
            value->rValue = model->BSIM4v0wxt;
            return(OK);
        case  BSIM4v0_MOD_WK1:
          value->rValue = model->BSIM4v0wk1;
            return(OK);
        case  BSIM4v0_MOD_WKT1:
          value->rValue = model->BSIM4v0wkt1;
            return(OK);
        case  BSIM4v0_MOD_WKT1L:
          value->rValue = model->BSIM4v0wkt1l;
            return(OK);
        case  BSIM4v0_MOD_WKT2 :
          value->rValue = model->BSIM4v0wkt2;
            return(OK);
        case  BSIM4v0_MOD_WK2 :
          value->rValue = model->BSIM4v0wk2;
            return(OK);
        case  BSIM4v0_MOD_WK3:
          value->rValue = model->BSIM4v0wk3;
            return(OK);
        case  BSIM4v0_MOD_WK3B:
          value->rValue = model->BSIM4v0wk3b;
            return(OK);
        case  BSIM4v0_MOD_WW0:
          value->rValue = model->BSIM4v0ww0;
            return(OK);
        case  BSIM4v0_MOD_WLPE0:
          value->rValue = model->BSIM4v0wlpe0;
            return(OK);
        case  BSIM4v0_MOD_WDVTP0:
          value->rValue = model->BSIM4v0wdvtp0;
            return(OK);
        case  BSIM4v0_MOD_WDVTP1:
          value->rValue = model->BSIM4v0wdvtp1;
            return(OK);
        case  BSIM4v0_MOD_WLPEB:
          value->rValue = model->BSIM4v0wlpeb;
            return(OK);
        case  BSIM4v0_MOD_WDVT0:                
          value->rValue = model->BSIM4v0wdvt0;
            return(OK);
        case  BSIM4v0_MOD_WDVT1 :             
          value->rValue = model->BSIM4v0wdvt1;
            return(OK);
        case  BSIM4v0_MOD_WDVT2 :             
          value->rValue = model->BSIM4v0wdvt2;
            return(OK);
        case  BSIM4v0_MOD_WDVT0W :                
          value->rValue = model->BSIM4v0wdvt0w;
            return(OK);
        case  BSIM4v0_MOD_WDVT1W :             
          value->rValue = model->BSIM4v0wdvt1w;
            return(OK);
        case  BSIM4v0_MOD_WDVT2W :             
          value->rValue = model->BSIM4v0wdvt2w;
            return(OK);
        case  BSIM4v0_MOD_WDROUT :           
          value->rValue = model->BSIM4v0wdrout;
            return(OK);
        case  BSIM4v0_MOD_WDSUB :           
          value->rValue = model->BSIM4v0wdsub;
            return(OK);
        case BSIM4v0_MOD_WVTH0:
            value->rValue = model->BSIM4v0wvth0; 
            return(OK);
        case BSIM4v0_MOD_WUA:
            value->rValue = model->BSIM4v0wua; 
            return(OK);
        case BSIM4v0_MOD_WUA1:
            value->rValue = model->BSIM4v0wua1; 
            return(OK);
        case BSIM4v0_MOD_WUB:
            value->rValue = model->BSIM4v0wub;  
            return(OK);
        case BSIM4v0_MOD_WUB1:
            value->rValue = model->BSIM4v0wub1;  
            return(OK);
        case BSIM4v0_MOD_WUC:
            value->rValue = model->BSIM4v0wuc; 
            return(OK);
        case BSIM4v0_MOD_WUC1:
            value->rValue = model->BSIM4v0wuc1; 
            return(OK);
        case BSIM4v0_MOD_WU0:
            value->rValue = model->BSIM4v0wu0;
            return(OK);
        case BSIM4v0_MOD_WUTE:
            value->rValue = model->BSIM4v0wute;
            return(OK);
        case BSIM4v0_MOD_WVOFF:
            value->rValue = model->BSIM4v0wvoff;
            return(OK);
        case BSIM4v0_MOD_WMINV:
            value->rValue = model->BSIM4v0wminv;
            return(OK);
        case BSIM4v0_MOD_WFPROUT:
            value->rValue = model->BSIM4v0wfprout;
            return(OK);
        case BSIM4v0_MOD_WPDITS:
            value->rValue = model->BSIM4v0wpdits;
            return(OK);
        case BSIM4v0_MOD_WPDITSD:
            value->rValue = model->BSIM4v0wpditsd;
            return(OK);
        case BSIM4v0_MOD_WDELTA:
            value->rValue = model->BSIM4v0wdelta;
            return(OK);
        case BSIM4v0_MOD_WRDSW:
            value->rValue = model->BSIM4v0wrdsw; 
            return(OK);             
        case BSIM4v0_MOD_WRDW:
            value->rValue = model->BSIM4v0wrdw;
            return(OK);
        case BSIM4v0_MOD_WRSW:
            value->rValue = model->BSIM4v0wrsw;
            return(OK);
        case BSIM4v0_MOD_WPRWB:
            value->rValue = model->BSIM4v0wprwb; 
            return(OK);             
        case BSIM4v0_MOD_WPRWG:
            value->rValue = model->BSIM4v0wprwg; 
            return(OK);             
        case BSIM4v0_MOD_WPRT:
            value->rValue = model->BSIM4v0wprt; 
            return(OK);              
        case BSIM4v0_MOD_WETA0:
            value->rValue = model->BSIM4v0weta0; 
            return(OK);               
        case BSIM4v0_MOD_WETAB:
            value->rValue = model->BSIM4v0wetab; 
            return(OK);               
        case BSIM4v0_MOD_WPCLM:
            value->rValue = model->BSIM4v0wpclm; 
            return(OK);               
        case BSIM4v0_MOD_WPDIBL1:
            value->rValue = model->BSIM4v0wpdibl1; 
            return(OK);               
        case BSIM4v0_MOD_WPDIBL2:
            value->rValue = model->BSIM4v0wpdibl2; 
            return(OK);               
        case BSIM4v0_MOD_WPDIBLB:
            value->rValue = model->BSIM4v0wpdiblb; 
            return(OK);               
        case BSIM4v0_MOD_WPSCBE1:
            value->rValue = model->BSIM4v0wpscbe1; 
            return(OK);               
        case BSIM4v0_MOD_WPSCBE2:
            value->rValue = model->BSIM4v0wpscbe2; 
            return(OK);               
        case BSIM4v0_MOD_WPVAG:
            value->rValue = model->BSIM4v0wpvag; 
            return(OK);               
        case BSIM4v0_MOD_WWR:
            value->rValue = model->BSIM4v0wwr;
            return(OK);
        case BSIM4v0_MOD_WDWG:
            value->rValue = model->BSIM4v0wdwg;
            return(OK);
        case BSIM4v0_MOD_WDWB:
            value->rValue = model->BSIM4v0wdwb;
            return(OK);
        case BSIM4v0_MOD_WB0:
            value->rValue = model->BSIM4v0wb0;
            return(OK);
        case BSIM4v0_MOD_WB1:
            value->rValue = model->BSIM4v0wb1;
            return(OK);
        case BSIM4v0_MOD_WALPHA0:
            value->rValue = model->BSIM4v0walpha0;
            return(OK);
        case BSIM4v0_MOD_WALPHA1:
            value->rValue = model->BSIM4v0walpha1;
            return(OK);
        case BSIM4v0_MOD_WBETA0:
            value->rValue = model->BSIM4v0wbeta0;
            return(OK);
        case BSIM4v0_MOD_WAGIDL:
            value->rValue = model->BSIM4v0wagidl;
            return(OK);
        case BSIM4v0_MOD_WBGIDL:
            value->rValue = model->BSIM4v0wbgidl;
            return(OK);
        case BSIM4v0_MOD_WCGIDL:
            value->rValue = model->BSIM4v0wcgidl;
            return(OK);
        case BSIM4v0_MOD_WEGIDL:
            value->rValue = model->BSIM4v0wegidl;
            return(OK);
        case BSIM4v0_MOD_WAIGC:
            value->rValue = model->BSIM4v0waigc;
            return(OK);
        case BSIM4v0_MOD_WBIGC:
            value->rValue = model->BSIM4v0wbigc;
            return(OK);
        case BSIM4v0_MOD_WCIGC:
            value->rValue = model->BSIM4v0wcigc;
            return(OK);
        case BSIM4v0_MOD_WAIGSD:
            value->rValue = model->BSIM4v0waigsd;
            return(OK);
        case BSIM4v0_MOD_WBIGSD:
            value->rValue = model->BSIM4v0wbigsd;
            return(OK);
        case BSIM4v0_MOD_WCIGSD:
            value->rValue = model->BSIM4v0wcigsd;
            return(OK);
        case BSIM4v0_MOD_WAIGBACC:
            value->rValue = model->BSIM4v0waigbacc;
            return(OK);
        case BSIM4v0_MOD_WBIGBACC:
            value->rValue = model->BSIM4v0wbigbacc;
            return(OK);
        case BSIM4v0_MOD_WCIGBACC:
            value->rValue = model->BSIM4v0wcigbacc;
            return(OK);
        case BSIM4v0_MOD_WAIGBINV:
            value->rValue = model->BSIM4v0waigbinv;
            return(OK);
        case BSIM4v0_MOD_WBIGBINV:
            value->rValue = model->BSIM4v0wbigbinv;
            return(OK);
        case BSIM4v0_MOD_WCIGBINV:
            value->rValue = model->BSIM4v0wcigbinv;
            return(OK);
        case BSIM4v0_MOD_WNIGC:
            value->rValue = model->BSIM4v0wnigc;
            return(OK);
        case BSIM4v0_MOD_WNIGBACC:
            value->rValue = model->BSIM4v0wnigbacc;
            return(OK);
        case BSIM4v0_MOD_WNIGBINV:
            value->rValue = model->BSIM4v0wnigbinv;
            return(OK);
        case BSIM4v0_MOD_WNTOX:
            value->rValue = model->BSIM4v0wntox;
            return(OK);
        case BSIM4v0_MOD_WEIGBINV:
            value->rValue = model->BSIM4v0weigbinv;
            return(OK);
        case BSIM4v0_MOD_WPIGCD:
            value->rValue = model->BSIM4v0wpigcd;
            return(OK);
        case BSIM4v0_MOD_WPOXEDGE:
            value->rValue = model->BSIM4v0wpoxedge;
            return(OK);
        case BSIM4v0_MOD_WPHIN:
            value->rValue = model->BSIM4v0wphin;
            return(OK);
        case BSIM4v0_MOD_WXRCRG1:
            value->rValue = model->BSIM4v0wxrcrg1;
            return(OK);
        case BSIM4v0_MOD_WXRCRG2:
            value->rValue = model->BSIM4v0wxrcrg2;
            return(OK);
        case BSIM4v0_MOD_WEU:
            value->rValue = model->BSIM4v0weu;
            return(OK);
        case BSIM4v0_MOD_WVFB:
            value->rValue = model->BSIM4v0wvfb;
            return(OK);

        case BSIM4v0_MOD_WCGSL:
            value->rValue = model->BSIM4v0wcgsl;
            return(OK);
        case BSIM4v0_MOD_WCGDL:
            value->rValue = model->BSIM4v0wcgdl;
            return(OK);
        case BSIM4v0_MOD_WCKAPPAS:
            value->rValue = model->BSIM4v0wckappas;
            return(OK);
        case BSIM4v0_MOD_WCKAPPAD:
            value->rValue = model->BSIM4v0wckappad;
            return(OK);
        case BSIM4v0_MOD_WCF:
            value->rValue = model->BSIM4v0wcf;
            return(OK);
        case BSIM4v0_MOD_WCLC:
            value->rValue = model->BSIM4v0wclc;
            return(OK);
        case BSIM4v0_MOD_WCLE:
            value->rValue = model->BSIM4v0wcle;
            return(OK);
        case BSIM4v0_MOD_WVFBCV:
            value->rValue = model->BSIM4v0wvfbcv;
            return(OK);
        case BSIM4v0_MOD_WACDE:
            value->rValue = model->BSIM4v0wacde;
            return(OK);
        case BSIM4v0_MOD_WMOIN:
            value->rValue = model->BSIM4v0wmoin;
            return(OK);
        case BSIM4v0_MOD_WNOFF:
            value->rValue = model->BSIM4v0wnoff;
            return(OK);
        case BSIM4v0_MOD_WVOFFCV:
            value->rValue = model->BSIM4v0wvoffcv;
            return(OK);

	/* Cross-term dependence */
        case  BSIM4v0_MOD_PCDSC :
          value->rValue = model->BSIM4v0pcdsc;
            return(OK);
        case  BSIM4v0_MOD_PCDSCB :
          value->rValue = model->BSIM4v0pcdscb;
            return(OK);
        case  BSIM4v0_MOD_PCDSCD :
          value->rValue = model->BSIM4v0pcdscd;
            return(OK);
         case  BSIM4v0_MOD_PCIT :
          value->rValue = model->BSIM4v0pcit;
            return(OK);
        case  BSIM4v0_MOD_PNFACTOR :
          value->rValue = model->BSIM4v0pnfactor;
            return(OK);
        case BSIM4v0_MOD_PXJ:
            value->rValue = model->BSIM4v0pxj;
            return(OK);
        case BSIM4v0_MOD_PVSAT:
            value->rValue = model->BSIM4v0pvsat;
            return(OK);
        case BSIM4v0_MOD_PAT:
            value->rValue = model->BSIM4v0pat;
            return(OK);
        case BSIM4v0_MOD_PA0:
            value->rValue = model->BSIM4v0pa0;
            return(OK);
        case BSIM4v0_MOD_PAGS:
            value->rValue = model->BSIM4v0pags;
            return(OK);
        case BSIM4v0_MOD_PA1:
            value->rValue = model->BSIM4v0pa1;
            return(OK);
        case BSIM4v0_MOD_PA2:
            value->rValue = model->BSIM4v0pa2;
            return(OK);
        case BSIM4v0_MOD_PKETA:
            value->rValue = model->BSIM4v0pketa;
            return(OK);   
        case BSIM4v0_MOD_PNSUB:
            value->rValue = model->BSIM4v0pnsub;
            return(OK);
        case BSIM4v0_MOD_PNDEP:
            value->rValue = model->BSIM4v0pndep;
            return(OK);
        case BSIM4v0_MOD_PNSD:
            value->rValue = model->BSIM4v0pnsd;
            return(OK);
        case BSIM4v0_MOD_PNGATE:
            value->rValue = model->BSIM4v0pngate;
            return(OK);
        case BSIM4v0_MOD_PGAMMA1:
            value->rValue = model->BSIM4v0pgamma1;
            return(OK);
        case BSIM4v0_MOD_PGAMMA2:
            value->rValue = model->BSIM4v0pgamma2;
            return(OK);
        case BSIM4v0_MOD_PVBX:
            value->rValue = model->BSIM4v0pvbx;
            return(OK);
        case BSIM4v0_MOD_PVBM:
            value->rValue = model->BSIM4v0pvbm;
            return(OK);
        case BSIM4v0_MOD_PXT:
            value->rValue = model->BSIM4v0pxt;
            return(OK);
        case  BSIM4v0_MOD_PK1:
          value->rValue = model->BSIM4v0pk1;
            return(OK);
        case  BSIM4v0_MOD_PKT1:
          value->rValue = model->BSIM4v0pkt1;
            return(OK);
        case  BSIM4v0_MOD_PKT1L:
          value->rValue = model->BSIM4v0pkt1l;
            return(OK);
        case  BSIM4v0_MOD_PKT2 :
          value->rValue = model->BSIM4v0pkt2;
            return(OK);
        case  BSIM4v0_MOD_PK2 :
          value->rValue = model->BSIM4v0pk2;
            return(OK);
        case  BSIM4v0_MOD_PK3:
          value->rValue = model->BSIM4v0pk3;
            return(OK);
        case  BSIM4v0_MOD_PK3B:
          value->rValue = model->BSIM4v0pk3b;
            return(OK);
        case  BSIM4v0_MOD_PW0:
          value->rValue = model->BSIM4v0pw0;
            return(OK);
        case  BSIM4v0_MOD_PLPE0:
          value->rValue = model->BSIM4v0plpe0;
            return(OK);
        case  BSIM4v0_MOD_PLPEB:
          value->rValue = model->BSIM4v0plpeb;
            return(OK);
        case  BSIM4v0_MOD_PDVTP0:
          value->rValue = model->BSIM4v0pdvtp0;
            return(OK);
        case  BSIM4v0_MOD_PDVTP1:
          value->rValue = model->BSIM4v0pdvtp1;
            return(OK);
        case  BSIM4v0_MOD_PDVT0 :                
          value->rValue = model->BSIM4v0pdvt0;
            return(OK);
        case  BSIM4v0_MOD_PDVT1 :             
          value->rValue = model->BSIM4v0pdvt1;
            return(OK);
        case  BSIM4v0_MOD_PDVT2 :             
          value->rValue = model->BSIM4v0pdvt2;
            return(OK);
        case  BSIM4v0_MOD_PDVT0W :                
          value->rValue = model->BSIM4v0pdvt0w;
            return(OK);
        case  BSIM4v0_MOD_PDVT1W :             
          value->rValue = model->BSIM4v0pdvt1w;
            return(OK);
        case  BSIM4v0_MOD_PDVT2W :             
          value->rValue = model->BSIM4v0pdvt2w;
            return(OK);
        case  BSIM4v0_MOD_PDROUT :           
          value->rValue = model->BSIM4v0pdrout;
            return(OK);
        case  BSIM4v0_MOD_PDSUB :           
          value->rValue = model->BSIM4v0pdsub;
            return(OK);
        case BSIM4v0_MOD_PVTH0:
            value->rValue = model->BSIM4v0pvth0; 
            return(OK);
        case BSIM4v0_MOD_PUA:
            value->rValue = model->BSIM4v0pua; 
            return(OK);
        case BSIM4v0_MOD_PUA1:
            value->rValue = model->BSIM4v0pua1; 
            return(OK);
        case BSIM4v0_MOD_PUB:
            value->rValue = model->BSIM4v0pub;  
            return(OK);
        case BSIM4v0_MOD_PUB1:
            value->rValue = model->BSIM4v0pub1;  
            return(OK);
        case BSIM4v0_MOD_PUC:
            value->rValue = model->BSIM4v0puc; 
            return(OK);
        case BSIM4v0_MOD_PUC1:
            value->rValue = model->BSIM4v0puc1; 
            return(OK);
        case BSIM4v0_MOD_PU0:
            value->rValue = model->BSIM4v0pu0;
            return(OK);
        case BSIM4v0_MOD_PUTE:
            value->rValue = model->BSIM4v0pute;
            return(OK);
        case BSIM4v0_MOD_PVOFF:
            value->rValue = model->BSIM4v0pvoff;
            return(OK);
        case BSIM4v0_MOD_PMINV:
            value->rValue = model->BSIM4v0pminv;
            return(OK);
        case BSIM4v0_MOD_PFPROUT:
            value->rValue = model->BSIM4v0pfprout;
            return(OK);
        case BSIM4v0_MOD_PPDITS:
            value->rValue = model->BSIM4v0ppdits;
            return(OK);
        case BSIM4v0_MOD_PPDITSD:
            value->rValue = model->BSIM4v0ppditsd;
            return(OK);
        case BSIM4v0_MOD_PDELTA:
            value->rValue = model->BSIM4v0pdelta;
            return(OK);
        case BSIM4v0_MOD_PRDSW:
            value->rValue = model->BSIM4v0prdsw; 
            return(OK);             
        case BSIM4v0_MOD_PRDW:
            value->rValue = model->BSIM4v0prdw;
            return(OK);
        case BSIM4v0_MOD_PRSW:
            value->rValue = model->BSIM4v0prsw;
            return(OK);
        case BSIM4v0_MOD_PPRWB:
            value->rValue = model->BSIM4v0pprwb; 
            return(OK);             
        case BSIM4v0_MOD_PPRWG:
            value->rValue = model->BSIM4v0pprwg; 
            return(OK);             
        case BSIM4v0_MOD_PPRT:
            value->rValue = model->BSIM4v0pprt; 
            return(OK);              
        case BSIM4v0_MOD_PETA0:
            value->rValue = model->BSIM4v0peta0; 
            return(OK);               
        case BSIM4v0_MOD_PETAB:
            value->rValue = model->BSIM4v0petab; 
            return(OK);               
        case BSIM4v0_MOD_PPCLM:
            value->rValue = model->BSIM4v0ppclm; 
            return(OK);               
        case BSIM4v0_MOD_PPDIBL1:
            value->rValue = model->BSIM4v0ppdibl1; 
            return(OK);               
        case BSIM4v0_MOD_PPDIBL2:
            value->rValue = model->BSIM4v0ppdibl2; 
            return(OK);               
        case BSIM4v0_MOD_PPDIBLB:
            value->rValue = model->BSIM4v0ppdiblb; 
            return(OK);               
        case BSIM4v0_MOD_PPSCBE1:
            value->rValue = model->BSIM4v0ppscbe1; 
            return(OK);               
        case BSIM4v0_MOD_PPSCBE2:
            value->rValue = model->BSIM4v0ppscbe2; 
            return(OK);               
        case BSIM4v0_MOD_PPVAG:
            value->rValue = model->BSIM4v0ppvag; 
            return(OK);               
        case BSIM4v0_MOD_PWR:
            value->rValue = model->BSIM4v0pwr;
            return(OK);
        case BSIM4v0_MOD_PDWG:
            value->rValue = model->BSIM4v0pdwg;
            return(OK);
        case BSIM4v0_MOD_PDWB:
            value->rValue = model->BSIM4v0pdwb;
            return(OK);
        case BSIM4v0_MOD_PB0:
            value->rValue = model->BSIM4v0pb0;
            return(OK);
        case BSIM4v0_MOD_PB1:
            value->rValue = model->BSIM4v0pb1;
            return(OK);
        case BSIM4v0_MOD_PALPHA0:
            value->rValue = model->BSIM4v0palpha0;
            return(OK);
        case BSIM4v0_MOD_PALPHA1:
            value->rValue = model->BSIM4v0palpha1;
            return(OK);
        case BSIM4v0_MOD_PBETA0:
            value->rValue = model->BSIM4v0pbeta0;
            return(OK);
        case BSIM4v0_MOD_PAGIDL:
            value->rValue = model->BSIM4v0pagidl;
            return(OK);
        case BSIM4v0_MOD_PBGIDL:
            value->rValue = model->BSIM4v0pbgidl;
            return(OK);
        case BSIM4v0_MOD_PCGIDL:
            value->rValue = model->BSIM4v0pcgidl;
            return(OK);
        case BSIM4v0_MOD_PEGIDL:
            value->rValue = model->BSIM4v0pegidl;
            return(OK);
        case BSIM4v0_MOD_PAIGC:
            value->rValue = model->BSIM4v0paigc;
            return(OK);
        case BSIM4v0_MOD_PBIGC:
            value->rValue = model->BSIM4v0pbigc;
            return(OK);
        case BSIM4v0_MOD_PCIGC:
            value->rValue = model->BSIM4v0pcigc;
            return(OK);
        case BSIM4v0_MOD_PAIGSD:
            value->rValue = model->BSIM4v0paigsd;
            return(OK);
        case BSIM4v0_MOD_PBIGSD:
            value->rValue = model->BSIM4v0pbigsd;
            return(OK);
        case BSIM4v0_MOD_PCIGSD:
            value->rValue = model->BSIM4v0pcigsd;
            return(OK);
        case BSIM4v0_MOD_PAIGBACC:
            value->rValue = model->BSIM4v0paigbacc;
            return(OK);
        case BSIM4v0_MOD_PBIGBACC:
            value->rValue = model->BSIM4v0pbigbacc;
            return(OK);
        case BSIM4v0_MOD_PCIGBACC:
            value->rValue = model->BSIM4v0pcigbacc;
            return(OK);
        case BSIM4v0_MOD_PAIGBINV:
            value->rValue = model->BSIM4v0paigbinv;
            return(OK);
        case BSIM4v0_MOD_PBIGBINV:
            value->rValue = model->BSIM4v0pbigbinv;
            return(OK);
        case BSIM4v0_MOD_PCIGBINV:
            value->rValue = model->BSIM4v0pcigbinv;
            return(OK);
        case BSIM4v0_MOD_PNIGC:
            value->rValue = model->BSIM4v0pnigc;
            return(OK);
        case BSIM4v0_MOD_PNIGBACC:
            value->rValue = model->BSIM4v0pnigbacc;
            return(OK);
        case BSIM4v0_MOD_PNIGBINV:
            value->rValue = model->BSIM4v0pnigbinv;
            return(OK);
        case BSIM4v0_MOD_PNTOX:
            value->rValue = model->BSIM4v0pntox;
            return(OK);
        case BSIM4v0_MOD_PEIGBINV:
            value->rValue = model->BSIM4v0peigbinv;
            return(OK);
        case BSIM4v0_MOD_PPIGCD:
            value->rValue = model->BSIM4v0ppigcd;
            return(OK);
        case BSIM4v0_MOD_PPOXEDGE:
            value->rValue = model->BSIM4v0ppoxedge;
            return(OK);
        case BSIM4v0_MOD_PPHIN:
            value->rValue = model->BSIM4v0pphin;
            return(OK);
        case BSIM4v0_MOD_PXRCRG1:
            value->rValue = model->BSIM4v0pxrcrg1;
            return(OK);
        case BSIM4v0_MOD_PXRCRG2:
            value->rValue = model->BSIM4v0pxrcrg2;
            return(OK);
        case BSIM4v0_MOD_PEU:
            value->rValue = model->BSIM4v0peu;
            return(OK);
        case BSIM4v0_MOD_PVFB:
            value->rValue = model->BSIM4v0pvfb;
            return(OK);

        case BSIM4v0_MOD_PCGSL:
            value->rValue = model->BSIM4v0pcgsl;
            return(OK);
        case BSIM4v0_MOD_PCGDL:
            value->rValue = model->BSIM4v0pcgdl;
            return(OK);
        case BSIM4v0_MOD_PCKAPPAS:
            value->rValue = model->BSIM4v0pckappas;
            return(OK);
        case BSIM4v0_MOD_PCKAPPAD:
            value->rValue = model->BSIM4v0pckappad;
            return(OK);
        case BSIM4v0_MOD_PCF:
            value->rValue = model->BSIM4v0pcf;
            return(OK);
        case BSIM4v0_MOD_PCLC:
            value->rValue = model->BSIM4v0pclc;
            return(OK);
        case BSIM4v0_MOD_PCLE:
            value->rValue = model->BSIM4v0pcle;
            return(OK);
        case BSIM4v0_MOD_PVFBCV:
            value->rValue = model->BSIM4v0pvfbcv;
            return(OK);
        case BSIM4v0_MOD_PACDE:
            value->rValue = model->BSIM4v0pacde;
            return(OK);
        case BSIM4v0_MOD_PMOIN:
            value->rValue = model->BSIM4v0pmoin;
            return(OK);
        case BSIM4v0_MOD_PNOFF:
            value->rValue = model->BSIM4v0pnoff;
            return(OK);
        case BSIM4v0_MOD_PVOFFCV:
            value->rValue = model->BSIM4v0pvoffcv;
            return(OK);

        case  BSIM4v0_MOD_TNOM :
          value->rValue = model->BSIM4v0tnom;
            return(OK);
        case BSIM4v0_MOD_CGSO:
            value->rValue = model->BSIM4v0cgso; 
            return(OK);
        case BSIM4v0_MOD_CGDO:
            value->rValue = model->BSIM4v0cgdo; 
            return(OK);
        case BSIM4v0_MOD_CGBO:
            value->rValue = model->BSIM4v0cgbo; 
            return(OK);
        case BSIM4v0_MOD_XPART:
            value->rValue = model->BSIM4v0xpart; 
            return(OK);
        case BSIM4v0_MOD_RSH:
            value->rValue = model->BSIM4v0sheetResistance; 
            return(OK);
        case BSIM4v0_MOD_JSS:
            value->rValue = model->BSIM4v0SjctSatCurDensity; 
            return(OK);
        case BSIM4v0_MOD_JSWS:
            value->rValue = model->BSIM4v0SjctSidewallSatCurDensity; 
            return(OK);
        case BSIM4v0_MOD_JSWGS:
            value->rValue = model->BSIM4v0SjctGateSidewallSatCurDensity;
            return(OK);
        case BSIM4v0_MOD_PBS:
            value->rValue = model->BSIM4v0SbulkJctPotential; 
            return(OK);
        case BSIM4v0_MOD_MJS:
            value->rValue = model->BSIM4v0SbulkJctBotGradingCoeff; 
            return(OK);
        case BSIM4v0_MOD_PBSWS:
            value->rValue = model->BSIM4v0SsidewallJctPotential; 
            return(OK);
        case BSIM4v0_MOD_MJSWS:
            value->rValue = model->BSIM4v0SbulkJctSideGradingCoeff; 
            return(OK);
        case BSIM4v0_MOD_CJS:
            value->rValue = model->BSIM4v0SunitAreaJctCap; 
            return(OK);
        case BSIM4v0_MOD_CJSWS:
            value->rValue = model->BSIM4v0SunitLengthSidewallJctCap; 
            return(OK);
        case BSIM4v0_MOD_PBSWGS:
            value->rValue = model->BSIM4v0SGatesidewallJctPotential; 
            return(OK);
        case BSIM4v0_MOD_MJSWGS:
            value->rValue = model->BSIM4v0SbulkJctGateSideGradingCoeff; 
            return(OK);
        case BSIM4v0_MOD_CJSWGS:
            value->rValue = model->BSIM4v0SunitLengthGateSidewallJctCap; 
            return(OK);
        case BSIM4v0_MOD_NJS:
            value->rValue = model->BSIM4v0SjctEmissionCoeff; 
            return(OK);
        case BSIM4v0_MOD_XTIS:
            value->rValue = model->BSIM4v0SjctTempExponent; 
            return(OK);
        case BSIM4v0_MOD_JSD:
            value->rValue = model->BSIM4v0DjctSatCurDensity;
            return(OK);
        case BSIM4v0_MOD_JSWD:
            value->rValue = model->BSIM4v0DjctSidewallSatCurDensity;
            return(OK);
        case BSIM4v0_MOD_JSWGD:
            value->rValue = model->BSIM4v0DjctGateSidewallSatCurDensity;
            return(OK);
        case BSIM4v0_MOD_PBD:
            value->rValue = model->BSIM4v0DbulkJctPotential;
            return(OK);
        case BSIM4v0_MOD_MJD:
            value->rValue = model->BSIM4v0DbulkJctBotGradingCoeff;
            return(OK);
        case BSIM4v0_MOD_PBSWD:
            value->rValue = model->BSIM4v0DsidewallJctPotential;
            return(OK);
        case BSIM4v0_MOD_MJSWD:
            value->rValue = model->BSIM4v0DbulkJctSideGradingCoeff;
            return(OK);
        case BSIM4v0_MOD_CJD:
            value->rValue = model->BSIM4v0DunitAreaJctCap;
            return(OK);
        case BSIM4v0_MOD_CJSWD:
            value->rValue = model->BSIM4v0DunitLengthSidewallJctCap;
            return(OK);
        case BSIM4v0_MOD_PBSWGD:
            value->rValue = model->BSIM4v0DGatesidewallJctPotential;
            return(OK);
        case BSIM4v0_MOD_MJSWGD:
            value->rValue = model->BSIM4v0DbulkJctGateSideGradingCoeff;
            return(OK);
        case BSIM4v0_MOD_CJSWGD:
            value->rValue = model->BSIM4v0DunitLengthGateSidewallJctCap;
            return(OK);
        case BSIM4v0_MOD_NJD:
            value->rValue = model->BSIM4v0DjctEmissionCoeff;
            return(OK);
        case BSIM4v0_MOD_XTID:
            value->rValue = model->BSIM4v0DjctTempExponent;
            return(OK);
        case BSIM4v0_MOD_LINT:
            value->rValue = model->BSIM4v0Lint; 
            return(OK);
        case BSIM4v0_MOD_LL:
            value->rValue = model->BSIM4v0Ll;
            return(OK);
        case BSIM4v0_MOD_LLC:
            value->rValue = model->BSIM4v0Llc;
            return(OK);
        case BSIM4v0_MOD_LLN:
            value->rValue = model->BSIM4v0Lln;
            return(OK);
        case BSIM4v0_MOD_LW:
            value->rValue = model->BSIM4v0Lw;
            return(OK);
        case BSIM4v0_MOD_LWC:
            value->rValue = model->BSIM4v0Lwc;
            return(OK);
        case BSIM4v0_MOD_LWN:
            value->rValue = model->BSIM4v0Lwn;
            return(OK);
        case BSIM4v0_MOD_LWL:
            value->rValue = model->BSIM4v0Lwl;
            return(OK);
        case BSIM4v0_MOD_LWLC:
            value->rValue = model->BSIM4v0Lwlc;
            return(OK);
        case BSIM4v0_MOD_LMIN:
            value->rValue = model->BSIM4v0Lmin;
            return(OK);
        case BSIM4v0_MOD_LMAX:
            value->rValue = model->BSIM4v0Lmax;
            return(OK);
        case BSIM4v0_MOD_WINT:
            value->rValue = model->BSIM4v0Wint;
            return(OK);
        case BSIM4v0_MOD_WL:
            value->rValue = model->BSIM4v0Wl;
            return(OK);
        case BSIM4v0_MOD_WLC:
            value->rValue = model->BSIM4v0Wlc;
            return(OK);
        case BSIM4v0_MOD_WLN:
            value->rValue = model->BSIM4v0Wln;
            return(OK);
        case BSIM4v0_MOD_WW:
            value->rValue = model->BSIM4v0Ww;
            return(OK);
        case BSIM4v0_MOD_WWC:
            value->rValue = model->BSIM4v0Wwc;
            return(OK);
        case BSIM4v0_MOD_WWN:
            value->rValue = model->BSIM4v0Wwn;
            return(OK);
        case BSIM4v0_MOD_WWL:
            value->rValue = model->BSIM4v0Wwl;
            return(OK);
        case BSIM4v0_MOD_WWLC:
            value->rValue = model->BSIM4v0Wwlc;
            return(OK);
        case BSIM4v0_MOD_WMIN:
            value->rValue = model->BSIM4v0Wmin;
            return(OK);
        case BSIM4v0_MOD_WMAX:
            value->rValue = model->BSIM4v0Wmax;
            return(OK);
        case BSIM4v0_MOD_NOIA:
            value->rValue = model->BSIM4v0oxideTrapDensityA;
            return(OK);
        case BSIM4v0_MOD_NOIB:
            value->rValue = model->BSIM4v0oxideTrapDensityB;
            return(OK);
        case BSIM4v0_MOD_NOIC:
            value->rValue = model->BSIM4v0oxideTrapDensityC;
            return(OK);
        case BSIM4v0_MOD_EM:
            value->rValue = model->BSIM4v0em;
            return(OK);
        case BSIM4v0_MOD_EF:
            value->rValue = model->BSIM4v0ef;
            return(OK);
        case BSIM4v0_MOD_AF:
            value->rValue = model->BSIM4v0af;
            return(OK);
        case BSIM4v0_MOD_KF:
            value->rValue = model->BSIM4v0kf;
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}



