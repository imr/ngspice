/**** BSIM4.3.0 Released by Xuemei(Jane) Xi 05/09/2003 ****/

/**********
 * Copyright 2003 Regents of the University of California. All rights reserved.
 * File: b4v3mask.c of BSIM4.3.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 * Modified by Xuemei Xi, 11/15/2002.
 * Modified by Xuemei Xi, 05/09/2003.
 **********/


#include "ngspice.h"
#include <stdio.h>
#include "ifsim.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "bsim4v3def.h"
#include "sperror.h"

int
BSIM4v3mAsk(
CKTcircuit *ckt,
GENmodel *inst,
int which,
IFvalue *value)
{
    BSIM4v3model *model = (BSIM4v3model *)inst;
    switch(which) 
    {   case BSIM4v3_MOD_MOBMOD :
            value->iValue = model->BSIM4v3mobMod; 
            return(OK);
        case BSIM4v3_MOD_PARAMCHK :
            value->iValue = model->BSIM4v3paramChk; 
            return(OK);
        case BSIM4v3_MOD_BINUNIT :
            value->iValue = model->BSIM4v3binUnit; 
            return(OK);
        case BSIM4v3_MOD_CAPMOD :
            value->iValue = model->BSIM4v3capMod; 
            return(OK);
        case BSIM4v3_MOD_DIOMOD :
            value->iValue = model->BSIM4v3dioMod;
            return(OK);
        case BSIM4v3_MOD_TRNQSMOD :
            value->iValue = model->BSIM4v3trnqsMod;
            return(OK);
        case BSIM4v3_MOD_ACNQSMOD :
            value->iValue = model->BSIM4v3acnqsMod;
            return(OK);
        case BSIM4v3_MOD_FNOIMOD :
            value->iValue = model->BSIM4v3fnoiMod; 
            return(OK);
        case BSIM4v3_MOD_TNOIMOD :
            value->iValue = model->BSIM4v3tnoiMod;
            return(OK);
        case BSIM4v3_MOD_RDSMOD :
            value->iValue = model->BSIM4v3rdsMod;
            return(OK);
        case BSIM4v3_MOD_RBODYMOD :
            value->iValue = model->BSIM4v3rbodyMod;
            return(OK);
        case BSIM4v3_MOD_RGATEMOD :
            value->iValue = model->BSIM4v3rgateMod;
            return(OK);
        case BSIM4v3_MOD_PERMOD :
            value->iValue = model->BSIM4v3perMod;
            return(OK);
        case BSIM4v3_MOD_GEOMOD :
            value->iValue = model->BSIM4v3geoMod;
            return(OK);
        case BSIM4v3_MOD_IGCMOD :
            value->iValue = model->BSIM4v3igcMod;
            return(OK);
        case BSIM4v3_MOD_IGBMOD :
            value->iValue = model->BSIM4v3igbMod;
            return(OK);
        case  BSIM4v3_MOD_TEMPMOD :
            value->iValue = model->BSIM4v3tempMod;
            return(OK);
        case  BSIM4v3_MOD_VERSION :
          value->sValue = model->BSIM4v3version;
            return(OK);
        case  BSIM4v3_MOD_TOXREF :
          value->rValue = model->BSIM4v3toxref;
          return(OK);
        case  BSIM4v3_MOD_TOXE :
          value->rValue = model->BSIM4v3toxe;
            return(OK);
        case  BSIM4v3_MOD_TOXP :
          value->rValue = model->BSIM4v3toxp;
            return(OK);
        case  BSIM4v3_MOD_TOXM :
          value->rValue = model->BSIM4v3toxm;
            return(OK);
        case  BSIM4v3_MOD_DTOX :
          value->rValue = model->BSIM4v3dtox;
            return(OK);
        case  BSIM4v3_MOD_EPSROX :
          value->rValue = model->BSIM4v3epsrox;
            return(OK);
        case  BSIM4v3_MOD_CDSC :
          value->rValue = model->BSIM4v3cdsc;
            return(OK);
        case  BSIM4v3_MOD_CDSCB :
          value->rValue = model->BSIM4v3cdscb;
            return(OK);

        case  BSIM4v3_MOD_CDSCD :
          value->rValue = model->BSIM4v3cdscd;
            return(OK);

        case  BSIM4v3_MOD_CIT :
          value->rValue = model->BSIM4v3cit;
            return(OK);
        case  BSIM4v3_MOD_NFACTOR :
          value->rValue = model->BSIM4v3nfactor;
            return(OK);
        case BSIM4v3_MOD_XJ:
            value->rValue = model->BSIM4v3xj;
            return(OK);
        case BSIM4v3_MOD_VSAT:
            value->rValue = model->BSIM4v3vsat;
            return(OK);
        case BSIM4v3_MOD_VTL:
            value->rValue = model->BSIM4v3vtl;
            return(OK);
        case BSIM4v3_MOD_XN:
            value->rValue = model->BSIM4v3xn;
            return(OK);
        case BSIM4v3_MOD_LC:
            value->rValue = model->BSIM4v3lc;
            return(OK);
        case BSIM4v3_MOD_LAMBDA:
            value->rValue = model->BSIM4v3lambda;
            return(OK);
        case BSIM4v3_MOD_AT:
            value->rValue = model->BSIM4v3at;
            return(OK);
        case BSIM4v3_MOD_A0:
            value->rValue = model->BSIM4v3a0;
            return(OK);

        case BSIM4v3_MOD_AGS:
            value->rValue = model->BSIM4v3ags;
            return(OK);

        case BSIM4v3_MOD_A1:
            value->rValue = model->BSIM4v3a1;
            return(OK);
        case BSIM4v3_MOD_A2:
            value->rValue = model->BSIM4v3a2;
            return(OK);
        case BSIM4v3_MOD_KETA:
            value->rValue = model->BSIM4v3keta;
            return(OK);   
        case BSIM4v3_MOD_NSUB:
            value->rValue = model->BSIM4v3nsub;
            return(OK);
        case BSIM4v3_MOD_NDEP:
            value->rValue = model->BSIM4v3ndep;
            return(OK);
        case BSIM4v3_MOD_NSD:
            value->rValue = model->BSIM4v3nsd;
            return(OK);
        case BSIM4v3_MOD_NGATE:
            value->rValue = model->BSIM4v3ngate;
            return(OK);
        case BSIM4v3_MOD_GAMMA1:
            value->rValue = model->BSIM4v3gamma1;
            return(OK);
        case BSIM4v3_MOD_GAMMA2:
            value->rValue = model->BSIM4v3gamma2;
            return(OK);
        case BSIM4v3_MOD_VBX:
            value->rValue = model->BSIM4v3vbx;
            return(OK);
        case BSIM4v3_MOD_VBM:
            value->rValue = model->BSIM4v3vbm;
            return(OK);
        case BSIM4v3_MOD_XT:
            value->rValue = model->BSIM4v3xt;
            return(OK);
        case  BSIM4v3_MOD_K1:
          value->rValue = model->BSIM4v3k1;
            return(OK);
        case  BSIM4v3_MOD_KT1:
          value->rValue = model->BSIM4v3kt1;
            return(OK);
        case  BSIM4v3_MOD_KT1L:
          value->rValue = model->BSIM4v3kt1l;
            return(OK);
        case  BSIM4v3_MOD_KT2 :
          value->rValue = model->BSIM4v3kt2;
            return(OK);
        case  BSIM4v3_MOD_K2 :
          value->rValue = model->BSIM4v3k2;
            return(OK);
        case  BSIM4v3_MOD_K3:
          value->rValue = model->BSIM4v3k3;
            return(OK);
        case  BSIM4v3_MOD_K3B:
          value->rValue = model->BSIM4v3k3b;
            return(OK);
        case  BSIM4v3_MOD_W0:
          value->rValue = model->BSIM4v3w0;
            return(OK);
        case  BSIM4v3_MOD_LPE0:
          value->rValue = model->BSIM4v3lpe0;
            return(OK);
        case  BSIM4v3_MOD_LPEB:
          value->rValue = model->BSIM4v3lpeb;
            return(OK);
        case  BSIM4v3_MOD_DVTP0:
          value->rValue = model->BSIM4v3dvtp0;
            return(OK);
        case  BSIM4v3_MOD_DVTP1:
          value->rValue = model->BSIM4v3dvtp1;
            return(OK);
        case  BSIM4v3_MOD_DVT0 :                
          value->rValue = model->BSIM4v3dvt0;
            return(OK);
        case  BSIM4v3_MOD_DVT1 :             
          value->rValue = model->BSIM4v3dvt1;
            return(OK);
        case  BSIM4v3_MOD_DVT2 :             
          value->rValue = model->BSIM4v3dvt2;
            return(OK);
        case  BSIM4v3_MOD_DVT0W :                
          value->rValue = model->BSIM4v3dvt0w;
            return(OK);
        case  BSIM4v3_MOD_DVT1W :             
          value->rValue = model->BSIM4v3dvt1w;
            return(OK);
        case  BSIM4v3_MOD_DVT2W :             
          value->rValue = model->BSIM4v3dvt2w;
            return(OK);
        case  BSIM4v3_MOD_DROUT :           
          value->rValue = model->BSIM4v3drout;
            return(OK);
        case  BSIM4v3_MOD_DSUB :           
          value->rValue = model->BSIM4v3dsub;
            return(OK);
        case BSIM4v3_MOD_VTH0:
            value->rValue = model->BSIM4v3vth0; 
            return(OK);
        case BSIM4v3_MOD_EU:
            value->rValue = model->BSIM4v3eu;
            return(OK);
        case BSIM4v3_MOD_UA:
            value->rValue = model->BSIM4v3ua; 
            return(OK);
        case BSIM4v3_MOD_UA1:
            value->rValue = model->BSIM4v3ua1; 
            return(OK);
        case BSIM4v3_MOD_UB:
            value->rValue = model->BSIM4v3ub;  
            return(OK);
        case BSIM4v3_MOD_UB1:
            value->rValue = model->BSIM4v3ub1;  
            return(OK);
        case BSIM4v3_MOD_UC:
            value->rValue = model->BSIM4v3uc; 
            return(OK);
        case BSIM4v3_MOD_UC1:
            value->rValue = model->BSIM4v3uc1; 
            return(OK);
        case BSIM4v3_MOD_U0:
            value->rValue = model->BSIM4v3u0;
            return(OK);
        case BSIM4v3_MOD_UTE:
            value->rValue = model->BSIM4v3ute;
            return(OK);
        case BSIM4v3_MOD_VOFF:
            value->rValue = model->BSIM4v3voff;
            return(OK);
        case BSIM4v3_MOD_VOFFL:
            value->rValue = model->BSIM4v3voffl;
            return(OK);
        case BSIM4v3_MOD_MINV:
            value->rValue = model->BSIM4v3minv;
            return(OK);
        case BSIM4v3_MOD_FPROUT:
            value->rValue = model->BSIM4v3fprout;
            return(OK);
        case BSIM4v3_MOD_PDITS:
            value->rValue = model->BSIM4v3pdits;
            return(OK);
        case BSIM4v3_MOD_PDITSD:
            value->rValue = model->BSIM4v3pditsd;
            return(OK);
        case BSIM4v3_MOD_PDITSL:
            value->rValue = model->BSIM4v3pditsl;
            return(OK);
        case BSIM4v3_MOD_DELTA:
            value->rValue = model->BSIM4v3delta;
            return(OK);
        case BSIM4v3_MOD_RDSW:
            value->rValue = model->BSIM4v3rdsw; 
            return(OK);
        case BSIM4v3_MOD_RDSWMIN:
            value->rValue = model->BSIM4v3rdswmin;
            return(OK);
        case BSIM4v3_MOD_RDWMIN:
            value->rValue = model->BSIM4v3rdwmin;
            return(OK);
        case BSIM4v3_MOD_RSWMIN:
            value->rValue = model->BSIM4v3rswmin;
            return(OK);
        case BSIM4v3_MOD_RDW:
            value->rValue = model->BSIM4v3rdw;
            return(OK);
        case BSIM4v3_MOD_RSW:
            value->rValue = model->BSIM4v3rsw;
            return(OK);
        case BSIM4v3_MOD_PRWG:
            value->rValue = model->BSIM4v3prwg; 
            return(OK);             
        case BSIM4v3_MOD_PRWB:
            value->rValue = model->BSIM4v3prwb; 
            return(OK);             
        case BSIM4v3_MOD_PRT:
            value->rValue = model->BSIM4v3prt; 
            return(OK);              
        case BSIM4v3_MOD_ETA0:
            value->rValue = model->BSIM4v3eta0; 
            return(OK);               
        case BSIM4v3_MOD_ETAB:
            value->rValue = model->BSIM4v3etab; 
            return(OK);               
        case BSIM4v3_MOD_PCLM:
            value->rValue = model->BSIM4v3pclm; 
            return(OK);               
        case BSIM4v3_MOD_PDIBL1:
            value->rValue = model->BSIM4v3pdibl1; 
            return(OK);               
        case BSIM4v3_MOD_PDIBL2:
            value->rValue = model->BSIM4v3pdibl2; 
            return(OK);               
        case BSIM4v3_MOD_PDIBLB:
            value->rValue = model->BSIM4v3pdiblb; 
            return(OK);               
        case BSIM4v3_MOD_PSCBE1:
            value->rValue = model->BSIM4v3pscbe1; 
            return(OK);               
        case BSIM4v3_MOD_PSCBE2:
            value->rValue = model->BSIM4v3pscbe2; 
            return(OK);               
        case BSIM4v3_MOD_PVAG:
            value->rValue = model->BSIM4v3pvag; 
            return(OK);               
        case BSIM4v3_MOD_WR:
            value->rValue = model->BSIM4v3wr;
            return(OK);
        case BSIM4v3_MOD_DWG:
            value->rValue = model->BSIM4v3dwg;
            return(OK);
        case BSIM4v3_MOD_DWB:
            value->rValue = model->BSIM4v3dwb;
            return(OK);
        case BSIM4v3_MOD_B0:
            value->rValue = model->BSIM4v3b0;
            return(OK);
        case BSIM4v3_MOD_B1:
            value->rValue = model->BSIM4v3b1;
            return(OK);
        case BSIM4v3_MOD_ALPHA0:
            value->rValue = model->BSIM4v3alpha0;
            return(OK);
        case BSIM4v3_MOD_ALPHA1:
            value->rValue = model->BSIM4v3alpha1;
            return(OK);
        case BSIM4v3_MOD_BETA0:
            value->rValue = model->BSIM4v3beta0;
            return(OK);
        case BSIM4v3_MOD_AGIDL:
            value->rValue = model->BSIM4v3agidl;
            return(OK);
        case BSIM4v3_MOD_BGIDL:
            value->rValue = model->BSIM4v3bgidl;
            return(OK);
        case BSIM4v3_MOD_CGIDL:
            value->rValue = model->BSIM4v3cgidl;
            return(OK);
        case BSIM4v3_MOD_EGIDL:
            value->rValue = model->BSIM4v3egidl;
            return(OK);
        case BSIM4v3_MOD_AIGC:
            value->rValue = model->BSIM4v3aigc;
            return(OK);
        case BSIM4v3_MOD_BIGC:
            value->rValue = model->BSIM4v3bigc;
            return(OK);
        case BSIM4v3_MOD_CIGC:
            value->rValue = model->BSIM4v3cigc;
            return(OK);
        case BSIM4v3_MOD_AIGSD:
            value->rValue = model->BSIM4v3aigsd;
            return(OK);
        case BSIM4v3_MOD_BIGSD:
            value->rValue = model->BSIM4v3bigsd;
            return(OK);
        case BSIM4v3_MOD_CIGSD:
            value->rValue = model->BSIM4v3cigsd;
            return(OK);
        case BSIM4v3_MOD_AIGBACC:
            value->rValue = model->BSIM4v3aigbacc;
            return(OK);
        case BSIM4v3_MOD_BIGBACC:
            value->rValue = model->BSIM4v3bigbacc;
            return(OK);
        case BSIM4v3_MOD_CIGBACC:
            value->rValue = model->BSIM4v3cigbacc;
            return(OK);
        case BSIM4v3_MOD_AIGBINV:
            value->rValue = model->BSIM4v3aigbinv;
            return(OK);
        case BSIM4v3_MOD_BIGBINV:
            value->rValue = model->BSIM4v3bigbinv;
            return(OK);
        case BSIM4v3_MOD_CIGBINV:
            value->rValue = model->BSIM4v3cigbinv;
            return(OK);
        case BSIM4v3_MOD_NIGC:
            value->rValue = model->BSIM4v3nigc;
            return(OK);
        case BSIM4v3_MOD_NIGBACC:
            value->rValue = model->BSIM4v3nigbacc;
            return(OK);
        case BSIM4v3_MOD_NIGBINV:
            value->rValue = model->BSIM4v3nigbinv;
            return(OK);
        case BSIM4v3_MOD_NTOX:
            value->rValue = model->BSIM4v3ntox;
            return(OK);
        case BSIM4v3_MOD_EIGBINV:
            value->rValue = model->BSIM4v3eigbinv;
            return(OK);
        case BSIM4v3_MOD_PIGCD:
            value->rValue = model->BSIM4v3pigcd;
            return(OK);
        case BSIM4v3_MOD_POXEDGE:
            value->rValue = model->BSIM4v3poxedge;
            return(OK);
        case BSIM4v3_MOD_PHIN:
            value->rValue = model->BSIM4v3phin;
            return(OK);
        case BSIM4v3_MOD_XRCRG1:
            value->rValue = model->BSIM4v3xrcrg1;
            return(OK);
        case BSIM4v3_MOD_XRCRG2:
            value->rValue = model->BSIM4v3xrcrg2;
            return(OK);
        case BSIM4v3_MOD_TNOIA:
            value->rValue = model->BSIM4v3tnoia;
            return(OK);
        case BSIM4v3_MOD_TNOIB:
            value->rValue = model->BSIM4v3tnoib;
            return(OK);
        case BSIM4v3_MOD_RNOIA:
            value->rValue = model->BSIM4v3rnoia;
            return(OK);
        case BSIM4v3_MOD_RNOIB:
            value->rValue = model->BSIM4v3rnoib;
            return(OK);
        case BSIM4v3_MOD_NTNOI:
            value->rValue = model->BSIM4v3ntnoi;
            return(OK);
        case BSIM4v3_MOD_IJTHDFWD:
            value->rValue = model->BSIM4v3ijthdfwd;
            return(OK);
        case BSIM4v3_MOD_IJTHSFWD:
            value->rValue = model->BSIM4v3ijthsfwd;
            return(OK);
        case BSIM4v3_MOD_IJTHDREV:
            value->rValue = model->BSIM4v3ijthdrev;
            return(OK);
        case BSIM4v3_MOD_IJTHSREV:
            value->rValue = model->BSIM4v3ijthsrev;
            return(OK);
        case BSIM4v3_MOD_XJBVD:
            value->rValue = model->BSIM4v3xjbvd;
            return(OK);
        case BSIM4v3_MOD_XJBVS:
            value->rValue = model->BSIM4v3xjbvs;
            return(OK);
        case BSIM4v3_MOD_BVD:
            value->rValue = model->BSIM4v3bvd;
            return(OK);
        case BSIM4v3_MOD_BVS:
            value->rValue = model->BSIM4v3bvs;
            return(OK);
        case BSIM4v3_MOD_VFB:
            value->rValue = model->BSIM4v3vfb;
            return(OK);

        case BSIM4v3_MOD_GBMIN:
            value->rValue = model->BSIM4v3gbmin;
            return(OK);
        case BSIM4v3_MOD_RBDB:
            value->rValue = model->BSIM4v3rbdb;
            return(OK);
        case BSIM4v3_MOD_RBPB:
            value->rValue = model->BSIM4v3rbpb;
            return(OK);
        case BSIM4v3_MOD_RBSB:
            value->rValue = model->BSIM4v3rbsb;
            return(OK);
        case BSIM4v3_MOD_RBPS:
            value->rValue = model->BSIM4v3rbps;
            return(OK);
        case BSIM4v3_MOD_RBPD:
            value->rValue = model->BSIM4v3rbpd;
            return(OK);

        case BSIM4v3_MOD_CGSL:
            value->rValue = model->BSIM4v3cgsl;
            return(OK);
        case BSIM4v3_MOD_CGDL:
            value->rValue = model->BSIM4v3cgdl;
            return(OK);
        case BSIM4v3_MOD_CKAPPAS:
            value->rValue = model->BSIM4v3ckappas;
            return(OK);
        case BSIM4v3_MOD_CKAPPAD:
            value->rValue = model->BSIM4v3ckappad;
            return(OK);
        case BSIM4v3_MOD_CF:
            value->rValue = model->BSIM4v3cf;
            return(OK);
        case BSIM4v3_MOD_CLC:
            value->rValue = model->BSIM4v3clc;
            return(OK);
        case BSIM4v3_MOD_CLE:
            value->rValue = model->BSIM4v3cle;
            return(OK);
        case BSIM4v3_MOD_DWC:
            value->rValue = model->BSIM4v3dwc;
            return(OK);
        case BSIM4v3_MOD_DLC:
            value->rValue = model->BSIM4v3dlc;
            return(OK);
        case BSIM4v3_MOD_XW:
            value->rValue = model->BSIM4v3xw;
            return(OK);
        case BSIM4v3_MOD_XL:
            value->rValue = model->BSIM4v3xl;
            return(OK);
        case BSIM4v3_MOD_DLCIG:
            value->rValue = model->BSIM4v3dlcig;
            return(OK);
        case BSIM4v3_MOD_DWJ:
            value->rValue = model->BSIM4v3dwj;
            return(OK);
        case BSIM4v3_MOD_VFBCV:
            value->rValue = model->BSIM4v3vfbcv; 
            return(OK);
        case BSIM4v3_MOD_ACDE:
            value->rValue = model->BSIM4v3acde;
            return(OK);
        case BSIM4v3_MOD_MOIN:
            value->rValue = model->BSIM4v3moin;
            return(OK);
        case BSIM4v3_MOD_NOFF:
            value->rValue = model->BSIM4v3noff;
            return(OK);
        case BSIM4v3_MOD_VOFFCV:
            value->rValue = model->BSIM4v3voffcv;
            return(OK);
        case BSIM4v3_MOD_DMCG:
            value->rValue = model->BSIM4v3dmcg;
            return(OK);
        case BSIM4v3_MOD_DMCI:
            value->rValue = model->BSIM4v3dmci;
            return(OK);
        case BSIM4v3_MOD_DMDG:
            value->rValue = model->BSIM4v3dmdg;
            return(OK);
        case BSIM4v3_MOD_DMCGT:
            value->rValue = model->BSIM4v3dmcgt;
            return(OK);
        case BSIM4v3_MOD_XGW:
            value->rValue = model->BSIM4v3xgw;
            return(OK);
        case BSIM4v3_MOD_XGL:
            value->rValue = model->BSIM4v3xgl;
            return(OK);
        case BSIM4v3_MOD_RSHG:
            value->rValue = model->BSIM4v3rshg;
            return(OK);
        case BSIM4v3_MOD_NGCON:
            value->rValue = model->BSIM4v3ngcon;
            return(OK);
        case BSIM4v3_MOD_TCJ:
            value->rValue = model->BSIM4v3tcj;
            return(OK);
        case BSIM4v3_MOD_TPB:
            value->rValue = model->BSIM4v3tpb;
            return(OK);
        case BSIM4v3_MOD_TCJSW:
            value->rValue = model->BSIM4v3tcjsw;
            return(OK);
        case BSIM4v3_MOD_TPBSW:
            value->rValue = model->BSIM4v3tpbsw;
            return(OK);
        case BSIM4v3_MOD_TCJSWG:
            value->rValue = model->BSIM4v3tcjswg;
            return(OK);
        case BSIM4v3_MOD_TPBSWG:
            value->rValue = model->BSIM4v3tpbswg;
            return(OK);

	/* Length dependence */
        case  BSIM4v3_MOD_LCDSC :
          value->rValue = model->BSIM4v3lcdsc;
            return(OK);
        case  BSIM4v3_MOD_LCDSCB :
          value->rValue = model->BSIM4v3lcdscb;
            return(OK);
        case  BSIM4v3_MOD_LCDSCD :
          value->rValue = model->BSIM4v3lcdscd;
            return(OK);
        case  BSIM4v3_MOD_LCIT :
          value->rValue = model->BSIM4v3lcit;
            return(OK);
        case  BSIM4v3_MOD_LNFACTOR :
          value->rValue = model->BSIM4v3lnfactor;
            return(OK);
        case BSIM4v3_MOD_LXJ:
            value->rValue = model->BSIM4v3lxj;
            return(OK);
        case BSIM4v3_MOD_LVSAT:
            value->rValue = model->BSIM4v3lvsat;
            return(OK);
        case BSIM4v3_MOD_LAT:
            value->rValue = model->BSIM4v3lat;
            return(OK);
        case BSIM4v3_MOD_LA0:
            value->rValue = model->BSIM4v3la0;
            return(OK);
        case BSIM4v3_MOD_LAGS:
            value->rValue = model->BSIM4v3lags;
            return(OK);
        case BSIM4v3_MOD_LA1:
            value->rValue = model->BSIM4v3la1;
            return(OK);
        case BSIM4v3_MOD_LA2:
            value->rValue = model->BSIM4v3la2;
            return(OK);
        case BSIM4v3_MOD_LKETA:
            value->rValue = model->BSIM4v3lketa;
            return(OK);   
        case BSIM4v3_MOD_LNSUB:
            value->rValue = model->BSIM4v3lnsub;
            return(OK);
        case BSIM4v3_MOD_LNDEP:
            value->rValue = model->BSIM4v3lndep;
            return(OK);
        case BSIM4v3_MOD_LNSD:
            value->rValue = model->BSIM4v3lnsd;
            return(OK);
        case BSIM4v3_MOD_LNGATE:
            value->rValue = model->BSIM4v3lngate;
            return(OK);
        case BSIM4v3_MOD_LGAMMA1:
            value->rValue = model->BSIM4v3lgamma1;
            return(OK);
        case BSIM4v3_MOD_LGAMMA2:
            value->rValue = model->BSIM4v3lgamma2;
            return(OK);
        case BSIM4v3_MOD_LVBX:
            value->rValue = model->BSIM4v3lvbx;
            return(OK);
        case BSIM4v3_MOD_LVBM:
            value->rValue = model->BSIM4v3lvbm;
            return(OK);
        case BSIM4v3_MOD_LXT:
            value->rValue = model->BSIM4v3lxt;
            return(OK);
        case  BSIM4v3_MOD_LK1:
          value->rValue = model->BSIM4v3lk1;
            return(OK);
        case  BSIM4v3_MOD_LKT1:
          value->rValue = model->BSIM4v3lkt1;
            return(OK);
        case  BSIM4v3_MOD_LKT1L:
          value->rValue = model->BSIM4v3lkt1l;
            return(OK);
        case  BSIM4v3_MOD_LKT2 :
          value->rValue = model->BSIM4v3lkt2;
            return(OK);
        case  BSIM4v3_MOD_LK2 :
          value->rValue = model->BSIM4v3lk2;
            return(OK);
        case  BSIM4v3_MOD_LK3:
          value->rValue = model->BSIM4v3lk3;
            return(OK);
        case  BSIM4v3_MOD_LK3B:
          value->rValue = model->BSIM4v3lk3b;
            return(OK);
        case  BSIM4v3_MOD_LW0:
          value->rValue = model->BSIM4v3lw0;
            return(OK);
        case  BSIM4v3_MOD_LLPE0:
          value->rValue = model->BSIM4v3llpe0;
            return(OK);
        case  BSIM4v3_MOD_LLPEB:
          value->rValue = model->BSIM4v3llpeb;
            return(OK);
        case  BSIM4v3_MOD_LDVTP0:
          value->rValue = model->BSIM4v3ldvtp0;
            return(OK);
        case  BSIM4v3_MOD_LDVTP1:
          value->rValue = model->BSIM4v3ldvtp1;
            return(OK);
        case  BSIM4v3_MOD_LDVT0:                
          value->rValue = model->BSIM4v3ldvt0;
            return(OK);
        case  BSIM4v3_MOD_LDVT1 :             
          value->rValue = model->BSIM4v3ldvt1;
            return(OK);
        case  BSIM4v3_MOD_LDVT2 :             
          value->rValue = model->BSIM4v3ldvt2;
            return(OK);
        case  BSIM4v3_MOD_LDVT0W :                
          value->rValue = model->BSIM4v3ldvt0w;
            return(OK);
        case  BSIM4v3_MOD_LDVT1W :             
          value->rValue = model->BSIM4v3ldvt1w;
            return(OK);
        case  BSIM4v3_MOD_LDVT2W :             
          value->rValue = model->BSIM4v3ldvt2w;
            return(OK);
        case  BSIM4v3_MOD_LDROUT :           
          value->rValue = model->BSIM4v3ldrout;
            return(OK);
        case  BSIM4v3_MOD_LDSUB :           
          value->rValue = model->BSIM4v3ldsub;
            return(OK);
        case BSIM4v3_MOD_LVTH0:
            value->rValue = model->BSIM4v3lvth0; 
            return(OK);
        case BSIM4v3_MOD_LUA:
            value->rValue = model->BSIM4v3lua; 
            return(OK);
        case BSIM4v3_MOD_LUA1:
            value->rValue = model->BSIM4v3lua1; 
            return(OK);
        case BSIM4v3_MOD_LUB:
            value->rValue = model->BSIM4v3lub;  
            return(OK);
        case BSIM4v3_MOD_LUB1:
            value->rValue = model->BSIM4v3lub1;  
            return(OK);
        case BSIM4v3_MOD_LUC:
            value->rValue = model->BSIM4v3luc; 
            return(OK);
        case BSIM4v3_MOD_LUC1:
            value->rValue = model->BSIM4v3luc1; 
            return(OK);
        case BSIM4v3_MOD_LU0:
            value->rValue = model->BSIM4v3lu0;
            return(OK);
        case BSIM4v3_MOD_LUTE:
            value->rValue = model->BSIM4v3lute;
            return(OK);
        case BSIM4v3_MOD_LVOFF:
            value->rValue = model->BSIM4v3lvoff;
            return(OK);
        case BSIM4v3_MOD_LMINV:
            value->rValue = model->BSIM4v3lminv;
            return(OK);
        case BSIM4v3_MOD_LFPROUT:
            value->rValue = model->BSIM4v3lfprout;
            return(OK);
        case BSIM4v3_MOD_LPDITS:
            value->rValue = model->BSIM4v3lpdits;
            return(OK);
        case BSIM4v3_MOD_LPDITSD:
            value->rValue = model->BSIM4v3lpditsd;
            return(OK);
        case BSIM4v3_MOD_LDELTA:
            value->rValue = model->BSIM4v3ldelta;
            return(OK);
        case BSIM4v3_MOD_LRDSW:
            value->rValue = model->BSIM4v3lrdsw; 
            return(OK);             
        case BSIM4v3_MOD_LRDW:
            value->rValue = model->BSIM4v3lrdw;
            return(OK);
        case BSIM4v3_MOD_LRSW:
            value->rValue = model->BSIM4v3lrsw;
            return(OK);
        case BSIM4v3_MOD_LPRWB:
            value->rValue = model->BSIM4v3lprwb; 
            return(OK);             
        case BSIM4v3_MOD_LPRWG:
            value->rValue = model->BSIM4v3lprwg; 
            return(OK);             
        case BSIM4v3_MOD_LPRT:
            value->rValue = model->BSIM4v3lprt; 
            return(OK);              
        case BSIM4v3_MOD_LETA0:
            value->rValue = model->BSIM4v3leta0; 
            return(OK);               
        case BSIM4v3_MOD_LETAB:
            value->rValue = model->BSIM4v3letab; 
            return(OK);               
        case BSIM4v3_MOD_LPCLM:
            value->rValue = model->BSIM4v3lpclm; 
            return(OK);               
        case BSIM4v3_MOD_LPDIBL1:
            value->rValue = model->BSIM4v3lpdibl1; 
            return(OK);               
        case BSIM4v3_MOD_LPDIBL2:
            value->rValue = model->BSIM4v3lpdibl2; 
            return(OK);               
        case BSIM4v3_MOD_LPDIBLB:
            value->rValue = model->BSIM4v3lpdiblb; 
            return(OK);               
        case BSIM4v3_MOD_LPSCBE1:
            value->rValue = model->BSIM4v3lpscbe1; 
            return(OK);               
        case BSIM4v3_MOD_LPSCBE2:
            value->rValue = model->BSIM4v3lpscbe2; 
            return(OK);               
        case BSIM4v3_MOD_LPVAG:
            value->rValue = model->BSIM4v3lpvag; 
            return(OK);               
        case BSIM4v3_MOD_LWR:
            value->rValue = model->BSIM4v3lwr;
            return(OK);
        case BSIM4v3_MOD_LDWG:
            value->rValue = model->BSIM4v3ldwg;
            return(OK);
        case BSIM4v3_MOD_LDWB:
            value->rValue = model->BSIM4v3ldwb;
            return(OK);
        case BSIM4v3_MOD_LB0:
            value->rValue = model->BSIM4v3lb0;
            return(OK);
        case BSIM4v3_MOD_LB1:
            value->rValue = model->BSIM4v3lb1;
            return(OK);
        case BSIM4v3_MOD_LALPHA0:
            value->rValue = model->BSIM4v3lalpha0;
            return(OK);
        case BSIM4v3_MOD_LALPHA1:
            value->rValue = model->BSIM4v3lalpha1;
            return(OK);
        case BSIM4v3_MOD_LBETA0:
            value->rValue = model->BSIM4v3lbeta0;
            return(OK);
        case BSIM4v3_MOD_LAGIDL:
            value->rValue = model->BSIM4v3lagidl;
            return(OK);
        case BSIM4v3_MOD_LBGIDL:
            value->rValue = model->BSIM4v3lbgidl;
            return(OK);
        case BSIM4v3_MOD_LCGIDL:
            value->rValue = model->BSIM4v3lcgidl;
            return(OK);
        case BSIM4v3_MOD_LEGIDL:
            value->rValue = model->BSIM4v3legidl;
            return(OK);
        case BSIM4v3_MOD_LAIGC:
            value->rValue = model->BSIM4v3laigc;
            return(OK);
        case BSIM4v3_MOD_LBIGC:
            value->rValue = model->BSIM4v3lbigc;
            return(OK);
        case BSIM4v3_MOD_LCIGC:
            value->rValue = model->BSIM4v3lcigc;
            return(OK);
        case BSIM4v3_MOD_LAIGSD:
            value->rValue = model->BSIM4v3laigsd;
            return(OK);
        case BSIM4v3_MOD_LBIGSD:
            value->rValue = model->BSIM4v3lbigsd;
            return(OK);
        case BSIM4v3_MOD_LCIGSD:
            value->rValue = model->BSIM4v3lcigsd;
            return(OK);
        case BSIM4v3_MOD_LAIGBACC:
            value->rValue = model->BSIM4v3laigbacc;
            return(OK);
        case BSIM4v3_MOD_LBIGBACC:
            value->rValue = model->BSIM4v3lbigbacc;
            return(OK);
        case BSIM4v3_MOD_LCIGBACC:
            value->rValue = model->BSIM4v3lcigbacc;
            return(OK);
        case BSIM4v3_MOD_LAIGBINV:
            value->rValue = model->BSIM4v3laigbinv;
            return(OK);
        case BSIM4v3_MOD_LBIGBINV:
            value->rValue = model->BSIM4v3lbigbinv;
            return(OK);
        case BSIM4v3_MOD_LCIGBINV:
            value->rValue = model->BSIM4v3lcigbinv;
            return(OK);
        case BSIM4v3_MOD_LNIGC:
            value->rValue = model->BSIM4v3lnigc;
            return(OK);
        case BSIM4v3_MOD_LNIGBACC:
            value->rValue = model->BSIM4v3lnigbacc;
            return(OK);
        case BSIM4v3_MOD_LNIGBINV:
            value->rValue = model->BSIM4v3lnigbinv;
            return(OK);
        case BSIM4v3_MOD_LNTOX:
            value->rValue = model->BSIM4v3lntox;
            return(OK);
        case BSIM4v3_MOD_LEIGBINV:
            value->rValue = model->BSIM4v3leigbinv;
            return(OK);
        case BSIM4v3_MOD_LPIGCD:
            value->rValue = model->BSIM4v3lpigcd;
            return(OK);
        case BSIM4v3_MOD_LPOXEDGE:
            value->rValue = model->BSIM4v3lpoxedge;
            return(OK);
        case BSIM4v3_MOD_LPHIN:
            value->rValue = model->BSIM4v3lphin;
            return(OK);
        case BSIM4v3_MOD_LXRCRG1:
            value->rValue = model->BSIM4v3lxrcrg1;
            return(OK);
        case BSIM4v3_MOD_LXRCRG2:
            value->rValue = model->BSIM4v3lxrcrg2;
            return(OK);
        case BSIM4v3_MOD_LEU:
            value->rValue = model->BSIM4v3leu;
            return(OK);
        case BSIM4v3_MOD_LVFB:
            value->rValue = model->BSIM4v3lvfb;
            return(OK);

        case BSIM4v3_MOD_LCGSL:
            value->rValue = model->BSIM4v3lcgsl;
            return(OK);
        case BSIM4v3_MOD_LCGDL:
            value->rValue = model->BSIM4v3lcgdl;
            return(OK);
        case BSIM4v3_MOD_LCKAPPAS:
            value->rValue = model->BSIM4v3lckappas;
            return(OK);
        case BSIM4v3_MOD_LCKAPPAD:
            value->rValue = model->BSIM4v3lckappad;
            return(OK);
        case BSIM4v3_MOD_LCF:
            value->rValue = model->BSIM4v3lcf;
            return(OK);
        case BSIM4v3_MOD_LCLC:
            value->rValue = model->BSIM4v3lclc;
            return(OK);
        case BSIM4v3_MOD_LCLE:
            value->rValue = model->BSIM4v3lcle;
            return(OK);
        case BSIM4v3_MOD_LVFBCV:
            value->rValue = model->BSIM4v3lvfbcv;
            return(OK);
        case BSIM4v3_MOD_LACDE:
            value->rValue = model->BSIM4v3lacde;
            return(OK);
        case BSIM4v3_MOD_LMOIN:
            value->rValue = model->BSIM4v3lmoin;
            return(OK);
        case BSIM4v3_MOD_LNOFF:
            value->rValue = model->BSIM4v3lnoff;
            return(OK);
        case BSIM4v3_MOD_LVOFFCV:
            value->rValue = model->BSIM4v3lvoffcv;
            return(OK);

	/* Width dependence */
        case  BSIM4v3_MOD_WCDSC :
          value->rValue = model->BSIM4v3wcdsc;
            return(OK);
        case  BSIM4v3_MOD_WCDSCB :
          value->rValue = model->BSIM4v3wcdscb;
            return(OK);
        case  BSIM4v3_MOD_WCDSCD :
          value->rValue = model->BSIM4v3wcdscd;
            return(OK);
        case  BSIM4v3_MOD_WCIT :
          value->rValue = model->BSIM4v3wcit;
            return(OK);
        case  BSIM4v3_MOD_WNFACTOR :
          value->rValue = model->BSIM4v3wnfactor;
            return(OK);
        case BSIM4v3_MOD_WXJ:
            value->rValue = model->BSIM4v3wxj;
            return(OK);
        case BSIM4v3_MOD_WVSAT:
            value->rValue = model->BSIM4v3wvsat;
            return(OK);
        case BSIM4v3_MOD_WAT:
            value->rValue = model->BSIM4v3wat;
            return(OK);
        case BSIM4v3_MOD_WA0:
            value->rValue = model->BSIM4v3wa0;
            return(OK);
        case BSIM4v3_MOD_WAGS:
            value->rValue = model->BSIM4v3wags;
            return(OK);
        case BSIM4v3_MOD_WA1:
            value->rValue = model->BSIM4v3wa1;
            return(OK);
        case BSIM4v3_MOD_WA2:
            value->rValue = model->BSIM4v3wa2;
            return(OK);
        case BSIM4v3_MOD_WKETA:
            value->rValue = model->BSIM4v3wketa;
            return(OK);   
        case BSIM4v3_MOD_WNSUB:
            value->rValue = model->BSIM4v3wnsub;
            return(OK);
        case BSIM4v3_MOD_WNDEP:
            value->rValue = model->BSIM4v3wndep;
            return(OK);
        case BSIM4v3_MOD_WNSD:
            value->rValue = model->BSIM4v3wnsd;
            return(OK);
        case BSIM4v3_MOD_WNGATE:
            value->rValue = model->BSIM4v3wngate;
            return(OK);
        case BSIM4v3_MOD_WGAMMA1:
            value->rValue = model->BSIM4v3wgamma1;
            return(OK);
        case BSIM4v3_MOD_WGAMMA2:
            value->rValue = model->BSIM4v3wgamma2;
            return(OK);
        case BSIM4v3_MOD_WVBX:
            value->rValue = model->BSIM4v3wvbx;
            return(OK);
        case BSIM4v3_MOD_WVBM:
            value->rValue = model->BSIM4v3wvbm;
            return(OK);
        case BSIM4v3_MOD_WXT:
            value->rValue = model->BSIM4v3wxt;
            return(OK);
        case  BSIM4v3_MOD_WK1:
          value->rValue = model->BSIM4v3wk1;
            return(OK);
        case  BSIM4v3_MOD_WKT1:
          value->rValue = model->BSIM4v3wkt1;
            return(OK);
        case  BSIM4v3_MOD_WKT1L:
          value->rValue = model->BSIM4v3wkt1l;
            return(OK);
        case  BSIM4v3_MOD_WKT2 :
          value->rValue = model->BSIM4v3wkt2;
            return(OK);
        case  BSIM4v3_MOD_WK2 :
          value->rValue = model->BSIM4v3wk2;
            return(OK);
        case  BSIM4v3_MOD_WK3:
          value->rValue = model->BSIM4v3wk3;
            return(OK);
        case  BSIM4v3_MOD_WK3B:
          value->rValue = model->BSIM4v3wk3b;
            return(OK);
        case  BSIM4v3_MOD_WW0:
          value->rValue = model->BSIM4v3ww0;
            return(OK);
        case  BSIM4v3_MOD_WLPE0:
          value->rValue = model->BSIM4v3wlpe0;
            return(OK);
        case  BSIM4v3_MOD_WDVTP0:
          value->rValue = model->BSIM4v3wdvtp0;
            return(OK);
        case  BSIM4v3_MOD_WDVTP1:
          value->rValue = model->BSIM4v3wdvtp1;
            return(OK);
        case  BSIM4v3_MOD_WLPEB:
          value->rValue = model->BSIM4v3wlpeb;
            return(OK);
        case  BSIM4v3_MOD_WDVT0:                
          value->rValue = model->BSIM4v3wdvt0;
            return(OK);
        case  BSIM4v3_MOD_WDVT1 :             
          value->rValue = model->BSIM4v3wdvt1;
            return(OK);
        case  BSIM4v3_MOD_WDVT2 :             
          value->rValue = model->BSIM4v3wdvt2;
            return(OK);
        case  BSIM4v3_MOD_WDVT0W :                
          value->rValue = model->BSIM4v3wdvt0w;
            return(OK);
        case  BSIM4v3_MOD_WDVT1W :             
          value->rValue = model->BSIM4v3wdvt1w;
            return(OK);
        case  BSIM4v3_MOD_WDVT2W :             
          value->rValue = model->BSIM4v3wdvt2w;
            return(OK);
        case  BSIM4v3_MOD_WDROUT :           
          value->rValue = model->BSIM4v3wdrout;
            return(OK);
        case  BSIM4v3_MOD_WDSUB :           
          value->rValue = model->BSIM4v3wdsub;
            return(OK);
        case BSIM4v3_MOD_WVTH0:
            value->rValue = model->BSIM4v3wvth0; 
            return(OK);
        case BSIM4v3_MOD_WUA:
            value->rValue = model->BSIM4v3wua; 
            return(OK);
        case BSIM4v3_MOD_WUA1:
            value->rValue = model->BSIM4v3wua1; 
            return(OK);
        case BSIM4v3_MOD_WUB:
            value->rValue = model->BSIM4v3wub;  
            return(OK);
        case BSIM4v3_MOD_WUB1:
            value->rValue = model->BSIM4v3wub1;  
            return(OK);
        case BSIM4v3_MOD_WUC:
            value->rValue = model->BSIM4v3wuc; 
            return(OK);
        case BSIM4v3_MOD_WUC1:
            value->rValue = model->BSIM4v3wuc1; 
            return(OK);
        case BSIM4v3_MOD_WU0:
            value->rValue = model->BSIM4v3wu0;
            return(OK);
        case BSIM4v3_MOD_WUTE:
            value->rValue = model->BSIM4v3wute;
            return(OK);
        case BSIM4v3_MOD_WVOFF:
            value->rValue = model->BSIM4v3wvoff;
            return(OK);
        case BSIM4v3_MOD_WMINV:
            value->rValue = model->BSIM4v3wminv;
            return(OK);
        case BSIM4v3_MOD_WFPROUT:
            value->rValue = model->BSIM4v3wfprout;
            return(OK);
        case BSIM4v3_MOD_WPDITS:
            value->rValue = model->BSIM4v3wpdits;
            return(OK);
        case BSIM4v3_MOD_WPDITSD:
            value->rValue = model->BSIM4v3wpditsd;
            return(OK);
        case BSIM4v3_MOD_WDELTA:
            value->rValue = model->BSIM4v3wdelta;
            return(OK);
        case BSIM4v3_MOD_WRDSW:
            value->rValue = model->BSIM4v3wrdsw; 
            return(OK);             
        case BSIM4v3_MOD_WRDW:
            value->rValue = model->BSIM4v3wrdw;
            return(OK);
        case BSIM4v3_MOD_WRSW:
            value->rValue = model->BSIM4v3wrsw;
            return(OK);
        case BSIM4v3_MOD_WPRWB:
            value->rValue = model->BSIM4v3wprwb; 
            return(OK);             
        case BSIM4v3_MOD_WPRWG:
            value->rValue = model->BSIM4v3wprwg; 
            return(OK);             
        case BSIM4v3_MOD_WPRT:
            value->rValue = model->BSIM4v3wprt; 
            return(OK);              
        case BSIM4v3_MOD_WETA0:
            value->rValue = model->BSIM4v3weta0; 
            return(OK);               
        case BSIM4v3_MOD_WETAB:
            value->rValue = model->BSIM4v3wetab; 
            return(OK);               
        case BSIM4v3_MOD_WPCLM:
            value->rValue = model->BSIM4v3wpclm; 
            return(OK);               
        case BSIM4v3_MOD_WPDIBL1:
            value->rValue = model->BSIM4v3wpdibl1; 
            return(OK);               
        case BSIM4v3_MOD_WPDIBL2:
            value->rValue = model->BSIM4v3wpdibl2; 
            return(OK);               
        case BSIM4v3_MOD_WPDIBLB:
            value->rValue = model->BSIM4v3wpdiblb; 
            return(OK);               
        case BSIM4v3_MOD_WPSCBE1:
            value->rValue = model->BSIM4v3wpscbe1; 
            return(OK);               
        case BSIM4v3_MOD_WPSCBE2:
            value->rValue = model->BSIM4v3wpscbe2; 
            return(OK);               
        case BSIM4v3_MOD_WPVAG:
            value->rValue = model->BSIM4v3wpvag; 
            return(OK);               
        case BSIM4v3_MOD_WWR:
            value->rValue = model->BSIM4v3wwr;
            return(OK);
        case BSIM4v3_MOD_WDWG:
            value->rValue = model->BSIM4v3wdwg;
            return(OK);
        case BSIM4v3_MOD_WDWB:
            value->rValue = model->BSIM4v3wdwb;
            return(OK);
        case BSIM4v3_MOD_WB0:
            value->rValue = model->BSIM4v3wb0;
            return(OK);
        case BSIM4v3_MOD_WB1:
            value->rValue = model->BSIM4v3wb1;
            return(OK);
        case BSIM4v3_MOD_WALPHA0:
            value->rValue = model->BSIM4v3walpha0;
            return(OK);
        case BSIM4v3_MOD_WALPHA1:
            value->rValue = model->BSIM4v3walpha1;
            return(OK);
        case BSIM4v3_MOD_WBETA0:
            value->rValue = model->BSIM4v3wbeta0;
            return(OK);
        case BSIM4v3_MOD_WAGIDL:
            value->rValue = model->BSIM4v3wagidl;
            return(OK);
        case BSIM4v3_MOD_WBGIDL:
            value->rValue = model->BSIM4v3wbgidl;
            return(OK);
        case BSIM4v3_MOD_WCGIDL:
            value->rValue = model->BSIM4v3wcgidl;
            return(OK);
        case BSIM4v3_MOD_WEGIDL:
            value->rValue = model->BSIM4v3wegidl;
            return(OK);
        case BSIM4v3_MOD_WAIGC:
            value->rValue = model->BSIM4v3waigc;
            return(OK);
        case BSIM4v3_MOD_WBIGC:
            value->rValue = model->BSIM4v3wbigc;
            return(OK);
        case BSIM4v3_MOD_WCIGC:
            value->rValue = model->BSIM4v3wcigc;
            return(OK);
        case BSIM4v3_MOD_WAIGSD:
            value->rValue = model->BSIM4v3waigsd;
            return(OK);
        case BSIM4v3_MOD_WBIGSD:
            value->rValue = model->BSIM4v3wbigsd;
            return(OK);
        case BSIM4v3_MOD_WCIGSD:
            value->rValue = model->BSIM4v3wcigsd;
            return(OK);
        case BSIM4v3_MOD_WAIGBACC:
            value->rValue = model->BSIM4v3waigbacc;
            return(OK);
        case BSIM4v3_MOD_WBIGBACC:
            value->rValue = model->BSIM4v3wbigbacc;
            return(OK);
        case BSIM4v3_MOD_WCIGBACC:
            value->rValue = model->BSIM4v3wcigbacc;
            return(OK);
        case BSIM4v3_MOD_WAIGBINV:
            value->rValue = model->BSIM4v3waigbinv;
            return(OK);
        case BSIM4v3_MOD_WBIGBINV:
            value->rValue = model->BSIM4v3wbigbinv;
            return(OK);
        case BSIM4v3_MOD_WCIGBINV:
            value->rValue = model->BSIM4v3wcigbinv;
            return(OK);
        case BSIM4v3_MOD_WNIGC:
            value->rValue = model->BSIM4v3wnigc;
            return(OK);
        case BSIM4v3_MOD_WNIGBACC:
            value->rValue = model->BSIM4v3wnigbacc;
            return(OK);
        case BSIM4v3_MOD_WNIGBINV:
            value->rValue = model->BSIM4v3wnigbinv;
            return(OK);
        case BSIM4v3_MOD_WNTOX:
            value->rValue = model->BSIM4v3wntox;
            return(OK);
        case BSIM4v3_MOD_WEIGBINV:
            value->rValue = model->BSIM4v3weigbinv;
            return(OK);
        case BSIM4v3_MOD_WPIGCD:
            value->rValue = model->BSIM4v3wpigcd;
            return(OK);
        case BSIM4v3_MOD_WPOXEDGE:
            value->rValue = model->BSIM4v3wpoxedge;
            return(OK);
        case BSIM4v3_MOD_WPHIN:
            value->rValue = model->BSIM4v3wphin;
            return(OK);
        case BSIM4v3_MOD_WXRCRG1:
            value->rValue = model->BSIM4v3wxrcrg1;
            return(OK);
        case BSIM4v3_MOD_WXRCRG2:
            value->rValue = model->BSIM4v3wxrcrg2;
            return(OK);
        case BSIM4v3_MOD_WEU:
            value->rValue = model->BSIM4v3weu;
            return(OK);
        case BSIM4v3_MOD_WVFB:
            value->rValue = model->BSIM4v3wvfb;
            return(OK);

        case BSIM4v3_MOD_WCGSL:
            value->rValue = model->BSIM4v3wcgsl;
            return(OK);
        case BSIM4v3_MOD_WCGDL:
            value->rValue = model->BSIM4v3wcgdl;
            return(OK);
        case BSIM4v3_MOD_WCKAPPAS:
            value->rValue = model->BSIM4v3wckappas;
            return(OK);
        case BSIM4v3_MOD_WCKAPPAD:
            value->rValue = model->BSIM4v3wckappad;
            return(OK);
        case BSIM4v3_MOD_WCF:
            value->rValue = model->BSIM4v3wcf;
            return(OK);
        case BSIM4v3_MOD_WCLC:
            value->rValue = model->BSIM4v3wclc;
            return(OK);
        case BSIM4v3_MOD_WCLE:
            value->rValue = model->BSIM4v3wcle;
            return(OK);
        case BSIM4v3_MOD_WVFBCV:
            value->rValue = model->BSIM4v3wvfbcv;
            return(OK);
        case BSIM4v3_MOD_WACDE:
            value->rValue = model->BSIM4v3wacde;
            return(OK);
        case BSIM4v3_MOD_WMOIN:
            value->rValue = model->BSIM4v3wmoin;
            return(OK);
        case BSIM4v3_MOD_WNOFF:
            value->rValue = model->BSIM4v3wnoff;
            return(OK);
        case BSIM4v3_MOD_WVOFFCV:
            value->rValue = model->BSIM4v3wvoffcv;
            return(OK);

	/* Cross-term dependence */
        case  BSIM4v3_MOD_PCDSC :
          value->rValue = model->BSIM4v3pcdsc;
            return(OK);
        case  BSIM4v3_MOD_PCDSCB :
          value->rValue = model->BSIM4v3pcdscb;
            return(OK);
        case  BSIM4v3_MOD_PCDSCD :
          value->rValue = model->BSIM4v3pcdscd;
            return(OK);
         case  BSIM4v3_MOD_PCIT :
          value->rValue = model->BSIM4v3pcit;
            return(OK);
        case  BSIM4v3_MOD_PNFACTOR :
          value->rValue = model->BSIM4v3pnfactor;
            return(OK);
        case BSIM4v3_MOD_PXJ:
            value->rValue = model->BSIM4v3pxj;
            return(OK);
        case BSIM4v3_MOD_PVSAT:
            value->rValue = model->BSIM4v3pvsat;
            return(OK);
        case BSIM4v3_MOD_PAT:
            value->rValue = model->BSIM4v3pat;
            return(OK);
        case BSIM4v3_MOD_PA0:
            value->rValue = model->BSIM4v3pa0;
            return(OK);
        case BSIM4v3_MOD_PAGS:
            value->rValue = model->BSIM4v3pags;
            return(OK);
        case BSIM4v3_MOD_PA1:
            value->rValue = model->BSIM4v3pa1;
            return(OK);
        case BSIM4v3_MOD_PA2:
            value->rValue = model->BSIM4v3pa2;
            return(OK);
        case BSIM4v3_MOD_PKETA:
            value->rValue = model->BSIM4v3pketa;
            return(OK);   
        case BSIM4v3_MOD_PNSUB:
            value->rValue = model->BSIM4v3pnsub;
            return(OK);
        case BSIM4v3_MOD_PNDEP:
            value->rValue = model->BSIM4v3pndep;
            return(OK);
        case BSIM4v3_MOD_PNSD:
            value->rValue = model->BSIM4v3pnsd;
            return(OK);
        case BSIM4v3_MOD_PNGATE:
            value->rValue = model->BSIM4v3pngate;
            return(OK);
        case BSIM4v3_MOD_PGAMMA1:
            value->rValue = model->BSIM4v3pgamma1;
            return(OK);
        case BSIM4v3_MOD_PGAMMA2:
            value->rValue = model->BSIM4v3pgamma2;
            return(OK);
        case BSIM4v3_MOD_PVBX:
            value->rValue = model->BSIM4v3pvbx;
            return(OK);
        case BSIM4v3_MOD_PVBM:
            value->rValue = model->BSIM4v3pvbm;
            return(OK);
        case BSIM4v3_MOD_PXT:
            value->rValue = model->BSIM4v3pxt;
            return(OK);
        case  BSIM4v3_MOD_PK1:
          value->rValue = model->BSIM4v3pk1;
            return(OK);
        case  BSIM4v3_MOD_PKT1:
          value->rValue = model->BSIM4v3pkt1;
            return(OK);
        case  BSIM4v3_MOD_PKT1L:
          value->rValue = model->BSIM4v3pkt1l;
            return(OK);
        case  BSIM4v3_MOD_PKT2 :
          value->rValue = model->BSIM4v3pkt2;
            return(OK);
        case  BSIM4v3_MOD_PK2 :
          value->rValue = model->BSIM4v3pk2;
            return(OK);
        case  BSIM4v3_MOD_PK3:
          value->rValue = model->BSIM4v3pk3;
            return(OK);
        case  BSIM4v3_MOD_PK3B:
          value->rValue = model->BSIM4v3pk3b;
            return(OK);
        case  BSIM4v3_MOD_PW0:
          value->rValue = model->BSIM4v3pw0;
            return(OK);
        case  BSIM4v3_MOD_PLPE0:
          value->rValue = model->BSIM4v3plpe0;
            return(OK);
        case  BSIM4v3_MOD_PLPEB:
          value->rValue = model->BSIM4v3plpeb;
            return(OK);
        case  BSIM4v3_MOD_PDVTP0:
          value->rValue = model->BSIM4v3pdvtp0;
            return(OK);
        case  BSIM4v3_MOD_PDVTP1:
          value->rValue = model->BSIM4v3pdvtp1;
            return(OK);
        case  BSIM4v3_MOD_PDVT0 :                
          value->rValue = model->BSIM4v3pdvt0;
            return(OK);
        case  BSIM4v3_MOD_PDVT1 :             
          value->rValue = model->BSIM4v3pdvt1;
            return(OK);
        case  BSIM4v3_MOD_PDVT2 :             
          value->rValue = model->BSIM4v3pdvt2;
            return(OK);
        case  BSIM4v3_MOD_PDVT0W :                
          value->rValue = model->BSIM4v3pdvt0w;
            return(OK);
        case  BSIM4v3_MOD_PDVT1W :             
          value->rValue = model->BSIM4v3pdvt1w;
            return(OK);
        case  BSIM4v3_MOD_PDVT2W :             
          value->rValue = model->BSIM4v3pdvt2w;
            return(OK);
        case  BSIM4v3_MOD_PDROUT :           
          value->rValue = model->BSIM4v3pdrout;
            return(OK);
        case  BSIM4v3_MOD_PDSUB :           
          value->rValue = model->BSIM4v3pdsub;
            return(OK);
        case BSIM4v3_MOD_PVTH0:
            value->rValue = model->BSIM4v3pvth0; 
            return(OK);
        case BSIM4v3_MOD_PUA:
            value->rValue = model->BSIM4v3pua; 
            return(OK);
        case BSIM4v3_MOD_PUA1:
            value->rValue = model->BSIM4v3pua1; 
            return(OK);
        case BSIM4v3_MOD_PUB:
            value->rValue = model->BSIM4v3pub;  
            return(OK);
        case BSIM4v3_MOD_PUB1:
            value->rValue = model->BSIM4v3pub1;  
            return(OK);
        case BSIM4v3_MOD_PUC:
            value->rValue = model->BSIM4v3puc; 
            return(OK);
        case BSIM4v3_MOD_PUC1:
            value->rValue = model->BSIM4v3puc1; 
            return(OK);
        case BSIM4v3_MOD_PU0:
            value->rValue = model->BSIM4v3pu0;
            return(OK);
        case BSIM4v3_MOD_PUTE:
            value->rValue = model->BSIM4v3pute;
            return(OK);
        case BSIM4v3_MOD_PVOFF:
            value->rValue = model->BSIM4v3pvoff;
            return(OK);
        case BSIM4v3_MOD_PMINV:
            value->rValue = model->BSIM4v3pminv;
            return(OK);
        case BSIM4v3_MOD_PFPROUT:
            value->rValue = model->BSIM4v3pfprout;
            return(OK);
        case BSIM4v3_MOD_PPDITS:
            value->rValue = model->BSIM4v3ppdits;
            return(OK);
        case BSIM4v3_MOD_PPDITSD:
            value->rValue = model->BSIM4v3ppditsd;
            return(OK);
        case BSIM4v3_MOD_PDELTA:
            value->rValue = model->BSIM4v3pdelta;
            return(OK);
        case BSIM4v3_MOD_PRDSW:
            value->rValue = model->BSIM4v3prdsw; 
            return(OK);             
        case BSIM4v3_MOD_PRDW:
            value->rValue = model->BSIM4v3prdw;
            return(OK);
        case BSIM4v3_MOD_PRSW:
            value->rValue = model->BSIM4v3prsw;
            return(OK);
        case BSIM4v3_MOD_PPRWB:
            value->rValue = model->BSIM4v3pprwb; 
            return(OK);             
        case BSIM4v3_MOD_PPRWG:
            value->rValue = model->BSIM4v3pprwg; 
            return(OK);             
        case BSIM4v3_MOD_PPRT:
            value->rValue = model->BSIM4v3pprt; 
            return(OK);              
        case BSIM4v3_MOD_PETA0:
            value->rValue = model->BSIM4v3peta0; 
            return(OK);               
        case BSIM4v3_MOD_PETAB:
            value->rValue = model->BSIM4v3petab; 
            return(OK);               
        case BSIM4v3_MOD_PPCLM:
            value->rValue = model->BSIM4v3ppclm; 
            return(OK);               
        case BSIM4v3_MOD_PPDIBL1:
            value->rValue = model->BSIM4v3ppdibl1; 
            return(OK);               
        case BSIM4v3_MOD_PPDIBL2:
            value->rValue = model->BSIM4v3ppdibl2; 
            return(OK);               
        case BSIM4v3_MOD_PPDIBLB:
            value->rValue = model->BSIM4v3ppdiblb; 
            return(OK);               
        case BSIM4v3_MOD_PPSCBE1:
            value->rValue = model->BSIM4v3ppscbe1; 
            return(OK);               
        case BSIM4v3_MOD_PPSCBE2:
            value->rValue = model->BSIM4v3ppscbe2; 
            return(OK);               
        case BSIM4v3_MOD_PPVAG:
            value->rValue = model->BSIM4v3ppvag; 
            return(OK);               
        case BSIM4v3_MOD_PWR:
            value->rValue = model->BSIM4v3pwr;
            return(OK);
        case BSIM4v3_MOD_PDWG:
            value->rValue = model->BSIM4v3pdwg;
            return(OK);
        case BSIM4v3_MOD_PDWB:
            value->rValue = model->BSIM4v3pdwb;
            return(OK);
        case BSIM4v3_MOD_PB0:
            value->rValue = model->BSIM4v3pb0;
            return(OK);
        case BSIM4v3_MOD_PB1:
            value->rValue = model->BSIM4v3pb1;
            return(OK);
        case BSIM4v3_MOD_PALPHA0:
            value->rValue = model->BSIM4v3palpha0;
            return(OK);
        case BSIM4v3_MOD_PALPHA1:
            value->rValue = model->BSIM4v3palpha1;
            return(OK);
        case BSIM4v3_MOD_PBETA0:
            value->rValue = model->BSIM4v3pbeta0;
            return(OK);
        case BSIM4v3_MOD_PAGIDL:
            value->rValue = model->BSIM4v3pagidl;
            return(OK);
        case BSIM4v3_MOD_PBGIDL:
            value->rValue = model->BSIM4v3pbgidl;
            return(OK);
        case BSIM4v3_MOD_PCGIDL:
            value->rValue = model->BSIM4v3pcgidl;
            return(OK);
        case BSIM4v3_MOD_PEGIDL:
            value->rValue = model->BSIM4v3pegidl;
            return(OK);
        case BSIM4v3_MOD_PAIGC:
            value->rValue = model->BSIM4v3paigc;
            return(OK);
        case BSIM4v3_MOD_PBIGC:
            value->rValue = model->BSIM4v3pbigc;
            return(OK);
        case BSIM4v3_MOD_PCIGC:
            value->rValue = model->BSIM4v3pcigc;
            return(OK);
        case BSIM4v3_MOD_PAIGSD:
            value->rValue = model->BSIM4v3paigsd;
            return(OK);
        case BSIM4v3_MOD_PBIGSD:
            value->rValue = model->BSIM4v3pbigsd;
            return(OK);
        case BSIM4v3_MOD_PCIGSD:
            value->rValue = model->BSIM4v3pcigsd;
            return(OK);
        case BSIM4v3_MOD_PAIGBACC:
            value->rValue = model->BSIM4v3paigbacc;
            return(OK);
        case BSIM4v3_MOD_PBIGBACC:
            value->rValue = model->BSIM4v3pbigbacc;
            return(OK);
        case BSIM4v3_MOD_PCIGBACC:
            value->rValue = model->BSIM4v3pcigbacc;
            return(OK);
        case BSIM4v3_MOD_PAIGBINV:
            value->rValue = model->BSIM4v3paigbinv;
            return(OK);
        case BSIM4v3_MOD_PBIGBINV:
            value->rValue = model->BSIM4v3pbigbinv;
            return(OK);
        case BSIM4v3_MOD_PCIGBINV:
            value->rValue = model->BSIM4v3pcigbinv;
            return(OK);
        case BSIM4v3_MOD_PNIGC:
            value->rValue = model->BSIM4v3pnigc;
            return(OK);
        case BSIM4v3_MOD_PNIGBACC:
            value->rValue = model->BSIM4v3pnigbacc;
            return(OK);
        case BSIM4v3_MOD_PNIGBINV:
            value->rValue = model->BSIM4v3pnigbinv;
            return(OK);
        case BSIM4v3_MOD_PNTOX:
            value->rValue = model->BSIM4v3pntox;
            return(OK);
        case BSIM4v3_MOD_PEIGBINV:
            value->rValue = model->BSIM4v3peigbinv;
            return(OK);
        case BSIM4v3_MOD_PPIGCD:
            value->rValue = model->BSIM4v3ppigcd;
            return(OK);
        case BSIM4v3_MOD_PPOXEDGE:
            value->rValue = model->BSIM4v3ppoxedge;
            return(OK);
        case BSIM4v3_MOD_PPHIN:
            value->rValue = model->BSIM4v3pphin;
            return(OK);
        case BSIM4v3_MOD_PXRCRG1:
            value->rValue = model->BSIM4v3pxrcrg1;
            return(OK);
        case BSIM4v3_MOD_PXRCRG2:
            value->rValue = model->BSIM4v3pxrcrg2;
            return(OK);
        case BSIM4v3_MOD_PEU:
            value->rValue = model->BSIM4v3peu;
            return(OK);
        case BSIM4v3_MOD_PVFB:
            value->rValue = model->BSIM4v3pvfb;
            return(OK);

        case BSIM4v3_MOD_PCGSL:
            value->rValue = model->BSIM4v3pcgsl;
            return(OK);
        case BSIM4v3_MOD_PCGDL:
            value->rValue = model->BSIM4v3pcgdl;
            return(OK);
        case BSIM4v3_MOD_PCKAPPAS:
            value->rValue = model->BSIM4v3pckappas;
            return(OK);
        case BSIM4v3_MOD_PCKAPPAD:
            value->rValue = model->BSIM4v3pckappad;
            return(OK);
        case BSIM4v3_MOD_PCF:
            value->rValue = model->BSIM4v3pcf;
            return(OK);
        case BSIM4v3_MOD_PCLC:
            value->rValue = model->BSIM4v3pclc;
            return(OK);
        case BSIM4v3_MOD_PCLE:
            value->rValue = model->BSIM4v3pcle;
            return(OK);
        case BSIM4v3_MOD_PVFBCV:
            value->rValue = model->BSIM4v3pvfbcv;
            return(OK);
        case BSIM4v3_MOD_PACDE:
            value->rValue = model->BSIM4v3pacde;
            return(OK);
        case BSIM4v3_MOD_PMOIN:
            value->rValue = model->BSIM4v3pmoin;
            return(OK);
        case BSIM4v3_MOD_PNOFF:
            value->rValue = model->BSIM4v3pnoff;
            return(OK);
        case BSIM4v3_MOD_PVOFFCV:
            value->rValue = model->BSIM4v3pvoffcv;
            return(OK);

        case  BSIM4v3_MOD_TNOM :
          value->rValue = model->BSIM4v3tnom;
            return(OK);
        case BSIM4v3_MOD_CGSO:
            value->rValue = model->BSIM4v3cgso; 
            return(OK);
        case BSIM4v3_MOD_CGDO:
            value->rValue = model->BSIM4v3cgdo; 
            return(OK);
        case BSIM4v3_MOD_CGBO:
            value->rValue = model->BSIM4v3cgbo; 
            return(OK);
        case BSIM4v3_MOD_XPART:
            value->rValue = model->BSIM4v3xpart; 
            return(OK);
        case BSIM4v3_MOD_RSH:
            value->rValue = model->BSIM4v3sheetResistance; 
            return(OK);
        case BSIM4v3_MOD_JSS:
            value->rValue = model->BSIM4v3SjctSatCurDensity; 
            return(OK);
        case BSIM4v3_MOD_JSWS:
            value->rValue = model->BSIM4v3SjctSidewallSatCurDensity; 
            return(OK);
        case BSIM4v3_MOD_JSWGS:
            value->rValue = model->BSIM4v3SjctGateSidewallSatCurDensity;
            return(OK);
        case BSIM4v3_MOD_PBS:
            value->rValue = model->BSIM4v3SbulkJctPotential; 
            return(OK);
        case BSIM4v3_MOD_MJS:
            value->rValue = model->BSIM4v3SbulkJctBotGradingCoeff; 
            return(OK);
        case BSIM4v3_MOD_PBSWS:
            value->rValue = model->BSIM4v3SsidewallJctPotential; 
            return(OK);
        case BSIM4v3_MOD_MJSWS:
            value->rValue = model->BSIM4v3SbulkJctSideGradingCoeff; 
            return(OK);
        case BSIM4v3_MOD_CJS:
            value->rValue = model->BSIM4v3SunitAreaJctCap; 
            return(OK);
        case BSIM4v3_MOD_CJSWS:
            value->rValue = model->BSIM4v3SunitLengthSidewallJctCap; 
            return(OK);
        case BSIM4v3_MOD_PBSWGS:
            value->rValue = model->BSIM4v3SGatesidewallJctPotential; 
            return(OK);
        case BSIM4v3_MOD_MJSWGS:
            value->rValue = model->BSIM4v3SbulkJctGateSideGradingCoeff; 
            return(OK);
        case BSIM4v3_MOD_CJSWGS:
            value->rValue = model->BSIM4v3SunitLengthGateSidewallJctCap; 
            return(OK);
        case BSIM4v3_MOD_NJS:
            value->rValue = model->BSIM4v3SjctEmissionCoeff; 
            return(OK);
        case BSIM4v3_MOD_XTIS:
            value->rValue = model->BSIM4v3SjctTempExponent; 
            return(OK);
        case BSIM4v3_MOD_JSD:
            value->rValue = model->BSIM4v3DjctSatCurDensity;
            return(OK);
        case BSIM4v3_MOD_JSWD:
            value->rValue = model->BSIM4v3DjctSidewallSatCurDensity;
            return(OK);
        case BSIM4v3_MOD_JSWGD:
            value->rValue = model->BSIM4v3DjctGateSidewallSatCurDensity;
            return(OK);
        case BSIM4v3_MOD_PBD:
            value->rValue = model->BSIM4v3DbulkJctPotential;
            return(OK);
        case BSIM4v3_MOD_MJD:
            value->rValue = model->BSIM4v3DbulkJctBotGradingCoeff;
            return(OK);
        case BSIM4v3_MOD_PBSWD:
            value->rValue = model->BSIM4v3DsidewallJctPotential;
            return(OK);
        case BSIM4v3_MOD_MJSWD:
            value->rValue = model->BSIM4v3DbulkJctSideGradingCoeff;
            return(OK);
        case BSIM4v3_MOD_CJD:
            value->rValue = model->BSIM4v3DunitAreaJctCap;
            return(OK);
        case BSIM4v3_MOD_CJSWD:
            value->rValue = model->BSIM4v3DunitLengthSidewallJctCap;
            return(OK);
        case BSIM4v3_MOD_PBSWGD:
            value->rValue = model->BSIM4v3DGatesidewallJctPotential;
            return(OK);
        case BSIM4v3_MOD_MJSWGD:
            value->rValue = model->BSIM4v3DbulkJctGateSideGradingCoeff;
            return(OK);
        case BSIM4v3_MOD_CJSWGD:
            value->rValue = model->BSIM4v3DunitLengthGateSidewallJctCap;
            return(OK);
        case BSIM4v3_MOD_NJD:
            value->rValue = model->BSIM4v3DjctEmissionCoeff;
            return(OK);
        case BSIM4v3_MOD_XTID:
            value->rValue = model->BSIM4v3DjctTempExponent;
            return(OK);
        case BSIM4v3_MOD_LINT:
            value->rValue = model->BSIM4v3Lint; 
            return(OK);
        case BSIM4v3_MOD_LL:
            value->rValue = model->BSIM4v3Ll;
            return(OK);
        case BSIM4v3_MOD_LLC:
            value->rValue = model->BSIM4v3Llc;
            return(OK);
        case BSIM4v3_MOD_LLN:
            value->rValue = model->BSIM4v3Lln;
            return(OK);
        case BSIM4v3_MOD_LW:
            value->rValue = model->BSIM4v3Lw;
            return(OK);
        case BSIM4v3_MOD_LWC:
            value->rValue = model->BSIM4v3Lwc;
            return(OK);
        case BSIM4v3_MOD_LWN:
            value->rValue = model->BSIM4v3Lwn;
            return(OK);
        case BSIM4v3_MOD_LWL:
            value->rValue = model->BSIM4v3Lwl;
            return(OK);
        case BSIM4v3_MOD_LWLC:
            value->rValue = model->BSIM4v3Lwlc;
            return(OK);
        case BSIM4v3_MOD_LMIN:
            value->rValue = model->BSIM4v3Lmin;
            return(OK);
        case BSIM4v3_MOD_LMAX:
            value->rValue = model->BSIM4v3Lmax;
            return(OK);
        case BSIM4v3_MOD_WINT:
            value->rValue = model->BSIM4v3Wint;
            return(OK);
        case BSIM4v3_MOD_WL:
            value->rValue = model->BSIM4v3Wl;
            return(OK);
        case BSIM4v3_MOD_WLC:
            value->rValue = model->BSIM4v3Wlc;
            return(OK);
        case BSIM4v3_MOD_WLN:
            value->rValue = model->BSIM4v3Wln;
            return(OK);
        case BSIM4v3_MOD_WW:
            value->rValue = model->BSIM4v3Ww;
            return(OK);
        case BSIM4v3_MOD_WWC:
            value->rValue = model->BSIM4v3Wwc;
            return(OK);
        case BSIM4v3_MOD_WWN:
            value->rValue = model->BSIM4v3Wwn;
            return(OK);
        case BSIM4v3_MOD_WWL:
            value->rValue = model->BSIM4v3Wwl;
            return(OK);
        case BSIM4v3_MOD_WWLC:
            value->rValue = model->BSIM4v3Wwlc;
            return(OK);
        case BSIM4v3_MOD_WMIN:
            value->rValue = model->BSIM4v3Wmin;
            return(OK);
        case BSIM4v3_MOD_WMAX:
            value->rValue = model->BSIM4v3Wmax;
            return(OK);

        /* stress effect */
        case BSIM4v3_MOD_SAREF:
            value->rValue = model->BSIM4v3saref;
            return(OK);
        case BSIM4v3_MOD_SBREF:
            value->rValue = model->BSIM4v3sbref;
            return(OK);
	case BSIM4v3_MOD_WLOD:
            value->rValue = model->BSIM4v3wlod;
            return(OK);
        case BSIM4v3_MOD_KU0:
            value->rValue = model->BSIM4v3ku0;
            return(OK);
        case BSIM4v3_MOD_KVSAT:
            value->rValue = model->BSIM4v3kvsat;
            return(OK);
        case BSIM4v3_MOD_KVTH0:
            value->rValue = model->BSIM4v3kvth0;
            return(OK);
        case BSIM4v3_MOD_TKU0:
            value->rValue = model->BSIM4v3tku0;
            return(OK);
        case BSIM4v3_MOD_LLODKU0:
            value->rValue = model->BSIM4v3llodku0;
            return(OK);
        case BSIM4v3_MOD_WLODKU0:
            value->rValue = model->BSIM4v3wlodku0;
            return(OK);
        case BSIM4v3_MOD_LLODVTH:
            value->rValue = model->BSIM4v3llodvth;
            return(OK);
        case BSIM4v3_MOD_WLODVTH:
            value->rValue = model->BSIM4v3wlodvth;
            return(OK);
        case BSIM4v3_MOD_LKU0:
            value->rValue = model->BSIM4v3lku0;
            return(OK);
        case BSIM4v3_MOD_WKU0:
            value->rValue = model->BSIM4v3wku0;
            return(OK);
        case BSIM4v3_MOD_PKU0:
            value->rValue = model->BSIM4v3pku0;
            return(OK);
        case BSIM4v3_MOD_LKVTH0:
            value->rValue = model->BSIM4v3lkvth0;
            return(OK);
        case BSIM4v3_MOD_WKVTH0:
            value->rValue = model->BSIM4v3wkvth0;
            return(OK);
        case BSIM4v3_MOD_PKVTH0:
            value->rValue = model->BSIM4v3pkvth0;
            return(OK);
        case BSIM4v3_MOD_STK2:
            value->rValue = model->BSIM4v3stk2;
            return(OK);
        case BSIM4v3_MOD_LODK2:
            value->rValue = model->BSIM4v3lodk2;
            return(OK);
        case BSIM4v3_MOD_STETA0:
            value->rValue = model->BSIM4v3steta0;
            return(OK);
        case BSIM4v3_MOD_LODETA0:
            value->rValue = model->BSIM4v3lodeta0;
            return(OK);

        case BSIM4v3_MOD_NOIA:
            value->rValue = model->BSIM4v3oxideTrapDensityA;
            return(OK);
        case BSIM4v3_MOD_NOIB:
            value->rValue = model->BSIM4v3oxideTrapDensityB;
            return(OK);
        case BSIM4v3_MOD_NOIC:
            value->rValue = model->BSIM4v3oxideTrapDensityC;
            return(OK);
        case BSIM4v3_MOD_EM:
            value->rValue = model->BSIM4v3em;
            return(OK);
        case BSIM4v3_MOD_EF:
            value->rValue = model->BSIM4v3ef;
            return(OK);
        case BSIM4v3_MOD_AF:
            value->rValue = model->BSIM4v3af;
            return(OK);
        case BSIM4v3_MOD_KF:
            value->rValue = model->BSIM4v3kf;
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}



