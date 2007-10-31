/**** BSIM4.4.0  Released by Xuemei (Jane) Xi 03/04/2004 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b4mask.c of BSIM4.4.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 * Modified by Xuemei Xi, 11/15/2002.
 * Modified by Xuemei Xi, 05/09/2003.
 **********/


#include "ngspice.h"
#include "ifsim.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "bsim4v4def.h"
#include "sperror.h"
#include "suffix.h"

int
BSIM4V4mAsk(ckt,inst,which,value)
CKTcircuit *ckt;
GENmodel *inst;
int which;
IFvalue *value;
{
    BSIM4V4model *model = (BSIM4V4model *)inst;
    switch(which) 
    {   case BSIM4V4_MOD_MOBMOD :
            value->iValue = model->BSIM4V4mobMod; 
            return(OK);
        case BSIM4V4_MOD_PARAMCHK :
            value->iValue = model->BSIM4V4paramChk; 
            return(OK);
        case BSIM4V4_MOD_BINUNIT :
            value->iValue = model->BSIM4V4binUnit; 
            return(OK);
        case BSIM4V4_MOD_CAPMOD :
            value->iValue = model->BSIM4V4capMod; 
            return(OK);
        case BSIM4V4_MOD_DIOMOD :
            value->iValue = model->BSIM4V4dioMod;
            return(OK);
        case BSIM4V4_MOD_TRNQSMOD :
            value->iValue = model->BSIM4V4trnqsMod;
            return(OK);
        case BSIM4V4_MOD_ACNQSMOD :
            value->iValue = model->BSIM4V4acnqsMod;
            return(OK);
        case BSIM4V4_MOD_FNOIMOD :
            value->iValue = model->BSIM4V4fnoiMod; 
            return(OK);
        case BSIM4V4_MOD_TNOIMOD :
            value->iValue = model->BSIM4V4tnoiMod;
            return(OK);
        case BSIM4V4_MOD_RDSMOD :
            value->iValue = model->BSIM4V4rdsMod;
            return(OK);
        case BSIM4V4_MOD_RBODYMOD :
            value->iValue = model->BSIM4V4rbodyMod;
            return(OK);
        case BSIM4V4_MOD_RGATEMOD :
            value->iValue = model->BSIM4V4rgateMod;
            return(OK);
        case BSIM4V4_MOD_PERMOD :
            value->iValue = model->BSIM4V4perMod;
            return(OK);
        case BSIM4V4_MOD_GEOMOD :
            value->iValue = model->BSIM4V4geoMod;
            return(OK);
        case BSIM4V4_MOD_IGCMOD :
            value->iValue = model->BSIM4V4igcMod;
            return(OK);
        case BSIM4V4_MOD_IGBMOD :
            value->iValue = model->BSIM4V4igbMod;
            return(OK);
        case  BSIM4V4_MOD_TEMPMOD :
            value->iValue = model->BSIM4V4tempMod;
            return(OK);
        case  BSIM4V4_MOD_VERSION :
          value->sValue = model->BSIM4V4version;
            return(OK);
        case  BSIM4V4_MOD_TOXREF :
          value->rValue = model->BSIM4V4toxref;
          return(OK);
        case  BSIM4V4_MOD_TOXE :
          value->rValue = model->BSIM4V4toxe;
            return(OK);
        case  BSIM4V4_MOD_TOXP :
          value->rValue = model->BSIM4V4toxp;
            return(OK);
        case  BSIM4V4_MOD_TOXM :
          value->rValue = model->BSIM4V4toxm;
            return(OK);
        case  BSIM4V4_MOD_DTOX :
          value->rValue = model->BSIM4V4dtox;
            return(OK);
        case  BSIM4V4_MOD_EPSROX :
          value->rValue = model->BSIM4V4epsrox;
            return(OK);
        case  BSIM4V4_MOD_CDSC :
          value->rValue = model->BSIM4V4cdsc;
            return(OK);
        case  BSIM4V4_MOD_CDSCB :
          value->rValue = model->BSIM4V4cdscb;
            return(OK);

        case  BSIM4V4_MOD_CDSCD :
          value->rValue = model->BSIM4V4cdscd;
            return(OK);

        case  BSIM4V4_MOD_CIT :
          value->rValue = model->BSIM4V4cit;
            return(OK);
        case  BSIM4V4_MOD_NFACTOR :
          value->rValue = model->BSIM4V4nfactor;
            return(OK);
        case BSIM4V4_MOD_XJ:
            value->rValue = model->BSIM4V4xj;
            return(OK);
        case BSIM4V4_MOD_VSAT:
            value->rValue = model->BSIM4V4vsat;
            return(OK);
        case BSIM4V4_MOD_VTL:
            value->rValue = model->BSIM4V4vtl;
            return(OK);
        case BSIM4V4_MOD_XN:
            value->rValue = model->BSIM4V4xn;
            return(OK);
        case BSIM4V4_MOD_LC:
            value->rValue = model->BSIM4V4lc;
            return(OK);
        case BSIM4V4_MOD_LAMBDA:
            value->rValue = model->BSIM4V4lambda;
            return(OK);
        case BSIM4V4_MOD_AT:
            value->rValue = model->BSIM4V4at;
            return(OK);
        case BSIM4V4_MOD_A0:
            value->rValue = model->BSIM4V4a0;
            return(OK);

        case BSIM4V4_MOD_AGS:
            value->rValue = model->BSIM4V4ags;
            return(OK);

        case BSIM4V4_MOD_A1:
            value->rValue = model->BSIM4V4a1;
            return(OK);
        case BSIM4V4_MOD_A2:
            value->rValue = model->BSIM4V4a2;
            return(OK);
        case BSIM4V4_MOD_KETA:
            value->rValue = model->BSIM4V4keta;
            return(OK);   
        case BSIM4V4_MOD_NSUB:
            value->rValue = model->BSIM4V4nsub;
            return(OK);
        case BSIM4V4_MOD_NDEP:
            value->rValue = model->BSIM4V4ndep;
            return(OK);
        case BSIM4V4_MOD_NSD:
            value->rValue = model->BSIM4V4nsd;
            return(OK);
        case BSIM4V4_MOD_NGATE:
            value->rValue = model->BSIM4V4ngate;
            return(OK);
        case BSIM4V4_MOD_GAMMA1:
            value->rValue = model->BSIM4V4gamma1;
            return(OK);
        case BSIM4V4_MOD_GAMMA2:
            value->rValue = model->BSIM4V4gamma2;
            return(OK);
        case BSIM4V4_MOD_VBX:
            value->rValue = model->BSIM4V4vbx;
            return(OK);
        case BSIM4V4_MOD_VBM:
            value->rValue = model->BSIM4V4vbm;
            return(OK);
        case BSIM4V4_MOD_XT:
            value->rValue = model->BSIM4V4xt;
            return(OK);
        case  BSIM4V4_MOD_K1:
          value->rValue = model->BSIM4V4k1;
            return(OK);
        case  BSIM4V4_MOD_KT1:
          value->rValue = model->BSIM4V4kt1;
            return(OK);
        case  BSIM4V4_MOD_KT1L:
          value->rValue = model->BSIM4V4kt1l;
            return(OK);
        case  BSIM4V4_MOD_KT2 :
          value->rValue = model->BSIM4V4kt2;
            return(OK);
        case  BSIM4V4_MOD_K2 :
          value->rValue = model->BSIM4V4k2;
            return(OK);
        case  BSIM4V4_MOD_K3:
          value->rValue = model->BSIM4V4k3;
            return(OK);
        case  BSIM4V4_MOD_K3B:
          value->rValue = model->BSIM4V4k3b;
            return(OK);
        case  BSIM4V4_MOD_W0:
          value->rValue = model->BSIM4V4w0;
            return(OK);
        case  BSIM4V4_MOD_LPE0:
          value->rValue = model->BSIM4V4lpe0;
            return(OK);
        case  BSIM4V4_MOD_LPEB:
          value->rValue = model->BSIM4V4lpeb;
            return(OK);
        case  BSIM4V4_MOD_DVTP0:
          value->rValue = model->BSIM4V4dvtp0;
            return(OK);
        case  BSIM4V4_MOD_DVTP1:
          value->rValue = model->BSIM4V4dvtp1;
            return(OK);
        case  BSIM4V4_MOD_DVT0 :                
          value->rValue = model->BSIM4V4dvt0;
            return(OK);
        case  BSIM4V4_MOD_DVT1 :             
          value->rValue = model->BSIM4V4dvt1;
            return(OK);
        case  BSIM4V4_MOD_DVT2 :             
          value->rValue = model->BSIM4V4dvt2;
            return(OK);
        case  BSIM4V4_MOD_DVT0W :                
          value->rValue = model->BSIM4V4dvt0w;
            return(OK);
        case  BSIM4V4_MOD_DVT1W :             
          value->rValue = model->BSIM4V4dvt1w;
            return(OK);
        case  BSIM4V4_MOD_DVT2W :             
          value->rValue = model->BSIM4V4dvt2w;
            return(OK);
        case  BSIM4V4_MOD_DROUT :           
          value->rValue = model->BSIM4V4drout;
            return(OK);
        case  BSIM4V4_MOD_DSUB :           
          value->rValue = model->BSIM4V4dsub;
            return(OK);
        case BSIM4V4_MOD_VTH0:
            value->rValue = model->BSIM4V4vth0; 
            return(OK);
        case BSIM4V4_MOD_EU:
            value->rValue = model->BSIM4V4eu;
            return(OK);
        case BSIM4V4_MOD_UA:
            value->rValue = model->BSIM4V4ua; 
            return(OK);
        case BSIM4V4_MOD_UA1:
            value->rValue = model->BSIM4V4ua1; 
            return(OK);
        case BSIM4V4_MOD_UB:
            value->rValue = model->BSIM4V4ub;  
            return(OK);
        case BSIM4V4_MOD_UB1:
            value->rValue = model->BSIM4V4ub1;  
            return(OK);
        case BSIM4V4_MOD_UC:
            value->rValue = model->BSIM4V4uc; 
            return(OK);
        case BSIM4V4_MOD_UC1:
            value->rValue = model->BSIM4V4uc1; 
            return(OK);
        case BSIM4V4_MOD_U0:
            value->rValue = model->BSIM4V4u0;
            return(OK);
        case BSIM4V4_MOD_UTE:
            value->rValue = model->BSIM4V4ute;
            return(OK);
        case BSIM4V4_MOD_VOFF:
            value->rValue = model->BSIM4V4voff;
            return(OK);
        case BSIM4V4_MOD_VOFFL:
            value->rValue = model->BSIM4V4voffl;
            return(OK);
        case BSIM4V4_MOD_MINV:
            value->rValue = model->BSIM4V4minv;
            return(OK);
        case BSIM4V4_MOD_FPROUT:
            value->rValue = model->BSIM4V4fprout;
            return(OK);
        case BSIM4V4_MOD_PDITS:
            value->rValue = model->BSIM4V4pdits;
            return(OK);
        case BSIM4V4_MOD_PDITSD:
            value->rValue = model->BSIM4V4pditsd;
            return(OK);
        case BSIM4V4_MOD_PDITSL:
            value->rValue = model->BSIM4V4pditsl;
            return(OK);
        case BSIM4V4_MOD_DELTA:
            value->rValue = model->BSIM4V4delta;
            return(OK);
        case BSIM4V4_MOD_RDSW:
            value->rValue = model->BSIM4V4rdsw; 
            return(OK);
        case BSIM4V4_MOD_RDSWMIN:
            value->rValue = model->BSIM4V4rdswmin;
            return(OK);
        case BSIM4V4_MOD_RDWMIN:
            value->rValue = model->BSIM4V4rdwmin;
            return(OK);
        case BSIM4V4_MOD_RSWMIN:
            value->rValue = model->BSIM4V4rswmin;
            return(OK);
        case BSIM4V4_MOD_RDW:
            value->rValue = model->BSIM4V4rdw;
            return(OK);
        case BSIM4V4_MOD_RSW:
            value->rValue = model->BSIM4V4rsw;
            return(OK);
        case BSIM4V4_MOD_PRWG:
            value->rValue = model->BSIM4V4prwg; 
            return(OK);             
        case BSIM4V4_MOD_PRWB:
            value->rValue = model->BSIM4V4prwb; 
            return(OK);             
        case BSIM4V4_MOD_PRT:
            value->rValue = model->BSIM4V4prt; 
            return(OK);              
        case BSIM4V4_MOD_ETA0:
            value->rValue = model->BSIM4V4eta0; 
            return(OK);               
        case BSIM4V4_MOD_ETAB:
            value->rValue = model->BSIM4V4etab; 
            return(OK);               
        case BSIM4V4_MOD_PCLM:
            value->rValue = model->BSIM4V4pclm; 
            return(OK);               
        case BSIM4V4_MOD_PDIBL1:
            value->rValue = model->BSIM4V4pdibl1; 
            return(OK);               
        case BSIM4V4_MOD_PDIBL2:
            value->rValue = model->BSIM4V4pdibl2; 
            return(OK);               
        case BSIM4V4_MOD_PDIBLB:
            value->rValue = model->BSIM4V4pdiblb; 
            return(OK);               
        case BSIM4V4_MOD_PSCBE1:
            value->rValue = model->BSIM4V4pscbe1; 
            return(OK);               
        case BSIM4V4_MOD_PSCBE2:
            value->rValue = model->BSIM4V4pscbe2; 
            return(OK);               
        case BSIM4V4_MOD_PVAG:
            value->rValue = model->BSIM4V4pvag; 
            return(OK);               
        case BSIM4V4_MOD_WR:
            value->rValue = model->BSIM4V4wr;
            return(OK);
        case BSIM4V4_MOD_DWG:
            value->rValue = model->BSIM4V4dwg;
            return(OK);
        case BSIM4V4_MOD_DWB:
            value->rValue = model->BSIM4V4dwb;
            return(OK);
        case BSIM4V4_MOD_B0:
            value->rValue = model->BSIM4V4b0;
            return(OK);
        case BSIM4V4_MOD_B1:
            value->rValue = model->BSIM4V4b1;
            return(OK);
        case BSIM4V4_MOD_ALPHA0:
            value->rValue = model->BSIM4V4alpha0;
            return(OK);
        case BSIM4V4_MOD_ALPHA1:
            value->rValue = model->BSIM4V4alpha1;
            return(OK);
        case BSIM4V4_MOD_BETA0:
            value->rValue = model->BSIM4V4beta0;
            return(OK);
        case BSIM4V4_MOD_AGIDL:
            value->rValue = model->BSIM4V4agidl;
            return(OK);
        case BSIM4V4_MOD_BGIDL:
            value->rValue = model->BSIM4V4bgidl;
            return(OK);
        case BSIM4V4_MOD_CGIDL:
            value->rValue = model->BSIM4V4cgidl;
            return(OK);
        case BSIM4V4_MOD_EGIDL:
            value->rValue = model->BSIM4V4egidl;
            return(OK);
        case BSIM4V4_MOD_AIGC:
            value->rValue = model->BSIM4V4aigc;
            return(OK);
        case BSIM4V4_MOD_BIGC:
            value->rValue = model->BSIM4V4bigc;
            return(OK);
        case BSIM4V4_MOD_CIGC:
            value->rValue = model->BSIM4V4cigc;
            return(OK);
        case BSIM4V4_MOD_AIGSD:
            value->rValue = model->BSIM4V4aigsd;
            return(OK);
        case BSIM4V4_MOD_BIGSD:
            value->rValue = model->BSIM4V4bigsd;
            return(OK);
        case BSIM4V4_MOD_CIGSD:
            value->rValue = model->BSIM4V4cigsd;
            return(OK);
        case BSIM4V4_MOD_AIGBACC:
            value->rValue = model->BSIM4V4aigbacc;
            return(OK);
        case BSIM4V4_MOD_BIGBACC:
            value->rValue = model->BSIM4V4bigbacc;
            return(OK);
        case BSIM4V4_MOD_CIGBACC:
            value->rValue = model->BSIM4V4cigbacc;
            return(OK);
        case BSIM4V4_MOD_AIGBINV:
            value->rValue = model->BSIM4V4aigbinv;
            return(OK);
        case BSIM4V4_MOD_BIGBINV:
            value->rValue = model->BSIM4V4bigbinv;
            return(OK);
        case BSIM4V4_MOD_CIGBINV:
            value->rValue = model->BSIM4V4cigbinv;
            return(OK);
        case BSIM4V4_MOD_NIGC:
            value->rValue = model->BSIM4V4nigc;
            return(OK);
        case BSIM4V4_MOD_NIGBACC:
            value->rValue = model->BSIM4V4nigbacc;
            return(OK);
        case BSIM4V4_MOD_NIGBINV:
            value->rValue = model->BSIM4V4nigbinv;
            return(OK);
        case BSIM4V4_MOD_NTOX:
            value->rValue = model->BSIM4V4ntox;
            return(OK);
        case BSIM4V4_MOD_EIGBINV:
            value->rValue = model->BSIM4V4eigbinv;
            return(OK);
        case BSIM4V4_MOD_PIGCD:
            value->rValue = model->BSIM4V4pigcd;
            return(OK);
        case BSIM4V4_MOD_POXEDGE:
            value->rValue = model->BSIM4V4poxedge;
            return(OK);
        case BSIM4V4_MOD_PHIN:
            value->rValue = model->BSIM4V4phin;
            return(OK);
        case BSIM4V4_MOD_XRCRG1:
            value->rValue = model->BSIM4V4xrcrg1;
            return(OK);
        case BSIM4V4_MOD_XRCRG2:
            value->rValue = model->BSIM4V4xrcrg2;
            return(OK);
        case BSIM4V4_MOD_TNOIA:
            value->rValue = model->BSIM4V4tnoia;
            return(OK);
        case BSIM4V4_MOD_TNOIB:
            value->rValue = model->BSIM4V4tnoib;
            return(OK);
        case BSIM4V4_MOD_RNOIA:
            value->rValue = model->BSIM4V4rnoia;
            return(OK);
        case BSIM4V4_MOD_RNOIB:
            value->rValue = model->BSIM4V4rnoib;
            return(OK);
        case BSIM4V4_MOD_NTNOI:
            value->rValue = model->BSIM4V4ntnoi;
            return(OK);
        case BSIM4V4_MOD_IJTHDFWD:
            value->rValue = model->BSIM4V4ijthdfwd;
            return(OK);
        case BSIM4V4_MOD_IJTHSFWD:
            value->rValue = model->BSIM4V4ijthsfwd;
            return(OK);
        case BSIM4V4_MOD_IJTHDREV:
            value->rValue = model->BSIM4V4ijthdrev;
            return(OK);
        case BSIM4V4_MOD_IJTHSREV:
            value->rValue = model->BSIM4V4ijthsrev;
            return(OK);
        case BSIM4V4_MOD_XJBVD:
            value->rValue = model->BSIM4V4xjbvd;
            return(OK);
        case BSIM4V4_MOD_XJBVS:
            value->rValue = model->BSIM4V4xjbvs;
            return(OK);
        case BSIM4V4_MOD_BVD:
            value->rValue = model->BSIM4V4bvd;
            return(OK);
        case BSIM4V4_MOD_BVS:
            value->rValue = model->BSIM4V4bvs;
            return(OK);
        case BSIM4V4_MOD_VFB:
            value->rValue = model->BSIM4V4vfb;
            return(OK);

        case BSIM4V4_MOD_GBMIN:
            value->rValue = model->BSIM4V4gbmin;
            return(OK);
        case BSIM4V4_MOD_RBDB:
            value->rValue = model->BSIM4V4rbdb;
            return(OK);
        case BSIM4V4_MOD_RBPB:
            value->rValue = model->BSIM4V4rbpb;
            return(OK);
        case BSIM4V4_MOD_RBSB:
            value->rValue = model->BSIM4V4rbsb;
            return(OK);
        case BSIM4V4_MOD_RBPS:
            value->rValue = model->BSIM4V4rbps;
            return(OK);
        case BSIM4V4_MOD_RBPD:
            value->rValue = model->BSIM4V4rbpd;
            return(OK);

        case BSIM4V4_MOD_CGSL:
            value->rValue = model->BSIM4V4cgsl;
            return(OK);
        case BSIM4V4_MOD_CGDL:
            value->rValue = model->BSIM4V4cgdl;
            return(OK);
        case BSIM4V4_MOD_CKAPPAS:
            value->rValue = model->BSIM4V4ckappas;
            return(OK);
        case BSIM4V4_MOD_CKAPPAD:
            value->rValue = model->BSIM4V4ckappad;
            return(OK);
        case BSIM4V4_MOD_CF:
            value->rValue = model->BSIM4V4cf;
            return(OK);
        case BSIM4V4_MOD_CLC:
            value->rValue = model->BSIM4V4clc;
            return(OK);
        case BSIM4V4_MOD_CLE:
            value->rValue = model->BSIM4V4cle;
            return(OK);
        case BSIM4V4_MOD_DWC:
            value->rValue = model->BSIM4V4dwc;
            return(OK);
        case BSIM4V4_MOD_DLC:
            value->rValue = model->BSIM4V4dlc;
            return(OK);
        case BSIM4V4_MOD_XW:
            value->rValue = model->BSIM4V4xw;
            return(OK);
        case BSIM4V4_MOD_XL:
            value->rValue = model->BSIM4V4xl;
            return(OK);
        case BSIM4V4_MOD_DLCIG:
            value->rValue = model->BSIM4V4dlcig;
            return(OK);
        case BSIM4V4_MOD_DWJ:
            value->rValue = model->BSIM4V4dwj;
            return(OK);
        case BSIM4V4_MOD_VFBCV:
            value->rValue = model->BSIM4V4vfbcv; 
            return(OK);
        case BSIM4V4_MOD_ACDE:
            value->rValue = model->BSIM4V4acde;
            return(OK);
        case BSIM4V4_MOD_MOIN:
            value->rValue = model->BSIM4V4moin;
            return(OK);
        case BSIM4V4_MOD_NOFF:
            value->rValue = model->BSIM4V4noff;
            return(OK);
        case BSIM4V4_MOD_VOFFCV:
            value->rValue = model->BSIM4V4voffcv;
            return(OK);
        case BSIM4V4_MOD_DMCG:
            value->rValue = model->BSIM4V4dmcg;
            return(OK);
        case BSIM4V4_MOD_DMCI:
            value->rValue = model->BSIM4V4dmci;
            return(OK);
        case BSIM4V4_MOD_DMDG:
            value->rValue = model->BSIM4V4dmdg;
            return(OK);
        case BSIM4V4_MOD_DMCGT:
            value->rValue = model->BSIM4V4dmcgt;
            return(OK);
        case BSIM4V4_MOD_XGW:
            value->rValue = model->BSIM4V4xgw;
            return(OK);
        case BSIM4V4_MOD_XGL:
            value->rValue = model->BSIM4V4xgl;
            return(OK);
        case BSIM4V4_MOD_RSHG:
            value->rValue = model->BSIM4V4rshg;
            return(OK);
        case BSIM4V4_MOD_NGCON:
            value->rValue = model->BSIM4V4ngcon;
            return(OK);
        case BSIM4V4_MOD_TCJ:
            value->rValue = model->BSIM4V4tcj;
            return(OK);
        case BSIM4V4_MOD_TPB:
            value->rValue = model->BSIM4V4tpb;
            return(OK);
        case BSIM4V4_MOD_TCJSW:
            value->rValue = model->BSIM4V4tcjsw;
            return(OK);
        case BSIM4V4_MOD_TPBSW:
            value->rValue = model->BSIM4V4tpbsw;
            return(OK);
        case BSIM4V4_MOD_TCJSWG:
            value->rValue = model->BSIM4V4tcjswg;
            return(OK);
        case BSIM4V4_MOD_TPBSWG:
            value->rValue = model->BSIM4V4tpbswg;
            return(OK);

	/* Length dependence */
        case  BSIM4V4_MOD_LCDSC :
          value->rValue = model->BSIM4V4lcdsc;
            return(OK);
        case  BSIM4V4_MOD_LCDSCB :
          value->rValue = model->BSIM4V4lcdscb;
            return(OK);
        case  BSIM4V4_MOD_LCDSCD :
          value->rValue = model->BSIM4V4lcdscd;
            return(OK);
        case  BSIM4V4_MOD_LCIT :
          value->rValue = model->BSIM4V4lcit;
            return(OK);
        case  BSIM4V4_MOD_LNFACTOR :
          value->rValue = model->BSIM4V4lnfactor;
            return(OK);
        case BSIM4V4_MOD_LXJ:
            value->rValue = model->BSIM4V4lxj;
            return(OK);
        case BSIM4V4_MOD_LVSAT:
            value->rValue = model->BSIM4V4lvsat;
            return(OK);
        case BSIM4V4_MOD_LAT:
            value->rValue = model->BSIM4V4lat;
            return(OK);
        case BSIM4V4_MOD_LA0:
            value->rValue = model->BSIM4V4la0;
            return(OK);
        case BSIM4V4_MOD_LAGS:
            value->rValue = model->BSIM4V4lags;
            return(OK);
        case BSIM4V4_MOD_LA1:
            value->rValue = model->BSIM4V4la1;
            return(OK);
        case BSIM4V4_MOD_LA2:
            value->rValue = model->BSIM4V4la2;
            return(OK);
        case BSIM4V4_MOD_LKETA:
            value->rValue = model->BSIM4V4lketa;
            return(OK);   
        case BSIM4V4_MOD_LNSUB:
            value->rValue = model->BSIM4V4lnsub;
            return(OK);
        case BSIM4V4_MOD_LNDEP:
            value->rValue = model->BSIM4V4lndep;
            return(OK);
        case BSIM4V4_MOD_LNSD:
            value->rValue = model->BSIM4V4lnsd;
            return(OK);
        case BSIM4V4_MOD_LNGATE:
            value->rValue = model->BSIM4V4lngate;
            return(OK);
        case BSIM4V4_MOD_LGAMMA1:
            value->rValue = model->BSIM4V4lgamma1;
            return(OK);
        case BSIM4V4_MOD_LGAMMA2:
            value->rValue = model->BSIM4V4lgamma2;
            return(OK);
        case BSIM4V4_MOD_LVBX:
            value->rValue = model->BSIM4V4lvbx;
            return(OK);
        case BSIM4V4_MOD_LVBM:
            value->rValue = model->BSIM4V4lvbm;
            return(OK);
        case BSIM4V4_MOD_LXT:
            value->rValue = model->BSIM4V4lxt;
            return(OK);
        case  BSIM4V4_MOD_LK1:
          value->rValue = model->BSIM4V4lk1;
            return(OK);
        case  BSIM4V4_MOD_LKT1:
          value->rValue = model->BSIM4V4lkt1;
            return(OK);
        case  BSIM4V4_MOD_LKT1L:
          value->rValue = model->BSIM4V4lkt1l;
            return(OK);
        case  BSIM4V4_MOD_LKT2 :
          value->rValue = model->BSIM4V4lkt2;
            return(OK);
        case  BSIM4V4_MOD_LK2 :
          value->rValue = model->BSIM4V4lk2;
            return(OK);
        case  BSIM4V4_MOD_LK3:
          value->rValue = model->BSIM4V4lk3;
            return(OK);
        case  BSIM4V4_MOD_LK3B:
          value->rValue = model->BSIM4V4lk3b;
            return(OK);
        case  BSIM4V4_MOD_LW0:
          value->rValue = model->BSIM4V4lw0;
            return(OK);
        case  BSIM4V4_MOD_LLPE0:
          value->rValue = model->BSIM4V4llpe0;
            return(OK);
        case  BSIM4V4_MOD_LLPEB:
          value->rValue = model->BSIM4V4llpeb;
            return(OK);
        case  BSIM4V4_MOD_LDVTP0:
          value->rValue = model->BSIM4V4ldvtp0;
            return(OK);
        case  BSIM4V4_MOD_LDVTP1:
          value->rValue = model->BSIM4V4ldvtp1;
            return(OK);
        case  BSIM4V4_MOD_LDVT0:                
          value->rValue = model->BSIM4V4ldvt0;
            return(OK);
        case  BSIM4V4_MOD_LDVT1 :             
          value->rValue = model->BSIM4V4ldvt1;
            return(OK);
        case  BSIM4V4_MOD_LDVT2 :             
          value->rValue = model->BSIM4V4ldvt2;
            return(OK);
        case  BSIM4V4_MOD_LDVT0W :                
          value->rValue = model->BSIM4V4ldvt0w;
            return(OK);
        case  BSIM4V4_MOD_LDVT1W :             
          value->rValue = model->BSIM4V4ldvt1w;
            return(OK);
        case  BSIM4V4_MOD_LDVT2W :             
          value->rValue = model->BSIM4V4ldvt2w;
            return(OK);
        case  BSIM4V4_MOD_LDROUT :           
          value->rValue = model->BSIM4V4ldrout;
            return(OK);
        case  BSIM4V4_MOD_LDSUB :           
          value->rValue = model->BSIM4V4ldsub;
            return(OK);
        case BSIM4V4_MOD_LVTH0:
            value->rValue = model->BSIM4V4lvth0; 
            return(OK);
        case BSIM4V4_MOD_LUA:
            value->rValue = model->BSIM4V4lua; 
            return(OK);
        case BSIM4V4_MOD_LUA1:
            value->rValue = model->BSIM4V4lua1; 
            return(OK);
        case BSIM4V4_MOD_LUB:
            value->rValue = model->BSIM4V4lub;  
            return(OK);
        case BSIM4V4_MOD_LUB1:
            value->rValue = model->BSIM4V4lub1;  
            return(OK);
        case BSIM4V4_MOD_LUC:
            value->rValue = model->BSIM4V4luc; 
            return(OK);
        case BSIM4V4_MOD_LUC1:
            value->rValue = model->BSIM4V4luc1; 
            return(OK);
        case BSIM4V4_MOD_LU0:
            value->rValue = model->BSIM4V4lu0;
            return(OK);
        case BSIM4V4_MOD_LUTE:
            value->rValue = model->BSIM4V4lute;
            return(OK);
        case BSIM4V4_MOD_LVOFF:
            value->rValue = model->BSIM4V4lvoff;
            return(OK);
        case BSIM4V4_MOD_LMINV:
            value->rValue = model->BSIM4V4lminv;
            return(OK);
        case BSIM4V4_MOD_LFPROUT:
            value->rValue = model->BSIM4V4lfprout;
            return(OK);
        case BSIM4V4_MOD_LPDITS:
            value->rValue = model->BSIM4V4lpdits;
            return(OK);
        case BSIM4V4_MOD_LPDITSD:
            value->rValue = model->BSIM4V4lpditsd;
            return(OK);
        case BSIM4V4_MOD_LDELTA:
            value->rValue = model->BSIM4V4ldelta;
            return(OK);
        case BSIM4V4_MOD_LRDSW:
            value->rValue = model->BSIM4V4lrdsw; 
            return(OK);             
        case BSIM4V4_MOD_LRDW:
            value->rValue = model->BSIM4V4lrdw;
            return(OK);
        case BSIM4V4_MOD_LRSW:
            value->rValue = model->BSIM4V4lrsw;
            return(OK);
        case BSIM4V4_MOD_LPRWB:
            value->rValue = model->BSIM4V4lprwb; 
            return(OK);             
        case BSIM4V4_MOD_LPRWG:
            value->rValue = model->BSIM4V4lprwg; 
            return(OK);             
        case BSIM4V4_MOD_LPRT:
            value->rValue = model->BSIM4V4lprt; 
            return(OK);              
        case BSIM4V4_MOD_LETA0:
            value->rValue = model->BSIM4V4leta0; 
            return(OK);               
        case BSIM4V4_MOD_LETAB:
            value->rValue = model->BSIM4V4letab; 
            return(OK);               
        case BSIM4V4_MOD_LPCLM:
            value->rValue = model->BSIM4V4lpclm; 
            return(OK);               
        case BSIM4V4_MOD_LPDIBL1:
            value->rValue = model->BSIM4V4lpdibl1; 
            return(OK);               
        case BSIM4V4_MOD_LPDIBL2:
            value->rValue = model->BSIM4V4lpdibl2; 
            return(OK);               
        case BSIM4V4_MOD_LPDIBLB:
            value->rValue = model->BSIM4V4lpdiblb; 
            return(OK);               
        case BSIM4V4_MOD_LPSCBE1:
            value->rValue = model->BSIM4V4lpscbe1; 
            return(OK);               
        case BSIM4V4_MOD_LPSCBE2:
            value->rValue = model->BSIM4V4lpscbe2; 
            return(OK);               
        case BSIM4V4_MOD_LPVAG:
            value->rValue = model->BSIM4V4lpvag; 
            return(OK);               
        case BSIM4V4_MOD_LWR:
            value->rValue = model->BSIM4V4lwr;
            return(OK);
        case BSIM4V4_MOD_LDWG:
            value->rValue = model->BSIM4V4ldwg;
            return(OK);
        case BSIM4V4_MOD_LDWB:
            value->rValue = model->BSIM4V4ldwb;
            return(OK);
        case BSIM4V4_MOD_LB0:
            value->rValue = model->BSIM4V4lb0;
            return(OK);
        case BSIM4V4_MOD_LB1:
            value->rValue = model->BSIM4V4lb1;
            return(OK);
        case BSIM4V4_MOD_LALPHA0:
            value->rValue = model->BSIM4V4lalpha0;
            return(OK);
        case BSIM4V4_MOD_LALPHA1:
            value->rValue = model->BSIM4V4lalpha1;
            return(OK);
        case BSIM4V4_MOD_LBETA0:
            value->rValue = model->BSIM4V4lbeta0;
            return(OK);
        case BSIM4V4_MOD_LAGIDL:
            value->rValue = model->BSIM4V4lagidl;
            return(OK);
        case BSIM4V4_MOD_LBGIDL:
            value->rValue = model->BSIM4V4lbgidl;
            return(OK);
        case BSIM4V4_MOD_LCGIDL:
            value->rValue = model->BSIM4V4lcgidl;
            return(OK);
        case BSIM4V4_MOD_LEGIDL:
            value->rValue = model->BSIM4V4legidl;
            return(OK);
        case BSIM4V4_MOD_LAIGC:
            value->rValue = model->BSIM4V4laigc;
            return(OK);
        case BSIM4V4_MOD_LBIGC:
            value->rValue = model->BSIM4V4lbigc;
            return(OK);
        case BSIM4V4_MOD_LCIGC:
            value->rValue = model->BSIM4V4lcigc;
            return(OK);
        case BSIM4V4_MOD_LAIGSD:
            value->rValue = model->BSIM4V4laigsd;
            return(OK);
        case BSIM4V4_MOD_LBIGSD:
            value->rValue = model->BSIM4V4lbigsd;
            return(OK);
        case BSIM4V4_MOD_LCIGSD:
            value->rValue = model->BSIM4V4lcigsd;
            return(OK);
        case BSIM4V4_MOD_LAIGBACC:
            value->rValue = model->BSIM4V4laigbacc;
            return(OK);
        case BSIM4V4_MOD_LBIGBACC:
            value->rValue = model->BSIM4V4lbigbacc;
            return(OK);
        case BSIM4V4_MOD_LCIGBACC:
            value->rValue = model->BSIM4V4lcigbacc;
            return(OK);
        case BSIM4V4_MOD_LAIGBINV:
            value->rValue = model->BSIM4V4laigbinv;
            return(OK);
        case BSIM4V4_MOD_LBIGBINV:
            value->rValue = model->BSIM4V4lbigbinv;
            return(OK);
        case BSIM4V4_MOD_LCIGBINV:
            value->rValue = model->BSIM4V4lcigbinv;
            return(OK);
        case BSIM4V4_MOD_LNIGC:
            value->rValue = model->BSIM4V4lnigc;
            return(OK);
        case BSIM4V4_MOD_LNIGBACC:
            value->rValue = model->BSIM4V4lnigbacc;
            return(OK);
        case BSIM4V4_MOD_LNIGBINV:
            value->rValue = model->BSIM4V4lnigbinv;
            return(OK);
        case BSIM4V4_MOD_LNTOX:
            value->rValue = model->BSIM4V4lntox;
            return(OK);
        case BSIM4V4_MOD_LEIGBINV:
            value->rValue = model->BSIM4V4leigbinv;
            return(OK);
        case BSIM4V4_MOD_LPIGCD:
            value->rValue = model->BSIM4V4lpigcd;
            return(OK);
        case BSIM4V4_MOD_LPOXEDGE:
            value->rValue = model->BSIM4V4lpoxedge;
            return(OK);
        case BSIM4V4_MOD_LPHIN:
            value->rValue = model->BSIM4V4lphin;
            return(OK);
        case BSIM4V4_MOD_LXRCRG1:
            value->rValue = model->BSIM4V4lxrcrg1;
            return(OK);
        case BSIM4V4_MOD_LXRCRG2:
            value->rValue = model->BSIM4V4lxrcrg2;
            return(OK);
        case BSIM4V4_MOD_LEU:
            value->rValue = model->BSIM4V4leu;
            return(OK);
        case BSIM4V4_MOD_LVFB:
            value->rValue = model->BSIM4V4lvfb;
            return(OK);

        case BSIM4V4_MOD_LCGSL:
            value->rValue = model->BSIM4V4lcgsl;
            return(OK);
        case BSIM4V4_MOD_LCGDL:
            value->rValue = model->BSIM4V4lcgdl;
            return(OK);
        case BSIM4V4_MOD_LCKAPPAS:
            value->rValue = model->BSIM4V4lckappas;
            return(OK);
        case BSIM4V4_MOD_LCKAPPAD:
            value->rValue = model->BSIM4V4lckappad;
            return(OK);
        case BSIM4V4_MOD_LCF:
            value->rValue = model->BSIM4V4lcf;
            return(OK);
        case BSIM4V4_MOD_LCLC:
            value->rValue = model->BSIM4V4lclc;
            return(OK);
        case BSIM4V4_MOD_LCLE:
            value->rValue = model->BSIM4V4lcle;
            return(OK);
        case BSIM4V4_MOD_LVFBCV:
            value->rValue = model->BSIM4V4lvfbcv;
            return(OK);
        case BSIM4V4_MOD_LACDE:
            value->rValue = model->BSIM4V4lacde;
            return(OK);
        case BSIM4V4_MOD_LMOIN:
            value->rValue = model->BSIM4V4lmoin;
            return(OK);
        case BSIM4V4_MOD_LNOFF:
            value->rValue = model->BSIM4V4lnoff;
            return(OK);
        case BSIM4V4_MOD_LVOFFCV:
            value->rValue = model->BSIM4V4lvoffcv;
            return(OK);

	/* Width dependence */
        case  BSIM4V4_MOD_WCDSC :
          value->rValue = model->BSIM4V4wcdsc;
            return(OK);
        case  BSIM4V4_MOD_WCDSCB :
          value->rValue = model->BSIM4V4wcdscb;
            return(OK);
        case  BSIM4V4_MOD_WCDSCD :
          value->rValue = model->BSIM4V4wcdscd;
            return(OK);
        case  BSIM4V4_MOD_WCIT :
          value->rValue = model->BSIM4V4wcit;
            return(OK);
        case  BSIM4V4_MOD_WNFACTOR :
          value->rValue = model->BSIM4V4wnfactor;
            return(OK);
        case BSIM4V4_MOD_WXJ:
            value->rValue = model->BSIM4V4wxj;
            return(OK);
        case BSIM4V4_MOD_WVSAT:
            value->rValue = model->BSIM4V4wvsat;
            return(OK);
        case BSIM4V4_MOD_WAT:
            value->rValue = model->BSIM4V4wat;
            return(OK);
        case BSIM4V4_MOD_WA0:
            value->rValue = model->BSIM4V4wa0;
            return(OK);
        case BSIM4V4_MOD_WAGS:
            value->rValue = model->BSIM4V4wags;
            return(OK);
        case BSIM4V4_MOD_WA1:
            value->rValue = model->BSIM4V4wa1;
            return(OK);
        case BSIM4V4_MOD_WA2:
            value->rValue = model->BSIM4V4wa2;
            return(OK);
        case BSIM4V4_MOD_WKETA:
            value->rValue = model->BSIM4V4wketa;
            return(OK);   
        case BSIM4V4_MOD_WNSUB:
            value->rValue = model->BSIM4V4wnsub;
            return(OK);
        case BSIM4V4_MOD_WNDEP:
            value->rValue = model->BSIM4V4wndep;
            return(OK);
        case BSIM4V4_MOD_WNSD:
            value->rValue = model->BSIM4V4wnsd;
            return(OK);
        case BSIM4V4_MOD_WNGATE:
            value->rValue = model->BSIM4V4wngate;
            return(OK);
        case BSIM4V4_MOD_WGAMMA1:
            value->rValue = model->BSIM4V4wgamma1;
            return(OK);
        case BSIM4V4_MOD_WGAMMA2:
            value->rValue = model->BSIM4V4wgamma2;
            return(OK);
        case BSIM4V4_MOD_WVBX:
            value->rValue = model->BSIM4V4wvbx;
            return(OK);
        case BSIM4V4_MOD_WVBM:
            value->rValue = model->BSIM4V4wvbm;
            return(OK);
        case BSIM4V4_MOD_WXT:
            value->rValue = model->BSIM4V4wxt;
            return(OK);
        case  BSIM4V4_MOD_WK1:
          value->rValue = model->BSIM4V4wk1;
            return(OK);
        case  BSIM4V4_MOD_WKT1:
          value->rValue = model->BSIM4V4wkt1;
            return(OK);
        case  BSIM4V4_MOD_WKT1L:
          value->rValue = model->BSIM4V4wkt1l;
            return(OK);
        case  BSIM4V4_MOD_WKT2 :
          value->rValue = model->BSIM4V4wkt2;
            return(OK);
        case  BSIM4V4_MOD_WK2 :
          value->rValue = model->BSIM4V4wk2;
            return(OK);
        case  BSIM4V4_MOD_WK3:
          value->rValue = model->BSIM4V4wk3;
            return(OK);
        case  BSIM4V4_MOD_WK3B:
          value->rValue = model->BSIM4V4wk3b;
            return(OK);
        case  BSIM4V4_MOD_WW0:
          value->rValue = model->BSIM4V4ww0;
            return(OK);
        case  BSIM4V4_MOD_WLPE0:
          value->rValue = model->BSIM4V4wlpe0;
            return(OK);
        case  BSIM4V4_MOD_WDVTP0:
          value->rValue = model->BSIM4V4wdvtp0;
            return(OK);
        case  BSIM4V4_MOD_WDVTP1:
          value->rValue = model->BSIM4V4wdvtp1;
            return(OK);
        case  BSIM4V4_MOD_WLPEB:
          value->rValue = model->BSIM4V4wlpeb;
            return(OK);
        case  BSIM4V4_MOD_WDVT0:                
          value->rValue = model->BSIM4V4wdvt0;
            return(OK);
        case  BSIM4V4_MOD_WDVT1 :             
          value->rValue = model->BSIM4V4wdvt1;
            return(OK);
        case  BSIM4V4_MOD_WDVT2 :             
          value->rValue = model->BSIM4V4wdvt2;
            return(OK);
        case  BSIM4V4_MOD_WDVT0W :                
          value->rValue = model->BSIM4V4wdvt0w;
            return(OK);
        case  BSIM4V4_MOD_WDVT1W :             
          value->rValue = model->BSIM4V4wdvt1w;
            return(OK);
        case  BSIM4V4_MOD_WDVT2W :             
          value->rValue = model->BSIM4V4wdvt2w;
            return(OK);
        case  BSIM4V4_MOD_WDROUT :           
          value->rValue = model->BSIM4V4wdrout;
            return(OK);
        case  BSIM4V4_MOD_WDSUB :           
          value->rValue = model->BSIM4V4wdsub;
            return(OK);
        case BSIM4V4_MOD_WVTH0:
            value->rValue = model->BSIM4V4wvth0; 
            return(OK);
        case BSIM4V4_MOD_WUA:
            value->rValue = model->BSIM4V4wua; 
            return(OK);
        case BSIM4V4_MOD_WUA1:
            value->rValue = model->BSIM4V4wua1; 
            return(OK);
        case BSIM4V4_MOD_WUB:
            value->rValue = model->BSIM4V4wub;  
            return(OK);
        case BSIM4V4_MOD_WUB1:
            value->rValue = model->BSIM4V4wub1;  
            return(OK);
        case BSIM4V4_MOD_WUC:
            value->rValue = model->BSIM4V4wuc; 
            return(OK);
        case BSIM4V4_MOD_WUC1:
            value->rValue = model->BSIM4V4wuc1; 
            return(OK);
        case BSIM4V4_MOD_WU0:
            value->rValue = model->BSIM4V4wu0;
            return(OK);
        case BSIM4V4_MOD_WUTE:
            value->rValue = model->BSIM4V4wute;
            return(OK);
        case BSIM4V4_MOD_WVOFF:
            value->rValue = model->BSIM4V4wvoff;
            return(OK);
        case BSIM4V4_MOD_WMINV:
            value->rValue = model->BSIM4V4wminv;
            return(OK);
        case BSIM4V4_MOD_WFPROUT:
            value->rValue = model->BSIM4V4wfprout;
            return(OK);
        case BSIM4V4_MOD_WPDITS:
            value->rValue = model->BSIM4V4wpdits;
            return(OK);
        case BSIM4V4_MOD_WPDITSD:
            value->rValue = model->BSIM4V4wpditsd;
            return(OK);
        case BSIM4V4_MOD_WDELTA:
            value->rValue = model->BSIM4V4wdelta;
            return(OK);
        case BSIM4V4_MOD_WRDSW:
            value->rValue = model->BSIM4V4wrdsw; 
            return(OK);             
        case BSIM4V4_MOD_WRDW:
            value->rValue = model->BSIM4V4wrdw;
            return(OK);
        case BSIM4V4_MOD_WRSW:
            value->rValue = model->BSIM4V4wrsw;
            return(OK);
        case BSIM4V4_MOD_WPRWB:
            value->rValue = model->BSIM4V4wprwb; 
            return(OK);             
        case BSIM4V4_MOD_WPRWG:
            value->rValue = model->BSIM4V4wprwg; 
            return(OK);             
        case BSIM4V4_MOD_WPRT:
            value->rValue = model->BSIM4V4wprt; 
            return(OK);              
        case BSIM4V4_MOD_WETA0:
            value->rValue = model->BSIM4V4weta0; 
            return(OK);               
        case BSIM4V4_MOD_WETAB:
            value->rValue = model->BSIM4V4wetab; 
            return(OK);               
        case BSIM4V4_MOD_WPCLM:
            value->rValue = model->BSIM4V4wpclm; 
            return(OK);               
        case BSIM4V4_MOD_WPDIBL1:
            value->rValue = model->BSIM4V4wpdibl1; 
            return(OK);               
        case BSIM4V4_MOD_WPDIBL2:
            value->rValue = model->BSIM4V4wpdibl2; 
            return(OK);               
        case BSIM4V4_MOD_WPDIBLB:
            value->rValue = model->BSIM4V4wpdiblb; 
            return(OK);               
        case BSIM4V4_MOD_WPSCBE1:
            value->rValue = model->BSIM4V4wpscbe1; 
            return(OK);               
        case BSIM4V4_MOD_WPSCBE2:
            value->rValue = model->BSIM4V4wpscbe2; 
            return(OK);               
        case BSIM4V4_MOD_WPVAG:
            value->rValue = model->BSIM4V4wpvag; 
            return(OK);               
        case BSIM4V4_MOD_WWR:
            value->rValue = model->BSIM4V4wwr;
            return(OK);
        case BSIM4V4_MOD_WDWG:
            value->rValue = model->BSIM4V4wdwg;
            return(OK);
        case BSIM4V4_MOD_WDWB:
            value->rValue = model->BSIM4V4wdwb;
            return(OK);
        case BSIM4V4_MOD_WB0:
            value->rValue = model->BSIM4V4wb0;
            return(OK);
        case BSIM4V4_MOD_WB1:
            value->rValue = model->BSIM4V4wb1;
            return(OK);
        case BSIM4V4_MOD_WALPHA0:
            value->rValue = model->BSIM4V4walpha0;
            return(OK);
        case BSIM4V4_MOD_WALPHA1:
            value->rValue = model->BSIM4V4walpha1;
            return(OK);
        case BSIM4V4_MOD_WBETA0:
            value->rValue = model->BSIM4V4wbeta0;
            return(OK);
        case BSIM4V4_MOD_WAGIDL:
            value->rValue = model->BSIM4V4wagidl;
            return(OK);
        case BSIM4V4_MOD_WBGIDL:
            value->rValue = model->BSIM4V4wbgidl;
            return(OK);
        case BSIM4V4_MOD_WCGIDL:
            value->rValue = model->BSIM4V4wcgidl;
            return(OK);
        case BSIM4V4_MOD_WEGIDL:
            value->rValue = model->BSIM4V4wegidl;
            return(OK);
        case BSIM4V4_MOD_WAIGC:
            value->rValue = model->BSIM4V4waigc;
            return(OK);
        case BSIM4V4_MOD_WBIGC:
            value->rValue = model->BSIM4V4wbigc;
            return(OK);
        case BSIM4V4_MOD_WCIGC:
            value->rValue = model->BSIM4V4wcigc;
            return(OK);
        case BSIM4V4_MOD_WAIGSD:
            value->rValue = model->BSIM4V4waigsd;
            return(OK);
        case BSIM4V4_MOD_WBIGSD:
            value->rValue = model->BSIM4V4wbigsd;
            return(OK);
        case BSIM4V4_MOD_WCIGSD:
            value->rValue = model->BSIM4V4wcigsd;
            return(OK);
        case BSIM4V4_MOD_WAIGBACC:
            value->rValue = model->BSIM4V4waigbacc;
            return(OK);
        case BSIM4V4_MOD_WBIGBACC:
            value->rValue = model->BSIM4V4wbigbacc;
            return(OK);
        case BSIM4V4_MOD_WCIGBACC:
            value->rValue = model->BSIM4V4wcigbacc;
            return(OK);
        case BSIM4V4_MOD_WAIGBINV:
            value->rValue = model->BSIM4V4waigbinv;
            return(OK);
        case BSIM4V4_MOD_WBIGBINV:
            value->rValue = model->BSIM4V4wbigbinv;
            return(OK);
        case BSIM4V4_MOD_WCIGBINV:
            value->rValue = model->BSIM4V4wcigbinv;
            return(OK);
        case BSIM4V4_MOD_WNIGC:
            value->rValue = model->BSIM4V4wnigc;
            return(OK);
        case BSIM4V4_MOD_WNIGBACC:
            value->rValue = model->BSIM4V4wnigbacc;
            return(OK);
        case BSIM4V4_MOD_WNIGBINV:
            value->rValue = model->BSIM4V4wnigbinv;
            return(OK);
        case BSIM4V4_MOD_WNTOX:
            value->rValue = model->BSIM4V4wntox;
            return(OK);
        case BSIM4V4_MOD_WEIGBINV:
            value->rValue = model->BSIM4V4weigbinv;
            return(OK);
        case BSIM4V4_MOD_WPIGCD:
            value->rValue = model->BSIM4V4wpigcd;
            return(OK);
        case BSIM4V4_MOD_WPOXEDGE:
            value->rValue = model->BSIM4V4wpoxedge;
            return(OK);
        case BSIM4V4_MOD_WPHIN:
            value->rValue = model->BSIM4V4wphin;
            return(OK);
        case BSIM4V4_MOD_WXRCRG1:
            value->rValue = model->BSIM4V4wxrcrg1;
            return(OK);
        case BSIM4V4_MOD_WXRCRG2:
            value->rValue = model->BSIM4V4wxrcrg2;
            return(OK);
        case BSIM4V4_MOD_WEU:
            value->rValue = model->BSIM4V4weu;
            return(OK);
        case BSIM4V4_MOD_WVFB:
            value->rValue = model->BSIM4V4wvfb;
            return(OK);

        case BSIM4V4_MOD_WCGSL:
            value->rValue = model->BSIM4V4wcgsl;
            return(OK);
        case BSIM4V4_MOD_WCGDL:
            value->rValue = model->BSIM4V4wcgdl;
            return(OK);
        case BSIM4V4_MOD_WCKAPPAS:
            value->rValue = model->BSIM4V4wckappas;
            return(OK);
        case BSIM4V4_MOD_WCKAPPAD:
            value->rValue = model->BSIM4V4wckappad;
            return(OK);
        case BSIM4V4_MOD_WCF:
            value->rValue = model->BSIM4V4wcf;
            return(OK);
        case BSIM4V4_MOD_WCLC:
            value->rValue = model->BSIM4V4wclc;
            return(OK);
        case BSIM4V4_MOD_WCLE:
            value->rValue = model->BSIM4V4wcle;
            return(OK);
        case BSIM4V4_MOD_WVFBCV:
            value->rValue = model->BSIM4V4wvfbcv;
            return(OK);
        case BSIM4V4_MOD_WACDE:
            value->rValue = model->BSIM4V4wacde;
            return(OK);
        case BSIM4V4_MOD_WMOIN:
            value->rValue = model->BSIM4V4wmoin;
            return(OK);
        case BSIM4V4_MOD_WNOFF:
            value->rValue = model->BSIM4V4wnoff;
            return(OK);
        case BSIM4V4_MOD_WVOFFCV:
            value->rValue = model->BSIM4V4wvoffcv;
            return(OK);

	/* Cross-term dependence */
        case  BSIM4V4_MOD_PCDSC :
          value->rValue = model->BSIM4V4pcdsc;
            return(OK);
        case  BSIM4V4_MOD_PCDSCB :
          value->rValue = model->BSIM4V4pcdscb;
            return(OK);
        case  BSIM4V4_MOD_PCDSCD :
          value->rValue = model->BSIM4V4pcdscd;
            return(OK);
         case  BSIM4V4_MOD_PCIT :
          value->rValue = model->BSIM4V4pcit;
            return(OK);
        case  BSIM4V4_MOD_PNFACTOR :
          value->rValue = model->BSIM4V4pnfactor;
            return(OK);
        case BSIM4V4_MOD_PXJ:
            value->rValue = model->BSIM4V4pxj;
            return(OK);
        case BSIM4V4_MOD_PVSAT:
            value->rValue = model->BSIM4V4pvsat;
            return(OK);
        case BSIM4V4_MOD_PAT:
            value->rValue = model->BSIM4V4pat;
            return(OK);
        case BSIM4V4_MOD_PA0:
            value->rValue = model->BSIM4V4pa0;
            return(OK);
        case BSIM4V4_MOD_PAGS:
            value->rValue = model->BSIM4V4pags;
            return(OK);
        case BSIM4V4_MOD_PA1:
            value->rValue = model->BSIM4V4pa1;
            return(OK);
        case BSIM4V4_MOD_PA2:
            value->rValue = model->BSIM4V4pa2;
            return(OK);
        case BSIM4V4_MOD_PKETA:
            value->rValue = model->BSIM4V4pketa;
            return(OK);   
        case BSIM4V4_MOD_PNSUB:
            value->rValue = model->BSIM4V4pnsub;
            return(OK);
        case BSIM4V4_MOD_PNDEP:
            value->rValue = model->BSIM4V4pndep;
            return(OK);
        case BSIM4V4_MOD_PNSD:
            value->rValue = model->BSIM4V4pnsd;
            return(OK);
        case BSIM4V4_MOD_PNGATE:
            value->rValue = model->BSIM4V4pngate;
            return(OK);
        case BSIM4V4_MOD_PGAMMA1:
            value->rValue = model->BSIM4V4pgamma1;
            return(OK);
        case BSIM4V4_MOD_PGAMMA2:
            value->rValue = model->BSIM4V4pgamma2;
            return(OK);
        case BSIM4V4_MOD_PVBX:
            value->rValue = model->BSIM4V4pvbx;
            return(OK);
        case BSIM4V4_MOD_PVBM:
            value->rValue = model->BSIM4V4pvbm;
            return(OK);
        case BSIM4V4_MOD_PXT:
            value->rValue = model->BSIM4V4pxt;
            return(OK);
        case  BSIM4V4_MOD_PK1:
          value->rValue = model->BSIM4V4pk1;
            return(OK);
        case  BSIM4V4_MOD_PKT1:
          value->rValue = model->BSIM4V4pkt1;
            return(OK);
        case  BSIM4V4_MOD_PKT1L:
          value->rValue = model->BSIM4V4pkt1l;
            return(OK);
        case  BSIM4V4_MOD_PKT2 :
          value->rValue = model->BSIM4V4pkt2;
            return(OK);
        case  BSIM4V4_MOD_PK2 :
          value->rValue = model->BSIM4V4pk2;
            return(OK);
        case  BSIM4V4_MOD_PK3:
          value->rValue = model->BSIM4V4pk3;
            return(OK);
        case  BSIM4V4_MOD_PK3B:
          value->rValue = model->BSIM4V4pk3b;
            return(OK);
        case  BSIM4V4_MOD_PW0:
          value->rValue = model->BSIM4V4pw0;
            return(OK);
        case  BSIM4V4_MOD_PLPE0:
          value->rValue = model->BSIM4V4plpe0;
            return(OK);
        case  BSIM4V4_MOD_PLPEB:
          value->rValue = model->BSIM4V4plpeb;
            return(OK);
        case  BSIM4V4_MOD_PDVTP0:
          value->rValue = model->BSIM4V4pdvtp0;
            return(OK);
        case  BSIM4V4_MOD_PDVTP1:
          value->rValue = model->BSIM4V4pdvtp1;
            return(OK);
        case  BSIM4V4_MOD_PDVT0 :                
          value->rValue = model->BSIM4V4pdvt0;
            return(OK);
        case  BSIM4V4_MOD_PDVT1 :             
          value->rValue = model->BSIM4V4pdvt1;
            return(OK);
        case  BSIM4V4_MOD_PDVT2 :             
          value->rValue = model->BSIM4V4pdvt2;
            return(OK);
        case  BSIM4V4_MOD_PDVT0W :                
          value->rValue = model->BSIM4V4pdvt0w;
            return(OK);
        case  BSIM4V4_MOD_PDVT1W :             
          value->rValue = model->BSIM4V4pdvt1w;
            return(OK);
        case  BSIM4V4_MOD_PDVT2W :             
          value->rValue = model->BSIM4V4pdvt2w;
            return(OK);
        case  BSIM4V4_MOD_PDROUT :           
          value->rValue = model->BSIM4V4pdrout;
            return(OK);
        case  BSIM4V4_MOD_PDSUB :           
          value->rValue = model->BSIM4V4pdsub;
            return(OK);
        case BSIM4V4_MOD_PVTH0:
            value->rValue = model->BSIM4V4pvth0; 
            return(OK);
        case BSIM4V4_MOD_PUA:
            value->rValue = model->BSIM4V4pua; 
            return(OK);
        case BSIM4V4_MOD_PUA1:
            value->rValue = model->BSIM4V4pua1; 
            return(OK);
        case BSIM4V4_MOD_PUB:
            value->rValue = model->BSIM4V4pub;  
            return(OK);
        case BSIM4V4_MOD_PUB1:
            value->rValue = model->BSIM4V4pub1;  
            return(OK);
        case BSIM4V4_MOD_PUC:
            value->rValue = model->BSIM4V4puc; 
            return(OK);
        case BSIM4V4_MOD_PUC1:
            value->rValue = model->BSIM4V4puc1; 
            return(OK);
        case BSIM4V4_MOD_PU0:
            value->rValue = model->BSIM4V4pu0;
            return(OK);
        case BSIM4V4_MOD_PUTE:
            value->rValue = model->BSIM4V4pute;
            return(OK);
        case BSIM4V4_MOD_PVOFF:
            value->rValue = model->BSIM4V4pvoff;
            return(OK);
        case BSIM4V4_MOD_PMINV:
            value->rValue = model->BSIM4V4pminv;
            return(OK);
        case BSIM4V4_MOD_PFPROUT:
            value->rValue = model->BSIM4V4pfprout;
            return(OK);
        case BSIM4V4_MOD_PPDITS:
            value->rValue = model->BSIM4V4ppdits;
            return(OK);
        case BSIM4V4_MOD_PPDITSD:
            value->rValue = model->BSIM4V4ppditsd;
            return(OK);
        case BSIM4V4_MOD_PDELTA:
            value->rValue = model->BSIM4V4pdelta;
            return(OK);
        case BSIM4V4_MOD_PRDSW:
            value->rValue = model->BSIM4V4prdsw; 
            return(OK);             
        case BSIM4V4_MOD_PRDW:
            value->rValue = model->BSIM4V4prdw;
            return(OK);
        case BSIM4V4_MOD_PRSW:
            value->rValue = model->BSIM4V4prsw;
            return(OK);
        case BSIM4V4_MOD_PPRWB:
            value->rValue = model->BSIM4V4pprwb; 
            return(OK);             
        case BSIM4V4_MOD_PPRWG:
            value->rValue = model->BSIM4V4pprwg; 
            return(OK);             
        case BSIM4V4_MOD_PPRT:
            value->rValue = model->BSIM4V4pprt; 
            return(OK);              
        case BSIM4V4_MOD_PETA0:
            value->rValue = model->BSIM4V4peta0; 
            return(OK);               
        case BSIM4V4_MOD_PETAB:
            value->rValue = model->BSIM4V4petab; 
            return(OK);               
        case BSIM4V4_MOD_PPCLM:
            value->rValue = model->BSIM4V4ppclm; 
            return(OK);               
        case BSIM4V4_MOD_PPDIBL1:
            value->rValue = model->BSIM4V4ppdibl1; 
            return(OK);               
        case BSIM4V4_MOD_PPDIBL2:
            value->rValue = model->BSIM4V4ppdibl2; 
            return(OK);               
        case BSIM4V4_MOD_PPDIBLB:
            value->rValue = model->BSIM4V4ppdiblb; 
            return(OK);               
        case BSIM4V4_MOD_PPSCBE1:
            value->rValue = model->BSIM4V4ppscbe1; 
            return(OK);               
        case BSIM4V4_MOD_PPSCBE2:
            value->rValue = model->BSIM4V4ppscbe2; 
            return(OK);               
        case BSIM4V4_MOD_PPVAG:
            value->rValue = model->BSIM4V4ppvag; 
            return(OK);               
        case BSIM4V4_MOD_PWR:
            value->rValue = model->BSIM4V4pwr;
            return(OK);
        case BSIM4V4_MOD_PDWG:
            value->rValue = model->BSIM4V4pdwg;
            return(OK);
        case BSIM4V4_MOD_PDWB:
            value->rValue = model->BSIM4V4pdwb;
            return(OK);
        case BSIM4V4_MOD_PB0:
            value->rValue = model->BSIM4V4pb0;
            return(OK);
        case BSIM4V4_MOD_PB1:
            value->rValue = model->BSIM4V4pb1;
            return(OK);
        case BSIM4V4_MOD_PALPHA0:
            value->rValue = model->BSIM4V4palpha0;
            return(OK);
        case BSIM4V4_MOD_PALPHA1:
            value->rValue = model->BSIM4V4palpha1;
            return(OK);
        case BSIM4V4_MOD_PBETA0:
            value->rValue = model->BSIM4V4pbeta0;
            return(OK);
        case BSIM4V4_MOD_PAGIDL:
            value->rValue = model->BSIM4V4pagidl;
            return(OK);
        case BSIM4V4_MOD_PBGIDL:
            value->rValue = model->BSIM4V4pbgidl;
            return(OK);
        case BSIM4V4_MOD_PCGIDL:
            value->rValue = model->BSIM4V4pcgidl;
            return(OK);
        case BSIM4V4_MOD_PEGIDL:
            value->rValue = model->BSIM4V4pegidl;
            return(OK);
        case BSIM4V4_MOD_PAIGC:
            value->rValue = model->BSIM4V4paigc;
            return(OK);
        case BSIM4V4_MOD_PBIGC:
            value->rValue = model->BSIM4V4pbigc;
            return(OK);
        case BSIM4V4_MOD_PCIGC:
            value->rValue = model->BSIM4V4pcigc;
            return(OK);
        case BSIM4V4_MOD_PAIGSD:
            value->rValue = model->BSIM4V4paigsd;
            return(OK);
        case BSIM4V4_MOD_PBIGSD:
            value->rValue = model->BSIM4V4pbigsd;
            return(OK);
        case BSIM4V4_MOD_PCIGSD:
            value->rValue = model->BSIM4V4pcigsd;
            return(OK);
        case BSIM4V4_MOD_PAIGBACC:
            value->rValue = model->BSIM4V4paigbacc;
            return(OK);
        case BSIM4V4_MOD_PBIGBACC:
            value->rValue = model->BSIM4V4pbigbacc;
            return(OK);
        case BSIM4V4_MOD_PCIGBACC:
            value->rValue = model->BSIM4V4pcigbacc;
            return(OK);
        case BSIM4V4_MOD_PAIGBINV:
            value->rValue = model->BSIM4V4paigbinv;
            return(OK);
        case BSIM4V4_MOD_PBIGBINV:
            value->rValue = model->BSIM4V4pbigbinv;
            return(OK);
        case BSIM4V4_MOD_PCIGBINV:
            value->rValue = model->BSIM4V4pcigbinv;
            return(OK);
        case BSIM4V4_MOD_PNIGC:
            value->rValue = model->BSIM4V4pnigc;
            return(OK);
        case BSIM4V4_MOD_PNIGBACC:
            value->rValue = model->BSIM4V4pnigbacc;
            return(OK);
        case BSIM4V4_MOD_PNIGBINV:
            value->rValue = model->BSIM4V4pnigbinv;
            return(OK);
        case BSIM4V4_MOD_PNTOX:
            value->rValue = model->BSIM4V4pntox;
            return(OK);
        case BSIM4V4_MOD_PEIGBINV:
            value->rValue = model->BSIM4V4peigbinv;
            return(OK);
        case BSIM4V4_MOD_PPIGCD:
            value->rValue = model->BSIM4V4ppigcd;
            return(OK);
        case BSIM4V4_MOD_PPOXEDGE:
            value->rValue = model->BSIM4V4ppoxedge;
            return(OK);
        case BSIM4V4_MOD_PPHIN:
            value->rValue = model->BSIM4V4pphin;
            return(OK);
        case BSIM4V4_MOD_PXRCRG1:
            value->rValue = model->BSIM4V4pxrcrg1;
            return(OK);
        case BSIM4V4_MOD_PXRCRG2:
            value->rValue = model->BSIM4V4pxrcrg2;
            return(OK);
        case BSIM4V4_MOD_PEU:
            value->rValue = model->BSIM4V4peu;
            return(OK);
        case BSIM4V4_MOD_PVFB:
            value->rValue = model->BSIM4V4pvfb;
            return(OK);

        case BSIM4V4_MOD_PCGSL:
            value->rValue = model->BSIM4V4pcgsl;
            return(OK);
        case BSIM4V4_MOD_PCGDL:
            value->rValue = model->BSIM4V4pcgdl;
            return(OK);
        case BSIM4V4_MOD_PCKAPPAS:
            value->rValue = model->BSIM4V4pckappas;
            return(OK);
        case BSIM4V4_MOD_PCKAPPAD:
            value->rValue = model->BSIM4V4pckappad;
            return(OK);
        case BSIM4V4_MOD_PCF:
            value->rValue = model->BSIM4V4pcf;
            return(OK);
        case BSIM4V4_MOD_PCLC:
            value->rValue = model->BSIM4V4pclc;
            return(OK);
        case BSIM4V4_MOD_PCLE:
            value->rValue = model->BSIM4V4pcle;
            return(OK);
        case BSIM4V4_MOD_PVFBCV:
            value->rValue = model->BSIM4V4pvfbcv;
            return(OK);
        case BSIM4V4_MOD_PACDE:
            value->rValue = model->BSIM4V4pacde;
            return(OK);
        case BSIM4V4_MOD_PMOIN:
            value->rValue = model->BSIM4V4pmoin;
            return(OK);
        case BSIM4V4_MOD_PNOFF:
            value->rValue = model->BSIM4V4pnoff;
            return(OK);
        case BSIM4V4_MOD_PVOFFCV:
            value->rValue = model->BSIM4V4pvoffcv;
            return(OK);

        case  BSIM4V4_MOD_TNOM :
          value->rValue = model->BSIM4V4tnom;
            return(OK);
        case BSIM4V4_MOD_CGSO:
            value->rValue = model->BSIM4V4cgso; 
            return(OK);
        case BSIM4V4_MOD_CGDO:
            value->rValue = model->BSIM4V4cgdo; 
            return(OK);
        case BSIM4V4_MOD_CGBO:
            value->rValue = model->BSIM4V4cgbo; 
            return(OK);
        case BSIM4V4_MOD_XPART:
            value->rValue = model->BSIM4V4xpart; 
            return(OK);
        case BSIM4V4_MOD_RSH:
            value->rValue = model->BSIM4V4sheetResistance; 
            return(OK);
        case BSIM4V4_MOD_JSS:
            value->rValue = model->BSIM4V4SjctSatCurDensity; 
            return(OK);
        case BSIM4V4_MOD_JSWS:
            value->rValue = model->BSIM4V4SjctSidewallSatCurDensity; 
            return(OK);
        case BSIM4V4_MOD_JSWGS:
            value->rValue = model->BSIM4V4SjctGateSidewallSatCurDensity;
            return(OK);
        case BSIM4V4_MOD_PBS:
            value->rValue = model->BSIM4V4SbulkJctPotential; 
            return(OK);
        case BSIM4V4_MOD_MJS:
            value->rValue = model->BSIM4V4SbulkJctBotGradingCoeff; 
            return(OK);
        case BSIM4V4_MOD_PBSWS:
            value->rValue = model->BSIM4V4SsidewallJctPotential; 
            return(OK);
        case BSIM4V4_MOD_MJSWS:
            value->rValue = model->BSIM4V4SbulkJctSideGradingCoeff; 
            return(OK);
        case BSIM4V4_MOD_CJS:
            value->rValue = model->BSIM4V4SunitAreaJctCap; 
            return(OK);
        case BSIM4V4_MOD_CJSWS:
            value->rValue = model->BSIM4V4SunitLengthSidewallJctCap; 
            return(OK);
        case BSIM4V4_MOD_PBSWGS:
            value->rValue = model->BSIM4V4SGatesidewallJctPotential; 
            return(OK);
        case BSIM4V4_MOD_MJSWGS:
            value->rValue = model->BSIM4V4SbulkJctGateSideGradingCoeff; 
            return(OK);
        case BSIM4V4_MOD_CJSWGS:
            value->rValue = model->BSIM4V4SunitLengthGateSidewallJctCap; 
            return(OK);
        case BSIM4V4_MOD_NJS:
            value->rValue = model->BSIM4V4SjctEmissionCoeff; 
            return(OK);
        case BSIM4V4_MOD_XTIS:
            value->rValue = model->BSIM4V4SjctTempExponent; 
            return(OK);
        case BSIM4V4_MOD_JSD:
            value->rValue = model->BSIM4V4DjctSatCurDensity;
            return(OK);
        case BSIM4V4_MOD_JSWD:
            value->rValue = model->BSIM4V4DjctSidewallSatCurDensity;
            return(OK);
        case BSIM4V4_MOD_JSWGD:
            value->rValue = model->BSIM4V4DjctGateSidewallSatCurDensity;
            return(OK);
        case BSIM4V4_MOD_PBD:
            value->rValue = model->BSIM4V4DbulkJctPotential;
            return(OK);
        case BSIM4V4_MOD_MJD:
            value->rValue = model->BSIM4V4DbulkJctBotGradingCoeff;
            return(OK);
        case BSIM4V4_MOD_PBSWD:
            value->rValue = model->BSIM4V4DsidewallJctPotential;
            return(OK);
        case BSIM4V4_MOD_MJSWD:
            value->rValue = model->BSIM4V4DbulkJctSideGradingCoeff;
            return(OK);
        case BSIM4V4_MOD_CJD:
            value->rValue = model->BSIM4V4DunitAreaJctCap;
            return(OK);
        case BSIM4V4_MOD_CJSWD:
            value->rValue = model->BSIM4V4DunitLengthSidewallJctCap;
            return(OK);
        case BSIM4V4_MOD_PBSWGD:
            value->rValue = model->BSIM4V4DGatesidewallJctPotential;
            return(OK);
        case BSIM4V4_MOD_MJSWGD:
            value->rValue = model->BSIM4V4DbulkJctGateSideGradingCoeff;
            return(OK);
        case BSIM4V4_MOD_CJSWGD:
            value->rValue = model->BSIM4V4DunitLengthGateSidewallJctCap;
            return(OK);
        case BSIM4V4_MOD_NJD:
            value->rValue = model->BSIM4V4DjctEmissionCoeff;
            return(OK);
        case BSIM4V4_MOD_XTID:
            value->rValue = model->BSIM4V4DjctTempExponent;
            return(OK);
        case BSIM4V4_MOD_LINT:
            value->rValue = model->BSIM4V4Lint; 
            return(OK);
        case BSIM4V4_MOD_LL:
            value->rValue = model->BSIM4V4Ll;
            return(OK);
        case BSIM4V4_MOD_LLC:
            value->rValue = model->BSIM4V4Llc;
            return(OK);
        case BSIM4V4_MOD_LLN:
            value->rValue = model->BSIM4V4Lln;
            return(OK);
        case BSIM4V4_MOD_LW:
            value->rValue = model->BSIM4V4Lw;
            return(OK);
        case BSIM4V4_MOD_LWC:
            value->rValue = model->BSIM4V4Lwc;
            return(OK);
        case BSIM4V4_MOD_LWN:
            value->rValue = model->BSIM4V4Lwn;
            return(OK);
        case BSIM4V4_MOD_LWL:
            value->rValue = model->BSIM4V4Lwl;
            return(OK);
        case BSIM4V4_MOD_LWLC:
            value->rValue = model->BSIM4V4Lwlc;
            return(OK);
        case BSIM4V4_MOD_LMIN:
            value->rValue = model->BSIM4V4Lmin;
            return(OK);
        case BSIM4V4_MOD_LMAX:
            value->rValue = model->BSIM4V4Lmax;
            return(OK);
        case BSIM4V4_MOD_WINT:
            value->rValue = model->BSIM4V4Wint;
            return(OK);
        case BSIM4V4_MOD_WL:
            value->rValue = model->BSIM4V4Wl;
            return(OK);
        case BSIM4V4_MOD_WLC:
            value->rValue = model->BSIM4V4Wlc;
            return(OK);
        case BSIM4V4_MOD_WLN:
            value->rValue = model->BSIM4V4Wln;
            return(OK);
        case BSIM4V4_MOD_WW:
            value->rValue = model->BSIM4V4Ww;
            return(OK);
        case BSIM4V4_MOD_WWC:
            value->rValue = model->BSIM4V4Wwc;
            return(OK);
        case BSIM4V4_MOD_WWN:
            value->rValue = model->BSIM4V4Wwn;
            return(OK);
        case BSIM4V4_MOD_WWL:
            value->rValue = model->BSIM4V4Wwl;
            return(OK);
        case BSIM4V4_MOD_WWLC:
            value->rValue = model->BSIM4V4Wwlc;
            return(OK);
        case BSIM4V4_MOD_WMIN:
            value->rValue = model->BSIM4V4Wmin;
            return(OK);
        case BSIM4V4_MOD_WMAX:
            value->rValue = model->BSIM4V4Wmax;
            return(OK);

        /* stress effect */
        case BSIM4V4_MOD_SAREF:
            value->rValue = model->BSIM4V4saref;
            return(OK);
        case BSIM4V4_MOD_SBREF:
            value->rValue = model->BSIM4V4sbref;
            return(OK);
	case BSIM4V4_MOD_WLOD:
            value->rValue = model->BSIM4V4wlod;
            return(OK);
        case BSIM4V4_MOD_KU0:
            value->rValue = model->BSIM4V4ku0;
            return(OK);
        case BSIM4V4_MOD_KVSAT:
            value->rValue = model->BSIM4V4kvsat;
            return(OK);
        case BSIM4V4_MOD_KVTH0:
            value->rValue = model->BSIM4V4kvth0;
            return(OK);
        case BSIM4V4_MOD_TKU0:
            value->rValue = model->BSIM4V4tku0;
            return(OK);
        case BSIM4V4_MOD_LLODKU0:
            value->rValue = model->BSIM4V4llodku0;
            return(OK);
        case BSIM4V4_MOD_WLODKU0:
            value->rValue = model->BSIM4V4wlodku0;
            return(OK);
        case BSIM4V4_MOD_LLODVTH:
            value->rValue = model->BSIM4V4llodvth;
            return(OK);
        case BSIM4V4_MOD_WLODVTH:
            value->rValue = model->BSIM4V4wlodvth;
            return(OK);
        case BSIM4V4_MOD_LKU0:
            value->rValue = model->BSIM4V4lku0;
            return(OK);
        case BSIM4V4_MOD_WKU0:
            value->rValue = model->BSIM4V4wku0;
            return(OK);
        case BSIM4V4_MOD_PKU0:
            value->rValue = model->BSIM4V4pku0;
            return(OK);
        case BSIM4V4_MOD_LKVTH0:
            value->rValue = model->BSIM4V4lkvth0;
            return(OK);
        case BSIM4V4_MOD_WKVTH0:
            value->rValue = model->BSIM4V4wkvth0;
            return(OK);
        case BSIM4V4_MOD_PKVTH0:
            value->rValue = model->BSIM4V4pkvth0;
            return(OK);
        case BSIM4V4_MOD_STK2:
            value->rValue = model->BSIM4V4stk2;
            return(OK);
        case BSIM4V4_MOD_LODK2:
            value->rValue = model->BSIM4V4lodk2;
            return(OK);
        case BSIM4V4_MOD_STETA0:
            value->rValue = model->BSIM4V4steta0;
            return(OK);
        case BSIM4V4_MOD_LODETA0:
            value->rValue = model->BSIM4V4lodeta0;
            return(OK);

        case BSIM4V4_MOD_NOIA:
            value->rValue = model->BSIM4V4oxideTrapDensityA;
            return(OK);
        case BSIM4V4_MOD_NOIB:
            value->rValue = model->BSIM4V4oxideTrapDensityB;
            return(OK);
        case BSIM4V4_MOD_NOIC:
            value->rValue = model->BSIM4V4oxideTrapDensityC;
            return(OK);
        case BSIM4V4_MOD_EM:
            value->rValue = model->BSIM4V4em;
            return(OK);
        case BSIM4V4_MOD_EF:
            value->rValue = model->BSIM4V4ef;
            return(OK);
        case BSIM4V4_MOD_AF:
            value->rValue = model->BSIM4V4af;
            return(OK);
        case BSIM4V4_MOD_KF:
            value->rValue = model->BSIM4V4kf;
            return(OK);
        case BSIM4V4_MOD_STIMOD:
            value->rValue = model->BSIM4V4stimod;
            return(OK);
        case BSIM4V4_MOD_RGEOMOD:
            value->rValue = model->BSIM4V4rgeomod;
            return(OK);
        case BSIM4V4_MOD_SA0:
            value->rValue = model->BSIM4V4sa0;
            return(OK);
        case BSIM4V4_MOD_SB0:
            value->rValue = model->BSIM4V4sb0;
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}



