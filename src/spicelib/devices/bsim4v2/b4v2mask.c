/**** BSIM4.2.1, Released by Xuemei Xi 10/05/2001 ****/

/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * File: b4mask.c of BSIM4.2.1.
 * Author: 2000 Weidong Liu
 * Authors: Xuemei Xi, Kanyu M. Cao, Hui Wan, Mansun Chan, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/


#include "ngspice.h"
#include <stdio.h>
#include "ifsim.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "bsim4v2def.h"
#include "sperror.h"


int
BSIM4v2mAsk(
CKTcircuit *ckt,
GENmodel *inst,
int which,
IFvalue *value)
{
    BSIM4v2model *model = (BSIM4v2model *)inst;

    NG_IGNORE(ckt);

    switch(which) 
    {   case BSIM4v2_MOD_MOBMOD :
            value->iValue = model->BSIM4v2mobMod; 
            return(OK);
        case BSIM4v2_MOD_PARAMCHK :
            value->iValue = model->BSIM4v2paramChk; 
            return(OK);
        case BSIM4v2_MOD_BINUNIT :
            value->iValue = model->BSIM4v2binUnit; 
            return(OK);
        case BSIM4v2_MOD_CAPMOD :
            value->iValue = model->BSIM4v2capMod; 
            return(OK);
        case BSIM4v2_MOD_DIOMOD :
            value->iValue = model->BSIM4v2dioMod;
            return(OK);
        case BSIM4v2_MOD_TRNQSMOD :
            value->iValue = model->BSIM4v2trnqsMod;
            return(OK);
        case BSIM4v2_MOD_ACNQSMOD :
            value->iValue = model->BSIM4v2acnqsMod;
            return(OK);
        case BSIM4v2_MOD_FNOIMOD :
            value->iValue = model->BSIM4v2fnoiMod; 
            return(OK);
        case BSIM4v2_MOD_TNOIMOD :
            value->iValue = model->BSIM4v2tnoiMod;
            return(OK);
        case BSIM4v2_MOD_RDSMOD :
            value->iValue = model->BSIM4v2rdsMod;
            return(OK);
        case BSIM4v2_MOD_RBODYMOD :
            value->iValue = model->BSIM4v2rbodyMod;
            return(OK);
        case BSIM4v2_MOD_RGATEMOD :
            value->iValue = model->BSIM4v2rgateMod;
            return(OK);
        case BSIM4v2_MOD_PERMOD :
            value->iValue = model->BSIM4v2perMod;
            return(OK);
        case BSIM4v2_MOD_GEOMOD :
            value->iValue = model->BSIM4v2geoMod;
            return(OK);
        case BSIM4v2_MOD_IGCMOD :
            value->iValue = model->BSIM4v2igcMod;
            return(OK);
        case BSIM4v2_MOD_IGBMOD :
            value->iValue = model->BSIM4v2igbMod;
            return(OK);
        case  BSIM4v2_MOD_VERSION :
          value->sValue = model->BSIM4v2version;
            return(OK);
        case  BSIM4v2_MOD_TOXREF :
          value->rValue = model->BSIM4v2toxref;
          return(OK);
        case  BSIM4v2_MOD_TOXE :
          value->rValue = model->BSIM4v2toxe;
            return(OK);
        case  BSIM4v2_MOD_TOXP :
          value->rValue = model->BSIM4v2toxp;
            return(OK);
        case  BSIM4v2_MOD_TOXM :
          value->rValue = model->BSIM4v2toxm;
            return(OK);
        case  BSIM4v2_MOD_DTOX :
          value->rValue = model->BSIM4v2dtox;
            return(OK);
        case  BSIM4v2_MOD_EPSROX :
          value->rValue = model->BSIM4v2epsrox;
            return(OK);
        case  BSIM4v2_MOD_CDSC :
          value->rValue = model->BSIM4v2cdsc;
            return(OK);
        case  BSIM4v2_MOD_CDSCB :
          value->rValue = model->BSIM4v2cdscb;
            return(OK);

        case  BSIM4v2_MOD_CDSCD :
          value->rValue = model->BSIM4v2cdscd;
            return(OK);

        case  BSIM4v2_MOD_CIT :
          value->rValue = model->BSIM4v2cit;
            return(OK);
        case  BSIM4v2_MOD_NFACTOR :
          value->rValue = model->BSIM4v2nfactor;
            return(OK);
        case BSIM4v2_MOD_XJ:
            value->rValue = model->BSIM4v2xj;
            return(OK);
        case BSIM4v2_MOD_VSAT:
            value->rValue = model->BSIM4v2vsat;
            return(OK);
        case BSIM4v2_MOD_AT:
            value->rValue = model->BSIM4v2at;
            return(OK);
        case BSIM4v2_MOD_A0:
            value->rValue = model->BSIM4v2a0;
            return(OK);

        case BSIM4v2_MOD_AGS:
            value->rValue = model->BSIM4v2ags;
            return(OK);

        case BSIM4v2_MOD_A1:
            value->rValue = model->BSIM4v2a1;
            return(OK);
        case BSIM4v2_MOD_A2:
            value->rValue = model->BSIM4v2a2;
            return(OK);
        case BSIM4v2_MOD_KETA:
            value->rValue = model->BSIM4v2keta;
            return(OK);   
        case BSIM4v2_MOD_NSUB:
            value->rValue = model->BSIM4v2nsub;
            return(OK);
        case BSIM4v2_MOD_NDEP:
            value->rValue = model->BSIM4v2ndep;
            return(OK);
        case BSIM4v2_MOD_NSD:
            value->rValue = model->BSIM4v2nsd;
            return(OK);
        case BSIM4v2_MOD_NGATE:
            value->rValue = model->BSIM4v2ngate;
            return(OK);
        case BSIM4v2_MOD_GAMMA1:
            value->rValue = model->BSIM4v2gamma1;
            return(OK);
        case BSIM4v2_MOD_GAMMA2:
            value->rValue = model->BSIM4v2gamma2;
            return(OK);
        case BSIM4v2_MOD_VBX:
            value->rValue = model->BSIM4v2vbx;
            return(OK);
        case BSIM4v2_MOD_VBM:
            value->rValue = model->BSIM4v2vbm;
            return(OK);
        case BSIM4v2_MOD_XT:
            value->rValue = model->BSIM4v2xt;
            return(OK);
        case  BSIM4v2_MOD_K1:
          value->rValue = model->BSIM4v2k1;
            return(OK);
        case  BSIM4v2_MOD_KT1:
          value->rValue = model->BSIM4v2kt1;
            return(OK);
        case  BSIM4v2_MOD_KT1L:
          value->rValue = model->BSIM4v2kt1l;
            return(OK);
        case  BSIM4v2_MOD_KT2 :
          value->rValue = model->BSIM4v2kt2;
            return(OK);
        case  BSIM4v2_MOD_K2 :
          value->rValue = model->BSIM4v2k2;
            return(OK);
        case  BSIM4v2_MOD_K3:
          value->rValue = model->BSIM4v2k3;
            return(OK);
        case  BSIM4v2_MOD_K3B:
          value->rValue = model->BSIM4v2k3b;
            return(OK);
        case  BSIM4v2_MOD_W0:
          value->rValue = model->BSIM4v2w0;
            return(OK);
        case  BSIM4v2_MOD_LPE0:
          value->rValue = model->BSIM4v2lpe0;
            return(OK);
        case  BSIM4v2_MOD_LPEB:
          value->rValue = model->BSIM4v2lpeb;
            return(OK);
        case  BSIM4v2_MOD_DVTP0:
          value->rValue = model->BSIM4v2dvtp0;
            return(OK);
        case  BSIM4v2_MOD_DVTP1:
          value->rValue = model->BSIM4v2dvtp1;
            return(OK);
        case  BSIM4v2_MOD_DVT0 :                
          value->rValue = model->BSIM4v2dvt0;
            return(OK);
        case  BSIM4v2_MOD_DVT1 :             
          value->rValue = model->BSIM4v2dvt1;
            return(OK);
        case  BSIM4v2_MOD_DVT2 :             
          value->rValue = model->BSIM4v2dvt2;
            return(OK);
        case  BSIM4v2_MOD_DVT0W :                
          value->rValue = model->BSIM4v2dvt0w;
            return(OK);
        case  BSIM4v2_MOD_DVT1W :             
          value->rValue = model->BSIM4v2dvt1w;
            return(OK);
        case  BSIM4v2_MOD_DVT2W :             
          value->rValue = model->BSIM4v2dvt2w;
            return(OK);
        case  BSIM4v2_MOD_DROUT :           
          value->rValue = model->BSIM4v2drout;
            return(OK);
        case  BSIM4v2_MOD_DSUB :           
          value->rValue = model->BSIM4v2dsub;
            return(OK);
        case BSIM4v2_MOD_VTH0:
            value->rValue = model->BSIM4v2vth0; 
            return(OK);
        case BSIM4v2_MOD_EU:
            value->rValue = model->BSIM4v2eu;
            return(OK);
        case BSIM4v2_MOD_UA:
            value->rValue = model->BSIM4v2ua; 
            return(OK);
        case BSIM4v2_MOD_UA1:
            value->rValue = model->BSIM4v2ua1; 
            return(OK);
        case BSIM4v2_MOD_UB:
            value->rValue = model->BSIM4v2ub;  
            return(OK);
        case BSIM4v2_MOD_UB1:
            value->rValue = model->BSIM4v2ub1;  
            return(OK);
        case BSIM4v2_MOD_UC:
            value->rValue = model->BSIM4v2uc; 
            return(OK);
        case BSIM4v2_MOD_UC1:
            value->rValue = model->BSIM4v2uc1; 
            return(OK);
        case BSIM4v2_MOD_U0:
            value->rValue = model->BSIM4v2u0;
            return(OK);
        case BSIM4v2_MOD_UTE:
            value->rValue = model->BSIM4v2ute;
            return(OK);
        case BSIM4v2_MOD_VOFF:
            value->rValue = model->BSIM4v2voff;
            return(OK);
        case BSIM4v2_MOD_VOFFL:
            value->rValue = model->BSIM4v2voffl;
            return(OK);
        case BSIM4v2_MOD_MINV:
            value->rValue = model->BSIM4v2minv;
            return(OK);
        case BSIM4v2_MOD_FPROUT:
            value->rValue = model->BSIM4v2fprout;
            return(OK);
        case BSIM4v2_MOD_PDITS:
            value->rValue = model->BSIM4v2pdits;
            return(OK);
        case BSIM4v2_MOD_PDITSD:
            value->rValue = model->BSIM4v2pditsd;
            return(OK);
        case BSIM4v2_MOD_PDITSL:
            value->rValue = model->BSIM4v2pditsl;
            return(OK);
        case BSIM4v2_MOD_DELTA:
            value->rValue = model->BSIM4v2delta;
            return(OK);
        case BSIM4v2_MOD_RDSW:
            value->rValue = model->BSIM4v2rdsw; 
            return(OK);
        case BSIM4v2_MOD_RDSWMIN:
            value->rValue = model->BSIM4v2rdswmin;
            return(OK);
        case BSIM4v2_MOD_RDWMIN:
            value->rValue = model->BSIM4v2rdwmin;
            return(OK);
        case BSIM4v2_MOD_RSWMIN:
            value->rValue = model->BSIM4v2rswmin;
            return(OK);
        case BSIM4v2_MOD_RDW:
            value->rValue = model->BSIM4v2rdw;
            return(OK);
        case BSIM4v2_MOD_RSW:
            value->rValue = model->BSIM4v2rsw;
            return(OK);
        case BSIM4v2_MOD_PRWG:
            value->rValue = model->BSIM4v2prwg; 
            return(OK);             
        case BSIM4v2_MOD_PRWB:
            value->rValue = model->BSIM4v2prwb; 
            return(OK);             
        case BSIM4v2_MOD_PRT:
            value->rValue = model->BSIM4v2prt; 
            return(OK);              
        case BSIM4v2_MOD_ETA0:
            value->rValue = model->BSIM4v2eta0; 
            return(OK);               
        case BSIM4v2_MOD_ETAB:
            value->rValue = model->BSIM4v2etab; 
            return(OK);               
        case BSIM4v2_MOD_PCLM:
            value->rValue = model->BSIM4v2pclm; 
            return(OK);               
        case BSIM4v2_MOD_PDIBL1:
            value->rValue = model->BSIM4v2pdibl1; 
            return(OK);               
        case BSIM4v2_MOD_PDIBL2:
            value->rValue = model->BSIM4v2pdibl2; 
            return(OK);               
        case BSIM4v2_MOD_PDIBLB:
            value->rValue = model->BSIM4v2pdiblb; 
            return(OK);               
        case BSIM4v2_MOD_PSCBE1:
            value->rValue = model->BSIM4v2pscbe1; 
            return(OK);               
        case BSIM4v2_MOD_PSCBE2:
            value->rValue = model->BSIM4v2pscbe2; 
            return(OK);               
        case BSIM4v2_MOD_PVAG:
            value->rValue = model->BSIM4v2pvag; 
            return(OK);               
        case BSIM4v2_MOD_WR:
            value->rValue = model->BSIM4v2wr;
            return(OK);
        case BSIM4v2_MOD_DWG:
            value->rValue = model->BSIM4v2dwg;
            return(OK);
        case BSIM4v2_MOD_DWB:
            value->rValue = model->BSIM4v2dwb;
            return(OK);
        case BSIM4v2_MOD_B0:
            value->rValue = model->BSIM4v2b0;
            return(OK);
        case BSIM4v2_MOD_B1:
            value->rValue = model->BSIM4v2b1;
            return(OK);
        case BSIM4v2_MOD_ALPHA0:
            value->rValue = model->BSIM4v2alpha0;
            return(OK);
        case BSIM4v2_MOD_ALPHA1:
            value->rValue = model->BSIM4v2alpha1;
            return(OK);
        case BSIM4v2_MOD_BETA0:
            value->rValue = model->BSIM4v2beta0;
            return(OK);
        case BSIM4v2_MOD_AGIDL:
            value->rValue = model->BSIM4v2agidl;
            return(OK);
        case BSIM4v2_MOD_BGIDL:
            value->rValue = model->BSIM4v2bgidl;
            return(OK);
        case BSIM4v2_MOD_CGIDL:
            value->rValue = model->BSIM4v2cgidl;
            return(OK);
        case BSIM4v2_MOD_EGIDL:
            value->rValue = model->BSIM4v2egidl;
            return(OK);
        case BSIM4v2_MOD_AIGC:
            value->rValue = model->BSIM4v2aigc;
            return(OK);
        case BSIM4v2_MOD_BIGC:
            value->rValue = model->BSIM4v2bigc;
            return(OK);
        case BSIM4v2_MOD_CIGC:
            value->rValue = model->BSIM4v2cigc;
            return(OK);
        case BSIM4v2_MOD_AIGSD:
            value->rValue = model->BSIM4v2aigsd;
            return(OK);
        case BSIM4v2_MOD_BIGSD:
            value->rValue = model->BSIM4v2bigsd;
            return(OK);
        case BSIM4v2_MOD_CIGSD:
            value->rValue = model->BSIM4v2cigsd;
            return(OK);
        case BSIM4v2_MOD_AIGBACC:
            value->rValue = model->BSIM4v2aigbacc;
            return(OK);
        case BSIM4v2_MOD_BIGBACC:
            value->rValue = model->BSIM4v2bigbacc;
            return(OK);
        case BSIM4v2_MOD_CIGBACC:
            value->rValue = model->BSIM4v2cigbacc;
            return(OK);
        case BSIM4v2_MOD_AIGBINV:
            value->rValue = model->BSIM4v2aigbinv;
            return(OK);
        case BSIM4v2_MOD_BIGBINV:
            value->rValue = model->BSIM4v2bigbinv;
            return(OK);
        case BSIM4v2_MOD_CIGBINV:
            value->rValue = model->BSIM4v2cigbinv;
            return(OK);
        case BSIM4v2_MOD_NIGC:
            value->rValue = model->BSIM4v2nigc;
            return(OK);
        case BSIM4v2_MOD_NIGBACC:
            value->rValue = model->BSIM4v2nigbacc;
            return(OK);
        case BSIM4v2_MOD_NIGBINV:
            value->rValue = model->BSIM4v2nigbinv;
            return(OK);
        case BSIM4v2_MOD_NTOX:
            value->rValue = model->BSIM4v2ntox;
            return(OK);
        case BSIM4v2_MOD_EIGBINV:
            value->rValue = model->BSIM4v2eigbinv;
            return(OK);
        case BSIM4v2_MOD_PIGCD:
            value->rValue = model->BSIM4v2pigcd;
            return(OK);
        case BSIM4v2_MOD_POXEDGE:
            value->rValue = model->BSIM4v2poxedge;
            return(OK);
        case BSIM4v2_MOD_PHIN:
            value->rValue = model->BSIM4v2phin;
            return(OK);
        case BSIM4v2_MOD_XRCRG1:
            value->rValue = model->BSIM4v2xrcrg1;
            return(OK);
        case BSIM4v2_MOD_XRCRG2:
            value->rValue = model->BSIM4v2xrcrg2;
            return(OK);
        case BSIM4v2_MOD_TNOIA:
            value->rValue = model->BSIM4v2tnoia;
            return(OK);
        case BSIM4v2_MOD_TNOIB:
            value->rValue = model->BSIM4v2tnoib;
            return(OK);
        case BSIM4v2_MOD_NTNOI:
            value->rValue = model->BSIM4v2ntnoi;
            return(OK);
        case BSIM4v2_MOD_IJTHDFWD:
            value->rValue = model->BSIM4v2ijthdfwd;
            return(OK);
        case BSIM4v2_MOD_IJTHSFWD:
            value->rValue = model->BSIM4v2ijthsfwd;
            return(OK);
        case BSIM4v2_MOD_IJTHDREV:
            value->rValue = model->BSIM4v2ijthdrev;
            return(OK);
        case BSIM4v2_MOD_IJTHSREV:
            value->rValue = model->BSIM4v2ijthsrev;
            return(OK);
        case BSIM4v2_MOD_XJBVD:
            value->rValue = model->BSIM4v2xjbvd;
            return(OK);
        case BSIM4v2_MOD_XJBVS:
            value->rValue = model->BSIM4v2xjbvs;
            return(OK);
        case BSIM4v2_MOD_BVD:
            value->rValue = model->BSIM4v2bvd;
            return(OK);
        case BSIM4v2_MOD_BVS:
            value->rValue = model->BSIM4v2bvs;
            return(OK);
        case BSIM4v2_MOD_VFB:
            value->rValue = model->BSIM4v2vfb;
            return(OK);

        case BSIM4v2_MOD_GBMIN:
            value->rValue = model->BSIM4v2gbmin;
            return(OK);
        case BSIM4v2_MOD_RBDB:
            value->rValue = model->BSIM4v2rbdb;
            return(OK);
        case BSIM4v2_MOD_RBPB:
            value->rValue = model->BSIM4v2rbpb;
            return(OK);
        case BSIM4v2_MOD_RBSB:
            value->rValue = model->BSIM4v2rbsb;
            return(OK);
        case BSIM4v2_MOD_RBPS:
            value->rValue = model->BSIM4v2rbps;
            return(OK);
        case BSIM4v2_MOD_RBPD:
            value->rValue = model->BSIM4v2rbpd;
            return(OK);

        case BSIM4v2_MOD_CGSL:
            value->rValue = model->BSIM4v2cgsl;
            return(OK);
        case BSIM4v2_MOD_CGDL:
            value->rValue = model->BSIM4v2cgdl;
            return(OK);
        case BSIM4v2_MOD_CKAPPAS:
            value->rValue = model->BSIM4v2ckappas;
            return(OK);
        case BSIM4v2_MOD_CKAPPAD:
            value->rValue = model->BSIM4v2ckappad;
            return(OK);
        case BSIM4v2_MOD_CF:
            value->rValue = model->BSIM4v2cf;
            return(OK);
        case BSIM4v2_MOD_CLC:
            value->rValue = model->BSIM4v2clc;
            return(OK);
        case BSIM4v2_MOD_CLE:
            value->rValue = model->BSIM4v2cle;
            return(OK);
        case BSIM4v2_MOD_DWC:
            value->rValue = model->BSIM4v2dwc;
            return(OK);
        case BSIM4v2_MOD_DLC:
            value->rValue = model->BSIM4v2dlc;
            return(OK);
        case BSIM4v2_MOD_XW:
            value->rValue = model->BSIM4v2xw;
            return(OK);
        case BSIM4v2_MOD_XL:
            value->rValue = model->BSIM4v2xl;
            return(OK);
        case BSIM4v2_MOD_DLCIG:
            value->rValue = model->BSIM4v2dlcig;
            return(OK);
        case BSIM4v2_MOD_DWJ:
            value->rValue = model->BSIM4v2dwj;
            return(OK);
        case BSIM4v2_MOD_VFBCV:
            value->rValue = model->BSIM4v2vfbcv; 
            return(OK);
        case BSIM4v2_MOD_ACDE:
            value->rValue = model->BSIM4v2acde;
            return(OK);
        case BSIM4v2_MOD_MOIN:
            value->rValue = model->BSIM4v2moin;
            return(OK);
        case BSIM4v2_MOD_NOFF:
            value->rValue = model->BSIM4v2noff;
            return(OK);
        case BSIM4v2_MOD_VOFFCV:
            value->rValue = model->BSIM4v2voffcv;
            return(OK);
        case BSIM4v2_MOD_DMCG:
            value->rValue = model->BSIM4v2dmcg;
            return(OK);
        case BSIM4v2_MOD_DMCI:
            value->rValue = model->BSIM4v2dmci;
            return(OK);
        case BSIM4v2_MOD_DMDG:
            value->rValue = model->BSIM4v2dmdg;
            return(OK);
        case BSIM4v2_MOD_DMCGT:
            value->rValue = model->BSIM4v2dmcgt;
            return(OK);
        case BSIM4v2_MOD_XGW:
            value->rValue = model->BSIM4v2xgw;
            return(OK);
        case BSIM4v2_MOD_XGL:
            value->rValue = model->BSIM4v2xgl;
            return(OK);
        case BSIM4v2_MOD_RSHG:
            value->rValue = model->BSIM4v2rshg;
            return(OK);
        case BSIM4v2_MOD_NGCON:
            value->rValue = model->BSIM4v2ngcon;
            return(OK);
        case BSIM4v2_MOD_TCJ:
            value->rValue = model->BSIM4v2tcj;
            return(OK);
        case BSIM4v2_MOD_TPB:
            value->rValue = model->BSIM4v2tpb;
            return(OK);
        case BSIM4v2_MOD_TCJSW:
            value->rValue = model->BSIM4v2tcjsw;
            return(OK);
        case BSIM4v2_MOD_TPBSW:
            value->rValue = model->BSIM4v2tpbsw;
            return(OK);
        case BSIM4v2_MOD_TCJSWG:
            value->rValue = model->BSIM4v2tcjswg;
            return(OK);
        case BSIM4v2_MOD_TPBSWG:
            value->rValue = model->BSIM4v2tpbswg;
            return(OK);

	/* Length dependence */
        case  BSIM4v2_MOD_LCDSC :
          value->rValue = model->BSIM4v2lcdsc;
            return(OK);
        case  BSIM4v2_MOD_LCDSCB :
          value->rValue = model->BSIM4v2lcdscb;
            return(OK);
        case  BSIM4v2_MOD_LCDSCD :
          value->rValue = model->BSIM4v2lcdscd;
            return(OK);
        case  BSIM4v2_MOD_LCIT :
          value->rValue = model->BSIM4v2lcit;
            return(OK);
        case  BSIM4v2_MOD_LNFACTOR :
          value->rValue = model->BSIM4v2lnfactor;
            return(OK);
        case BSIM4v2_MOD_LXJ:
            value->rValue = model->BSIM4v2lxj;
            return(OK);
        case BSIM4v2_MOD_LVSAT:
            value->rValue = model->BSIM4v2lvsat;
            return(OK);
        case BSIM4v2_MOD_LAT:
            value->rValue = model->BSIM4v2lat;
            return(OK);
        case BSIM4v2_MOD_LA0:
            value->rValue = model->BSIM4v2la0;
            return(OK);
        case BSIM4v2_MOD_LAGS:
            value->rValue = model->BSIM4v2lags;
            return(OK);
        case BSIM4v2_MOD_LA1:
            value->rValue = model->BSIM4v2la1;
            return(OK);
        case BSIM4v2_MOD_LA2:
            value->rValue = model->BSIM4v2la2;
            return(OK);
        case BSIM4v2_MOD_LKETA:
            value->rValue = model->BSIM4v2lketa;
            return(OK);   
        case BSIM4v2_MOD_LNSUB:
            value->rValue = model->BSIM4v2lnsub;
            return(OK);
        case BSIM4v2_MOD_LNDEP:
            value->rValue = model->BSIM4v2lndep;
            return(OK);
        case BSIM4v2_MOD_LNSD:
            value->rValue = model->BSIM4v2lnsd;
            return(OK);
        case BSIM4v2_MOD_LNGATE:
            value->rValue = model->BSIM4v2lngate;
            return(OK);
        case BSIM4v2_MOD_LGAMMA1:
            value->rValue = model->BSIM4v2lgamma1;
            return(OK);
        case BSIM4v2_MOD_LGAMMA2:
            value->rValue = model->BSIM4v2lgamma2;
            return(OK);
        case BSIM4v2_MOD_LVBX:
            value->rValue = model->BSIM4v2lvbx;
            return(OK);
        case BSIM4v2_MOD_LVBM:
            value->rValue = model->BSIM4v2lvbm;
            return(OK);
        case BSIM4v2_MOD_LXT:
            value->rValue = model->BSIM4v2lxt;
            return(OK);
        case  BSIM4v2_MOD_LK1:
          value->rValue = model->BSIM4v2lk1;
            return(OK);
        case  BSIM4v2_MOD_LKT1:
          value->rValue = model->BSIM4v2lkt1;
            return(OK);
        case  BSIM4v2_MOD_LKT1L:
          value->rValue = model->BSIM4v2lkt1l;
            return(OK);
        case  BSIM4v2_MOD_LKT2 :
          value->rValue = model->BSIM4v2lkt2;
            return(OK);
        case  BSIM4v2_MOD_LK2 :
          value->rValue = model->BSIM4v2lk2;
            return(OK);
        case  BSIM4v2_MOD_LK3:
          value->rValue = model->BSIM4v2lk3;
            return(OK);
        case  BSIM4v2_MOD_LK3B:
          value->rValue = model->BSIM4v2lk3b;
            return(OK);
        case  BSIM4v2_MOD_LW0:
          value->rValue = model->BSIM4v2lw0;
            return(OK);
        case  BSIM4v2_MOD_LLPE0:
          value->rValue = model->BSIM4v2llpe0;
            return(OK);
        case  BSIM4v2_MOD_LLPEB:
          value->rValue = model->BSIM4v2llpeb;
            return(OK);
        case  BSIM4v2_MOD_LDVTP0:
          value->rValue = model->BSIM4v2ldvtp0;
            return(OK);
        case  BSIM4v2_MOD_LDVTP1:
          value->rValue = model->BSIM4v2ldvtp1;
            return(OK);
        case  BSIM4v2_MOD_LDVT0:                
          value->rValue = model->BSIM4v2ldvt0;
            return(OK);
        case  BSIM4v2_MOD_LDVT1 :             
          value->rValue = model->BSIM4v2ldvt1;
            return(OK);
        case  BSIM4v2_MOD_LDVT2 :             
          value->rValue = model->BSIM4v2ldvt2;
            return(OK);
        case  BSIM4v2_MOD_LDVT0W :                
          value->rValue = model->BSIM4v2ldvt0w;
            return(OK);
        case  BSIM4v2_MOD_LDVT1W :             
          value->rValue = model->BSIM4v2ldvt1w;
            return(OK);
        case  BSIM4v2_MOD_LDVT2W :             
          value->rValue = model->BSIM4v2ldvt2w;
            return(OK);
        case  BSIM4v2_MOD_LDROUT :           
          value->rValue = model->BSIM4v2ldrout;
            return(OK);
        case  BSIM4v2_MOD_LDSUB :           
          value->rValue = model->BSIM4v2ldsub;
            return(OK);
        case BSIM4v2_MOD_LVTH0:
            value->rValue = model->BSIM4v2lvth0; 
            return(OK);
        case BSIM4v2_MOD_LUA:
            value->rValue = model->BSIM4v2lua; 
            return(OK);
        case BSIM4v2_MOD_LUA1:
            value->rValue = model->BSIM4v2lua1; 
            return(OK);
        case BSIM4v2_MOD_LUB:
            value->rValue = model->BSIM4v2lub;  
            return(OK);
        case BSIM4v2_MOD_LUB1:
            value->rValue = model->BSIM4v2lub1;  
            return(OK);
        case BSIM4v2_MOD_LUC:
            value->rValue = model->BSIM4v2luc; 
            return(OK);
        case BSIM4v2_MOD_LUC1:
            value->rValue = model->BSIM4v2luc1; 
            return(OK);
        case BSIM4v2_MOD_LU0:
            value->rValue = model->BSIM4v2lu0;
            return(OK);
        case BSIM4v2_MOD_LUTE:
            value->rValue = model->BSIM4v2lute;
            return(OK);
        case BSIM4v2_MOD_LVOFF:
            value->rValue = model->BSIM4v2lvoff;
            return(OK);
        case BSIM4v2_MOD_LMINV:
            value->rValue = model->BSIM4v2lminv;
            return(OK);
        case BSIM4v2_MOD_LFPROUT:
            value->rValue = model->BSIM4v2lfprout;
            return(OK);
        case BSIM4v2_MOD_LPDITS:
            value->rValue = model->BSIM4v2lpdits;
            return(OK);
        case BSIM4v2_MOD_LPDITSD:
            value->rValue = model->BSIM4v2lpditsd;
            return(OK);
        case BSIM4v2_MOD_LDELTA:
            value->rValue = model->BSIM4v2ldelta;
            return(OK);
        case BSIM4v2_MOD_LRDSW:
            value->rValue = model->BSIM4v2lrdsw; 
            return(OK);             
        case BSIM4v2_MOD_LRDW:
            value->rValue = model->BSIM4v2lrdw;
            return(OK);
        case BSIM4v2_MOD_LRSW:
            value->rValue = model->BSIM4v2lrsw;
            return(OK);
        case BSIM4v2_MOD_LPRWB:
            value->rValue = model->BSIM4v2lprwb; 
            return(OK);             
        case BSIM4v2_MOD_LPRWG:
            value->rValue = model->BSIM4v2lprwg; 
            return(OK);             
        case BSIM4v2_MOD_LPRT:
            value->rValue = model->BSIM4v2lprt; 
            return(OK);              
        case BSIM4v2_MOD_LETA0:
            value->rValue = model->BSIM4v2leta0; 
            return(OK);               
        case BSIM4v2_MOD_LETAB:
            value->rValue = model->BSIM4v2letab; 
            return(OK);               
        case BSIM4v2_MOD_LPCLM:
            value->rValue = model->BSIM4v2lpclm; 
            return(OK);               
        case BSIM4v2_MOD_LPDIBL1:
            value->rValue = model->BSIM4v2lpdibl1; 
            return(OK);               
        case BSIM4v2_MOD_LPDIBL2:
            value->rValue = model->BSIM4v2lpdibl2; 
            return(OK);               
        case BSIM4v2_MOD_LPDIBLB:
            value->rValue = model->BSIM4v2lpdiblb; 
            return(OK);               
        case BSIM4v2_MOD_LPSCBE1:
            value->rValue = model->BSIM4v2lpscbe1; 
            return(OK);               
        case BSIM4v2_MOD_LPSCBE2:
            value->rValue = model->BSIM4v2lpscbe2; 
            return(OK);               
        case BSIM4v2_MOD_LPVAG:
            value->rValue = model->BSIM4v2lpvag; 
            return(OK);               
        case BSIM4v2_MOD_LWR:
            value->rValue = model->BSIM4v2lwr;
            return(OK);
        case BSIM4v2_MOD_LDWG:
            value->rValue = model->BSIM4v2ldwg;
            return(OK);
        case BSIM4v2_MOD_LDWB:
            value->rValue = model->BSIM4v2ldwb;
            return(OK);
        case BSIM4v2_MOD_LB0:
            value->rValue = model->BSIM4v2lb0;
            return(OK);
        case BSIM4v2_MOD_LB1:
            value->rValue = model->BSIM4v2lb1;
            return(OK);
        case BSIM4v2_MOD_LALPHA0:
            value->rValue = model->BSIM4v2lalpha0;
            return(OK);
        case BSIM4v2_MOD_LALPHA1:
            value->rValue = model->BSIM4v2lalpha1;
            return(OK);
        case BSIM4v2_MOD_LBETA0:
            value->rValue = model->BSIM4v2lbeta0;
            return(OK);
        case BSIM4v2_MOD_LAGIDL:
            value->rValue = model->BSIM4v2lagidl;
            return(OK);
        case BSIM4v2_MOD_LBGIDL:
            value->rValue = model->BSIM4v2lbgidl;
            return(OK);
        case BSIM4v2_MOD_LCGIDL:
            value->rValue = model->BSIM4v2lcgidl;
            return(OK);
        case BSIM4v2_MOD_LEGIDL:
            value->rValue = model->BSIM4v2legidl;
            return(OK);
        case BSIM4v2_MOD_LAIGC:
            value->rValue = model->BSIM4v2laigc;
            return(OK);
        case BSIM4v2_MOD_LBIGC:
            value->rValue = model->BSIM4v2lbigc;
            return(OK);
        case BSIM4v2_MOD_LCIGC:
            value->rValue = model->BSIM4v2lcigc;
            return(OK);
        case BSIM4v2_MOD_LAIGSD:
            value->rValue = model->BSIM4v2laigsd;
            return(OK);
        case BSIM4v2_MOD_LBIGSD:
            value->rValue = model->BSIM4v2lbigsd;
            return(OK);
        case BSIM4v2_MOD_LCIGSD:
            value->rValue = model->BSIM4v2lcigsd;
            return(OK);
        case BSIM4v2_MOD_LAIGBACC:
            value->rValue = model->BSIM4v2laigbacc;
            return(OK);
        case BSIM4v2_MOD_LBIGBACC:
            value->rValue = model->BSIM4v2lbigbacc;
            return(OK);
        case BSIM4v2_MOD_LCIGBACC:
            value->rValue = model->BSIM4v2lcigbacc;
            return(OK);
        case BSIM4v2_MOD_LAIGBINV:
            value->rValue = model->BSIM4v2laigbinv;
            return(OK);
        case BSIM4v2_MOD_LBIGBINV:
            value->rValue = model->BSIM4v2lbigbinv;
            return(OK);
        case BSIM4v2_MOD_LCIGBINV:
            value->rValue = model->BSIM4v2lcigbinv;
            return(OK);
        case BSIM4v2_MOD_LNIGC:
            value->rValue = model->BSIM4v2lnigc;
            return(OK);
        case BSIM4v2_MOD_LNIGBACC:
            value->rValue = model->BSIM4v2lnigbacc;
            return(OK);
        case BSIM4v2_MOD_LNIGBINV:
            value->rValue = model->BSIM4v2lnigbinv;
            return(OK);
        case BSIM4v2_MOD_LNTOX:
            value->rValue = model->BSIM4v2lntox;
            return(OK);
        case BSIM4v2_MOD_LEIGBINV:
            value->rValue = model->BSIM4v2leigbinv;
            return(OK);
        case BSIM4v2_MOD_LPIGCD:
            value->rValue = model->BSIM4v2lpigcd;
            return(OK);
        case BSIM4v2_MOD_LPOXEDGE:
            value->rValue = model->BSIM4v2lpoxedge;
            return(OK);
        case BSIM4v2_MOD_LPHIN:
            value->rValue = model->BSIM4v2lphin;
            return(OK);
        case BSIM4v2_MOD_LXRCRG1:
            value->rValue = model->BSIM4v2lxrcrg1;
            return(OK);
        case BSIM4v2_MOD_LXRCRG2:
            value->rValue = model->BSIM4v2lxrcrg2;
            return(OK);
        case BSIM4v2_MOD_LEU:
            value->rValue = model->BSIM4v2leu;
            return(OK);
        case BSIM4v2_MOD_LVFB:
            value->rValue = model->BSIM4v2lvfb;
            return(OK);

        case BSIM4v2_MOD_LCGSL:
            value->rValue = model->BSIM4v2lcgsl;
            return(OK);
        case BSIM4v2_MOD_LCGDL:
            value->rValue = model->BSIM4v2lcgdl;
            return(OK);
        case BSIM4v2_MOD_LCKAPPAS:
            value->rValue = model->BSIM4v2lckappas;
            return(OK);
        case BSIM4v2_MOD_LCKAPPAD:
            value->rValue = model->BSIM4v2lckappad;
            return(OK);
        case BSIM4v2_MOD_LCF:
            value->rValue = model->BSIM4v2lcf;
            return(OK);
        case BSIM4v2_MOD_LCLC:
            value->rValue = model->BSIM4v2lclc;
            return(OK);
        case BSIM4v2_MOD_LCLE:
            value->rValue = model->BSIM4v2lcle;
            return(OK);
        case BSIM4v2_MOD_LVFBCV:
            value->rValue = model->BSIM4v2lvfbcv;
            return(OK);
        case BSIM4v2_MOD_LACDE:
            value->rValue = model->BSIM4v2lacde;
            return(OK);
        case BSIM4v2_MOD_LMOIN:
            value->rValue = model->BSIM4v2lmoin;
            return(OK);
        case BSIM4v2_MOD_LNOFF:
            value->rValue = model->BSIM4v2lnoff;
            return(OK);
        case BSIM4v2_MOD_LVOFFCV:
            value->rValue = model->BSIM4v2lvoffcv;
            return(OK);

	/* Width dependence */
        case  BSIM4v2_MOD_WCDSC :
          value->rValue = model->BSIM4v2wcdsc;
            return(OK);
        case  BSIM4v2_MOD_WCDSCB :
          value->rValue = model->BSIM4v2wcdscb;
            return(OK);
        case  BSIM4v2_MOD_WCDSCD :
          value->rValue = model->BSIM4v2wcdscd;
            return(OK);
        case  BSIM4v2_MOD_WCIT :
          value->rValue = model->BSIM4v2wcit;
            return(OK);
        case  BSIM4v2_MOD_WNFACTOR :
          value->rValue = model->BSIM4v2wnfactor;
            return(OK);
        case BSIM4v2_MOD_WXJ:
            value->rValue = model->BSIM4v2wxj;
            return(OK);
        case BSIM4v2_MOD_WVSAT:
            value->rValue = model->BSIM4v2wvsat;
            return(OK);
        case BSIM4v2_MOD_WAT:
            value->rValue = model->BSIM4v2wat;
            return(OK);
        case BSIM4v2_MOD_WA0:
            value->rValue = model->BSIM4v2wa0;
            return(OK);
        case BSIM4v2_MOD_WAGS:
            value->rValue = model->BSIM4v2wags;
            return(OK);
        case BSIM4v2_MOD_WA1:
            value->rValue = model->BSIM4v2wa1;
            return(OK);
        case BSIM4v2_MOD_WA2:
            value->rValue = model->BSIM4v2wa2;
            return(OK);
        case BSIM4v2_MOD_WKETA:
            value->rValue = model->BSIM4v2wketa;
            return(OK);   
        case BSIM4v2_MOD_WNSUB:
            value->rValue = model->BSIM4v2wnsub;
            return(OK);
        case BSIM4v2_MOD_WNDEP:
            value->rValue = model->BSIM4v2wndep;
            return(OK);
        case BSIM4v2_MOD_WNSD:
            value->rValue = model->BSIM4v2wnsd;
            return(OK);
        case BSIM4v2_MOD_WNGATE:
            value->rValue = model->BSIM4v2wngate;
            return(OK);
        case BSIM4v2_MOD_WGAMMA1:
            value->rValue = model->BSIM4v2wgamma1;
            return(OK);
        case BSIM4v2_MOD_WGAMMA2:
            value->rValue = model->BSIM4v2wgamma2;
            return(OK);
        case BSIM4v2_MOD_WVBX:
            value->rValue = model->BSIM4v2wvbx;
            return(OK);
        case BSIM4v2_MOD_WVBM:
            value->rValue = model->BSIM4v2wvbm;
            return(OK);
        case BSIM4v2_MOD_WXT:
            value->rValue = model->BSIM4v2wxt;
            return(OK);
        case  BSIM4v2_MOD_WK1:
          value->rValue = model->BSIM4v2wk1;
            return(OK);
        case  BSIM4v2_MOD_WKT1:
          value->rValue = model->BSIM4v2wkt1;
            return(OK);
        case  BSIM4v2_MOD_WKT1L:
          value->rValue = model->BSIM4v2wkt1l;
            return(OK);
        case  BSIM4v2_MOD_WKT2 :
          value->rValue = model->BSIM4v2wkt2;
            return(OK);
        case  BSIM4v2_MOD_WK2 :
          value->rValue = model->BSIM4v2wk2;
            return(OK);
        case  BSIM4v2_MOD_WK3:
          value->rValue = model->BSIM4v2wk3;
            return(OK);
        case  BSIM4v2_MOD_WK3B:
          value->rValue = model->BSIM4v2wk3b;
            return(OK);
        case  BSIM4v2_MOD_WW0:
          value->rValue = model->BSIM4v2ww0;
            return(OK);
        case  BSIM4v2_MOD_WLPE0:
          value->rValue = model->BSIM4v2wlpe0;
            return(OK);
        case  BSIM4v2_MOD_WDVTP0:
          value->rValue = model->BSIM4v2wdvtp0;
            return(OK);
        case  BSIM4v2_MOD_WDVTP1:
          value->rValue = model->BSIM4v2wdvtp1;
            return(OK);
        case  BSIM4v2_MOD_WLPEB:
          value->rValue = model->BSIM4v2wlpeb;
            return(OK);
        case  BSIM4v2_MOD_WDVT0:                
          value->rValue = model->BSIM4v2wdvt0;
            return(OK);
        case  BSIM4v2_MOD_WDVT1 :             
          value->rValue = model->BSIM4v2wdvt1;
            return(OK);
        case  BSIM4v2_MOD_WDVT2 :             
          value->rValue = model->BSIM4v2wdvt2;
            return(OK);
        case  BSIM4v2_MOD_WDVT0W :                
          value->rValue = model->BSIM4v2wdvt0w;
            return(OK);
        case  BSIM4v2_MOD_WDVT1W :             
          value->rValue = model->BSIM4v2wdvt1w;
            return(OK);
        case  BSIM4v2_MOD_WDVT2W :             
          value->rValue = model->BSIM4v2wdvt2w;
            return(OK);
        case  BSIM4v2_MOD_WDROUT :           
          value->rValue = model->BSIM4v2wdrout;
            return(OK);
        case  BSIM4v2_MOD_WDSUB :           
          value->rValue = model->BSIM4v2wdsub;
            return(OK);
        case BSIM4v2_MOD_WVTH0:
            value->rValue = model->BSIM4v2wvth0; 
            return(OK);
        case BSIM4v2_MOD_WUA:
            value->rValue = model->BSIM4v2wua; 
            return(OK);
        case BSIM4v2_MOD_WUA1:
            value->rValue = model->BSIM4v2wua1; 
            return(OK);
        case BSIM4v2_MOD_WUB:
            value->rValue = model->BSIM4v2wub;  
            return(OK);
        case BSIM4v2_MOD_WUB1:
            value->rValue = model->BSIM4v2wub1;  
            return(OK);
        case BSIM4v2_MOD_WUC:
            value->rValue = model->BSIM4v2wuc; 
            return(OK);
        case BSIM4v2_MOD_WUC1:
            value->rValue = model->BSIM4v2wuc1; 
            return(OK);
        case BSIM4v2_MOD_WU0:
            value->rValue = model->BSIM4v2wu0;
            return(OK);
        case BSIM4v2_MOD_WUTE:
            value->rValue = model->BSIM4v2wute;
            return(OK);
        case BSIM4v2_MOD_WVOFF:
            value->rValue = model->BSIM4v2wvoff;
            return(OK);
        case BSIM4v2_MOD_WMINV:
            value->rValue = model->BSIM4v2wminv;
            return(OK);
        case BSIM4v2_MOD_WFPROUT:
            value->rValue = model->BSIM4v2wfprout;
            return(OK);
        case BSIM4v2_MOD_WPDITS:
            value->rValue = model->BSIM4v2wpdits;
            return(OK);
        case BSIM4v2_MOD_WPDITSD:
            value->rValue = model->BSIM4v2wpditsd;
            return(OK);
        case BSIM4v2_MOD_WDELTA:
            value->rValue = model->BSIM4v2wdelta;
            return(OK);
        case BSIM4v2_MOD_WRDSW:
            value->rValue = model->BSIM4v2wrdsw; 
            return(OK);             
        case BSIM4v2_MOD_WRDW:
            value->rValue = model->BSIM4v2wrdw;
            return(OK);
        case BSIM4v2_MOD_WRSW:
            value->rValue = model->BSIM4v2wrsw;
            return(OK);
        case BSIM4v2_MOD_WPRWB:
            value->rValue = model->BSIM4v2wprwb; 
            return(OK);             
        case BSIM4v2_MOD_WPRWG:
            value->rValue = model->BSIM4v2wprwg; 
            return(OK);             
        case BSIM4v2_MOD_WPRT:
            value->rValue = model->BSIM4v2wprt; 
            return(OK);              
        case BSIM4v2_MOD_WETA0:
            value->rValue = model->BSIM4v2weta0; 
            return(OK);               
        case BSIM4v2_MOD_WETAB:
            value->rValue = model->BSIM4v2wetab; 
            return(OK);               
        case BSIM4v2_MOD_WPCLM:
            value->rValue = model->BSIM4v2wpclm; 
            return(OK);               
        case BSIM4v2_MOD_WPDIBL1:
            value->rValue = model->BSIM4v2wpdibl1; 
            return(OK);               
        case BSIM4v2_MOD_WPDIBL2:
            value->rValue = model->BSIM4v2wpdibl2; 
            return(OK);               
        case BSIM4v2_MOD_WPDIBLB:
            value->rValue = model->BSIM4v2wpdiblb; 
            return(OK);               
        case BSIM4v2_MOD_WPSCBE1:
            value->rValue = model->BSIM4v2wpscbe1; 
            return(OK);               
        case BSIM4v2_MOD_WPSCBE2:
            value->rValue = model->BSIM4v2wpscbe2; 
            return(OK);               
        case BSIM4v2_MOD_WPVAG:
            value->rValue = model->BSIM4v2wpvag; 
            return(OK);               
        case BSIM4v2_MOD_WWR:
            value->rValue = model->BSIM4v2wwr;
            return(OK);
        case BSIM4v2_MOD_WDWG:
            value->rValue = model->BSIM4v2wdwg;
            return(OK);
        case BSIM4v2_MOD_WDWB:
            value->rValue = model->BSIM4v2wdwb;
            return(OK);
        case BSIM4v2_MOD_WB0:
            value->rValue = model->BSIM4v2wb0;
            return(OK);
        case BSIM4v2_MOD_WB1:
            value->rValue = model->BSIM4v2wb1;
            return(OK);
        case BSIM4v2_MOD_WALPHA0:
            value->rValue = model->BSIM4v2walpha0;
            return(OK);
        case BSIM4v2_MOD_WALPHA1:
            value->rValue = model->BSIM4v2walpha1;
            return(OK);
        case BSIM4v2_MOD_WBETA0:
            value->rValue = model->BSIM4v2wbeta0;
            return(OK);
        case BSIM4v2_MOD_WAGIDL:
            value->rValue = model->BSIM4v2wagidl;
            return(OK);
        case BSIM4v2_MOD_WBGIDL:
            value->rValue = model->BSIM4v2wbgidl;
            return(OK);
        case BSIM4v2_MOD_WCGIDL:
            value->rValue = model->BSIM4v2wcgidl;
            return(OK);
        case BSIM4v2_MOD_WEGIDL:
            value->rValue = model->BSIM4v2wegidl;
            return(OK);
        case BSIM4v2_MOD_WAIGC:
            value->rValue = model->BSIM4v2waigc;
            return(OK);
        case BSIM4v2_MOD_WBIGC:
            value->rValue = model->BSIM4v2wbigc;
            return(OK);
        case BSIM4v2_MOD_WCIGC:
            value->rValue = model->BSIM4v2wcigc;
            return(OK);
        case BSIM4v2_MOD_WAIGSD:
            value->rValue = model->BSIM4v2waigsd;
            return(OK);
        case BSIM4v2_MOD_WBIGSD:
            value->rValue = model->BSIM4v2wbigsd;
            return(OK);
        case BSIM4v2_MOD_WCIGSD:
            value->rValue = model->BSIM4v2wcigsd;
            return(OK);
        case BSIM4v2_MOD_WAIGBACC:
            value->rValue = model->BSIM4v2waigbacc;
            return(OK);
        case BSIM4v2_MOD_WBIGBACC:
            value->rValue = model->BSIM4v2wbigbacc;
            return(OK);
        case BSIM4v2_MOD_WCIGBACC:
            value->rValue = model->BSIM4v2wcigbacc;
            return(OK);
        case BSIM4v2_MOD_WAIGBINV:
            value->rValue = model->BSIM4v2waigbinv;
            return(OK);
        case BSIM4v2_MOD_WBIGBINV:
            value->rValue = model->BSIM4v2wbigbinv;
            return(OK);
        case BSIM4v2_MOD_WCIGBINV:
            value->rValue = model->BSIM4v2wcigbinv;
            return(OK);
        case BSIM4v2_MOD_WNIGC:
            value->rValue = model->BSIM4v2wnigc;
            return(OK);
        case BSIM4v2_MOD_WNIGBACC:
            value->rValue = model->BSIM4v2wnigbacc;
            return(OK);
        case BSIM4v2_MOD_WNIGBINV:
            value->rValue = model->BSIM4v2wnigbinv;
            return(OK);
        case BSIM4v2_MOD_WNTOX:
            value->rValue = model->BSIM4v2wntox;
            return(OK);
        case BSIM4v2_MOD_WEIGBINV:
            value->rValue = model->BSIM4v2weigbinv;
            return(OK);
        case BSIM4v2_MOD_WPIGCD:
            value->rValue = model->BSIM4v2wpigcd;
            return(OK);
        case BSIM4v2_MOD_WPOXEDGE:
            value->rValue = model->BSIM4v2wpoxedge;
            return(OK);
        case BSIM4v2_MOD_WPHIN:
            value->rValue = model->BSIM4v2wphin;
            return(OK);
        case BSIM4v2_MOD_WXRCRG1:
            value->rValue = model->BSIM4v2wxrcrg1;
            return(OK);
        case BSIM4v2_MOD_WXRCRG2:
            value->rValue = model->BSIM4v2wxrcrg2;
            return(OK);
        case BSIM4v2_MOD_WEU:
            value->rValue = model->BSIM4v2weu;
            return(OK);
        case BSIM4v2_MOD_WVFB:
            value->rValue = model->BSIM4v2wvfb;
            return(OK);

        case BSIM4v2_MOD_WCGSL:
            value->rValue = model->BSIM4v2wcgsl;
            return(OK);
        case BSIM4v2_MOD_WCGDL:
            value->rValue = model->BSIM4v2wcgdl;
            return(OK);
        case BSIM4v2_MOD_WCKAPPAS:
            value->rValue = model->BSIM4v2wckappas;
            return(OK);
        case BSIM4v2_MOD_WCKAPPAD:
            value->rValue = model->BSIM4v2wckappad;
            return(OK);
        case BSIM4v2_MOD_WCF:
            value->rValue = model->BSIM4v2wcf;
            return(OK);
        case BSIM4v2_MOD_WCLC:
            value->rValue = model->BSIM4v2wclc;
            return(OK);
        case BSIM4v2_MOD_WCLE:
            value->rValue = model->BSIM4v2wcle;
            return(OK);
        case BSIM4v2_MOD_WVFBCV:
            value->rValue = model->BSIM4v2wvfbcv;
            return(OK);
        case BSIM4v2_MOD_WACDE:
            value->rValue = model->BSIM4v2wacde;
            return(OK);
        case BSIM4v2_MOD_WMOIN:
            value->rValue = model->BSIM4v2wmoin;
            return(OK);
        case BSIM4v2_MOD_WNOFF:
            value->rValue = model->BSIM4v2wnoff;
            return(OK);
        case BSIM4v2_MOD_WVOFFCV:
            value->rValue = model->BSIM4v2wvoffcv;
            return(OK);

	/* Cross-term dependence */
        case  BSIM4v2_MOD_PCDSC :
          value->rValue = model->BSIM4v2pcdsc;
            return(OK);
        case  BSIM4v2_MOD_PCDSCB :
          value->rValue = model->BSIM4v2pcdscb;
            return(OK);
        case  BSIM4v2_MOD_PCDSCD :
          value->rValue = model->BSIM4v2pcdscd;
            return(OK);
         case  BSIM4v2_MOD_PCIT :
          value->rValue = model->BSIM4v2pcit;
            return(OK);
        case  BSIM4v2_MOD_PNFACTOR :
          value->rValue = model->BSIM4v2pnfactor;
            return(OK);
        case BSIM4v2_MOD_PXJ:
            value->rValue = model->BSIM4v2pxj;
            return(OK);
        case BSIM4v2_MOD_PVSAT:
            value->rValue = model->BSIM4v2pvsat;
            return(OK);
        case BSIM4v2_MOD_PAT:
            value->rValue = model->BSIM4v2pat;
            return(OK);
        case BSIM4v2_MOD_PA0:
            value->rValue = model->BSIM4v2pa0;
            return(OK);
        case BSIM4v2_MOD_PAGS:
            value->rValue = model->BSIM4v2pags;
            return(OK);
        case BSIM4v2_MOD_PA1:
            value->rValue = model->BSIM4v2pa1;
            return(OK);
        case BSIM4v2_MOD_PA2:
            value->rValue = model->BSIM4v2pa2;
            return(OK);
        case BSIM4v2_MOD_PKETA:
            value->rValue = model->BSIM4v2pketa;
            return(OK);   
        case BSIM4v2_MOD_PNSUB:
            value->rValue = model->BSIM4v2pnsub;
            return(OK);
        case BSIM4v2_MOD_PNDEP:
            value->rValue = model->BSIM4v2pndep;
            return(OK);
        case BSIM4v2_MOD_PNSD:
            value->rValue = model->BSIM4v2pnsd;
            return(OK);
        case BSIM4v2_MOD_PNGATE:
            value->rValue = model->BSIM4v2pngate;
            return(OK);
        case BSIM4v2_MOD_PGAMMA1:
            value->rValue = model->BSIM4v2pgamma1;
            return(OK);
        case BSIM4v2_MOD_PGAMMA2:
            value->rValue = model->BSIM4v2pgamma2;
            return(OK);
        case BSIM4v2_MOD_PVBX:
            value->rValue = model->BSIM4v2pvbx;
            return(OK);
        case BSIM4v2_MOD_PVBM:
            value->rValue = model->BSIM4v2pvbm;
            return(OK);
        case BSIM4v2_MOD_PXT:
            value->rValue = model->BSIM4v2pxt;
            return(OK);
        case  BSIM4v2_MOD_PK1:
          value->rValue = model->BSIM4v2pk1;
            return(OK);
        case  BSIM4v2_MOD_PKT1:
          value->rValue = model->BSIM4v2pkt1;
            return(OK);
        case  BSIM4v2_MOD_PKT1L:
          value->rValue = model->BSIM4v2pkt1l;
            return(OK);
        case  BSIM4v2_MOD_PKT2 :
          value->rValue = model->BSIM4v2pkt2;
            return(OK);
        case  BSIM4v2_MOD_PK2 :
          value->rValue = model->BSIM4v2pk2;
            return(OK);
        case  BSIM4v2_MOD_PK3:
          value->rValue = model->BSIM4v2pk3;
            return(OK);
        case  BSIM4v2_MOD_PK3B:
          value->rValue = model->BSIM4v2pk3b;
            return(OK);
        case  BSIM4v2_MOD_PW0:
          value->rValue = model->BSIM4v2pw0;
            return(OK);
        case  BSIM4v2_MOD_PLPE0:
          value->rValue = model->BSIM4v2plpe0;
            return(OK);
        case  BSIM4v2_MOD_PLPEB:
          value->rValue = model->BSIM4v2plpeb;
            return(OK);
        case  BSIM4v2_MOD_PDVTP0:
          value->rValue = model->BSIM4v2pdvtp0;
            return(OK);
        case  BSIM4v2_MOD_PDVTP1:
          value->rValue = model->BSIM4v2pdvtp1;
            return(OK);
        case  BSIM4v2_MOD_PDVT0 :                
          value->rValue = model->BSIM4v2pdvt0;
            return(OK);
        case  BSIM4v2_MOD_PDVT1 :             
          value->rValue = model->BSIM4v2pdvt1;
            return(OK);
        case  BSIM4v2_MOD_PDVT2 :             
          value->rValue = model->BSIM4v2pdvt2;
            return(OK);
        case  BSIM4v2_MOD_PDVT0W :                
          value->rValue = model->BSIM4v2pdvt0w;
            return(OK);
        case  BSIM4v2_MOD_PDVT1W :             
          value->rValue = model->BSIM4v2pdvt1w;
            return(OK);
        case  BSIM4v2_MOD_PDVT2W :             
          value->rValue = model->BSIM4v2pdvt2w;
            return(OK);
        case  BSIM4v2_MOD_PDROUT :           
          value->rValue = model->BSIM4v2pdrout;
            return(OK);
        case  BSIM4v2_MOD_PDSUB :           
          value->rValue = model->BSIM4v2pdsub;
            return(OK);
        case BSIM4v2_MOD_PVTH0:
            value->rValue = model->BSIM4v2pvth0; 
            return(OK);
        case BSIM4v2_MOD_PUA:
            value->rValue = model->BSIM4v2pua; 
            return(OK);
        case BSIM4v2_MOD_PUA1:
            value->rValue = model->BSIM4v2pua1; 
            return(OK);
        case BSIM4v2_MOD_PUB:
            value->rValue = model->BSIM4v2pub;  
            return(OK);
        case BSIM4v2_MOD_PUB1:
            value->rValue = model->BSIM4v2pub1;  
            return(OK);
        case BSIM4v2_MOD_PUC:
            value->rValue = model->BSIM4v2puc; 
            return(OK);
        case BSIM4v2_MOD_PUC1:
            value->rValue = model->BSIM4v2puc1; 
            return(OK);
        case BSIM4v2_MOD_PU0:
            value->rValue = model->BSIM4v2pu0;
            return(OK);
        case BSIM4v2_MOD_PUTE:
            value->rValue = model->BSIM4v2pute;
            return(OK);
        case BSIM4v2_MOD_PVOFF:
            value->rValue = model->BSIM4v2pvoff;
            return(OK);
        case BSIM4v2_MOD_PMINV:
            value->rValue = model->BSIM4v2pminv;
            return(OK);
        case BSIM4v2_MOD_PFPROUT:
            value->rValue = model->BSIM4v2pfprout;
            return(OK);
        case BSIM4v2_MOD_PPDITS:
            value->rValue = model->BSIM4v2ppdits;
            return(OK);
        case BSIM4v2_MOD_PPDITSD:
            value->rValue = model->BSIM4v2ppditsd;
            return(OK);
        case BSIM4v2_MOD_PDELTA:
            value->rValue = model->BSIM4v2pdelta;
            return(OK);
        case BSIM4v2_MOD_PRDSW:
            value->rValue = model->BSIM4v2prdsw; 
            return(OK);             
        case BSIM4v2_MOD_PRDW:
            value->rValue = model->BSIM4v2prdw;
            return(OK);
        case BSIM4v2_MOD_PRSW:
            value->rValue = model->BSIM4v2prsw;
            return(OK);
        case BSIM4v2_MOD_PPRWB:
            value->rValue = model->BSIM4v2pprwb; 
            return(OK);             
        case BSIM4v2_MOD_PPRWG:
            value->rValue = model->BSIM4v2pprwg; 
            return(OK);             
        case BSIM4v2_MOD_PPRT:
            value->rValue = model->BSIM4v2pprt; 
            return(OK);              
        case BSIM4v2_MOD_PETA0:
            value->rValue = model->BSIM4v2peta0; 
            return(OK);               
        case BSIM4v2_MOD_PETAB:
            value->rValue = model->BSIM4v2petab; 
            return(OK);               
        case BSIM4v2_MOD_PPCLM:
            value->rValue = model->BSIM4v2ppclm; 
            return(OK);               
        case BSIM4v2_MOD_PPDIBL1:
            value->rValue = model->BSIM4v2ppdibl1; 
            return(OK);               
        case BSIM4v2_MOD_PPDIBL2:
            value->rValue = model->BSIM4v2ppdibl2; 
            return(OK);               
        case BSIM4v2_MOD_PPDIBLB:
            value->rValue = model->BSIM4v2ppdiblb; 
            return(OK);               
        case BSIM4v2_MOD_PPSCBE1:
            value->rValue = model->BSIM4v2ppscbe1; 
            return(OK);               
        case BSIM4v2_MOD_PPSCBE2:
            value->rValue = model->BSIM4v2ppscbe2; 
            return(OK);               
        case BSIM4v2_MOD_PPVAG:
            value->rValue = model->BSIM4v2ppvag; 
            return(OK);               
        case BSIM4v2_MOD_PWR:
            value->rValue = model->BSIM4v2pwr;
            return(OK);
        case BSIM4v2_MOD_PDWG:
            value->rValue = model->BSIM4v2pdwg;
            return(OK);
        case BSIM4v2_MOD_PDWB:
            value->rValue = model->BSIM4v2pdwb;
            return(OK);
        case BSIM4v2_MOD_PB0:
            value->rValue = model->BSIM4v2pb0;
            return(OK);
        case BSIM4v2_MOD_PB1:
            value->rValue = model->BSIM4v2pb1;
            return(OK);
        case BSIM4v2_MOD_PALPHA0:
            value->rValue = model->BSIM4v2palpha0;
            return(OK);
        case BSIM4v2_MOD_PALPHA1:
            value->rValue = model->BSIM4v2palpha1;
            return(OK);
        case BSIM4v2_MOD_PBETA0:
            value->rValue = model->BSIM4v2pbeta0;
            return(OK);
        case BSIM4v2_MOD_PAGIDL:
            value->rValue = model->BSIM4v2pagidl;
            return(OK);
        case BSIM4v2_MOD_PBGIDL:
            value->rValue = model->BSIM4v2pbgidl;
            return(OK);
        case BSIM4v2_MOD_PCGIDL:
            value->rValue = model->BSIM4v2pcgidl;
            return(OK);
        case BSIM4v2_MOD_PEGIDL:
            value->rValue = model->BSIM4v2pegidl;
            return(OK);
        case BSIM4v2_MOD_PAIGC:
            value->rValue = model->BSIM4v2paigc;
            return(OK);
        case BSIM4v2_MOD_PBIGC:
            value->rValue = model->BSIM4v2pbigc;
            return(OK);
        case BSIM4v2_MOD_PCIGC:
            value->rValue = model->BSIM4v2pcigc;
            return(OK);
        case BSIM4v2_MOD_PAIGSD:
            value->rValue = model->BSIM4v2paigsd;
            return(OK);
        case BSIM4v2_MOD_PBIGSD:
            value->rValue = model->BSIM4v2pbigsd;
            return(OK);
        case BSIM4v2_MOD_PCIGSD:
            value->rValue = model->BSIM4v2pcigsd;
            return(OK);
        case BSIM4v2_MOD_PAIGBACC:
            value->rValue = model->BSIM4v2paigbacc;
            return(OK);
        case BSIM4v2_MOD_PBIGBACC:
            value->rValue = model->BSIM4v2pbigbacc;
            return(OK);
        case BSIM4v2_MOD_PCIGBACC:
            value->rValue = model->BSIM4v2pcigbacc;
            return(OK);
        case BSIM4v2_MOD_PAIGBINV:
            value->rValue = model->BSIM4v2paigbinv;
            return(OK);
        case BSIM4v2_MOD_PBIGBINV:
            value->rValue = model->BSIM4v2pbigbinv;
            return(OK);
        case BSIM4v2_MOD_PCIGBINV:
            value->rValue = model->BSIM4v2pcigbinv;
            return(OK);
        case BSIM4v2_MOD_PNIGC:
            value->rValue = model->BSIM4v2pnigc;
            return(OK);
        case BSIM4v2_MOD_PNIGBACC:
            value->rValue = model->BSIM4v2pnigbacc;
            return(OK);
        case BSIM4v2_MOD_PNIGBINV:
            value->rValue = model->BSIM4v2pnigbinv;
            return(OK);
        case BSIM4v2_MOD_PNTOX:
            value->rValue = model->BSIM4v2pntox;
            return(OK);
        case BSIM4v2_MOD_PEIGBINV:
            value->rValue = model->BSIM4v2peigbinv;
            return(OK);
        case BSIM4v2_MOD_PPIGCD:
            value->rValue = model->BSIM4v2ppigcd;
            return(OK);
        case BSIM4v2_MOD_PPOXEDGE:
            value->rValue = model->BSIM4v2ppoxedge;
            return(OK);
        case BSIM4v2_MOD_PPHIN:
            value->rValue = model->BSIM4v2pphin;
            return(OK);
        case BSIM4v2_MOD_PXRCRG1:
            value->rValue = model->BSIM4v2pxrcrg1;
            return(OK);
        case BSIM4v2_MOD_PXRCRG2:
            value->rValue = model->BSIM4v2pxrcrg2;
            return(OK);
        case BSIM4v2_MOD_PEU:
            value->rValue = model->BSIM4v2peu;
            return(OK);
        case BSIM4v2_MOD_PVFB:
            value->rValue = model->BSIM4v2pvfb;
            return(OK);

        case BSIM4v2_MOD_PCGSL:
            value->rValue = model->BSIM4v2pcgsl;
            return(OK);
        case BSIM4v2_MOD_PCGDL:
            value->rValue = model->BSIM4v2pcgdl;
            return(OK);
        case BSIM4v2_MOD_PCKAPPAS:
            value->rValue = model->BSIM4v2pckappas;
            return(OK);
        case BSIM4v2_MOD_PCKAPPAD:
            value->rValue = model->BSIM4v2pckappad;
            return(OK);
        case BSIM4v2_MOD_PCF:
            value->rValue = model->BSIM4v2pcf;
            return(OK);
        case BSIM4v2_MOD_PCLC:
            value->rValue = model->BSIM4v2pclc;
            return(OK);
        case BSIM4v2_MOD_PCLE:
            value->rValue = model->BSIM4v2pcle;
            return(OK);
        case BSIM4v2_MOD_PVFBCV:
            value->rValue = model->BSIM4v2pvfbcv;
            return(OK);
        case BSIM4v2_MOD_PACDE:
            value->rValue = model->BSIM4v2pacde;
            return(OK);
        case BSIM4v2_MOD_PMOIN:
            value->rValue = model->BSIM4v2pmoin;
            return(OK);
        case BSIM4v2_MOD_PNOFF:
            value->rValue = model->BSIM4v2pnoff;
            return(OK);
        case BSIM4v2_MOD_PVOFFCV:
            value->rValue = model->BSIM4v2pvoffcv;
            return(OK);

        case  BSIM4v2_MOD_TNOM :
          value->rValue = model->BSIM4v2tnom;
            return(OK);
        case BSIM4v2_MOD_CGSO:
            value->rValue = model->BSIM4v2cgso; 
            return(OK);
        case BSIM4v2_MOD_CGDO:
            value->rValue = model->BSIM4v2cgdo; 
            return(OK);
        case BSIM4v2_MOD_CGBO:
            value->rValue = model->BSIM4v2cgbo; 
            return(OK);
        case BSIM4v2_MOD_XPART:
            value->rValue = model->BSIM4v2xpart; 
            return(OK);
        case BSIM4v2_MOD_RSH:
            value->rValue = model->BSIM4v2sheetResistance; 
            return(OK);
        case BSIM4v2_MOD_JSS:
            value->rValue = model->BSIM4v2SjctSatCurDensity; 
            return(OK);
        case BSIM4v2_MOD_JSWS:
            value->rValue = model->BSIM4v2SjctSidewallSatCurDensity; 
            return(OK);
        case BSIM4v2_MOD_JSWGS:
            value->rValue = model->BSIM4v2SjctGateSidewallSatCurDensity;
            return(OK);
        case BSIM4v2_MOD_PBS:
            value->rValue = model->BSIM4v2SbulkJctPotential; 
            return(OK);
        case BSIM4v2_MOD_MJS:
            value->rValue = model->BSIM4v2SbulkJctBotGradingCoeff; 
            return(OK);
        case BSIM4v2_MOD_PBSWS:
            value->rValue = model->BSIM4v2SsidewallJctPotential; 
            return(OK);
        case BSIM4v2_MOD_MJSWS:
            value->rValue = model->BSIM4v2SbulkJctSideGradingCoeff; 
            return(OK);
        case BSIM4v2_MOD_CJS:
            value->rValue = model->BSIM4v2SunitAreaJctCap; 
            return(OK);
        case BSIM4v2_MOD_CJSWS:
            value->rValue = model->BSIM4v2SunitLengthSidewallJctCap; 
            return(OK);
        case BSIM4v2_MOD_PBSWGS:
            value->rValue = model->BSIM4v2SGatesidewallJctPotential; 
            return(OK);
        case BSIM4v2_MOD_MJSWGS:
            value->rValue = model->BSIM4v2SbulkJctGateSideGradingCoeff; 
            return(OK);
        case BSIM4v2_MOD_CJSWGS:
            value->rValue = model->BSIM4v2SunitLengthGateSidewallJctCap; 
            return(OK);
        case BSIM4v2_MOD_NJS:
            value->rValue = model->BSIM4v2SjctEmissionCoeff; 
            return(OK);
        case BSIM4v2_MOD_XTIS:
            value->rValue = model->BSIM4v2SjctTempExponent; 
            return(OK);
        case BSIM4v2_MOD_JSD:
            value->rValue = model->BSIM4v2DjctSatCurDensity;
            return(OK);
        case BSIM4v2_MOD_JSWD:
            value->rValue = model->BSIM4v2DjctSidewallSatCurDensity;
            return(OK);
        case BSIM4v2_MOD_JSWGD:
            value->rValue = model->BSIM4v2DjctGateSidewallSatCurDensity;
            return(OK);
        case BSIM4v2_MOD_PBD:
            value->rValue = model->BSIM4v2DbulkJctPotential;
            return(OK);
        case BSIM4v2_MOD_MJD:
            value->rValue = model->BSIM4v2DbulkJctBotGradingCoeff;
            return(OK);
        case BSIM4v2_MOD_PBSWD:
            value->rValue = model->BSIM4v2DsidewallJctPotential;
            return(OK);
        case BSIM4v2_MOD_MJSWD:
            value->rValue = model->BSIM4v2DbulkJctSideGradingCoeff;
            return(OK);
        case BSIM4v2_MOD_CJD:
            value->rValue = model->BSIM4v2DunitAreaJctCap;
            return(OK);
        case BSIM4v2_MOD_CJSWD:
            value->rValue = model->BSIM4v2DunitLengthSidewallJctCap;
            return(OK);
        case BSIM4v2_MOD_PBSWGD:
            value->rValue = model->BSIM4v2DGatesidewallJctPotential;
            return(OK);
        case BSIM4v2_MOD_MJSWGD:
            value->rValue = model->BSIM4v2DbulkJctGateSideGradingCoeff;
            return(OK);
        case BSIM4v2_MOD_CJSWGD:
            value->rValue = model->BSIM4v2DunitLengthGateSidewallJctCap;
            return(OK);
        case BSIM4v2_MOD_NJD:
            value->rValue = model->BSIM4v2DjctEmissionCoeff;
            return(OK);
        case BSIM4v2_MOD_XTID:
            value->rValue = model->BSIM4v2DjctTempExponent;
            return(OK);
        case BSIM4v2_MOD_LINT:
            value->rValue = model->BSIM4v2Lint; 
            return(OK);
        case BSIM4v2_MOD_LL:
            value->rValue = model->BSIM4v2Ll;
            return(OK);
        case BSIM4v2_MOD_LLC:
            value->rValue = model->BSIM4v2Llc;
            return(OK);
        case BSIM4v2_MOD_LLN:
            value->rValue = model->BSIM4v2Lln;
            return(OK);
        case BSIM4v2_MOD_LW:
            value->rValue = model->BSIM4v2Lw;
            return(OK);
        case BSIM4v2_MOD_LWC:
            value->rValue = model->BSIM4v2Lwc;
            return(OK);
        case BSIM4v2_MOD_LWN:
            value->rValue = model->BSIM4v2Lwn;
            return(OK);
        case BSIM4v2_MOD_LWL:
            value->rValue = model->BSIM4v2Lwl;
            return(OK);
        case BSIM4v2_MOD_LWLC:
            value->rValue = model->BSIM4v2Lwlc;
            return(OK);
        case BSIM4v2_MOD_LMIN:
            value->rValue = model->BSIM4v2Lmin;
            return(OK);
        case BSIM4v2_MOD_LMAX:
            value->rValue = model->BSIM4v2Lmax;
            return(OK);
        case BSIM4v2_MOD_WINT:
            value->rValue = model->BSIM4v2Wint;
            return(OK);
        case BSIM4v2_MOD_WL:
            value->rValue = model->BSIM4v2Wl;
            return(OK);
        case BSIM4v2_MOD_WLC:
            value->rValue = model->BSIM4v2Wlc;
            return(OK);
        case BSIM4v2_MOD_WLN:
            value->rValue = model->BSIM4v2Wln;
            return(OK);
        case BSIM4v2_MOD_WW:
            value->rValue = model->BSIM4v2Ww;
            return(OK);
        case BSIM4v2_MOD_WWC:
            value->rValue = model->BSIM4v2Wwc;
            return(OK);
        case BSIM4v2_MOD_WWN:
            value->rValue = model->BSIM4v2Wwn;
            return(OK);
        case BSIM4v2_MOD_WWL:
            value->rValue = model->BSIM4v2Wwl;
            return(OK);
        case BSIM4v2_MOD_WWLC:
            value->rValue = model->BSIM4v2Wwlc;
            return(OK);
        case BSIM4v2_MOD_WMIN:
            value->rValue = model->BSIM4v2Wmin;
            return(OK);
        case BSIM4v2_MOD_WMAX:
            value->rValue = model->BSIM4v2Wmax;
            return(OK);
        case BSIM4v2_MOD_NOIA:
            value->rValue = model->BSIM4v2oxideTrapDensityA;
            return(OK);
        case BSIM4v2_MOD_NOIB:
            value->rValue = model->BSIM4v2oxideTrapDensityB;
            return(OK);
        case BSIM4v2_MOD_NOIC:
            value->rValue = model->BSIM4v2oxideTrapDensityC;
            return(OK);
        case BSIM4v2_MOD_EM:
            value->rValue = model->BSIM4v2em;
            return(OK);
        case BSIM4v2_MOD_EF:
            value->rValue = model->BSIM4v2ef;
            return(OK);
        case BSIM4v2_MOD_AF:
            value->rValue = model->BSIM4v2af;
            return(OK);
        case BSIM4v2_MOD_KF:
            value->rValue = model->BSIM4v2kf;
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}



