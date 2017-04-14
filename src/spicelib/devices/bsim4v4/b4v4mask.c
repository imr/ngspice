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


#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "bsim4v4def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
BSIM4v4mAsk(ckt,inst,which,value)
CKTcircuit *ckt;
GENmodel *inst;
int which;
IFvalue *value;
{
    BSIM4v4model *model = (BSIM4v4model *)inst;
    switch(which)
    {   case BSIM4v4_MOD_MOBMOD :
            value->iValue = model->BSIM4v4mobMod;
            return(OK);
        case BSIM4v4_MOD_PARAMCHK :
            value->iValue = model->BSIM4v4paramChk;
            return(OK);
        case BSIM4v4_MOD_BINUNIT :
            value->iValue = model->BSIM4v4binUnit;
            return(OK);
        case BSIM4v4_MOD_CAPMOD :
            value->iValue = model->BSIM4v4capMod;
            return(OK);
        case BSIM4v4_MOD_DIOMOD :
            value->iValue = model->BSIM4v4dioMod;
            return(OK);
        case BSIM4v4_MOD_TRNQSMOD :
            value->iValue = model->BSIM4v4trnqsMod;
            return(OK);
        case BSIM4v4_MOD_ACNQSMOD :
            value->iValue = model->BSIM4v4acnqsMod;
            return(OK);
        case BSIM4v4_MOD_FNOIMOD :
            value->iValue = model->BSIM4v4fnoiMod;
            return(OK);
        case BSIM4v4_MOD_TNOIMOD :
            value->iValue = model->BSIM4v4tnoiMod;
            return(OK);
        case BSIM4v4_MOD_RDSMOD :
            value->iValue = model->BSIM4v4rdsMod;
            return(OK);
        case BSIM4v4_MOD_RBODYMOD :
            value->iValue = model->BSIM4v4rbodyMod;
            return(OK);
        case BSIM4v4_MOD_RGATEMOD :
            value->iValue = model->BSIM4v4rgateMod;
            return(OK);
        case BSIM4v4_MOD_PERMOD :
            value->iValue = model->BSIM4v4perMod;
            return(OK);
        case BSIM4v4_MOD_GEOMOD :
            value->iValue = model->BSIM4v4geoMod;
            return(OK);
        case BSIM4v4_MOD_IGCMOD :
            value->iValue = model->BSIM4v4igcMod;
            return(OK);
        case BSIM4v4_MOD_IGBMOD :
            value->iValue = model->BSIM4v4igbMod;
            return(OK);
        case  BSIM4v4_MOD_TEMPMOD :
            value->iValue = model->BSIM4v4tempMod;
            return(OK);
        case  BSIM4v4_MOD_VERSION :
          value->sValue = model->BSIM4v4version;
            return(OK);
        case  BSIM4v4_MOD_TOXREF :
          value->rValue = model->BSIM4v4toxref;
          return(OK);
        case  BSIM4v4_MOD_TOXE :
          value->rValue = model->BSIM4v4toxe;
            return(OK);
        case  BSIM4v4_MOD_TOXP :
          value->rValue = model->BSIM4v4toxp;
            return(OK);
        case  BSIM4v4_MOD_TOXM :
          value->rValue = model->BSIM4v4toxm;
            return(OK);
        case  BSIM4v4_MOD_DTOX :
          value->rValue = model->BSIM4v4dtox;
            return(OK);
        case  BSIM4v4_MOD_EPSROX :
          value->rValue = model->BSIM4v4epsrox;
            return(OK);
        case  BSIM4v4_MOD_CDSC :
          value->rValue = model->BSIM4v4cdsc;
            return(OK);
        case  BSIM4v4_MOD_CDSCB :
          value->rValue = model->BSIM4v4cdscb;
            return(OK);

        case  BSIM4v4_MOD_CDSCD :
          value->rValue = model->BSIM4v4cdscd;
            return(OK);

        case  BSIM4v4_MOD_CIT :
          value->rValue = model->BSIM4v4cit;
            return(OK);
        case  BSIM4v4_MOD_NFACTOR :
          value->rValue = model->BSIM4v4nfactor;
            return(OK);
        case BSIM4v4_MOD_XJ:
            value->rValue = model->BSIM4v4xj;
            return(OK);
        case BSIM4v4_MOD_VSAT:
            value->rValue = model->BSIM4v4vsat;
            return(OK);
        case BSIM4v4_MOD_VTL:
            value->rValue = model->BSIM4v4vtl;
            return(OK);
        case BSIM4v4_MOD_XN:
            value->rValue = model->BSIM4v4xn;
            return(OK);
        case BSIM4v4_MOD_LC:
            value->rValue = model->BSIM4v4lc;
            return(OK);
        case BSIM4v4_MOD_LAMBDA:
            value->rValue = model->BSIM4v4lambda;
            return(OK);
        case BSIM4v4_MOD_AT:
            value->rValue = model->BSIM4v4at;
            return(OK);
        case BSIM4v4_MOD_A0:
            value->rValue = model->BSIM4v4a0;
            return(OK);

        case BSIM4v4_MOD_AGS:
            value->rValue = model->BSIM4v4ags;
            return(OK);

        case BSIM4v4_MOD_A1:
            value->rValue = model->BSIM4v4a1;
            return(OK);
        case BSIM4v4_MOD_A2:
            value->rValue = model->BSIM4v4a2;
            return(OK);
        case BSIM4v4_MOD_KETA:
            value->rValue = model->BSIM4v4keta;
            return(OK);
        case BSIM4v4_MOD_NSUB:
            value->rValue = model->BSIM4v4nsub;
            return(OK);
        case BSIM4v4_MOD_NDEP:
            value->rValue = model->BSIM4v4ndep;
            return(OK);
        case BSIM4v4_MOD_NSD:
            value->rValue = model->BSIM4v4nsd;
            return(OK);
        case BSIM4v4_MOD_NGATE:
            value->rValue = model->BSIM4v4ngate;
            return(OK);
        case BSIM4v4_MOD_GAMMA1:
            value->rValue = model->BSIM4v4gamma1;
            return(OK);
        case BSIM4v4_MOD_GAMMA2:
            value->rValue = model->BSIM4v4gamma2;
            return(OK);
        case BSIM4v4_MOD_VBX:
            value->rValue = model->BSIM4v4vbx;
            return(OK);
        case BSIM4v4_MOD_VBM:
            value->rValue = model->BSIM4v4vbm;
            return(OK);
        case BSIM4v4_MOD_XT:
            value->rValue = model->BSIM4v4xt;
            return(OK);
        case  BSIM4v4_MOD_K1:
          value->rValue = model->BSIM4v4k1;
            return(OK);
        case  BSIM4v4_MOD_KT1:
          value->rValue = model->BSIM4v4kt1;
            return(OK);
        case  BSIM4v4_MOD_KT1L:
          value->rValue = model->BSIM4v4kt1l;
            return(OK);
        case  BSIM4v4_MOD_KT2 :
          value->rValue = model->BSIM4v4kt2;
            return(OK);
        case  BSIM4v4_MOD_K2 :
          value->rValue = model->BSIM4v4k2;
            return(OK);
        case  BSIM4v4_MOD_K3:
          value->rValue = model->BSIM4v4k3;
            return(OK);
        case  BSIM4v4_MOD_K3B:
          value->rValue = model->BSIM4v4k3b;
            return(OK);
        case  BSIM4v4_MOD_W0:
          value->rValue = model->BSIM4v4w0;
            return(OK);
        case  BSIM4v4_MOD_LPE0:
          value->rValue = model->BSIM4v4lpe0;
            return(OK);
        case  BSIM4v4_MOD_LPEB:
          value->rValue = model->BSIM4v4lpeb;
            return(OK);
        case  BSIM4v4_MOD_DVTP0:
          value->rValue = model->BSIM4v4dvtp0;
            return(OK);
        case  BSIM4v4_MOD_DVTP1:
          value->rValue = model->BSIM4v4dvtp1;
            return(OK);
        case  BSIM4v4_MOD_DVT0 :
          value->rValue = model->BSIM4v4dvt0;
            return(OK);
        case  BSIM4v4_MOD_DVT1 :
          value->rValue = model->BSIM4v4dvt1;
            return(OK);
        case  BSIM4v4_MOD_DVT2 :
          value->rValue = model->BSIM4v4dvt2;
            return(OK);
        case  BSIM4v4_MOD_DVT0W :
          value->rValue = model->BSIM4v4dvt0w;
            return(OK);
        case  BSIM4v4_MOD_DVT1W :
          value->rValue = model->BSIM4v4dvt1w;
            return(OK);
        case  BSIM4v4_MOD_DVT2W :
          value->rValue = model->BSIM4v4dvt2w;
            return(OK);
        case  BSIM4v4_MOD_DROUT :
          value->rValue = model->BSIM4v4drout;
            return(OK);
        case  BSIM4v4_MOD_DSUB :
          value->rValue = model->BSIM4v4dsub;
            return(OK);
        case BSIM4v4_MOD_VTH0:
            value->rValue = model->BSIM4v4vth0;
            return(OK);
        case BSIM4v4_MOD_EU:
            value->rValue = model->BSIM4v4eu;
            return(OK);
        case BSIM4v4_MOD_UA:
            value->rValue = model->BSIM4v4ua;
            return(OK);
        case BSIM4v4_MOD_UA1:
            value->rValue = model->BSIM4v4ua1;
            return(OK);
        case BSIM4v4_MOD_UB:
            value->rValue = model->BSIM4v4ub;
            return(OK);
        case BSIM4v4_MOD_UB1:
            value->rValue = model->BSIM4v4ub1;
            return(OK);
        case BSIM4v4_MOD_UC:
            value->rValue = model->BSIM4v4uc;
            return(OK);
        case BSIM4v4_MOD_UC1:
            value->rValue = model->BSIM4v4uc1;
            return(OK);
        case BSIM4v4_MOD_U0:
            value->rValue = model->BSIM4v4u0;
            return(OK);
        case BSIM4v4_MOD_UTE:
            value->rValue = model->BSIM4v4ute;
            return(OK);
        case BSIM4v4_MOD_VOFF:
            value->rValue = model->BSIM4v4voff;
            return(OK);
        case BSIM4v4_MOD_VOFFL:
            value->rValue = model->BSIM4v4voffl;
            return(OK);
        case BSIM4v4_MOD_MINV:
            value->rValue = model->BSIM4v4minv;
            return(OK);
        case BSIM4v4_MOD_FPROUT:
            value->rValue = model->BSIM4v4fprout;
            return(OK);
        case BSIM4v4_MOD_PDITS:
            value->rValue = model->BSIM4v4pdits;
            return(OK);
        case BSIM4v4_MOD_PDITSD:
            value->rValue = model->BSIM4v4pditsd;
            return(OK);
        case BSIM4v4_MOD_PDITSL:
            value->rValue = model->BSIM4v4pditsl;
            return(OK);
        case BSIM4v4_MOD_DELTA:
            value->rValue = model->BSIM4v4delta;
            return(OK);
        case BSIM4v4_MOD_RDSW:
            value->rValue = model->BSIM4v4rdsw;
            return(OK);
        case BSIM4v4_MOD_RDSWMIN:
            value->rValue = model->BSIM4v4rdswmin;
            return(OK);
        case BSIM4v4_MOD_RDWMIN:
            value->rValue = model->BSIM4v4rdwmin;
            return(OK);
        case BSIM4v4_MOD_RSWMIN:
            value->rValue = model->BSIM4v4rswmin;
            return(OK);
        case BSIM4v4_MOD_RDW:
            value->rValue = model->BSIM4v4rdw;
            return(OK);
        case BSIM4v4_MOD_RSW:
            value->rValue = model->BSIM4v4rsw;
            return(OK);
        case BSIM4v4_MOD_PRWG:
            value->rValue = model->BSIM4v4prwg;
            return(OK);
        case BSIM4v4_MOD_PRWB:
            value->rValue = model->BSIM4v4prwb;
            return(OK);
        case BSIM4v4_MOD_PRT:
            value->rValue = model->BSIM4v4prt;
            return(OK);
        case BSIM4v4_MOD_ETA0:
            value->rValue = model->BSIM4v4eta0;
            return(OK);
        case BSIM4v4_MOD_ETAB:
            value->rValue = model->BSIM4v4etab;
            return(OK);
        case BSIM4v4_MOD_PCLM:
            value->rValue = model->BSIM4v4pclm;
            return(OK);
        case BSIM4v4_MOD_PDIBL1:
            value->rValue = model->BSIM4v4pdibl1;
            return(OK);
        case BSIM4v4_MOD_PDIBL2:
            value->rValue = model->BSIM4v4pdibl2;
            return(OK);
        case BSIM4v4_MOD_PDIBLB:
            value->rValue = model->BSIM4v4pdiblb;
            return(OK);
        case BSIM4v4_MOD_PSCBE1:
            value->rValue = model->BSIM4v4pscbe1;
            return(OK);
        case BSIM4v4_MOD_PSCBE2:
            value->rValue = model->BSIM4v4pscbe2;
            return(OK);
        case BSIM4v4_MOD_PVAG:
            value->rValue = model->BSIM4v4pvag;
            return(OK);
        case BSIM4v4_MOD_WR:
            value->rValue = model->BSIM4v4wr;
            return(OK);
        case BSIM4v4_MOD_DWG:
            value->rValue = model->BSIM4v4dwg;
            return(OK);
        case BSIM4v4_MOD_DWB:
            value->rValue = model->BSIM4v4dwb;
            return(OK);
        case BSIM4v4_MOD_B0:
            value->rValue = model->BSIM4v4b0;
            return(OK);
        case BSIM4v4_MOD_B1:
            value->rValue = model->BSIM4v4b1;
            return(OK);
        case BSIM4v4_MOD_ALPHA0:
            value->rValue = model->BSIM4v4alpha0;
            return(OK);
        case BSIM4v4_MOD_ALPHA1:
            value->rValue = model->BSIM4v4alpha1;
            return(OK);
        case BSIM4v4_MOD_BETA0:
            value->rValue = model->BSIM4v4beta0;
            return(OK);
        case BSIM4v4_MOD_AGIDL:
            value->rValue = model->BSIM4v4agidl;
            return(OK);
        case BSIM4v4_MOD_BGIDL:
            value->rValue = model->BSIM4v4bgidl;
            return(OK);
        case BSIM4v4_MOD_CGIDL:
            value->rValue = model->BSIM4v4cgidl;
            return(OK);
        case BSIM4v4_MOD_EGIDL:
            value->rValue = model->BSIM4v4egidl;
            return(OK);
        case BSIM4v4_MOD_AIGC:
            value->rValue = model->BSIM4v4aigc;
            return(OK);
        case BSIM4v4_MOD_BIGC:
            value->rValue = model->BSIM4v4bigc;
            return(OK);
        case BSIM4v4_MOD_CIGC:
            value->rValue = model->BSIM4v4cigc;
            return(OK);
        case BSIM4v4_MOD_AIGSD:
            value->rValue = model->BSIM4v4aigsd;
            return(OK);
        case BSIM4v4_MOD_BIGSD:
            value->rValue = model->BSIM4v4bigsd;
            return(OK);
        case BSIM4v4_MOD_CIGSD:
            value->rValue = model->BSIM4v4cigsd;
            return(OK);
        case BSIM4v4_MOD_AIGBACC:
            value->rValue = model->BSIM4v4aigbacc;
            return(OK);
        case BSIM4v4_MOD_BIGBACC:
            value->rValue = model->BSIM4v4bigbacc;
            return(OK);
        case BSIM4v4_MOD_CIGBACC:
            value->rValue = model->BSIM4v4cigbacc;
            return(OK);
        case BSIM4v4_MOD_AIGBINV:
            value->rValue = model->BSIM4v4aigbinv;
            return(OK);
        case BSIM4v4_MOD_BIGBINV:
            value->rValue = model->BSIM4v4bigbinv;
            return(OK);
        case BSIM4v4_MOD_CIGBINV:
            value->rValue = model->BSIM4v4cigbinv;
            return(OK);
        case BSIM4v4_MOD_NIGC:
            value->rValue = model->BSIM4v4nigc;
            return(OK);
        case BSIM4v4_MOD_NIGBACC:
            value->rValue = model->BSIM4v4nigbacc;
            return(OK);
        case BSIM4v4_MOD_NIGBINV:
            value->rValue = model->BSIM4v4nigbinv;
            return(OK);
        case BSIM4v4_MOD_NTOX:
            value->rValue = model->BSIM4v4ntox;
            return(OK);
        case BSIM4v4_MOD_EIGBINV:
            value->rValue = model->BSIM4v4eigbinv;
            return(OK);
        case BSIM4v4_MOD_PIGCD:
            value->rValue = model->BSIM4v4pigcd;
            return(OK);
        case BSIM4v4_MOD_POXEDGE:
            value->rValue = model->BSIM4v4poxedge;
            return(OK);
        case BSIM4v4_MOD_PHIN:
            value->rValue = model->BSIM4v4phin;
            return(OK);
        case BSIM4v4_MOD_XRCRG1:
            value->rValue = model->BSIM4v4xrcrg1;
            return(OK);
        case BSIM4v4_MOD_XRCRG2:
            value->rValue = model->BSIM4v4xrcrg2;
            return(OK);
        case BSIM4v4_MOD_TNOIA:
            value->rValue = model->BSIM4v4tnoia;
            return(OK);
        case BSIM4v4_MOD_TNOIB:
            value->rValue = model->BSIM4v4tnoib;
            return(OK);
        case BSIM4v4_MOD_RNOIA:
            value->rValue = model->BSIM4v4rnoia;
            return(OK);
        case BSIM4v4_MOD_RNOIB:
            value->rValue = model->BSIM4v4rnoib;
            return(OK);
        case BSIM4v4_MOD_NTNOI:
            value->rValue = model->BSIM4v4ntnoi;
            return(OK);
        case BSIM4v4_MOD_IJTHDFWD:
            value->rValue = model->BSIM4v4ijthdfwd;
            return(OK);
        case BSIM4v4_MOD_IJTHSFWD:
            value->rValue = model->BSIM4v4ijthsfwd;
            return(OK);
        case BSIM4v4_MOD_IJTHDREV:
            value->rValue = model->BSIM4v4ijthdrev;
            return(OK);
        case BSIM4v4_MOD_IJTHSREV:
            value->rValue = model->BSIM4v4ijthsrev;
            return(OK);
        case BSIM4v4_MOD_XJBVD:
            value->rValue = model->BSIM4v4xjbvd;
            return(OK);
        case BSIM4v4_MOD_XJBVS:
            value->rValue = model->BSIM4v4xjbvs;
            return(OK);
        case BSIM4v4_MOD_BVD:
            value->rValue = model->BSIM4v4bvd;
            return(OK);
        case BSIM4v4_MOD_BVS:
            value->rValue = model->BSIM4v4bvs;
            return(OK);
        case BSIM4v4_MOD_VFB:
            value->rValue = model->BSIM4v4vfb;
            return(OK);

        case BSIM4v4_MOD_GBMIN:
            value->rValue = model->BSIM4v4gbmin;
            return(OK);
        case BSIM4v4_MOD_RBDB:
            value->rValue = model->BSIM4v4rbdb;
            return(OK);
        case BSIM4v4_MOD_RBPB:
            value->rValue = model->BSIM4v4rbpb;
            return(OK);
        case BSIM4v4_MOD_RBSB:
            value->rValue = model->BSIM4v4rbsb;
            return(OK);
        case BSIM4v4_MOD_RBPS:
            value->rValue = model->BSIM4v4rbps;
            return(OK);
        case BSIM4v4_MOD_RBPD:
            value->rValue = model->BSIM4v4rbpd;
            return(OK);

        case BSIM4v4_MOD_CGSL:
            value->rValue = model->BSIM4v4cgsl;
            return(OK);
        case BSIM4v4_MOD_CGDL:
            value->rValue = model->BSIM4v4cgdl;
            return(OK);
        case BSIM4v4_MOD_CKAPPAS:
            value->rValue = model->BSIM4v4ckappas;
            return(OK);
        case BSIM4v4_MOD_CKAPPAD:
            value->rValue = model->BSIM4v4ckappad;
            return(OK);
        case BSIM4v4_MOD_CF:
            value->rValue = model->BSIM4v4cf;
            return(OK);
        case BSIM4v4_MOD_CLC:
            value->rValue = model->BSIM4v4clc;
            return(OK);
        case BSIM4v4_MOD_CLE:
            value->rValue = model->BSIM4v4cle;
            return(OK);
        case BSIM4v4_MOD_DWC:
            value->rValue = model->BSIM4v4dwc;
            return(OK);
        case BSIM4v4_MOD_DLC:
            value->rValue = model->BSIM4v4dlc;
            return(OK);
        case BSIM4v4_MOD_XW:
            value->rValue = model->BSIM4v4xw;
            return(OK);
        case BSIM4v4_MOD_XL:
            value->rValue = model->BSIM4v4xl;
            return(OK);
        case BSIM4v4_MOD_DLCIG:
            value->rValue = model->BSIM4v4dlcig;
            return(OK);
        case BSIM4v4_MOD_DWJ:
            value->rValue = model->BSIM4v4dwj;
            return(OK);
        case BSIM4v4_MOD_VFBCV:
            value->rValue = model->BSIM4v4vfbcv;
            return(OK);
        case BSIM4v4_MOD_ACDE:
            value->rValue = model->BSIM4v4acde;
            return(OK);
        case BSIM4v4_MOD_MOIN:
            value->rValue = model->BSIM4v4moin;
            return(OK);
        case BSIM4v4_MOD_NOFF:
            value->rValue = model->BSIM4v4noff;
            return(OK);
        case BSIM4v4_MOD_VOFFCV:
            value->rValue = model->BSIM4v4voffcv;
            return(OK);
        case BSIM4v4_MOD_DMCG:
            value->rValue = model->BSIM4v4dmcg;
            return(OK);
        case BSIM4v4_MOD_DMCI:
            value->rValue = model->BSIM4v4dmci;
            return(OK);
        case BSIM4v4_MOD_DMDG:
            value->rValue = model->BSIM4v4dmdg;
            return(OK);
        case BSIM4v4_MOD_DMCGT:
            value->rValue = model->BSIM4v4dmcgt;
            return(OK);
        case BSIM4v4_MOD_XGW:
            value->rValue = model->BSIM4v4xgw;
            return(OK);
        case BSIM4v4_MOD_XGL:
            value->rValue = model->BSIM4v4xgl;
            return(OK);
        case BSIM4v4_MOD_RSHG:
            value->rValue = model->BSIM4v4rshg;
            return(OK);
        case BSIM4v4_MOD_NGCON:
            value->rValue = model->BSIM4v4ngcon;
            return(OK);
        case BSIM4v4_MOD_TCJ:
            value->rValue = model->BSIM4v4tcj;
            return(OK);
        case BSIM4v4_MOD_TPB:
            value->rValue = model->BSIM4v4tpb;
            return(OK);
        case BSIM4v4_MOD_TCJSW:
            value->rValue = model->BSIM4v4tcjsw;
            return(OK);
        case BSIM4v4_MOD_TPBSW:
            value->rValue = model->BSIM4v4tpbsw;
            return(OK);
        case BSIM4v4_MOD_TCJSWG:
            value->rValue = model->BSIM4v4tcjswg;
            return(OK);
        case BSIM4v4_MOD_TPBSWG:
            value->rValue = model->BSIM4v4tpbswg;
            return(OK);

	/* Length dependence */
        case  BSIM4v4_MOD_LCDSC :
          value->rValue = model->BSIM4v4lcdsc;
            return(OK);
        case  BSIM4v4_MOD_LCDSCB :
          value->rValue = model->BSIM4v4lcdscb;
            return(OK);
        case  BSIM4v4_MOD_LCDSCD :
          value->rValue = model->BSIM4v4lcdscd;
            return(OK);
        case  BSIM4v4_MOD_LCIT :
          value->rValue = model->BSIM4v4lcit;
            return(OK);
        case  BSIM4v4_MOD_LNFACTOR :
          value->rValue = model->BSIM4v4lnfactor;
            return(OK);
        case BSIM4v4_MOD_LXJ:
            value->rValue = model->BSIM4v4lxj;
            return(OK);
        case BSIM4v4_MOD_LVSAT:
            value->rValue = model->BSIM4v4lvsat;
            return(OK);
        case BSIM4v4_MOD_LAT:
            value->rValue = model->BSIM4v4lat;
            return(OK);
        case BSIM4v4_MOD_LA0:
            value->rValue = model->BSIM4v4la0;
            return(OK);
        case BSIM4v4_MOD_LAGS:
            value->rValue = model->BSIM4v4lags;
            return(OK);
        case BSIM4v4_MOD_LA1:
            value->rValue = model->BSIM4v4la1;
            return(OK);
        case BSIM4v4_MOD_LA2:
            value->rValue = model->BSIM4v4la2;
            return(OK);
        case BSIM4v4_MOD_LKETA:
            value->rValue = model->BSIM4v4lketa;
            return(OK);
        case BSIM4v4_MOD_LNSUB:
            value->rValue = model->BSIM4v4lnsub;
            return(OK);
        case BSIM4v4_MOD_LNDEP:
            value->rValue = model->BSIM4v4lndep;
            return(OK);
        case BSIM4v4_MOD_LNSD:
            value->rValue = model->BSIM4v4lnsd;
            return(OK);
        case BSIM4v4_MOD_LNGATE:
            value->rValue = model->BSIM4v4lngate;
            return(OK);
        case BSIM4v4_MOD_LGAMMA1:
            value->rValue = model->BSIM4v4lgamma1;
            return(OK);
        case BSIM4v4_MOD_LGAMMA2:
            value->rValue = model->BSIM4v4lgamma2;
            return(OK);
        case BSIM4v4_MOD_LVBX:
            value->rValue = model->BSIM4v4lvbx;
            return(OK);
        case BSIM4v4_MOD_LVBM:
            value->rValue = model->BSIM4v4lvbm;
            return(OK);
        case BSIM4v4_MOD_LXT:
            value->rValue = model->BSIM4v4lxt;
            return(OK);
        case  BSIM4v4_MOD_LK1:
          value->rValue = model->BSIM4v4lk1;
            return(OK);
        case  BSIM4v4_MOD_LKT1:
          value->rValue = model->BSIM4v4lkt1;
            return(OK);
        case  BSIM4v4_MOD_LKT1L:
          value->rValue = model->BSIM4v4lkt1l;
            return(OK);
        case  BSIM4v4_MOD_LKT2 :
          value->rValue = model->BSIM4v4lkt2;
            return(OK);
        case  BSIM4v4_MOD_LK2 :
          value->rValue = model->BSIM4v4lk2;
            return(OK);
        case  BSIM4v4_MOD_LK3:
          value->rValue = model->BSIM4v4lk3;
            return(OK);
        case  BSIM4v4_MOD_LK3B:
          value->rValue = model->BSIM4v4lk3b;
            return(OK);
        case  BSIM4v4_MOD_LW0:
          value->rValue = model->BSIM4v4lw0;
            return(OK);
        case  BSIM4v4_MOD_LLPE0:
          value->rValue = model->BSIM4v4llpe0;
            return(OK);
        case  BSIM4v4_MOD_LLPEB:
          value->rValue = model->BSIM4v4llpeb;
            return(OK);
        case  BSIM4v4_MOD_LDVTP0:
          value->rValue = model->BSIM4v4ldvtp0;
            return(OK);
        case  BSIM4v4_MOD_LDVTP1:
          value->rValue = model->BSIM4v4ldvtp1;
            return(OK);
        case  BSIM4v4_MOD_LDVT0:
          value->rValue = model->BSIM4v4ldvt0;
            return(OK);
        case  BSIM4v4_MOD_LDVT1 :
          value->rValue = model->BSIM4v4ldvt1;
            return(OK);
        case  BSIM4v4_MOD_LDVT2 :
          value->rValue = model->BSIM4v4ldvt2;
            return(OK);
        case  BSIM4v4_MOD_LDVT0W :
          value->rValue = model->BSIM4v4ldvt0w;
            return(OK);
        case  BSIM4v4_MOD_LDVT1W :
          value->rValue = model->BSIM4v4ldvt1w;
            return(OK);
        case  BSIM4v4_MOD_LDVT2W :
          value->rValue = model->BSIM4v4ldvt2w;
            return(OK);
        case  BSIM4v4_MOD_LDROUT :
          value->rValue = model->BSIM4v4ldrout;
            return(OK);
        case  BSIM4v4_MOD_LDSUB :
          value->rValue = model->BSIM4v4ldsub;
            return(OK);
        case BSIM4v4_MOD_LVTH0:
            value->rValue = model->BSIM4v4lvth0;
            return(OK);
        case BSIM4v4_MOD_LUA:
            value->rValue = model->BSIM4v4lua;
            return(OK);
        case BSIM4v4_MOD_LUA1:
            value->rValue = model->BSIM4v4lua1;
            return(OK);
        case BSIM4v4_MOD_LUB:
            value->rValue = model->BSIM4v4lub;
            return(OK);
        case BSIM4v4_MOD_LUB1:
            value->rValue = model->BSIM4v4lub1;
            return(OK);
        case BSIM4v4_MOD_LUC:
            value->rValue = model->BSIM4v4luc;
            return(OK);
        case BSIM4v4_MOD_LUC1:
            value->rValue = model->BSIM4v4luc1;
            return(OK);
        case BSIM4v4_MOD_LU0:
            value->rValue = model->BSIM4v4lu0;
            return(OK);
        case BSIM4v4_MOD_LUTE:
            value->rValue = model->BSIM4v4lute;
            return(OK);
        case BSIM4v4_MOD_LVOFF:
            value->rValue = model->BSIM4v4lvoff;
            return(OK);
        case BSIM4v4_MOD_LMINV:
            value->rValue = model->BSIM4v4lminv;
            return(OK);
        case BSIM4v4_MOD_LFPROUT:
            value->rValue = model->BSIM4v4lfprout;
            return(OK);
        case BSIM4v4_MOD_LPDITS:
            value->rValue = model->BSIM4v4lpdits;
            return(OK);
        case BSIM4v4_MOD_LPDITSD:
            value->rValue = model->BSIM4v4lpditsd;
            return(OK);
        case BSIM4v4_MOD_LDELTA:
            value->rValue = model->BSIM4v4ldelta;
            return(OK);
        case BSIM4v4_MOD_LRDSW:
            value->rValue = model->BSIM4v4lrdsw;
            return(OK);
        case BSIM4v4_MOD_LRDW:
            value->rValue = model->BSIM4v4lrdw;
            return(OK);
        case BSIM4v4_MOD_LRSW:
            value->rValue = model->BSIM4v4lrsw;
            return(OK);
        case BSIM4v4_MOD_LPRWB:
            value->rValue = model->BSIM4v4lprwb;
            return(OK);
        case BSIM4v4_MOD_LPRWG:
            value->rValue = model->BSIM4v4lprwg;
            return(OK);
        case BSIM4v4_MOD_LPRT:
            value->rValue = model->BSIM4v4lprt;
            return(OK);
        case BSIM4v4_MOD_LETA0:
            value->rValue = model->BSIM4v4leta0;
            return(OK);
        case BSIM4v4_MOD_LETAB:
            value->rValue = model->BSIM4v4letab;
            return(OK);
        case BSIM4v4_MOD_LPCLM:
            value->rValue = model->BSIM4v4lpclm;
            return(OK);
        case BSIM4v4_MOD_LPDIBL1:
            value->rValue = model->BSIM4v4lpdibl1;
            return(OK);
        case BSIM4v4_MOD_LPDIBL2:
            value->rValue = model->BSIM4v4lpdibl2;
            return(OK);
        case BSIM4v4_MOD_LPDIBLB:
            value->rValue = model->BSIM4v4lpdiblb;
            return(OK);
        case BSIM4v4_MOD_LPSCBE1:
            value->rValue = model->BSIM4v4lpscbe1;
            return(OK);
        case BSIM4v4_MOD_LPSCBE2:
            value->rValue = model->BSIM4v4lpscbe2;
            return(OK);
        case BSIM4v4_MOD_LPVAG:
            value->rValue = model->BSIM4v4lpvag;
            return(OK);
        case BSIM4v4_MOD_LWR:
            value->rValue = model->BSIM4v4lwr;
            return(OK);
        case BSIM4v4_MOD_LDWG:
            value->rValue = model->BSIM4v4ldwg;
            return(OK);
        case BSIM4v4_MOD_LDWB:
            value->rValue = model->BSIM4v4ldwb;
            return(OK);
        case BSIM4v4_MOD_LB0:
            value->rValue = model->BSIM4v4lb0;
            return(OK);
        case BSIM4v4_MOD_LB1:
            value->rValue = model->BSIM4v4lb1;
            return(OK);
        case BSIM4v4_MOD_LALPHA0:
            value->rValue = model->BSIM4v4lalpha0;
            return(OK);
        case BSIM4v4_MOD_LALPHA1:
            value->rValue = model->BSIM4v4lalpha1;
            return(OK);
        case BSIM4v4_MOD_LBETA0:
            value->rValue = model->BSIM4v4lbeta0;
            return(OK);
        case BSIM4v4_MOD_LAGIDL:
            value->rValue = model->BSIM4v4lagidl;
            return(OK);
        case BSIM4v4_MOD_LBGIDL:
            value->rValue = model->BSIM4v4lbgidl;
            return(OK);
        case BSIM4v4_MOD_LCGIDL:
            value->rValue = model->BSIM4v4lcgidl;
            return(OK);
        case BSIM4v4_MOD_LEGIDL:
            value->rValue = model->BSIM4v4legidl;
            return(OK);
        case BSIM4v4_MOD_LAIGC:
            value->rValue = model->BSIM4v4laigc;
            return(OK);
        case BSIM4v4_MOD_LBIGC:
            value->rValue = model->BSIM4v4lbigc;
            return(OK);
        case BSIM4v4_MOD_LCIGC:
            value->rValue = model->BSIM4v4lcigc;
            return(OK);
        case BSIM4v4_MOD_LAIGSD:
            value->rValue = model->BSIM4v4laigsd;
            return(OK);
        case BSIM4v4_MOD_LBIGSD:
            value->rValue = model->BSIM4v4lbigsd;
            return(OK);
        case BSIM4v4_MOD_LCIGSD:
            value->rValue = model->BSIM4v4lcigsd;
            return(OK);
        case BSIM4v4_MOD_LAIGBACC:
            value->rValue = model->BSIM4v4laigbacc;
            return(OK);
        case BSIM4v4_MOD_LBIGBACC:
            value->rValue = model->BSIM4v4lbigbacc;
            return(OK);
        case BSIM4v4_MOD_LCIGBACC:
            value->rValue = model->BSIM4v4lcigbacc;
            return(OK);
        case BSIM4v4_MOD_LAIGBINV:
            value->rValue = model->BSIM4v4laigbinv;
            return(OK);
        case BSIM4v4_MOD_LBIGBINV:
            value->rValue = model->BSIM4v4lbigbinv;
            return(OK);
        case BSIM4v4_MOD_LCIGBINV:
            value->rValue = model->BSIM4v4lcigbinv;
            return(OK);
        case BSIM4v4_MOD_LNIGC:
            value->rValue = model->BSIM4v4lnigc;
            return(OK);
        case BSIM4v4_MOD_LNIGBACC:
            value->rValue = model->BSIM4v4lnigbacc;
            return(OK);
        case BSIM4v4_MOD_LNIGBINV:
            value->rValue = model->BSIM4v4lnigbinv;
            return(OK);
        case BSIM4v4_MOD_LNTOX:
            value->rValue = model->BSIM4v4lntox;
            return(OK);
        case BSIM4v4_MOD_LEIGBINV:
            value->rValue = model->BSIM4v4leigbinv;
            return(OK);
        case BSIM4v4_MOD_LPIGCD:
            value->rValue = model->BSIM4v4lpigcd;
            return(OK);
        case BSIM4v4_MOD_LPOXEDGE:
            value->rValue = model->BSIM4v4lpoxedge;
            return(OK);
        case BSIM4v4_MOD_LPHIN:
            value->rValue = model->BSIM4v4lphin;
            return(OK);
        case BSIM4v4_MOD_LXRCRG1:
            value->rValue = model->BSIM4v4lxrcrg1;
            return(OK);
        case BSIM4v4_MOD_LXRCRG2:
            value->rValue = model->BSIM4v4lxrcrg2;
            return(OK);
        case BSIM4v4_MOD_LEU:
            value->rValue = model->BSIM4v4leu;
            return(OK);
        case BSIM4v4_MOD_LVFB:
            value->rValue = model->BSIM4v4lvfb;
            return(OK);

        case BSIM4v4_MOD_LCGSL:
            value->rValue = model->BSIM4v4lcgsl;
            return(OK);
        case BSIM4v4_MOD_LCGDL:
            value->rValue = model->BSIM4v4lcgdl;
            return(OK);
        case BSIM4v4_MOD_LCKAPPAS:
            value->rValue = model->BSIM4v4lckappas;
            return(OK);
        case BSIM4v4_MOD_LCKAPPAD:
            value->rValue = model->BSIM4v4lckappad;
            return(OK);
        case BSIM4v4_MOD_LCF:
            value->rValue = model->BSIM4v4lcf;
            return(OK);
        case BSIM4v4_MOD_LCLC:
            value->rValue = model->BSIM4v4lclc;
            return(OK);
        case BSIM4v4_MOD_LCLE:
            value->rValue = model->BSIM4v4lcle;
            return(OK);
        case BSIM4v4_MOD_LVFBCV:
            value->rValue = model->BSIM4v4lvfbcv;
            return(OK);
        case BSIM4v4_MOD_LACDE:
            value->rValue = model->BSIM4v4lacde;
            return(OK);
        case BSIM4v4_MOD_LMOIN:
            value->rValue = model->BSIM4v4lmoin;
            return(OK);
        case BSIM4v4_MOD_LNOFF:
            value->rValue = model->BSIM4v4lnoff;
            return(OK);
        case BSIM4v4_MOD_LVOFFCV:
            value->rValue = model->BSIM4v4lvoffcv;
            return(OK);

	/* Width dependence */
        case  BSIM4v4_MOD_WCDSC :
          value->rValue = model->BSIM4v4wcdsc;
            return(OK);
        case  BSIM4v4_MOD_WCDSCB :
          value->rValue = model->BSIM4v4wcdscb;
            return(OK);
        case  BSIM4v4_MOD_WCDSCD :
          value->rValue = model->BSIM4v4wcdscd;
            return(OK);
        case  BSIM4v4_MOD_WCIT :
          value->rValue = model->BSIM4v4wcit;
            return(OK);
        case  BSIM4v4_MOD_WNFACTOR :
          value->rValue = model->BSIM4v4wnfactor;
            return(OK);
        case BSIM4v4_MOD_WXJ:
            value->rValue = model->BSIM4v4wxj;
            return(OK);
        case BSIM4v4_MOD_WVSAT:
            value->rValue = model->BSIM4v4wvsat;
            return(OK);
        case BSIM4v4_MOD_WAT:
            value->rValue = model->BSIM4v4wat;
            return(OK);
        case BSIM4v4_MOD_WA0:
            value->rValue = model->BSIM4v4wa0;
            return(OK);
        case BSIM4v4_MOD_WAGS:
            value->rValue = model->BSIM4v4wags;
            return(OK);
        case BSIM4v4_MOD_WA1:
            value->rValue = model->BSIM4v4wa1;
            return(OK);
        case BSIM4v4_MOD_WA2:
            value->rValue = model->BSIM4v4wa2;
            return(OK);
        case BSIM4v4_MOD_WKETA:
            value->rValue = model->BSIM4v4wketa;
            return(OK);
        case BSIM4v4_MOD_WNSUB:
            value->rValue = model->BSIM4v4wnsub;
            return(OK);
        case BSIM4v4_MOD_WNDEP:
            value->rValue = model->BSIM4v4wndep;
            return(OK);
        case BSIM4v4_MOD_WNSD:
            value->rValue = model->BSIM4v4wnsd;
            return(OK);
        case BSIM4v4_MOD_WNGATE:
            value->rValue = model->BSIM4v4wngate;
            return(OK);
        case BSIM4v4_MOD_WGAMMA1:
            value->rValue = model->BSIM4v4wgamma1;
            return(OK);
        case BSIM4v4_MOD_WGAMMA2:
            value->rValue = model->BSIM4v4wgamma2;
            return(OK);
        case BSIM4v4_MOD_WVBX:
            value->rValue = model->BSIM4v4wvbx;
            return(OK);
        case BSIM4v4_MOD_WVBM:
            value->rValue = model->BSIM4v4wvbm;
            return(OK);
        case BSIM4v4_MOD_WXT:
            value->rValue = model->BSIM4v4wxt;
            return(OK);
        case  BSIM4v4_MOD_WK1:
          value->rValue = model->BSIM4v4wk1;
            return(OK);
        case  BSIM4v4_MOD_WKT1:
          value->rValue = model->BSIM4v4wkt1;
            return(OK);
        case  BSIM4v4_MOD_WKT1L:
          value->rValue = model->BSIM4v4wkt1l;
            return(OK);
        case  BSIM4v4_MOD_WKT2 :
          value->rValue = model->BSIM4v4wkt2;
            return(OK);
        case  BSIM4v4_MOD_WK2 :
          value->rValue = model->BSIM4v4wk2;
            return(OK);
        case  BSIM4v4_MOD_WK3:
          value->rValue = model->BSIM4v4wk3;
            return(OK);
        case  BSIM4v4_MOD_WK3B:
          value->rValue = model->BSIM4v4wk3b;
            return(OK);
        case  BSIM4v4_MOD_WW0:
          value->rValue = model->BSIM4v4ww0;
            return(OK);
        case  BSIM4v4_MOD_WLPE0:
          value->rValue = model->BSIM4v4wlpe0;
            return(OK);
        case  BSIM4v4_MOD_WDVTP0:
          value->rValue = model->BSIM4v4wdvtp0;
            return(OK);
        case  BSIM4v4_MOD_WDVTP1:
          value->rValue = model->BSIM4v4wdvtp1;
            return(OK);
        case  BSIM4v4_MOD_WLPEB:
          value->rValue = model->BSIM4v4wlpeb;
            return(OK);
        case  BSIM4v4_MOD_WDVT0:
          value->rValue = model->BSIM4v4wdvt0;
            return(OK);
        case  BSIM4v4_MOD_WDVT1 :
          value->rValue = model->BSIM4v4wdvt1;
            return(OK);
        case  BSIM4v4_MOD_WDVT2 :
          value->rValue = model->BSIM4v4wdvt2;
            return(OK);
        case  BSIM4v4_MOD_WDVT0W :
          value->rValue = model->BSIM4v4wdvt0w;
            return(OK);
        case  BSIM4v4_MOD_WDVT1W :
          value->rValue = model->BSIM4v4wdvt1w;
            return(OK);
        case  BSIM4v4_MOD_WDVT2W :
          value->rValue = model->BSIM4v4wdvt2w;
            return(OK);
        case  BSIM4v4_MOD_WDROUT :
          value->rValue = model->BSIM4v4wdrout;
            return(OK);
        case  BSIM4v4_MOD_WDSUB :
          value->rValue = model->BSIM4v4wdsub;
            return(OK);
        case BSIM4v4_MOD_WVTH0:
            value->rValue = model->BSIM4v4wvth0;
            return(OK);
        case BSIM4v4_MOD_WUA:
            value->rValue = model->BSIM4v4wua;
            return(OK);
        case BSIM4v4_MOD_WUA1:
            value->rValue = model->BSIM4v4wua1;
            return(OK);
        case BSIM4v4_MOD_WUB:
            value->rValue = model->BSIM4v4wub;
            return(OK);
        case BSIM4v4_MOD_WUB1:
            value->rValue = model->BSIM4v4wub1;
            return(OK);
        case BSIM4v4_MOD_WUC:
            value->rValue = model->BSIM4v4wuc;
            return(OK);
        case BSIM4v4_MOD_WUC1:
            value->rValue = model->BSIM4v4wuc1;
            return(OK);
        case BSIM4v4_MOD_WU0:
            value->rValue = model->BSIM4v4wu0;
            return(OK);
        case BSIM4v4_MOD_WUTE:
            value->rValue = model->BSIM4v4wute;
            return(OK);
        case BSIM4v4_MOD_WVOFF:
            value->rValue = model->BSIM4v4wvoff;
            return(OK);
        case BSIM4v4_MOD_WMINV:
            value->rValue = model->BSIM4v4wminv;
            return(OK);
        case BSIM4v4_MOD_WFPROUT:
            value->rValue = model->BSIM4v4wfprout;
            return(OK);
        case BSIM4v4_MOD_WPDITS:
            value->rValue = model->BSIM4v4wpdits;
            return(OK);
        case BSIM4v4_MOD_WPDITSD:
            value->rValue = model->BSIM4v4wpditsd;
            return(OK);
        case BSIM4v4_MOD_WDELTA:
            value->rValue = model->BSIM4v4wdelta;
            return(OK);
        case BSIM4v4_MOD_WRDSW:
            value->rValue = model->BSIM4v4wrdsw;
            return(OK);
        case BSIM4v4_MOD_WRDW:
            value->rValue = model->BSIM4v4wrdw;
            return(OK);
        case BSIM4v4_MOD_WRSW:
            value->rValue = model->BSIM4v4wrsw;
            return(OK);
        case BSIM4v4_MOD_WPRWB:
            value->rValue = model->BSIM4v4wprwb;
            return(OK);
        case BSIM4v4_MOD_WPRWG:
            value->rValue = model->BSIM4v4wprwg;
            return(OK);
        case BSIM4v4_MOD_WPRT:
            value->rValue = model->BSIM4v4wprt;
            return(OK);
        case BSIM4v4_MOD_WETA0:
            value->rValue = model->BSIM4v4weta0;
            return(OK);
        case BSIM4v4_MOD_WETAB:
            value->rValue = model->BSIM4v4wetab;
            return(OK);
        case BSIM4v4_MOD_WPCLM:
            value->rValue = model->BSIM4v4wpclm;
            return(OK);
        case BSIM4v4_MOD_WPDIBL1:
            value->rValue = model->BSIM4v4wpdibl1;
            return(OK);
        case BSIM4v4_MOD_WPDIBL2:
            value->rValue = model->BSIM4v4wpdibl2;
            return(OK);
        case BSIM4v4_MOD_WPDIBLB:
            value->rValue = model->BSIM4v4wpdiblb;
            return(OK);
        case BSIM4v4_MOD_WPSCBE1:
            value->rValue = model->BSIM4v4wpscbe1;
            return(OK);
        case BSIM4v4_MOD_WPSCBE2:
            value->rValue = model->BSIM4v4wpscbe2;
            return(OK);
        case BSIM4v4_MOD_WPVAG:
            value->rValue = model->BSIM4v4wpvag;
            return(OK);
        case BSIM4v4_MOD_WWR:
            value->rValue = model->BSIM4v4wwr;
            return(OK);
        case BSIM4v4_MOD_WDWG:
            value->rValue = model->BSIM4v4wdwg;
            return(OK);
        case BSIM4v4_MOD_WDWB:
            value->rValue = model->BSIM4v4wdwb;
            return(OK);
        case BSIM4v4_MOD_WB0:
            value->rValue = model->BSIM4v4wb0;
            return(OK);
        case BSIM4v4_MOD_WB1:
            value->rValue = model->BSIM4v4wb1;
            return(OK);
        case BSIM4v4_MOD_WALPHA0:
            value->rValue = model->BSIM4v4walpha0;
            return(OK);
        case BSIM4v4_MOD_WALPHA1:
            value->rValue = model->BSIM4v4walpha1;
            return(OK);
        case BSIM4v4_MOD_WBETA0:
            value->rValue = model->BSIM4v4wbeta0;
            return(OK);
        case BSIM4v4_MOD_WAGIDL:
            value->rValue = model->BSIM4v4wagidl;
            return(OK);
        case BSIM4v4_MOD_WBGIDL:
            value->rValue = model->BSIM4v4wbgidl;
            return(OK);
        case BSIM4v4_MOD_WCGIDL:
            value->rValue = model->BSIM4v4wcgidl;
            return(OK);
        case BSIM4v4_MOD_WEGIDL:
            value->rValue = model->BSIM4v4wegidl;
            return(OK);
        case BSIM4v4_MOD_WAIGC:
            value->rValue = model->BSIM4v4waigc;
            return(OK);
        case BSIM4v4_MOD_WBIGC:
            value->rValue = model->BSIM4v4wbigc;
            return(OK);
        case BSIM4v4_MOD_WCIGC:
            value->rValue = model->BSIM4v4wcigc;
            return(OK);
        case BSIM4v4_MOD_WAIGSD:
            value->rValue = model->BSIM4v4waigsd;
            return(OK);
        case BSIM4v4_MOD_WBIGSD:
            value->rValue = model->BSIM4v4wbigsd;
            return(OK);
        case BSIM4v4_MOD_WCIGSD:
            value->rValue = model->BSIM4v4wcigsd;
            return(OK);
        case BSIM4v4_MOD_WAIGBACC:
            value->rValue = model->BSIM4v4waigbacc;
            return(OK);
        case BSIM4v4_MOD_WBIGBACC:
            value->rValue = model->BSIM4v4wbigbacc;
            return(OK);
        case BSIM4v4_MOD_WCIGBACC:
            value->rValue = model->BSIM4v4wcigbacc;
            return(OK);
        case BSIM4v4_MOD_WAIGBINV:
            value->rValue = model->BSIM4v4waigbinv;
            return(OK);
        case BSIM4v4_MOD_WBIGBINV:
            value->rValue = model->BSIM4v4wbigbinv;
            return(OK);
        case BSIM4v4_MOD_WCIGBINV:
            value->rValue = model->BSIM4v4wcigbinv;
            return(OK);
        case BSIM4v4_MOD_WNIGC:
            value->rValue = model->BSIM4v4wnigc;
            return(OK);
        case BSIM4v4_MOD_WNIGBACC:
            value->rValue = model->BSIM4v4wnigbacc;
            return(OK);
        case BSIM4v4_MOD_WNIGBINV:
            value->rValue = model->BSIM4v4wnigbinv;
            return(OK);
        case BSIM4v4_MOD_WNTOX:
            value->rValue = model->BSIM4v4wntox;
            return(OK);
        case BSIM4v4_MOD_WEIGBINV:
            value->rValue = model->BSIM4v4weigbinv;
            return(OK);
        case BSIM4v4_MOD_WPIGCD:
            value->rValue = model->BSIM4v4wpigcd;
            return(OK);
        case BSIM4v4_MOD_WPOXEDGE:
            value->rValue = model->BSIM4v4wpoxedge;
            return(OK);
        case BSIM4v4_MOD_WPHIN:
            value->rValue = model->BSIM4v4wphin;
            return(OK);
        case BSIM4v4_MOD_WXRCRG1:
            value->rValue = model->BSIM4v4wxrcrg1;
            return(OK);
        case BSIM4v4_MOD_WXRCRG2:
            value->rValue = model->BSIM4v4wxrcrg2;
            return(OK);
        case BSIM4v4_MOD_WEU:
            value->rValue = model->BSIM4v4weu;
            return(OK);
        case BSIM4v4_MOD_WVFB:
            value->rValue = model->BSIM4v4wvfb;
            return(OK);

        case BSIM4v4_MOD_WCGSL:
            value->rValue = model->BSIM4v4wcgsl;
            return(OK);
        case BSIM4v4_MOD_WCGDL:
            value->rValue = model->BSIM4v4wcgdl;
            return(OK);
        case BSIM4v4_MOD_WCKAPPAS:
            value->rValue = model->BSIM4v4wckappas;
            return(OK);
        case BSIM4v4_MOD_WCKAPPAD:
            value->rValue = model->BSIM4v4wckappad;
            return(OK);
        case BSIM4v4_MOD_WCF:
            value->rValue = model->BSIM4v4wcf;
            return(OK);
        case BSIM4v4_MOD_WCLC:
            value->rValue = model->BSIM4v4wclc;
            return(OK);
        case BSIM4v4_MOD_WCLE:
            value->rValue = model->BSIM4v4wcle;
            return(OK);
        case BSIM4v4_MOD_WVFBCV:
            value->rValue = model->BSIM4v4wvfbcv;
            return(OK);
        case BSIM4v4_MOD_WACDE:
            value->rValue = model->BSIM4v4wacde;
            return(OK);
        case BSIM4v4_MOD_WMOIN:
            value->rValue = model->BSIM4v4wmoin;
            return(OK);
        case BSIM4v4_MOD_WNOFF:
            value->rValue = model->BSIM4v4wnoff;
            return(OK);
        case BSIM4v4_MOD_WVOFFCV:
            value->rValue = model->BSIM4v4wvoffcv;
            return(OK);

	/* Cross-term dependence */
        case  BSIM4v4_MOD_PCDSC :
          value->rValue = model->BSIM4v4pcdsc;
            return(OK);
        case  BSIM4v4_MOD_PCDSCB :
          value->rValue = model->BSIM4v4pcdscb;
            return(OK);
        case  BSIM4v4_MOD_PCDSCD :
          value->rValue = model->BSIM4v4pcdscd;
            return(OK);
         case  BSIM4v4_MOD_PCIT :
          value->rValue = model->BSIM4v4pcit;
            return(OK);
        case  BSIM4v4_MOD_PNFACTOR :
          value->rValue = model->BSIM4v4pnfactor;
            return(OK);
        case BSIM4v4_MOD_PXJ:
            value->rValue = model->BSIM4v4pxj;
            return(OK);
        case BSIM4v4_MOD_PVSAT:
            value->rValue = model->BSIM4v4pvsat;
            return(OK);
        case BSIM4v4_MOD_PAT:
            value->rValue = model->BSIM4v4pat;
            return(OK);
        case BSIM4v4_MOD_PA0:
            value->rValue = model->BSIM4v4pa0;
            return(OK);
        case BSIM4v4_MOD_PAGS:
            value->rValue = model->BSIM4v4pags;
            return(OK);
        case BSIM4v4_MOD_PA1:
            value->rValue = model->BSIM4v4pa1;
            return(OK);
        case BSIM4v4_MOD_PA2:
            value->rValue = model->BSIM4v4pa2;
            return(OK);
        case BSIM4v4_MOD_PKETA:
            value->rValue = model->BSIM4v4pketa;
            return(OK);
        case BSIM4v4_MOD_PNSUB:
            value->rValue = model->BSIM4v4pnsub;
            return(OK);
        case BSIM4v4_MOD_PNDEP:
            value->rValue = model->BSIM4v4pndep;
            return(OK);
        case BSIM4v4_MOD_PNSD:
            value->rValue = model->BSIM4v4pnsd;
            return(OK);
        case BSIM4v4_MOD_PNGATE:
            value->rValue = model->BSIM4v4pngate;
            return(OK);
        case BSIM4v4_MOD_PGAMMA1:
            value->rValue = model->BSIM4v4pgamma1;
            return(OK);
        case BSIM4v4_MOD_PGAMMA2:
            value->rValue = model->BSIM4v4pgamma2;
            return(OK);
        case BSIM4v4_MOD_PVBX:
            value->rValue = model->BSIM4v4pvbx;
            return(OK);
        case BSIM4v4_MOD_PVBM:
            value->rValue = model->BSIM4v4pvbm;
            return(OK);
        case BSIM4v4_MOD_PXT:
            value->rValue = model->BSIM4v4pxt;
            return(OK);
        case  BSIM4v4_MOD_PK1:
          value->rValue = model->BSIM4v4pk1;
            return(OK);
        case  BSIM4v4_MOD_PKT1:
          value->rValue = model->BSIM4v4pkt1;
            return(OK);
        case  BSIM4v4_MOD_PKT1L:
          value->rValue = model->BSIM4v4pkt1l;
            return(OK);
        case  BSIM4v4_MOD_PKT2 :
          value->rValue = model->BSIM4v4pkt2;
            return(OK);
        case  BSIM4v4_MOD_PK2 :
          value->rValue = model->BSIM4v4pk2;
            return(OK);
        case  BSIM4v4_MOD_PK3:
          value->rValue = model->BSIM4v4pk3;
            return(OK);
        case  BSIM4v4_MOD_PK3B:
          value->rValue = model->BSIM4v4pk3b;
            return(OK);
        case  BSIM4v4_MOD_PW0:
          value->rValue = model->BSIM4v4pw0;
            return(OK);
        case  BSIM4v4_MOD_PLPE0:
          value->rValue = model->BSIM4v4plpe0;
            return(OK);
        case  BSIM4v4_MOD_PLPEB:
          value->rValue = model->BSIM4v4plpeb;
            return(OK);
        case  BSIM4v4_MOD_PDVTP0:
          value->rValue = model->BSIM4v4pdvtp0;
            return(OK);
        case  BSIM4v4_MOD_PDVTP1:
          value->rValue = model->BSIM4v4pdvtp1;
            return(OK);
        case  BSIM4v4_MOD_PDVT0 :
          value->rValue = model->BSIM4v4pdvt0;
            return(OK);
        case  BSIM4v4_MOD_PDVT1 :
          value->rValue = model->BSIM4v4pdvt1;
            return(OK);
        case  BSIM4v4_MOD_PDVT2 :
          value->rValue = model->BSIM4v4pdvt2;
            return(OK);
        case  BSIM4v4_MOD_PDVT0W :
          value->rValue = model->BSIM4v4pdvt0w;
            return(OK);
        case  BSIM4v4_MOD_PDVT1W :
          value->rValue = model->BSIM4v4pdvt1w;
            return(OK);
        case  BSIM4v4_MOD_PDVT2W :
          value->rValue = model->BSIM4v4pdvt2w;
            return(OK);
        case  BSIM4v4_MOD_PDROUT :
          value->rValue = model->BSIM4v4pdrout;
            return(OK);
        case  BSIM4v4_MOD_PDSUB :
          value->rValue = model->BSIM4v4pdsub;
            return(OK);
        case BSIM4v4_MOD_PVTH0:
            value->rValue = model->BSIM4v4pvth0;
            return(OK);
        case BSIM4v4_MOD_PUA:
            value->rValue = model->BSIM4v4pua;
            return(OK);
        case BSIM4v4_MOD_PUA1:
            value->rValue = model->BSIM4v4pua1;
            return(OK);
        case BSIM4v4_MOD_PUB:
            value->rValue = model->BSIM4v4pub;
            return(OK);
        case BSIM4v4_MOD_PUB1:
            value->rValue = model->BSIM4v4pub1;
            return(OK);
        case BSIM4v4_MOD_PUC:
            value->rValue = model->BSIM4v4puc;
            return(OK);
        case BSIM4v4_MOD_PUC1:
            value->rValue = model->BSIM4v4puc1;
            return(OK);
        case BSIM4v4_MOD_PU0:
            value->rValue = model->BSIM4v4pu0;
            return(OK);
        case BSIM4v4_MOD_PUTE:
            value->rValue = model->BSIM4v4pute;
            return(OK);
        case BSIM4v4_MOD_PVOFF:
            value->rValue = model->BSIM4v4pvoff;
            return(OK);
        case BSIM4v4_MOD_PMINV:
            value->rValue = model->BSIM4v4pminv;
            return(OK);
        case BSIM4v4_MOD_PFPROUT:
            value->rValue = model->BSIM4v4pfprout;
            return(OK);
        case BSIM4v4_MOD_PPDITS:
            value->rValue = model->BSIM4v4ppdits;
            return(OK);
        case BSIM4v4_MOD_PPDITSD:
            value->rValue = model->BSIM4v4ppditsd;
            return(OK);
        case BSIM4v4_MOD_PDELTA:
            value->rValue = model->BSIM4v4pdelta;
            return(OK);
        case BSIM4v4_MOD_PRDSW:
            value->rValue = model->BSIM4v4prdsw;
            return(OK);
        case BSIM4v4_MOD_PRDW:
            value->rValue = model->BSIM4v4prdw;
            return(OK);
        case BSIM4v4_MOD_PRSW:
            value->rValue = model->BSIM4v4prsw;
            return(OK);
        case BSIM4v4_MOD_PPRWB:
            value->rValue = model->BSIM4v4pprwb;
            return(OK);
        case BSIM4v4_MOD_PPRWG:
            value->rValue = model->BSIM4v4pprwg;
            return(OK);
        case BSIM4v4_MOD_PPRT:
            value->rValue = model->BSIM4v4pprt;
            return(OK);
        case BSIM4v4_MOD_PETA0:
            value->rValue = model->BSIM4v4peta0;
            return(OK);
        case BSIM4v4_MOD_PETAB:
            value->rValue = model->BSIM4v4petab;
            return(OK);
        case BSIM4v4_MOD_PPCLM:
            value->rValue = model->BSIM4v4ppclm;
            return(OK);
        case BSIM4v4_MOD_PPDIBL1:
            value->rValue = model->BSIM4v4ppdibl1;
            return(OK);
        case BSIM4v4_MOD_PPDIBL2:
            value->rValue = model->BSIM4v4ppdibl2;
            return(OK);
        case BSIM4v4_MOD_PPDIBLB:
            value->rValue = model->BSIM4v4ppdiblb;
            return(OK);
        case BSIM4v4_MOD_PPSCBE1:
            value->rValue = model->BSIM4v4ppscbe1;
            return(OK);
        case BSIM4v4_MOD_PPSCBE2:
            value->rValue = model->BSIM4v4ppscbe2;
            return(OK);
        case BSIM4v4_MOD_PPVAG:
            value->rValue = model->BSIM4v4ppvag;
            return(OK);
        case BSIM4v4_MOD_PWR:
            value->rValue = model->BSIM4v4pwr;
            return(OK);
        case BSIM4v4_MOD_PDWG:
            value->rValue = model->BSIM4v4pdwg;
            return(OK);
        case BSIM4v4_MOD_PDWB:
            value->rValue = model->BSIM4v4pdwb;
            return(OK);
        case BSIM4v4_MOD_PB0:
            value->rValue = model->BSIM4v4pb0;
            return(OK);
        case BSIM4v4_MOD_PB1:
            value->rValue = model->BSIM4v4pb1;
            return(OK);
        case BSIM4v4_MOD_PALPHA0:
            value->rValue = model->BSIM4v4palpha0;
            return(OK);
        case BSIM4v4_MOD_PALPHA1:
            value->rValue = model->BSIM4v4palpha1;
            return(OK);
        case BSIM4v4_MOD_PBETA0:
            value->rValue = model->BSIM4v4pbeta0;
            return(OK);
        case BSIM4v4_MOD_PAGIDL:
            value->rValue = model->BSIM4v4pagidl;
            return(OK);
        case BSIM4v4_MOD_PBGIDL:
            value->rValue = model->BSIM4v4pbgidl;
            return(OK);
        case BSIM4v4_MOD_PCGIDL:
            value->rValue = model->BSIM4v4pcgidl;
            return(OK);
        case BSIM4v4_MOD_PEGIDL:
            value->rValue = model->BSIM4v4pegidl;
            return(OK);
        case BSIM4v4_MOD_PAIGC:
            value->rValue = model->BSIM4v4paigc;
            return(OK);
        case BSIM4v4_MOD_PBIGC:
            value->rValue = model->BSIM4v4pbigc;
            return(OK);
        case BSIM4v4_MOD_PCIGC:
            value->rValue = model->BSIM4v4pcigc;
            return(OK);
        case BSIM4v4_MOD_PAIGSD:
            value->rValue = model->BSIM4v4paigsd;
            return(OK);
        case BSIM4v4_MOD_PBIGSD:
            value->rValue = model->BSIM4v4pbigsd;
            return(OK);
        case BSIM4v4_MOD_PCIGSD:
            value->rValue = model->BSIM4v4pcigsd;
            return(OK);
        case BSIM4v4_MOD_PAIGBACC:
            value->rValue = model->BSIM4v4paigbacc;
            return(OK);
        case BSIM4v4_MOD_PBIGBACC:
            value->rValue = model->BSIM4v4pbigbacc;
            return(OK);
        case BSIM4v4_MOD_PCIGBACC:
            value->rValue = model->BSIM4v4pcigbacc;
            return(OK);
        case BSIM4v4_MOD_PAIGBINV:
            value->rValue = model->BSIM4v4paigbinv;
            return(OK);
        case BSIM4v4_MOD_PBIGBINV:
            value->rValue = model->BSIM4v4pbigbinv;
            return(OK);
        case BSIM4v4_MOD_PCIGBINV:
            value->rValue = model->BSIM4v4pcigbinv;
            return(OK);
        case BSIM4v4_MOD_PNIGC:
            value->rValue = model->BSIM4v4pnigc;
            return(OK);
        case BSIM4v4_MOD_PNIGBACC:
            value->rValue = model->BSIM4v4pnigbacc;
            return(OK);
        case BSIM4v4_MOD_PNIGBINV:
            value->rValue = model->BSIM4v4pnigbinv;
            return(OK);
        case BSIM4v4_MOD_PNTOX:
            value->rValue = model->BSIM4v4pntox;
            return(OK);
        case BSIM4v4_MOD_PEIGBINV:
            value->rValue = model->BSIM4v4peigbinv;
            return(OK);
        case BSIM4v4_MOD_PPIGCD:
            value->rValue = model->BSIM4v4ppigcd;
            return(OK);
        case BSIM4v4_MOD_PPOXEDGE:
            value->rValue = model->BSIM4v4ppoxedge;
            return(OK);
        case BSIM4v4_MOD_PPHIN:
            value->rValue = model->BSIM4v4pphin;
            return(OK);
        case BSIM4v4_MOD_PXRCRG1:
            value->rValue = model->BSIM4v4pxrcrg1;
            return(OK);
        case BSIM4v4_MOD_PXRCRG2:
            value->rValue = model->BSIM4v4pxrcrg2;
            return(OK);
        case BSIM4v4_MOD_PEU:
            value->rValue = model->BSIM4v4peu;
            return(OK);
        case BSIM4v4_MOD_PVFB:
            value->rValue = model->BSIM4v4pvfb;
            return(OK);

        case BSIM4v4_MOD_PCGSL:
            value->rValue = model->BSIM4v4pcgsl;
            return(OK);
        case BSIM4v4_MOD_PCGDL:
            value->rValue = model->BSIM4v4pcgdl;
            return(OK);
        case BSIM4v4_MOD_PCKAPPAS:
            value->rValue = model->BSIM4v4pckappas;
            return(OK);
        case BSIM4v4_MOD_PCKAPPAD:
            value->rValue = model->BSIM4v4pckappad;
            return(OK);
        case BSIM4v4_MOD_PCF:
            value->rValue = model->BSIM4v4pcf;
            return(OK);
        case BSIM4v4_MOD_PCLC:
            value->rValue = model->BSIM4v4pclc;
            return(OK);
        case BSIM4v4_MOD_PCLE:
            value->rValue = model->BSIM4v4pcle;
            return(OK);
        case BSIM4v4_MOD_PVFBCV:
            value->rValue = model->BSIM4v4pvfbcv;
            return(OK);
        case BSIM4v4_MOD_PACDE:
            value->rValue = model->BSIM4v4pacde;
            return(OK);
        case BSIM4v4_MOD_PMOIN:
            value->rValue = model->BSIM4v4pmoin;
            return(OK);
        case BSIM4v4_MOD_PNOFF:
            value->rValue = model->BSIM4v4pnoff;
            return(OK);
        case BSIM4v4_MOD_PVOFFCV:
            value->rValue = model->BSIM4v4pvoffcv;
            return(OK);

        case  BSIM4v4_MOD_TNOM :
          value->rValue = model->BSIM4v4tnom;
            return(OK);
        case BSIM4v4_MOD_CGSO:
            value->rValue = model->BSIM4v4cgso;
            return(OK);
        case BSIM4v4_MOD_CGDO:
            value->rValue = model->BSIM4v4cgdo;
            return(OK);
        case BSIM4v4_MOD_CGBO:
            value->rValue = model->BSIM4v4cgbo;
            return(OK);
        case BSIM4v4_MOD_XPART:
            value->rValue = model->BSIM4v4xpart;
            return(OK);
        case BSIM4v4_MOD_RSH:
            value->rValue = model->BSIM4v4sheetResistance;
            return(OK);
        case BSIM4v4_MOD_JSS:
            value->rValue = model->BSIM4v4SjctSatCurDensity;
            return(OK);
        case BSIM4v4_MOD_JSWS:
            value->rValue = model->BSIM4v4SjctSidewallSatCurDensity;
            return(OK);
        case BSIM4v4_MOD_JSWGS:
            value->rValue = model->BSIM4v4SjctGateSidewallSatCurDensity;
            return(OK);
        case BSIM4v4_MOD_PBS:
            value->rValue = model->BSIM4v4SbulkJctPotential;
            return(OK);
        case BSIM4v4_MOD_MJS:
            value->rValue = model->BSIM4v4SbulkJctBotGradingCoeff;
            return(OK);
        case BSIM4v4_MOD_PBSWS:
            value->rValue = model->BSIM4v4SsidewallJctPotential;
            return(OK);
        case BSIM4v4_MOD_MJSWS:
            value->rValue = model->BSIM4v4SbulkJctSideGradingCoeff;
            return(OK);
        case BSIM4v4_MOD_CJS:
            value->rValue = model->BSIM4v4SunitAreaJctCap;
            return(OK);
        case BSIM4v4_MOD_CJSWS:
            value->rValue = model->BSIM4v4SunitLengthSidewallJctCap;
            return(OK);
        case BSIM4v4_MOD_PBSWGS:
            value->rValue = model->BSIM4v4SGatesidewallJctPotential;
            return(OK);
        case BSIM4v4_MOD_MJSWGS:
            value->rValue = model->BSIM4v4SbulkJctGateSideGradingCoeff;
            return(OK);
        case BSIM4v4_MOD_CJSWGS:
            value->rValue = model->BSIM4v4SunitLengthGateSidewallJctCap;
            return(OK);
        case BSIM4v4_MOD_NJS:
            value->rValue = model->BSIM4v4SjctEmissionCoeff;
            return(OK);
        case BSIM4v4_MOD_XTIS:
            value->rValue = model->BSIM4v4SjctTempExponent;
            return(OK);
        case BSIM4v4_MOD_JSD:
            value->rValue = model->BSIM4v4DjctSatCurDensity;
            return(OK);
        case BSIM4v4_MOD_JSWD:
            value->rValue = model->BSIM4v4DjctSidewallSatCurDensity;
            return(OK);
        case BSIM4v4_MOD_JSWGD:
            value->rValue = model->BSIM4v4DjctGateSidewallSatCurDensity;
            return(OK);
        case BSIM4v4_MOD_PBD:
            value->rValue = model->BSIM4v4DbulkJctPotential;
            return(OK);
        case BSIM4v4_MOD_MJD:
            value->rValue = model->BSIM4v4DbulkJctBotGradingCoeff;
            return(OK);
        case BSIM4v4_MOD_PBSWD:
            value->rValue = model->BSIM4v4DsidewallJctPotential;
            return(OK);
        case BSIM4v4_MOD_MJSWD:
            value->rValue = model->BSIM4v4DbulkJctSideGradingCoeff;
            return(OK);
        case BSIM4v4_MOD_CJD:
            value->rValue = model->BSIM4v4DunitAreaJctCap;
            return(OK);
        case BSIM4v4_MOD_CJSWD:
            value->rValue = model->BSIM4v4DunitLengthSidewallJctCap;
            return(OK);
        case BSIM4v4_MOD_PBSWGD:
            value->rValue = model->BSIM4v4DGatesidewallJctPotential;
            return(OK);
        case BSIM4v4_MOD_MJSWGD:
            value->rValue = model->BSIM4v4DbulkJctGateSideGradingCoeff;
            return(OK);
        case BSIM4v4_MOD_CJSWGD:
            value->rValue = model->BSIM4v4DunitLengthGateSidewallJctCap;
            return(OK);
        case BSIM4v4_MOD_NJD:
            value->rValue = model->BSIM4v4DjctEmissionCoeff;
            return(OK);
        case BSIM4v4_MOD_XTID:
            value->rValue = model->BSIM4v4DjctTempExponent;
            return(OK);
        case BSIM4v4_MOD_LINT:
            value->rValue = model->BSIM4v4Lint;
            return(OK);
        case BSIM4v4_MOD_LL:
            value->rValue = model->BSIM4v4Ll;
            return(OK);
        case BSIM4v4_MOD_LLC:
            value->rValue = model->BSIM4v4Llc;
            return(OK);
        case BSIM4v4_MOD_LLN:
            value->rValue = model->BSIM4v4Lln;
            return(OK);
        case BSIM4v4_MOD_LW:
            value->rValue = model->BSIM4v4Lw;
            return(OK);
        case BSIM4v4_MOD_LWC:
            value->rValue = model->BSIM4v4Lwc;
            return(OK);
        case BSIM4v4_MOD_LWN:
            value->rValue = model->BSIM4v4Lwn;
            return(OK);
        case BSIM4v4_MOD_LWL:
            value->rValue = model->BSIM4v4Lwl;
            return(OK);
        case BSIM4v4_MOD_LWLC:
            value->rValue = model->BSIM4v4Lwlc;
            return(OK);
        case BSIM4v4_MOD_LMIN:
            value->rValue = model->BSIM4v4Lmin;
            return(OK);
        case BSIM4v4_MOD_LMAX:
            value->rValue = model->BSIM4v4Lmax;
            return(OK);
        case BSIM4v4_MOD_WINT:
            value->rValue = model->BSIM4v4Wint;
            return(OK);
        case BSIM4v4_MOD_WL:
            value->rValue = model->BSIM4v4Wl;
            return(OK);
        case BSIM4v4_MOD_WLC:
            value->rValue = model->BSIM4v4Wlc;
            return(OK);
        case BSIM4v4_MOD_WLN:
            value->rValue = model->BSIM4v4Wln;
            return(OK);
        case BSIM4v4_MOD_WW:
            value->rValue = model->BSIM4v4Ww;
            return(OK);
        case BSIM4v4_MOD_WWC:
            value->rValue = model->BSIM4v4Wwc;
            return(OK);
        case BSIM4v4_MOD_WWN:
            value->rValue = model->BSIM4v4Wwn;
            return(OK);
        case BSIM4v4_MOD_WWL:
            value->rValue = model->BSIM4v4Wwl;
            return(OK);
        case BSIM4v4_MOD_WWLC:
            value->rValue = model->BSIM4v4Wwlc;
            return(OK);
        case BSIM4v4_MOD_WMIN:
            value->rValue = model->BSIM4v4Wmin;
            return(OK);
        case BSIM4v4_MOD_WMAX:
            value->rValue = model->BSIM4v4Wmax;
            return(OK);

        /* stress effect */
        case BSIM4v4_MOD_SAREF:
            value->rValue = model->BSIM4v4saref;
            return(OK);
        case BSIM4v4_MOD_SBREF:
            value->rValue = model->BSIM4v4sbref;
            return(OK);
	case BSIM4v4_MOD_WLOD:
            value->rValue = model->BSIM4v4wlod;
            return(OK);
        case BSIM4v4_MOD_KU0:
            value->rValue = model->BSIM4v4ku0;
            return(OK);
        case BSIM4v4_MOD_KVSAT:
            value->rValue = model->BSIM4v4kvsat;
            return(OK);
        case BSIM4v4_MOD_KVTH0:
            value->rValue = model->BSIM4v4kvth0;
            return(OK);
        case BSIM4v4_MOD_TKU0:
            value->rValue = model->BSIM4v4tku0;
            return(OK);
        case BSIM4v4_MOD_LLODKU0:
            value->rValue = model->BSIM4v4llodku0;
            return(OK);
        case BSIM4v4_MOD_WLODKU0:
            value->rValue = model->BSIM4v4wlodku0;
            return(OK);
        case BSIM4v4_MOD_LLODVTH:
            value->rValue = model->BSIM4v4llodvth;
            return(OK);
        case BSIM4v4_MOD_WLODVTH:
            value->rValue = model->BSIM4v4wlodvth;
            return(OK);
        case BSIM4v4_MOD_LKU0:
            value->rValue = model->BSIM4v4lku0;
            return(OK);
        case BSIM4v4_MOD_WKU0:
            value->rValue = model->BSIM4v4wku0;
            return(OK);
        case BSIM4v4_MOD_PKU0:
            value->rValue = model->BSIM4v4pku0;
            return(OK);
        case BSIM4v4_MOD_LKVTH0:
            value->rValue = model->BSIM4v4lkvth0;
            return(OK);
        case BSIM4v4_MOD_WKVTH0:
            value->rValue = model->BSIM4v4wkvth0;
            return(OK);
        case BSIM4v4_MOD_PKVTH0:
            value->rValue = model->BSIM4v4pkvth0;
            return(OK);
        case BSIM4v4_MOD_STK2:
            value->rValue = model->BSIM4v4stk2;
            return(OK);
        case BSIM4v4_MOD_LODK2:
            value->rValue = model->BSIM4v4lodk2;
            return(OK);
        case BSIM4v4_MOD_STETA0:
            value->rValue = model->BSIM4v4steta0;
            return(OK);
        case BSIM4v4_MOD_LODETA0:
            value->rValue = model->BSIM4v4lodeta0;
            return(OK);

        case BSIM4v4_MOD_NOIA:
            value->rValue = model->BSIM4v4oxideTrapDensityA;
            return(OK);
        case BSIM4v4_MOD_NOIB:
            value->rValue = model->BSIM4v4oxideTrapDensityB;
            return(OK);
        case BSIM4v4_MOD_NOIC:
            value->rValue = model->BSIM4v4oxideTrapDensityC;
            return(OK);
        case BSIM4v4_MOD_EM:
            value->rValue = model->BSIM4v4em;
            return(OK);
        case BSIM4v4_MOD_EF:
            value->rValue = model->BSIM4v4ef;
            return(OK);
        case BSIM4v4_MOD_AF:
            value->rValue = model->BSIM4v4af;
            return(OK);
        case BSIM4v4_MOD_KF:
            value->rValue = model->BSIM4v4kf;
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}



