/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: Weidong Liu and Pin Su         Feb 1999
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
Modified by Wei Jin 99/9/27
File: b3soiddmask.c          98/5/01
Modified by Paolo Nenzi 2002
**********/


#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "b3soidddef.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
B3SOIDDmAsk(CKTcircuit *ckt, GENmodel *inst, int which, IFvalue *value)
{
    B3SOIDDmodel *model = (B3SOIDDmodel *)inst;

    NG_IGNORE(ckt);

    switch(which) 
    {   case B3SOIDD_MOD_MOBMOD:
            value->iValue = model->B3SOIDDmobMod; 
            return(OK);
        case B3SOIDD_MOD_PARAMCHK:
            value->iValue = model->B3SOIDDparamChk; 
            return(OK);
        case B3SOIDD_MOD_BINUNIT:
            value->iValue = model->B3SOIDDbinUnit; 
            return(OK);
        case B3SOIDD_MOD_CAPMOD:
            value->iValue = model->B3SOIDDcapMod; 
            return(OK);
        case B3SOIDD_MOD_SHMOD:
            value->iValue = model->B3SOIDDshMod; 
            return(OK);
        case B3SOIDD_MOD_NOIMOD:
            value->iValue = model->B3SOIDDnoiMod; 
            return(OK);
        case  B3SOIDD_MOD_VERSION :
          value->rValue = model->B3SOIDDversion;
            return(OK);
        case  B3SOIDD_MOD_TOX :
          value->rValue = model->B3SOIDDtox;
            return(OK);
        case  B3SOIDD_MOD_CDSC :
          value->rValue = model->B3SOIDDcdsc;
            return(OK);
        case  B3SOIDD_MOD_CDSCB :
          value->rValue = model->B3SOIDDcdscb;
            return(OK);

        case  B3SOIDD_MOD_CDSCD :
          value->rValue = model->B3SOIDDcdscd;
            return(OK);

        case  B3SOIDD_MOD_CIT :
          value->rValue = model->B3SOIDDcit;
            return(OK);
        case  B3SOIDD_MOD_NFACTOR :
          value->rValue = model->B3SOIDDnfactor;
            return(OK);
        case B3SOIDD_MOD_VSAT:
            value->rValue = model->B3SOIDDvsat;
            return(OK);
        case B3SOIDD_MOD_AT:
            value->rValue = model->B3SOIDDat;
            return(OK);
        case B3SOIDD_MOD_A0:
            value->rValue = model->B3SOIDDa0;
            return(OK);

        case B3SOIDD_MOD_AGS:
            value->rValue = model->B3SOIDDags;
            return(OK);

        case B3SOIDD_MOD_A1:
            value->rValue = model->B3SOIDDa1;
            return(OK);
        case B3SOIDD_MOD_A2:
            value->rValue = model->B3SOIDDa2;
            return(OK);
        case B3SOIDD_MOD_KETA:
            value->rValue = model->B3SOIDDketa;
            return(OK);   
        case B3SOIDD_MOD_NSUB:
            value->rValue = model->B3SOIDDnsub;
            return(OK);
        case B3SOIDD_MOD_NPEAK:
            value->rValue = model->B3SOIDDnpeak;
            return(OK);
        case B3SOIDD_MOD_NGATE:
            value->rValue = model->B3SOIDDngate;
            return(OK);
        case B3SOIDD_MOD_GAMMA1:
            value->rValue = model->B3SOIDDgamma1;
            return(OK);
        case B3SOIDD_MOD_GAMMA2:
            value->rValue = model->B3SOIDDgamma2;
            return(OK);
        case B3SOIDD_MOD_VBX:
            value->rValue = model->B3SOIDDvbx;
            return(OK);
        case B3SOIDD_MOD_VBM:
            value->rValue = model->B3SOIDDvbm;
            return(OK);
        case B3SOIDD_MOD_XT:
            value->rValue = model->B3SOIDDxt;
            return(OK);
        case  B3SOIDD_MOD_K1:
          value->rValue = model->B3SOIDDk1;
            return(OK);
        case  B3SOIDD_MOD_KT1:
          value->rValue = model->B3SOIDDkt1;
            return(OK);
        case  B3SOIDD_MOD_KT1L:
          value->rValue = model->B3SOIDDkt1l;
            return(OK);
        case  B3SOIDD_MOD_KT2 :
          value->rValue = model->B3SOIDDkt2;
            return(OK);
        case  B3SOIDD_MOD_K2 :
          value->rValue = model->B3SOIDDk2;
            return(OK);
        case  B3SOIDD_MOD_K3:
          value->rValue = model->B3SOIDDk3;
            return(OK);
        case  B3SOIDD_MOD_K3B:
          value->rValue = model->B3SOIDDk3b;
            return(OK);
        case  B3SOIDD_MOD_W0:
          value->rValue = model->B3SOIDDw0;
            return(OK);
        case  B3SOIDD_MOD_NLX:
          value->rValue = model->B3SOIDDnlx;
            return(OK);
        case  B3SOIDD_MOD_DVT0 :                
          value->rValue = model->B3SOIDDdvt0;
            return(OK);
        case  B3SOIDD_MOD_DVT1 :             
          value->rValue = model->B3SOIDDdvt1;
            return(OK);
        case  B3SOIDD_MOD_DVT2 :             
          value->rValue = model->B3SOIDDdvt2;
            return(OK);
        case  B3SOIDD_MOD_DVT0W :                
          value->rValue = model->B3SOIDDdvt0w;
            return(OK);
        case  B3SOIDD_MOD_DVT1W :             
          value->rValue = model->B3SOIDDdvt1w;
            return(OK);
        case  B3SOIDD_MOD_DVT2W :             
          value->rValue = model->B3SOIDDdvt2w;
            return(OK);
        case  B3SOIDD_MOD_DROUT :           
          value->rValue = model->B3SOIDDdrout;
            return(OK);
        case  B3SOIDD_MOD_DSUB :           
          value->rValue = model->B3SOIDDdsub;
            return(OK);
        case B3SOIDD_MOD_VTH0:
            value->rValue = model->B3SOIDDvth0; 
            return(OK);
        case B3SOIDD_MOD_UA:
            value->rValue = model->B3SOIDDua; 
            return(OK);
        case B3SOIDD_MOD_UA1:
            value->rValue = model->B3SOIDDua1; 
            return(OK);
        case B3SOIDD_MOD_UB:
            value->rValue = model->B3SOIDDub;  
            return(OK);
        case B3SOIDD_MOD_UB1:
            value->rValue = model->B3SOIDDub1;  
            return(OK);
        case B3SOIDD_MOD_UC:
            value->rValue = model->B3SOIDDuc; 
            return(OK);
        case B3SOIDD_MOD_UC1:
            value->rValue = model->B3SOIDDuc1; 
            return(OK);
        case B3SOIDD_MOD_U0:
            value->rValue = model->B3SOIDDu0;
            return(OK);
        case B3SOIDD_MOD_UTE:
            value->rValue = model->B3SOIDDute;
            return(OK);
        case B3SOIDD_MOD_VOFF:
            value->rValue = model->B3SOIDDvoff;
            return(OK);
        case B3SOIDD_MOD_DELTA:
            value->rValue = model->B3SOIDDdelta;
            return(OK);
        case B3SOIDD_MOD_RDSW:
            value->rValue = model->B3SOIDDrdsw; 
            return(OK);             
        case B3SOIDD_MOD_PRWG:
            value->rValue = model->B3SOIDDprwg; 
            return(OK);             
        case B3SOIDD_MOD_PRWB:
            value->rValue = model->B3SOIDDprwb; 
            return(OK);             
        case B3SOIDD_MOD_PRT:
            value->rValue = model->B3SOIDDprt; 
            return(OK);              
        case B3SOIDD_MOD_ETA0:
            value->rValue = model->B3SOIDDeta0; 
            return(OK);               
        case B3SOIDD_MOD_ETAB:
            value->rValue = model->B3SOIDDetab; 
            return(OK);               
        case B3SOIDD_MOD_PCLM:
            value->rValue = model->B3SOIDDpclm; 
            return(OK);               
        case B3SOIDD_MOD_PDIBL1:
            value->rValue = model->B3SOIDDpdibl1; 
            return(OK);               
        case B3SOIDD_MOD_PDIBL2:
            value->rValue = model->B3SOIDDpdibl2; 
            return(OK);               
        case B3SOIDD_MOD_PDIBLB:
            value->rValue = model->B3SOIDDpdiblb; 
            return(OK);               
        case B3SOIDD_MOD_PVAG:
            value->rValue = model->B3SOIDDpvag; 
            return(OK);               
        case B3SOIDD_MOD_WR:
            value->rValue = model->B3SOIDDwr;
            return(OK);
        case B3SOIDD_MOD_DWG:
            value->rValue = model->B3SOIDDdwg;
            return(OK);
        case B3SOIDD_MOD_DWB:
            value->rValue = model->B3SOIDDdwb;
            return(OK);
        case B3SOIDD_MOD_B0:
            value->rValue = model->B3SOIDDb0;
            return(OK);
        case B3SOIDD_MOD_B1:
            value->rValue = model->B3SOIDDb1;
            return(OK);
        case B3SOIDD_MOD_ALPHA0:
            value->rValue = model->B3SOIDDalpha0;
            return(OK);
        case B3SOIDD_MOD_ALPHA1:
            value->rValue = model->B3SOIDDalpha1;
            return(OK);
        case B3SOIDD_MOD_BETA0:
            value->rValue = model->B3SOIDDbeta0;
            return(OK);

        case B3SOIDD_MOD_CGSL:
            value->rValue = model->B3SOIDDcgsl;
            return(OK);
        case B3SOIDD_MOD_CGDL:
            value->rValue = model->B3SOIDDcgdl;
            return(OK);
        case B3SOIDD_MOD_CKAPPA:
            value->rValue = model->B3SOIDDckappa;
            return(OK);
        case B3SOIDD_MOD_CF:
            value->rValue = model->B3SOIDDcf;
            return(OK);
        case B3SOIDD_MOD_CLC:
            value->rValue = model->B3SOIDDclc;
            return(OK);
        case B3SOIDD_MOD_CLE:
            value->rValue = model->B3SOIDDcle;
            return(OK);
        case B3SOIDD_MOD_DWC:
            value->rValue = model->B3SOIDDdwc;
            return(OK);
        case B3SOIDD_MOD_DLC:
            value->rValue = model->B3SOIDDdlc;
            return(OK);

        case B3SOIDD_MOD_TBOX:
            value->rValue = model->B3SOIDDtbox; 
            return(OK);
        case B3SOIDD_MOD_TSI:
            value->rValue = model->B3SOIDDtsi; 
            return(OK);
        case B3SOIDD_MOD_KB1:
            value->rValue = model->B3SOIDDkb1; 
            return(OK);
        case B3SOIDD_MOD_KB3:
            value->rValue = model->B3SOIDDkb3; 
            return(OK);
        case B3SOIDD_MOD_DVBD0:
            value->rValue = model->B3SOIDDdvbd0; 
            return(OK);
        case B3SOIDD_MOD_DVBD1:
            value->rValue = model->B3SOIDDdvbd1; 
            return(OK);
        case B3SOIDD_MOD_DELP:
            value->rValue = model->B3SOIDDdelp; 
            return(OK);
        case B3SOIDD_MOD_VBSA:
            value->rValue = model->B3SOIDDvbsa; 
            return(OK);
        case B3SOIDD_MOD_RBODY:
            value->rValue = model->B3SOIDDrbody; 
            return(OK);
        case B3SOIDD_MOD_RBSH:
            value->rValue = model->B3SOIDDrbsh; 
            return(OK);
        case B3SOIDD_MOD_ADICE0:
            value->rValue = model->B3SOIDDadice0; 
            return(OK);
        case B3SOIDD_MOD_ABP:
            value->rValue = model->B3SOIDDabp; 
            return(OK);
        case B3SOIDD_MOD_MXC:
            value->rValue = model->B3SOIDDmxc; 
            return(OK);
        case B3SOIDD_MOD_RTH0:
            value->rValue = model->B3SOIDDrth0; 
            return(OK);
        case B3SOIDD_MOD_CTH0:
            value->rValue = model->B3SOIDDcth0; 
            return(OK);
        case B3SOIDD_MOD_AII:
            value->rValue = model->B3SOIDDaii; 
            return(OK);
        case B3SOIDD_MOD_BII:
            value->rValue = model->B3SOIDDbii; 
            return(OK);
        case B3SOIDD_MOD_CII:
            value->rValue = model->B3SOIDDcii; 
            return(OK);
        case B3SOIDD_MOD_DII:
            value->rValue = model->B3SOIDDdii; 
            return(OK);
        case B3SOIDD_MOD_NDIODE:
            value->rValue = model->B3SOIDDndiode; 
            return(OK);
        case B3SOIDD_MOD_NTUN:
            value->rValue = model->B3SOIDDntun; 
            return(OK);
        case B3SOIDD_MOD_ISBJT:
            value->rValue = model->B3SOIDDisbjt; 
            return(OK);
        case B3SOIDD_MOD_ISDIF:
            value->rValue = model->B3SOIDDisdif; 
            return(OK);
        case B3SOIDD_MOD_ISREC:
            value->rValue = model->B3SOIDDisrec; 
            return(OK);
        case B3SOIDD_MOD_ISTUN:
            value->rValue = model->B3SOIDDistun; 
            return(OK);
        case B3SOIDD_MOD_XBJT:
            value->rValue = model->B3SOIDDxbjt; 
            return(OK);
        case B3SOIDD_MOD_XREC:
            value->rValue = model->B3SOIDDxrec; 
            return(OK);
        case B3SOIDD_MOD_XTUN:
            value->rValue = model->B3SOIDDxtun; 
            return(OK);
        case B3SOIDD_MOD_EDL:
            value->rValue = model->B3SOIDDedl; 
            return(OK);
        case B3SOIDD_MOD_KBJT1:
            value->rValue = model->B3SOIDDkbjt1; 
            return(OK);
        case B3SOIDD_MOD_TT:
            value->rValue = model->B3SOIDDtt; 
            return(OK);
        case B3SOIDD_MOD_VSDTH:
            value->rValue = model->B3SOIDDvsdth; 
            return(OK);
        case B3SOIDD_MOD_VSDFB:
            value->rValue = model->B3SOIDDvsdfb; 
            return(OK);
        case B3SOIDD_MOD_CSDMIN:
            value->rValue = model->B3SOIDDcsdmin; 
            return(OK);
        case B3SOIDD_MOD_ASD:
            value->rValue = model->B3SOIDDasd; 
            return(OK);

        case  B3SOIDD_MOD_TNOM :
          value->rValue = model->B3SOIDDtnom;
            return(OK);
        case B3SOIDD_MOD_CGSO:
            value->rValue = model->B3SOIDDcgso; 
            return(OK);
        case B3SOIDD_MOD_CGDO:
            value->rValue = model->B3SOIDDcgdo; 
            return(OK);
        case B3SOIDD_MOD_CGEO:
            value->rValue = model->B3SOIDDcgeo; 
            return(OK);
        case B3SOIDD_MOD_XPART:
            value->rValue = model->B3SOIDDxpart; 
            return(OK);
        case B3SOIDD_MOD_RSH:
            value->rValue = model->B3SOIDDsheetResistance; 
            return(OK);
        case B3SOIDD_MOD_PBSWG:
            value->rValue = model->B3SOIDDGatesidewallJctPotential; 
            return(OK);
        case B3SOIDD_MOD_MJSWG:
            value->rValue = model->B3SOIDDbodyJctGateSideGradingCoeff; 
            return(OK);
        case B3SOIDD_MOD_CJSWG:
            value->rValue = model->B3SOIDDunitLengthGateSidewallJctCap; 
            return(OK);
        case B3SOIDD_MOD_CSDESW:
            value->rValue = model->B3SOIDDcsdesw; 
            return(OK);
        case B3SOIDD_MOD_LINT:
            value->rValue = model->B3SOIDDLint; 
            return(OK);
        case B3SOIDD_MOD_LL:
            value->rValue = model->B3SOIDDLl;
            return(OK);
        case B3SOIDD_MOD_LLN:
            value->rValue = model->B3SOIDDLln;
            return(OK);
        case B3SOIDD_MOD_LW:
            value->rValue = model->B3SOIDDLw;
            return(OK);
        case B3SOIDD_MOD_LWN:
            value->rValue = model->B3SOIDDLwn;
            return(OK);
        case B3SOIDD_MOD_LWL:
            value->rValue = model->B3SOIDDLwl;
            return(OK);
        case B3SOIDD_MOD_WINT:
            value->rValue = model->B3SOIDDWint;
            return(OK);
        case B3SOIDD_MOD_WL:
            value->rValue = model->B3SOIDDWl;
            return(OK);
        case B3SOIDD_MOD_WLN:
            value->rValue = model->B3SOIDDWln;
            return(OK);
        case B3SOIDD_MOD_WW:
            value->rValue = model->B3SOIDDWw;
            return(OK);
        case B3SOIDD_MOD_WWN:
            value->rValue = model->B3SOIDDWwn;
            return(OK);
        case B3SOIDD_MOD_WWL:
            value->rValue = model->B3SOIDDWwl;
            return(OK);
        case B3SOIDD_MOD_NOIA:
            value->rValue = model->B3SOIDDoxideTrapDensityA;
            return(OK);
        case B3SOIDD_MOD_NOIB:
            value->rValue = model->B3SOIDDoxideTrapDensityB;
            return(OK);
        case B3SOIDD_MOD_NOIC:
            value->rValue = model->B3SOIDDoxideTrapDensityC;
            return(OK);
        case B3SOIDD_MOD_NOIF:
            value->rValue = model->B3SOIDDnoif;
            return(OK);
        case B3SOIDD_MOD_EM:
            value->rValue = model->B3SOIDDem;
            return(OK);
        case B3SOIDD_MOD_EF:
            value->rValue = model->B3SOIDDef;
            return(OK);
        case B3SOIDD_MOD_AF:
            value->rValue = model->B3SOIDDaf;
            return(OK);
        case B3SOIDD_MOD_KF:
            value->rValue = model->B3SOIDDkf;
            return(OK);

/* Added for binning - START */
        /* Length Dependence */
        case B3SOIDD_MOD_LNPEAK:
            value->rValue = model->B3SOIDDlnpeak;
            return(OK);
        case B3SOIDD_MOD_LNSUB:
            value->rValue = model->B3SOIDDlnsub;
            return(OK);
        case B3SOIDD_MOD_LNGATE:
            value->rValue = model->B3SOIDDlngate;
            return(OK);
        case B3SOIDD_MOD_LVTH0:
            value->rValue = model->B3SOIDDlvth0;
            return(OK);
        case  B3SOIDD_MOD_LK1:
          value->rValue = model->B3SOIDDlk1;
            return(OK);
        case  B3SOIDD_MOD_LK2:
          value->rValue = model->B3SOIDDlk2;
            return(OK);
        case  B3SOIDD_MOD_LK3:
          value->rValue = model->B3SOIDDlk3;
            return(OK);
        case  B3SOIDD_MOD_LK3B:
          value->rValue = model->B3SOIDDlk3b;
            return(OK);
        case  B3SOIDD_MOD_LVBSA:
          value->rValue = model->B3SOIDDlvbsa;
            return(OK);
        case  B3SOIDD_MOD_LDELP:
          value->rValue = model->B3SOIDDldelp;
            return(OK);
        case  B3SOIDD_MOD_LKB1:
            value->rValue = model->B3SOIDDlkb1;
            return(OK);
        case  B3SOIDD_MOD_LKB3:
            value->rValue = model->B3SOIDDlkb3;
            return(OK);
        case  B3SOIDD_MOD_LDVBD0:
            value->rValue = model->B3SOIDDdvbd0;
            return(OK);
        case  B3SOIDD_MOD_LDVBD1:
            value->rValue = model->B3SOIDDdvbd1;
            return(OK);
        case  B3SOIDD_MOD_LW0:
          value->rValue = model->B3SOIDDlw0;
            return(OK);
        case  B3SOIDD_MOD_LNLX:
          value->rValue = model->B3SOIDDlnlx;
            return(OK);
        case  B3SOIDD_MOD_LDVT0 :
          value->rValue = model->B3SOIDDldvt0;
            return(OK);
        case  B3SOIDD_MOD_LDVT1 :
          value->rValue = model->B3SOIDDldvt1;
            return(OK);
        case  B3SOIDD_MOD_LDVT2 :
          value->rValue = model->B3SOIDDldvt2;
            return(OK);
        case  B3SOIDD_MOD_LDVT0W :
          value->rValue = model->B3SOIDDldvt0w;
            return(OK);
        case  B3SOIDD_MOD_LDVT1W :
          value->rValue = model->B3SOIDDldvt1w;
            return(OK);
        case  B3SOIDD_MOD_LDVT2W :
          value->rValue = model->B3SOIDDldvt2w;
            return(OK);
        case B3SOIDD_MOD_LU0:
            value->rValue = model->B3SOIDDlu0;
            return(OK);
        case B3SOIDD_MOD_LUA:
            value->rValue = model->B3SOIDDlua;
            return(OK);
        case B3SOIDD_MOD_LUB:
            value->rValue = model->B3SOIDDlub;
            return(OK);
        case B3SOIDD_MOD_LUC:
            value->rValue = model->B3SOIDDluc;
            return(OK);
        case B3SOIDD_MOD_LVSAT:
            value->rValue = model->B3SOIDDlvsat;
            return(OK);
        case B3SOIDD_MOD_LA0:
            value->rValue = model->B3SOIDDla0;
            return(OK);
        case B3SOIDD_MOD_LAGS:
            value->rValue = model->B3SOIDDlags;
            return(OK);
        case B3SOIDD_MOD_LB0:
            value->rValue = model->B3SOIDDlb0;
            return(OK);
        case B3SOIDD_MOD_LB1:
            value->rValue = model->B3SOIDDlb1;
            return(OK);
        case B3SOIDD_MOD_LKETA:
            value->rValue = model->B3SOIDDlketa;
            return(OK);
        case B3SOIDD_MOD_LABP:
            value->rValue = model->B3SOIDDlabp;
            return(OK);
        case B3SOIDD_MOD_LMXC:
            value->rValue = model->B3SOIDDlmxc;
            return(OK);
        case B3SOIDD_MOD_LADICE0:
            value->rValue = model->B3SOIDDladice0;
            return(OK);
        case B3SOIDD_MOD_LA1:
            value->rValue = model->B3SOIDDla1;
            return(OK);
        case B3SOIDD_MOD_LA2:
            value->rValue = model->B3SOIDDla2;
            return(OK);
        case B3SOIDD_MOD_LRDSW:
            value->rValue = model->B3SOIDDlrdsw;
            return(OK);
        case B3SOIDD_MOD_LPRWB:
            value->rValue = model->B3SOIDDlprwb;
            return(OK);
        case B3SOIDD_MOD_LPRWG:
            value->rValue = model->B3SOIDDlprwg;
            return(OK);
        case B3SOIDD_MOD_LWR:
            value->rValue = model->B3SOIDDlwr;
            return(OK);
        case  B3SOIDD_MOD_LNFACTOR :
          value->rValue = model->B3SOIDDlnfactor;
            return(OK);
        case B3SOIDD_MOD_LDWG:
            value->rValue = model->B3SOIDDldwg;
            return(OK);
        case B3SOIDD_MOD_LDWB:
            value->rValue = model->B3SOIDDldwb;
            return(OK);
        case B3SOIDD_MOD_LVOFF:
            value->rValue = model->B3SOIDDlvoff;
            return(OK);
        case B3SOIDD_MOD_LETA0:
            value->rValue = model->B3SOIDDleta0;
            return(OK);
        case B3SOIDD_MOD_LETAB:
            value->rValue = model->B3SOIDDletab;
            return(OK);
        case  B3SOIDD_MOD_LDSUB :
          value->rValue = model->B3SOIDDldsub;
            return(OK);
        case  B3SOIDD_MOD_LCIT :
          value->rValue = model->B3SOIDDlcit;
            return(OK);
        case  B3SOIDD_MOD_LCDSC :
          value->rValue = model->B3SOIDDlcdsc;
            return(OK);
        case  B3SOIDD_MOD_LCDSCB :
          value->rValue = model->B3SOIDDlcdscb;
            return(OK);
        case  B3SOIDD_MOD_LCDSCD :
          value->rValue = model->B3SOIDDlcdscd;
            return(OK);
        case B3SOIDD_MOD_LPCLM:
            value->rValue = model->B3SOIDDlpclm;
            return(OK);
        case B3SOIDD_MOD_LPDIBL1:
            value->rValue = model->B3SOIDDlpdibl1;
            return(OK);
        case B3SOIDD_MOD_LPDIBL2:
            value->rValue = model->B3SOIDDlpdibl2;
            return(OK);
        case B3SOIDD_MOD_LPDIBLB:
            value->rValue = model->B3SOIDDlpdiblb;
            return(OK);
        case  B3SOIDD_MOD_LDROUT :
          value->rValue = model->B3SOIDDldrout;
            return(OK);
        case B3SOIDD_MOD_LPVAG:
            value->rValue = model->B3SOIDDlpvag;
            return(OK);
        case B3SOIDD_MOD_LDELTA:
            value->rValue = model->B3SOIDDldelta;
            return(OK);
        case B3SOIDD_MOD_LAII:
            value->rValue = model->B3SOIDDlaii;
            return(OK);
        case B3SOIDD_MOD_LBII:
            value->rValue = model->B3SOIDDlbii;
            return(OK);
        case B3SOIDD_MOD_LCII:
            value->rValue = model->B3SOIDDlcii;
            return(OK);
        case B3SOIDD_MOD_LDII:
            value->rValue = model->B3SOIDDldii;
            return(OK);
        case B3SOIDD_MOD_LALPHA0:
            value->rValue = model->B3SOIDDlalpha0;
            return(OK);
        case B3SOIDD_MOD_LALPHA1:
            value->rValue = model->B3SOIDDlalpha1;
            return(OK);
        case B3SOIDD_MOD_LBETA0:
            value->rValue = model->B3SOIDDlbeta0;
            return(OK);
        case B3SOIDD_MOD_LAGIDL:
            value->rValue = model->B3SOIDDlagidl;
            return(OK);
        case B3SOIDD_MOD_LBGIDL:
            value->rValue = model->B3SOIDDlbgidl;
            return(OK);
        case B3SOIDD_MOD_LNGIDL:
            value->rValue = model->B3SOIDDlngidl;
            return(OK);
        case B3SOIDD_MOD_LNTUN:
            value->rValue = model->B3SOIDDlntun;
            return(OK);
        case B3SOIDD_MOD_LNDIODE:
            value->rValue = model->B3SOIDDlndiode;
            return(OK);
        case B3SOIDD_MOD_LISBJT:
            value->rValue = model->B3SOIDDlisbjt;
            return(OK);
        case B3SOIDD_MOD_LISDIF:
            value->rValue = model->B3SOIDDlisdif;
            return(OK);
        case B3SOIDD_MOD_LISREC:
            value->rValue = model->B3SOIDDlisrec;
            return(OK);
        case B3SOIDD_MOD_LISTUN:
            value->rValue = model->B3SOIDDlistun;
            return(OK);
        case B3SOIDD_MOD_LEDL:
            value->rValue = model->B3SOIDDledl;
            return(OK);
        case B3SOIDD_MOD_LKBJT1:
            value->rValue = model->B3SOIDDlkbjt1;
            return(OK);
	/* CV Model */
        case B3SOIDD_MOD_LVSDFB:
            value->rValue = model->B3SOIDDlvsdfb;
            return(OK);
        case B3SOIDD_MOD_LVSDTH:
            value->rValue = model->B3SOIDDlvsdth;
            return(OK);
        /* Width Dependence */
        case B3SOIDD_MOD_WNPEAK:
            value->rValue = model->B3SOIDDwnpeak;
            return(OK);
        case B3SOIDD_MOD_WNSUB:
            value->rValue = model->B3SOIDDwnsub;
            return(OK);
        case B3SOIDD_MOD_WNGATE:
            value->rValue = model->B3SOIDDwngate;
            return(OK);
        case B3SOIDD_MOD_WVTH0:
            value->rValue = model->B3SOIDDwvth0;
            return(OK);
        case  B3SOIDD_MOD_WK1:
          value->rValue = model->B3SOIDDwk1;
            return(OK);
        case  B3SOIDD_MOD_WK2:
          value->rValue = model->B3SOIDDwk2;
            return(OK);
        case  B3SOIDD_MOD_WK3:
          value->rValue = model->B3SOIDDwk3;
            return(OK);
        case  B3SOIDD_MOD_WK3B:
          value->rValue = model->B3SOIDDwk3b;
            return(OK);
        case  B3SOIDD_MOD_WVBSA:
          value->rValue = model->B3SOIDDwvbsa;
            return(OK);
        case  B3SOIDD_MOD_WDELP:
          value->rValue = model->B3SOIDDwdelp;
            return(OK);
        case  B3SOIDD_MOD_WKB1:
            value->rValue = model->B3SOIDDwkb1;
            return(OK);
        case  B3SOIDD_MOD_WKB3:
            value->rValue = model->B3SOIDDwkb3;
            return(OK);
        case  B3SOIDD_MOD_WDVBD0:
            value->rValue = model->B3SOIDDdvbd0;
            return(OK);
        case  B3SOIDD_MOD_WDVBD1:
            value->rValue = model->B3SOIDDdvbd1;
            return(OK);
        case  B3SOIDD_MOD_WW0:
          value->rValue = model->B3SOIDDww0;
            return(OK);
        case  B3SOIDD_MOD_WNLX:
          value->rValue = model->B3SOIDDwnlx;
            return(OK);
        case  B3SOIDD_MOD_WDVT0 :
          value->rValue = model->B3SOIDDwdvt0;
            return(OK);
        case  B3SOIDD_MOD_WDVT1 :
          value->rValue = model->B3SOIDDwdvt1;
            return(OK);
        case  B3SOIDD_MOD_WDVT2 :
          value->rValue = model->B3SOIDDwdvt2;
            return(OK);
        case  B3SOIDD_MOD_WDVT0W :
          value->rValue = model->B3SOIDDwdvt0w;
            return(OK);
        case  B3SOIDD_MOD_WDVT1W :
          value->rValue = model->B3SOIDDwdvt1w;
            return(OK);
        case  B3SOIDD_MOD_WDVT2W :
          value->rValue = model->B3SOIDDwdvt2w;
            return(OK);
        case B3SOIDD_MOD_WU0:
            value->rValue = model->B3SOIDDwu0;
            return(OK);
        case B3SOIDD_MOD_WUA:
            value->rValue = model->B3SOIDDwua;
            return(OK);
        case B3SOIDD_MOD_WUB:
            value->rValue = model->B3SOIDDwub;
            return(OK);
        case B3SOIDD_MOD_WUC:
            value->rValue = model->B3SOIDDwuc;
            return(OK);
        case B3SOIDD_MOD_WVSAT:
            value->rValue = model->B3SOIDDwvsat;
            return(OK);
        case B3SOIDD_MOD_WA0:
            value->rValue = model->B3SOIDDwa0;
            return(OK);
        case B3SOIDD_MOD_WAGS:
            value->rValue = model->B3SOIDDwags;
            return(OK);
        case B3SOIDD_MOD_WB0:
            value->rValue = model->B3SOIDDwb0;
            return(OK);
        case B3SOIDD_MOD_WB1:
            value->rValue = model->B3SOIDDwb1;
            return(OK);
        case B3SOIDD_MOD_WKETA:
            value->rValue = model->B3SOIDDwketa;
            return(OK);
        case B3SOIDD_MOD_WABP:
            value->rValue = model->B3SOIDDwabp;
            return(OK);
        case B3SOIDD_MOD_WMXC:
            value->rValue = model->B3SOIDDwmxc;
            return(OK);
        case B3SOIDD_MOD_WADICE0:
            value->rValue = model->B3SOIDDwadice0;
            return(OK);
        case B3SOIDD_MOD_WA1:
            value->rValue = model->B3SOIDDwa1;
            return(OK);
        case B3SOIDD_MOD_WA2:
            value->rValue = model->B3SOIDDwa2;
            return(OK);
        case B3SOIDD_MOD_WRDSW:
            value->rValue = model->B3SOIDDwrdsw;
            return(OK);
        case B3SOIDD_MOD_WPRWB:
            value->rValue = model->B3SOIDDwprwb;
            return(OK);
        case B3SOIDD_MOD_WPRWG:
            value->rValue = model->B3SOIDDwprwg;
            return(OK);
        case B3SOIDD_MOD_WWR:
            value->rValue = model->B3SOIDDwwr;
            return(OK);
        case  B3SOIDD_MOD_WNFACTOR :
          value->rValue = model->B3SOIDDwnfactor;
            return(OK);
        case B3SOIDD_MOD_WDWG:
            value->rValue = model->B3SOIDDwdwg;
            return(OK);
        case B3SOIDD_MOD_WDWB:
            value->rValue = model->B3SOIDDwdwb;
            return(OK);
        case B3SOIDD_MOD_WVOFF:
            value->rValue = model->B3SOIDDwvoff;
            return(OK);
        case B3SOIDD_MOD_WETA0:
            value->rValue = model->B3SOIDDweta0;
            return(OK);
        case B3SOIDD_MOD_WETAB:
            value->rValue = model->B3SOIDDwetab;
            return(OK);
        case  B3SOIDD_MOD_WDSUB :
          value->rValue = model->B3SOIDDwdsub;
            return(OK);
        case  B3SOIDD_MOD_WCIT :
          value->rValue = model->B3SOIDDwcit;
            return(OK);
        case  B3SOIDD_MOD_WCDSC :
          value->rValue = model->B3SOIDDwcdsc;
            return(OK);
        case  B3SOIDD_MOD_WCDSCB :
          value->rValue = model->B3SOIDDwcdscb;
            return(OK);
        case  B3SOIDD_MOD_WCDSCD :
          value->rValue = model->B3SOIDDwcdscd;
            return(OK);
        case B3SOIDD_MOD_WPCLM:
            value->rValue = model->B3SOIDDwpclm;
            return(OK);
        case B3SOIDD_MOD_WPDIBL1:
            value->rValue = model->B3SOIDDwpdibl1;
            return(OK);
        case B3SOIDD_MOD_WPDIBL2:
            value->rValue = model->B3SOIDDwpdibl2;
            return(OK);
        case B3SOIDD_MOD_WPDIBLB:
            value->rValue = model->B3SOIDDwpdiblb;
            return(OK);
        case  B3SOIDD_MOD_WDROUT :
          value->rValue = model->B3SOIDDwdrout;
            return(OK);
        case B3SOIDD_MOD_WPVAG:
            value->rValue = model->B3SOIDDwpvag;
            return(OK);
        case B3SOIDD_MOD_WDELTA:
            value->rValue = model->B3SOIDDwdelta;
            return(OK);
        case B3SOIDD_MOD_WAII:
            value->rValue = model->B3SOIDDwaii;
            return(OK);
        case B3SOIDD_MOD_WBII:
            value->rValue = model->B3SOIDDwbii;
            return(OK);
        case B3SOIDD_MOD_WCII:
            value->rValue = model->B3SOIDDwcii;
            return(OK);
        case B3SOIDD_MOD_WDII:
            value->rValue = model->B3SOIDDwdii;
            return(OK);
        case B3SOIDD_MOD_WALPHA0:
            value->rValue = model->B3SOIDDwalpha0;
            return(OK);
        case B3SOIDD_MOD_WALPHA1:
            value->rValue = model->B3SOIDDwalpha1;
            return(OK);
        case B3SOIDD_MOD_WBETA0:
            value->rValue = model->B3SOIDDwbeta0;
            return(OK);
        case B3SOIDD_MOD_WAGIDL:
            value->rValue = model->B3SOIDDwagidl;
            return(OK);
        case B3SOIDD_MOD_WBGIDL:
            value->rValue = model->B3SOIDDwbgidl;
            return(OK);
        case B3SOIDD_MOD_WNGIDL:
            value->rValue = model->B3SOIDDwngidl;
            return(OK);
        case B3SOIDD_MOD_WNTUN:
            value->rValue = model->B3SOIDDwntun;
            return(OK);
        case B3SOIDD_MOD_WNDIODE:
            value->rValue = model->B3SOIDDwndiode;
            return(OK);
        case B3SOIDD_MOD_WISBJT:
            value->rValue = model->B3SOIDDwisbjt;
            return(OK);
        case B3SOIDD_MOD_WISDIF:
            value->rValue = model->B3SOIDDwisdif;
            return(OK);
        case B3SOIDD_MOD_WISREC:
            value->rValue = model->B3SOIDDwisrec;
            return(OK);
        case B3SOIDD_MOD_WISTUN:
            value->rValue = model->B3SOIDDwistun;
            return(OK);
        case B3SOIDD_MOD_WEDL:
            value->rValue = model->B3SOIDDwedl;
            return(OK);
        case B3SOIDD_MOD_WKBJT1:
            value->rValue = model->B3SOIDDwkbjt1;
            return(OK);
	/* CV Model */
        case B3SOIDD_MOD_WVSDFB:
            value->rValue = model->B3SOIDDwvsdfb;
            return(OK);
        case B3SOIDD_MOD_WVSDTH:
            value->rValue = model->B3SOIDDwvsdth;
            return(OK);
        /* Cross-term Dependence */
        case B3SOIDD_MOD_PNPEAK:
            value->rValue = model->B3SOIDDpnpeak;
            return(OK);
        case B3SOIDD_MOD_PNSUB:
            value->rValue = model->B3SOIDDpnsub;
            return(OK);
        case B3SOIDD_MOD_PNGATE:
            value->rValue = model->B3SOIDDpngate;
            return(OK);
        case B3SOIDD_MOD_PVTH0:
            value->rValue = model->B3SOIDDpvth0;
            return(OK);
        case  B3SOIDD_MOD_PK1:
          value->rValue = model->B3SOIDDpk1;
            return(OK);
        case  B3SOIDD_MOD_PK2:
          value->rValue = model->B3SOIDDpk2;
            return(OK);
        case  B3SOIDD_MOD_PK3:
          value->rValue = model->B3SOIDDpk3;
            return(OK);
        case  B3SOIDD_MOD_PK3B:
          value->rValue = model->B3SOIDDpk3b;
            return(OK);
        case  B3SOIDD_MOD_PVBSA:
          value->rValue = model->B3SOIDDpvbsa;
            return(OK);
        case  B3SOIDD_MOD_PDELP:
          value->rValue = model->B3SOIDDpdelp;
            return(OK);
        case  B3SOIDD_MOD_PKB1:
            value->rValue = model->B3SOIDDpkb1;
            return(OK);
        case  B3SOIDD_MOD_PKB3:
            value->rValue = model->B3SOIDDpkb3;
            return(OK);
        case  B3SOIDD_MOD_PDVBD0:
            value->rValue = model->B3SOIDDdvbd0;
            return(OK);
        case  B3SOIDD_MOD_PDVBD1:
            value->rValue = model->B3SOIDDdvbd1;
            return(OK);
        case  B3SOIDD_MOD_PW0:
          value->rValue = model->B3SOIDDpw0;
            return(OK);
        case  B3SOIDD_MOD_PNLX:
          value->rValue = model->B3SOIDDpnlx;
            return(OK);
        case  B3SOIDD_MOD_PDVT0 :
          value->rValue = model->B3SOIDDpdvt0;
            return(OK);
        case  B3SOIDD_MOD_PDVT1 :
          value->rValue = model->B3SOIDDpdvt1;
            return(OK);
        case  B3SOIDD_MOD_PDVT2 :
          value->rValue = model->B3SOIDDpdvt2;
            return(OK);
        case  B3SOIDD_MOD_PDVT0W :
          value->rValue = model->B3SOIDDpdvt0w;
            return(OK);
        case  B3SOIDD_MOD_PDVT1W :
          value->rValue = model->B3SOIDDpdvt1w;
            return(OK);
        case  B3SOIDD_MOD_PDVT2W :
          value->rValue = model->B3SOIDDpdvt2w;
            return(OK);
        case B3SOIDD_MOD_PU0:
            value->rValue = model->B3SOIDDpu0;
            return(OK);
        case B3SOIDD_MOD_PUA:
            value->rValue = model->B3SOIDDpua;
            return(OK);
        case B3SOIDD_MOD_PUB:
            value->rValue = model->B3SOIDDpub;
            return(OK);
        case B3SOIDD_MOD_PUC:
            value->rValue = model->B3SOIDDpuc;
            return(OK);
        case B3SOIDD_MOD_PVSAT:
            value->rValue = model->B3SOIDDpvsat;
            return(OK);
        case B3SOIDD_MOD_PA0:
            value->rValue = model->B3SOIDDpa0;
            return(OK);
        case B3SOIDD_MOD_PAGS:
            value->rValue = model->B3SOIDDpags;
            return(OK);
        case B3SOIDD_MOD_PB0:
            value->rValue = model->B3SOIDDpb0;
            return(OK);
        case B3SOIDD_MOD_PB1:
            value->rValue = model->B3SOIDDpb1;
            return(OK);
        case B3SOIDD_MOD_PKETA:
            value->rValue = model->B3SOIDDpketa;
            return(OK);
        case B3SOIDD_MOD_PABP:
            value->rValue = model->B3SOIDDpabp;
            return(OK);
        case B3SOIDD_MOD_PMXC:
            value->rValue = model->B3SOIDDpmxc;
            return(OK);
        case B3SOIDD_MOD_PADICE0:
            value->rValue = model->B3SOIDDpadice0;
            return(OK);
        case B3SOIDD_MOD_PA1:
            value->rValue = model->B3SOIDDpa1;
            return(OK);
        case B3SOIDD_MOD_PA2:
            value->rValue = model->B3SOIDDpa2;
            return(OK);
        case B3SOIDD_MOD_PRDSW:
            value->rValue = model->B3SOIDDprdsw;
            return(OK);
        case B3SOIDD_MOD_PPRWB:
            value->rValue = model->B3SOIDDpprwb;
            return(OK);
        case B3SOIDD_MOD_PPRWG:
            value->rValue = model->B3SOIDDpprwg;
            return(OK);
        case B3SOIDD_MOD_PWR:
            value->rValue = model->B3SOIDDpwr;
            return(OK);
        case  B3SOIDD_MOD_PNFACTOR :
          value->rValue = model->B3SOIDDpnfactor;
            return(OK);
        case B3SOIDD_MOD_PDWG:
            value->rValue = model->B3SOIDDpdwg;
            return(OK);
        case B3SOIDD_MOD_PDWB:
            value->rValue = model->B3SOIDDpdwb;
            return(OK);
        case B3SOIDD_MOD_PVOFF:
            value->rValue = model->B3SOIDDpvoff;
            return(OK);
        case B3SOIDD_MOD_PETA0:
            value->rValue = model->B3SOIDDpeta0;
            return(OK);
        case B3SOIDD_MOD_PETAB:
            value->rValue = model->B3SOIDDpetab;
            return(OK);
        case  B3SOIDD_MOD_PDSUB :
          value->rValue = model->B3SOIDDpdsub;
            return(OK);
        case  B3SOIDD_MOD_PCIT :
          value->rValue = model->B3SOIDDpcit;
            return(OK);
        case  B3SOIDD_MOD_PCDSC :
          value->rValue = model->B3SOIDDpcdsc;
            return(OK);
        case  B3SOIDD_MOD_PCDSCB :
          value->rValue = model->B3SOIDDpcdscb;
            return(OK);
        case  B3SOIDD_MOD_PCDSCD :
          value->rValue = model->B3SOIDDpcdscd;
            return(OK);
        case B3SOIDD_MOD_PPCLM:
            value->rValue = model->B3SOIDDppclm;
            return(OK);
        case B3SOIDD_MOD_PPDIBL1:
            value->rValue = model->B3SOIDDppdibl1;
            return(OK);
        case B3SOIDD_MOD_PPDIBL2:
            value->rValue = model->B3SOIDDppdibl2;
            return(OK);
        case B3SOIDD_MOD_PPDIBLB:
            value->rValue = model->B3SOIDDppdiblb;
            return(OK);
        case  B3SOIDD_MOD_PDROUT :
          value->rValue = model->B3SOIDDpdrout;
            return(OK);
        case B3SOIDD_MOD_PPVAG:
            value->rValue = model->B3SOIDDppvag;
            return(OK);
        case B3SOIDD_MOD_PDELTA:
            value->rValue = model->B3SOIDDpdelta;
            return(OK);
        case B3SOIDD_MOD_PAII:
            value->rValue = model->B3SOIDDpaii;
            return(OK);
        case B3SOIDD_MOD_PBII:
            value->rValue = model->B3SOIDDpbii;
            return(OK);
        case B3SOIDD_MOD_PCII:
            value->rValue = model->B3SOIDDpcii;
            return(OK);
        case B3SOIDD_MOD_PDII:
            value->rValue = model->B3SOIDDpdii;
            return(OK);
        case B3SOIDD_MOD_PALPHA0:
            value->rValue = model->B3SOIDDpalpha0;
            return(OK);
        case B3SOIDD_MOD_PALPHA1:
            value->rValue = model->B3SOIDDpalpha1;
            return(OK);
        case B3SOIDD_MOD_PBETA0:
            value->rValue = model->B3SOIDDpbeta0;
            return(OK);
        case B3SOIDD_MOD_PAGIDL:
            value->rValue = model->B3SOIDDpagidl;
            return(OK);
        case B3SOIDD_MOD_PBGIDL:
            value->rValue = model->B3SOIDDpbgidl;
            return(OK);
        case B3SOIDD_MOD_PNGIDL:
            value->rValue = model->B3SOIDDpngidl;
            return(OK);
        case B3SOIDD_MOD_PNTUN:
            value->rValue = model->B3SOIDDpntun;
            return(OK);
        case B3SOIDD_MOD_PNDIODE:
            value->rValue = model->B3SOIDDpndiode;
            return(OK);
        case B3SOIDD_MOD_PISBJT:
            value->rValue = model->B3SOIDDpisbjt;
            return(OK);
        case B3SOIDD_MOD_PISDIF:
            value->rValue = model->B3SOIDDpisdif;
            return(OK);
        case B3SOIDD_MOD_PISREC:
            value->rValue = model->B3SOIDDpisrec;
            return(OK);
        case B3SOIDD_MOD_PISTUN:
            value->rValue = model->B3SOIDDpistun;
            return(OK);
        case B3SOIDD_MOD_PEDL:
            value->rValue = model->B3SOIDDpedl;
            return(OK);
        case B3SOIDD_MOD_PKBJT1:
            value->rValue = model->B3SOIDDpkbjt1;
            return(OK);
	/* CV Model */
        case B3SOIDD_MOD_PVSDFB:
            value->rValue = model->B3SOIDDpvsdfb;
            return(OK);
        case B3SOIDD_MOD_PVSDTH:
            value->rValue = model->B3SOIDDpvsdth;
            return(OK);
/* Added for binning - END */

        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}



