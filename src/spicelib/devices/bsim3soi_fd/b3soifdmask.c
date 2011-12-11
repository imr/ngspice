/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: Weidong Liu and Pin Su         Feb 1999
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soifdmask.c          98/5/01
Modified by Wei Jin 99/9/27
Modified by Paolo Nenzi 2002
**********/

/*
 * Revision 2.1  99/9/27 Pin Su 
 * BSIMFD2.1 release
 */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "b3soifddef.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
B3SOIFDmAsk(CKTcircuit *ckt, GENmodel *inst, int which, IFvalue *value)
{
    B3SOIFDmodel *model = (B3SOIFDmodel *)inst;

    NG_IGNORE(ckt);

    switch(which) 
    {   case B3SOIFD_MOD_MOBMOD:
            value->iValue = model->B3SOIFDmobMod; 
            return(OK);
        case B3SOIFD_MOD_PARAMCHK:
            value->iValue = model->B3SOIFDparamChk; 
            return(OK);
        case B3SOIFD_MOD_BINUNIT:
            value->iValue = model->B3SOIFDbinUnit; 
            return(OK);
        case B3SOIFD_MOD_CAPMOD:
            value->iValue = model->B3SOIFDcapMod; 
            return(OK);
        case B3SOIFD_MOD_SHMOD:
            value->iValue = model->B3SOIFDshMod; 
            return(OK);
        case B3SOIFD_MOD_NOIMOD:
            value->iValue = model->B3SOIFDnoiMod; 
            return(OK);
        case  B3SOIFD_MOD_VERSION :
          value->rValue = model->B3SOIFDversion;
            return(OK);
        case  B3SOIFD_MOD_TOX :
          value->rValue = model->B3SOIFDtox;
            return(OK);
        case  B3SOIFD_MOD_CDSC :
          value->rValue = model->B3SOIFDcdsc;
            return(OK);
        case  B3SOIFD_MOD_CDSCB :
          value->rValue = model->B3SOIFDcdscb;
            return(OK);

        case  B3SOIFD_MOD_CDSCD :
          value->rValue = model->B3SOIFDcdscd;
            return(OK);

        case  B3SOIFD_MOD_CIT :
          value->rValue = model->B3SOIFDcit;
            return(OK);
        case  B3SOIFD_MOD_NFACTOR :
          value->rValue = model->B3SOIFDnfactor;
            return(OK);
        case B3SOIFD_MOD_VSAT:
            value->rValue = model->B3SOIFDvsat;
            return(OK);
        case B3SOIFD_MOD_AT:
            value->rValue = model->B3SOIFDat;
            return(OK);
        case B3SOIFD_MOD_A0:
            value->rValue = model->B3SOIFDa0;
            return(OK);

        case B3SOIFD_MOD_AGS:
            value->rValue = model->B3SOIFDags;
            return(OK);

        case B3SOIFD_MOD_A1:
            value->rValue = model->B3SOIFDa1;
            return(OK);
        case B3SOIFD_MOD_A2:
            value->rValue = model->B3SOIFDa2;
            return(OK);
        case B3SOIFD_MOD_KETA:
            value->rValue = model->B3SOIFDketa;
            return(OK);   
        case B3SOIFD_MOD_NSUB:
            value->rValue = model->B3SOIFDnsub;
            return(OK);
        case B3SOIFD_MOD_NPEAK:
            value->rValue = model->B3SOIFDnpeak;
            return(OK);
        case B3SOIFD_MOD_NGATE:
            value->rValue = model->B3SOIFDngate;
            return(OK);
        case B3SOIFD_MOD_GAMMA1:
            value->rValue = model->B3SOIFDgamma1;
            return(OK);
        case B3SOIFD_MOD_GAMMA2:
            value->rValue = model->B3SOIFDgamma2;
            return(OK);
        case B3SOIFD_MOD_VBX:
            value->rValue = model->B3SOIFDvbx;
            return(OK);
        case B3SOIFD_MOD_VBM:
            value->rValue = model->B3SOIFDvbm;
            return(OK);
        case B3SOIFD_MOD_XT:
            value->rValue = model->B3SOIFDxt;
            return(OK);
        case  B3SOIFD_MOD_K1:
          value->rValue = model->B3SOIFDk1;
            return(OK);
        case  B3SOIFD_MOD_KT1:
          value->rValue = model->B3SOIFDkt1;
            return(OK);
        case  B3SOIFD_MOD_KT1L:
          value->rValue = model->B3SOIFDkt1l;
            return(OK);
        case  B3SOIFD_MOD_KT2 :
          value->rValue = model->B3SOIFDkt2;
            return(OK);
        case  B3SOIFD_MOD_K2 :
          value->rValue = model->B3SOIFDk2;
            return(OK);
        case  B3SOIFD_MOD_K3:
          value->rValue = model->B3SOIFDk3;
            return(OK);
        case  B3SOIFD_MOD_K3B:
          value->rValue = model->B3SOIFDk3b;
            return(OK);
        case  B3SOIFD_MOD_W0:
          value->rValue = model->B3SOIFDw0;
            return(OK);
        case  B3SOIFD_MOD_NLX:
          value->rValue = model->B3SOIFDnlx;
            return(OK);
        case  B3SOIFD_MOD_DVT0 :                
          value->rValue = model->B3SOIFDdvt0;
            return(OK);
        case  B3SOIFD_MOD_DVT1 :             
          value->rValue = model->B3SOIFDdvt1;
            return(OK);
        case  B3SOIFD_MOD_DVT2 :             
          value->rValue = model->B3SOIFDdvt2;
            return(OK);
        case  B3SOIFD_MOD_DVT0W :                
          value->rValue = model->B3SOIFDdvt0w;
            return(OK);
        case  B3SOIFD_MOD_DVT1W :             
          value->rValue = model->B3SOIFDdvt1w;
            return(OK);
        case  B3SOIFD_MOD_DVT2W :             
          value->rValue = model->B3SOIFDdvt2w;
            return(OK);
        case  B3SOIFD_MOD_DROUT :           
          value->rValue = model->B3SOIFDdrout;
            return(OK);
        case  B3SOIFD_MOD_DSUB :           
          value->rValue = model->B3SOIFDdsub;
            return(OK);
        case B3SOIFD_MOD_VTH0:
            value->rValue = model->B3SOIFDvth0; 
            return(OK);
        case B3SOIFD_MOD_UA:
            value->rValue = model->B3SOIFDua; 
            return(OK);
        case B3SOIFD_MOD_UA1:
            value->rValue = model->B3SOIFDua1; 
            return(OK);
        case B3SOIFD_MOD_UB:
            value->rValue = model->B3SOIFDub;  
            return(OK);
        case B3SOIFD_MOD_UB1:
            value->rValue = model->B3SOIFDub1;  
            return(OK);
        case B3SOIFD_MOD_UC:
            value->rValue = model->B3SOIFDuc; 
            return(OK);
        case B3SOIFD_MOD_UC1:
            value->rValue = model->B3SOIFDuc1; 
            return(OK);
        case B3SOIFD_MOD_U0:
            value->rValue = model->B3SOIFDu0;
            return(OK);
        case B3SOIFD_MOD_UTE:
            value->rValue = model->B3SOIFDute;
            return(OK);
        case B3SOIFD_MOD_VOFF:
            value->rValue = model->B3SOIFDvoff;
            return(OK);
        case B3SOIFD_MOD_DELTA:
            value->rValue = model->B3SOIFDdelta;
            return(OK);
        case B3SOIFD_MOD_RDSW:
            value->rValue = model->B3SOIFDrdsw; 
            return(OK);             
        case B3SOIFD_MOD_PRWG:
            value->rValue = model->B3SOIFDprwg; 
            return(OK);             
        case B3SOIFD_MOD_PRWB:
            value->rValue = model->B3SOIFDprwb; 
            return(OK);             
        case B3SOIFD_MOD_PRT:
            value->rValue = model->B3SOIFDprt; 
            return(OK);              
        case B3SOIFD_MOD_ETA0:
            value->rValue = model->B3SOIFDeta0; 
            return(OK);               
        case B3SOIFD_MOD_ETAB:
            value->rValue = model->B3SOIFDetab; 
            return(OK);               
        case B3SOIFD_MOD_PCLM:
            value->rValue = model->B3SOIFDpclm; 
            return(OK);               
        case B3SOIFD_MOD_PDIBL1:
            value->rValue = model->B3SOIFDpdibl1; 
            return(OK);               
        case B3SOIFD_MOD_PDIBL2:
            value->rValue = model->B3SOIFDpdibl2; 
            return(OK);               
        case B3SOIFD_MOD_PDIBLB:
            value->rValue = model->B3SOIFDpdiblb; 
            return(OK);               
        case B3SOIFD_MOD_PVAG:
            value->rValue = model->B3SOIFDpvag; 
            return(OK);               
        case B3SOIFD_MOD_WR:
            value->rValue = model->B3SOIFDwr;
            return(OK);
        case B3SOIFD_MOD_DWG:
            value->rValue = model->B3SOIFDdwg;
            return(OK);
        case B3SOIFD_MOD_DWB:
            value->rValue = model->B3SOIFDdwb;
            return(OK);
        case B3SOIFD_MOD_B0:
            value->rValue = model->B3SOIFDb0;
            return(OK);
        case B3SOIFD_MOD_B1:
            value->rValue = model->B3SOIFDb1;
            return(OK);
        case B3SOIFD_MOD_ALPHA0:
            value->rValue = model->B3SOIFDalpha0;
            return(OK);
        case B3SOIFD_MOD_ALPHA1:
            value->rValue = model->B3SOIFDalpha1;
            return(OK);
        case B3SOIFD_MOD_BETA0:
            value->rValue = model->B3SOIFDbeta0;
            return(OK);

        case B3SOIFD_MOD_CGSL:
            value->rValue = model->B3SOIFDcgsl;
            return(OK);
        case B3SOIFD_MOD_CGDL:
            value->rValue = model->B3SOIFDcgdl;
            return(OK);
        case B3SOIFD_MOD_CKAPPA:
            value->rValue = model->B3SOIFDckappa;
            return(OK);
        case B3SOIFD_MOD_CF:
            value->rValue = model->B3SOIFDcf;
            return(OK);
        case B3SOIFD_MOD_CLC:
            value->rValue = model->B3SOIFDclc;
            return(OK);
        case B3SOIFD_MOD_CLE:
            value->rValue = model->B3SOIFDcle;
            return(OK);
        case B3SOIFD_MOD_DWC:
            value->rValue = model->B3SOIFDdwc;
            return(OK);
        case B3SOIFD_MOD_DLC:
            value->rValue = model->B3SOIFDdlc;
            return(OK);

        case B3SOIFD_MOD_TBOX:
            value->rValue = model->B3SOIFDtbox; 
            return(OK);
        case B3SOIFD_MOD_TSI:
            value->rValue = model->B3SOIFDtsi; 
            return(OK);
        case B3SOIFD_MOD_KB1:
            value->rValue = model->B3SOIFDkb1; 
            return(OK);
        case B3SOIFD_MOD_KB3:
            value->rValue = model->B3SOIFDkb3; 
            return(OK);
        case B3SOIFD_MOD_DVBD0:
            value->rValue = model->B3SOIFDdvbd0; 
            return(OK);
        case B3SOIFD_MOD_DVBD1:
            value->rValue = model->B3SOIFDdvbd1; 
            return(OK);
        case B3SOIFD_MOD_DELP:
            value->rValue = model->B3SOIFDdelp; 
            return(OK);
        case B3SOIFD_MOD_VBSA:
            value->rValue = model->B3SOIFDvbsa; 
            return(OK);
        case B3SOIFD_MOD_RBODY:
            value->rValue = model->B3SOIFDrbody; 
            return(OK);
        case B3SOIFD_MOD_RBSH:
            value->rValue = model->B3SOIFDrbsh; 
            return(OK);
        case B3SOIFD_MOD_ADICE0:
            value->rValue = model->B3SOIFDadice0; 
            return(OK);
        case B3SOIFD_MOD_ABP:
            value->rValue = model->B3SOIFDabp; 
            return(OK);
        case B3SOIFD_MOD_MXC:
            value->rValue = model->B3SOIFDmxc; 
            return(OK);
        case B3SOIFD_MOD_RTH0:
            value->rValue = model->B3SOIFDrth0; 
            return(OK);
        case B3SOIFD_MOD_CTH0:
            value->rValue = model->B3SOIFDcth0; 
            return(OK);
        case B3SOIFD_MOD_AII:
            value->rValue = model->B3SOIFDaii; 
            return(OK);
        case B3SOIFD_MOD_BII:
            value->rValue = model->B3SOIFDbii; 
            return(OK);
        case B3SOIFD_MOD_CII:
            value->rValue = model->B3SOIFDcii; 
            return(OK);
        case B3SOIFD_MOD_DII:
            value->rValue = model->B3SOIFDdii; 
            return(OK);
        case B3SOIFD_MOD_NDIODE:
            value->rValue = model->B3SOIFDndiode; 
            return(OK);
        case B3SOIFD_MOD_NTUN:
            value->rValue = model->B3SOIFDntun; 
            return(OK);
        case B3SOIFD_MOD_ISBJT:
            value->rValue = model->B3SOIFDisbjt; 
            return(OK);
        case B3SOIFD_MOD_ISDIF:
            value->rValue = model->B3SOIFDisdif; 
            return(OK);
        case B3SOIFD_MOD_ISREC:
            value->rValue = model->B3SOIFDisrec; 
            return(OK);
        case B3SOIFD_MOD_ISTUN:
            value->rValue = model->B3SOIFDistun; 
            return(OK);
        case B3SOIFD_MOD_XBJT:
            value->rValue = model->B3SOIFDxbjt; 
            return(OK);
        case B3SOIFD_MOD_XREC:
            value->rValue = model->B3SOIFDxrec; 
            return(OK);
        case B3SOIFD_MOD_XTUN:
            value->rValue = model->B3SOIFDxtun; 
            return(OK);
        case B3SOIFD_MOD_EDL:
            value->rValue = model->B3SOIFDedl; 
            return(OK);
        case B3SOIFD_MOD_KBJT1:
            value->rValue = model->B3SOIFDkbjt1; 
            return(OK);
        case B3SOIFD_MOD_TT:
            value->rValue = model->B3SOIFDtt; 
            return(OK);
        case B3SOIFD_MOD_VSDTH:
            value->rValue = model->B3SOIFDvsdth; 
            return(OK);
        case B3SOIFD_MOD_VSDFB:
            value->rValue = model->B3SOIFDvsdfb; 
            return(OK);
        case B3SOIFD_MOD_CSDMIN:
            value->rValue = model->B3SOIFDcsdmin; 
            return(OK);
        case B3SOIFD_MOD_ASD:
            value->rValue = model->B3SOIFDasd; 
            return(OK);

        case  B3SOIFD_MOD_TNOM :
          value->rValue = model->B3SOIFDtnom;
            return(OK);
        case B3SOIFD_MOD_CGSO:
            value->rValue = model->B3SOIFDcgso; 
            return(OK);
        case B3SOIFD_MOD_CGDO:
            value->rValue = model->B3SOIFDcgdo; 
            return(OK);
        case B3SOIFD_MOD_CGEO:
            value->rValue = model->B3SOIFDcgeo; 
            return(OK);
        case B3SOIFD_MOD_XPART:
            value->rValue = model->B3SOIFDxpart; 
            return(OK);
        case B3SOIFD_MOD_RSH:
            value->rValue = model->B3SOIFDsheetResistance; 
            return(OK);
        case B3SOIFD_MOD_PBSWG:
            value->rValue = model->B3SOIFDGatesidewallJctPotential; 
            return(OK);
        case B3SOIFD_MOD_MJSWG:
            value->rValue = model->B3SOIFDbodyJctGateSideGradingCoeff; 
            return(OK);
        case B3SOIFD_MOD_CJSWG:
            value->rValue = model->B3SOIFDunitLengthGateSidewallJctCap; 
            return(OK);
        case B3SOIFD_MOD_CSDESW:
            value->rValue = model->B3SOIFDcsdesw; 
            return(OK);
        case B3SOIFD_MOD_LINT:
            value->rValue = model->B3SOIFDLint; 
            return(OK);
        case B3SOIFD_MOD_LL:
            value->rValue = model->B3SOIFDLl;
            return(OK);
        case B3SOIFD_MOD_LLN:
            value->rValue = model->B3SOIFDLln;
            return(OK);
        case B3SOIFD_MOD_LW:
            value->rValue = model->B3SOIFDLw;
            return(OK);
        case B3SOIFD_MOD_LWN:
            value->rValue = model->B3SOIFDLwn;
            return(OK);
        case B3SOIFD_MOD_LWL:
            value->rValue = model->B3SOIFDLwl;
            return(OK);
        case B3SOIFD_MOD_WINT:
            value->rValue = model->B3SOIFDWint;
            return(OK);
        case B3SOIFD_MOD_WL:
            value->rValue = model->B3SOIFDWl;
            return(OK);
        case B3SOIFD_MOD_WLN:
            value->rValue = model->B3SOIFDWln;
            return(OK);
        case B3SOIFD_MOD_WW:
            value->rValue = model->B3SOIFDWw;
            return(OK);
        case B3SOIFD_MOD_WWN:
            value->rValue = model->B3SOIFDWwn;
            return(OK);
        case B3SOIFD_MOD_WWL:
            value->rValue = model->B3SOIFDWwl;
            return(OK);
        case B3SOIFD_MOD_NOIA:
            value->rValue = model->B3SOIFDoxideTrapDensityA;
            return(OK);
        case B3SOIFD_MOD_NOIB:
            value->rValue = model->B3SOIFDoxideTrapDensityB;
            return(OK);
        case B3SOIFD_MOD_NOIC:
            value->rValue = model->B3SOIFDoxideTrapDensityC;
            return(OK);
        case B3SOIFD_MOD_NOIF:
            value->rValue = model->B3SOIFDnoif;
            return(OK);
        case B3SOIFD_MOD_EM:
            value->rValue = model->B3SOIFDem;
            return(OK);
        case B3SOIFD_MOD_EF:
            value->rValue = model->B3SOIFDef;
            return(OK);
        case B3SOIFD_MOD_AF:
            value->rValue = model->B3SOIFDaf;
            return(OK);
        case B3SOIFD_MOD_KF:
            value->rValue = model->B3SOIFDkf;
            return(OK);

/* Added for binning - START */
        /* Length Dependence */
        case B3SOIFD_MOD_LNPEAK:
            value->rValue = model->B3SOIFDlnpeak;
            return(OK);
        case B3SOIFD_MOD_LNSUB:
            value->rValue = model->B3SOIFDlnsub;
            return(OK);
        case B3SOIFD_MOD_LNGATE:
            value->rValue = model->B3SOIFDlngate;
            return(OK);
        case B3SOIFD_MOD_LVTH0:
            value->rValue = model->B3SOIFDlvth0;
            return(OK);
        case  B3SOIFD_MOD_LK1:
          value->rValue = model->B3SOIFDlk1;
            return(OK);
        case  B3SOIFD_MOD_LK2:
          value->rValue = model->B3SOIFDlk2;
            return(OK);
        case  B3SOIFD_MOD_LK3:
          value->rValue = model->B3SOIFDlk3;
            return(OK);
        case  B3SOIFD_MOD_LK3B:
          value->rValue = model->B3SOIFDlk3b;
            return(OK);
        case  B3SOIFD_MOD_LVBSA:
          value->rValue = model->B3SOIFDlvbsa;
            return(OK);
        case  B3SOIFD_MOD_LDELP:
          value->rValue = model->B3SOIFDldelp;
            return(OK);
        case  B3SOIFD_MOD_LKB1:
            value->rValue = model->B3SOIFDlkb1;
            return(OK);
        case  B3SOIFD_MOD_LKB3:
            value->rValue = model->B3SOIFDlkb3;
            return(OK);
        case  B3SOIFD_MOD_LDVBD0:
            value->rValue = model->B3SOIFDdvbd0;
            return(OK);
        case  B3SOIFD_MOD_LDVBD1:
            value->rValue = model->B3SOIFDdvbd1;
            return(OK);
        case  B3SOIFD_MOD_LW0:
          value->rValue = model->B3SOIFDlw0;
            return(OK);
        case  B3SOIFD_MOD_LNLX:
          value->rValue = model->B3SOIFDlnlx;
            return(OK);
        case  B3SOIFD_MOD_LDVT0 :
          value->rValue = model->B3SOIFDldvt0;
            return(OK);
        case  B3SOIFD_MOD_LDVT1 :
          value->rValue = model->B3SOIFDldvt1;
            return(OK);
        case  B3SOIFD_MOD_LDVT2 :
          value->rValue = model->B3SOIFDldvt2;
            return(OK);
        case  B3SOIFD_MOD_LDVT0W :
          value->rValue = model->B3SOIFDldvt0w;
            return(OK);
        case  B3SOIFD_MOD_LDVT1W :
          value->rValue = model->B3SOIFDldvt1w;
            return(OK);
        case  B3SOIFD_MOD_LDVT2W :
          value->rValue = model->B3SOIFDldvt2w;
            return(OK);
        case B3SOIFD_MOD_LU0:
            value->rValue = model->B3SOIFDlu0;
            return(OK);
        case B3SOIFD_MOD_LUA:
            value->rValue = model->B3SOIFDlua;
            return(OK);
        case B3SOIFD_MOD_LUB:
            value->rValue = model->B3SOIFDlub;
            return(OK);
        case B3SOIFD_MOD_LUC:
            value->rValue = model->B3SOIFDluc;
            return(OK);
        case B3SOIFD_MOD_LVSAT:
            value->rValue = model->B3SOIFDlvsat;
            return(OK);
        case B3SOIFD_MOD_LA0:
            value->rValue = model->B3SOIFDla0;
            return(OK);
        case B3SOIFD_MOD_LAGS:
            value->rValue = model->B3SOIFDlags;
            return(OK);
        case B3SOIFD_MOD_LB0:
            value->rValue = model->B3SOIFDlb0;
            return(OK);
        case B3SOIFD_MOD_LB1:
            value->rValue = model->B3SOIFDlb1;
            return(OK);
        case B3SOIFD_MOD_LKETA:
            value->rValue = model->B3SOIFDlketa;
            return(OK);
        case B3SOIFD_MOD_LABP:
            value->rValue = model->B3SOIFDlabp;
            return(OK);
        case B3SOIFD_MOD_LMXC:
            value->rValue = model->B3SOIFDlmxc;
            return(OK);
        case B3SOIFD_MOD_LADICE0:
            value->rValue = model->B3SOIFDladice0;
            return(OK);
        case B3SOIFD_MOD_LA1:
            value->rValue = model->B3SOIFDla1;
            return(OK);
        case B3SOIFD_MOD_LA2:
            value->rValue = model->B3SOIFDla2;
            return(OK);
        case B3SOIFD_MOD_LRDSW:
            value->rValue = model->B3SOIFDlrdsw;
            return(OK);
        case B3SOIFD_MOD_LPRWB:
            value->rValue = model->B3SOIFDlprwb;
            return(OK);
        case B3SOIFD_MOD_LPRWG:
            value->rValue = model->B3SOIFDlprwg;
            return(OK);
        case B3SOIFD_MOD_LWR:
            value->rValue = model->B3SOIFDlwr;
            return(OK);
        case  B3SOIFD_MOD_LNFACTOR :
          value->rValue = model->B3SOIFDlnfactor;
            return(OK);
        case B3SOIFD_MOD_LDWG:
            value->rValue = model->B3SOIFDldwg;
            return(OK);
        case B3SOIFD_MOD_LDWB:
            value->rValue = model->B3SOIFDldwb;
            return(OK);
        case B3SOIFD_MOD_LVOFF:
            value->rValue = model->B3SOIFDlvoff;
            return(OK);
        case B3SOIFD_MOD_LETA0:
            value->rValue = model->B3SOIFDleta0;
            return(OK);
        case B3SOIFD_MOD_LETAB:
            value->rValue = model->B3SOIFDletab;
            return(OK);
        case  B3SOIFD_MOD_LDSUB :
          value->rValue = model->B3SOIFDldsub;
            return(OK);
        case  B3SOIFD_MOD_LCIT :
          value->rValue = model->B3SOIFDlcit;
            return(OK);
        case  B3SOIFD_MOD_LCDSC :
          value->rValue = model->B3SOIFDlcdsc;
            return(OK);
        case  B3SOIFD_MOD_LCDSCB :
          value->rValue = model->B3SOIFDlcdscb;
            return(OK);
        case  B3SOIFD_MOD_LCDSCD :
          value->rValue = model->B3SOIFDlcdscd;
            return(OK);
        case B3SOIFD_MOD_LPCLM:
            value->rValue = model->B3SOIFDlpclm;
            return(OK);
        case B3SOIFD_MOD_LPDIBL1:
            value->rValue = model->B3SOIFDlpdibl1;
            return(OK);
        case B3SOIFD_MOD_LPDIBL2:
            value->rValue = model->B3SOIFDlpdibl2;
            return(OK);
        case B3SOIFD_MOD_LPDIBLB:
            value->rValue = model->B3SOIFDlpdiblb;
            return(OK);
        case  B3SOIFD_MOD_LDROUT :
          value->rValue = model->B3SOIFDldrout;
            return(OK);
        case B3SOIFD_MOD_LPVAG:
            value->rValue = model->B3SOIFDlpvag;
            return(OK);
        case B3SOIFD_MOD_LDELTA:
            value->rValue = model->B3SOIFDldelta;
            return(OK);
        case B3SOIFD_MOD_LAII:
            value->rValue = model->B3SOIFDlaii;
            return(OK);
        case B3SOIFD_MOD_LBII:
            value->rValue = model->B3SOIFDlbii;
            return(OK);
        case B3SOIFD_MOD_LCII:
            value->rValue = model->B3SOIFDlcii;
            return(OK);
        case B3SOIFD_MOD_LDII:
            value->rValue = model->B3SOIFDldii;
            return(OK);
        case B3SOIFD_MOD_LALPHA0:
            value->rValue = model->B3SOIFDlalpha0;
            return(OK);
        case B3SOIFD_MOD_LALPHA1:
            value->rValue = model->B3SOIFDlalpha1;
            return(OK);
        case B3SOIFD_MOD_LBETA0:
            value->rValue = model->B3SOIFDlbeta0;
            return(OK);
        case B3SOIFD_MOD_LAGIDL:
            value->rValue = model->B3SOIFDlagidl;
            return(OK);
        case B3SOIFD_MOD_LBGIDL:
            value->rValue = model->B3SOIFDlbgidl;
            return(OK);
        case B3SOIFD_MOD_LNGIDL:
            value->rValue = model->B3SOIFDlngidl;
            return(OK);
        case B3SOIFD_MOD_LNTUN:
            value->rValue = model->B3SOIFDlntun;
            return(OK);
        case B3SOIFD_MOD_LNDIODE:
            value->rValue = model->B3SOIFDlndiode;
            return(OK);
        case B3SOIFD_MOD_LISBJT:
            value->rValue = model->B3SOIFDlisbjt;
            return(OK);
        case B3SOIFD_MOD_LISDIF:
            value->rValue = model->B3SOIFDlisdif;
            return(OK);
        case B3SOIFD_MOD_LISREC:
            value->rValue = model->B3SOIFDlisrec;
            return(OK);
        case B3SOIFD_MOD_LISTUN:
            value->rValue = model->B3SOIFDlistun;
            return(OK);
        case B3SOIFD_MOD_LEDL:
            value->rValue = model->B3SOIFDledl;
            return(OK);
        case B3SOIFD_MOD_LKBJT1:
            value->rValue = model->B3SOIFDlkbjt1;
            return(OK);
	/* CV Model */
        case B3SOIFD_MOD_LVSDFB:
            value->rValue = model->B3SOIFDlvsdfb;
            return(OK);
        case B3SOIFD_MOD_LVSDTH:
            value->rValue = model->B3SOIFDlvsdth;
            return(OK);
        /* Width Dependence */
        case B3SOIFD_MOD_WNPEAK:
            value->rValue = model->B3SOIFDwnpeak;
            return(OK);
        case B3SOIFD_MOD_WNSUB:
            value->rValue = model->B3SOIFDwnsub;
            return(OK);
        case B3SOIFD_MOD_WNGATE:
            value->rValue = model->B3SOIFDwngate;
            return(OK);
        case B3SOIFD_MOD_WVTH0:
            value->rValue = model->B3SOIFDwvth0;
            return(OK);
        case  B3SOIFD_MOD_WK1:
          value->rValue = model->B3SOIFDwk1;
            return(OK);
        case  B3SOIFD_MOD_WK2:
          value->rValue = model->B3SOIFDwk2;
            return(OK);
        case  B3SOIFD_MOD_WK3:
          value->rValue = model->B3SOIFDwk3;
            return(OK);
        case  B3SOIFD_MOD_WK3B:
          value->rValue = model->B3SOIFDwk3b;
            return(OK);
        case  B3SOIFD_MOD_WVBSA:
          value->rValue = model->B3SOIFDwvbsa;
            return(OK);
        case  B3SOIFD_MOD_WDELP:
          value->rValue = model->B3SOIFDwdelp;
            return(OK);
        case  B3SOIFD_MOD_WKB1:
            value->rValue = model->B3SOIFDwkb1;
            return(OK);
        case  B3SOIFD_MOD_WKB3:
            value->rValue = model->B3SOIFDwkb3;
            return(OK);
        case  B3SOIFD_MOD_WDVBD0:
            value->rValue = model->B3SOIFDdvbd0;
            return(OK);
        case  B3SOIFD_MOD_WDVBD1:
            value->rValue = model->B3SOIFDdvbd1;
            return(OK);
        case  B3SOIFD_MOD_WW0:
          value->rValue = model->B3SOIFDww0;
            return(OK);
        case  B3SOIFD_MOD_WNLX:
          value->rValue = model->B3SOIFDwnlx;
            return(OK);
        case  B3SOIFD_MOD_WDVT0 :
          value->rValue = model->B3SOIFDwdvt0;
            return(OK);
        case  B3SOIFD_MOD_WDVT1 :
          value->rValue = model->B3SOIFDwdvt1;
            return(OK);
        case  B3SOIFD_MOD_WDVT2 :
          value->rValue = model->B3SOIFDwdvt2;
            return(OK);
        case  B3SOIFD_MOD_WDVT0W :
          value->rValue = model->B3SOIFDwdvt0w;
            return(OK);
        case  B3SOIFD_MOD_WDVT1W :
          value->rValue = model->B3SOIFDwdvt1w;
            return(OK);
        case  B3SOIFD_MOD_WDVT2W :
          value->rValue = model->B3SOIFDwdvt2w;
            return(OK);
        case B3SOIFD_MOD_WU0:
            value->rValue = model->B3SOIFDwu0;
            return(OK);
        case B3SOIFD_MOD_WUA:
            value->rValue = model->B3SOIFDwua;
            return(OK);
        case B3SOIFD_MOD_WUB:
            value->rValue = model->B3SOIFDwub;
            return(OK);
        case B3SOIFD_MOD_WUC:
            value->rValue = model->B3SOIFDwuc;
            return(OK);
        case B3SOIFD_MOD_WVSAT:
            value->rValue = model->B3SOIFDwvsat;
            return(OK);
        case B3SOIFD_MOD_WA0:
            value->rValue = model->B3SOIFDwa0;
            return(OK);
        case B3SOIFD_MOD_WAGS:
            value->rValue = model->B3SOIFDwags;
            return(OK);
        case B3SOIFD_MOD_WB0:
            value->rValue = model->B3SOIFDwb0;
            return(OK);
        case B3SOIFD_MOD_WB1:
            value->rValue = model->B3SOIFDwb1;
            return(OK);
        case B3SOIFD_MOD_WKETA:
            value->rValue = model->B3SOIFDwketa;
            return(OK);
        case B3SOIFD_MOD_WABP:
            value->rValue = model->B3SOIFDwabp;
            return(OK);
        case B3SOIFD_MOD_WMXC:
            value->rValue = model->B3SOIFDwmxc;
            return(OK);
        case B3SOIFD_MOD_WADICE0:
            value->rValue = model->B3SOIFDwadice0;
            return(OK);
        case B3SOIFD_MOD_WA1:
            value->rValue = model->B3SOIFDwa1;
            return(OK);
        case B3SOIFD_MOD_WA2:
            value->rValue = model->B3SOIFDwa2;
            return(OK);
        case B3SOIFD_MOD_WRDSW:
            value->rValue = model->B3SOIFDwrdsw;
            return(OK);
        case B3SOIFD_MOD_WPRWB:
            value->rValue = model->B3SOIFDwprwb;
            return(OK);
        case B3SOIFD_MOD_WPRWG:
            value->rValue = model->B3SOIFDwprwg;
            return(OK);
        case B3SOIFD_MOD_WWR:
            value->rValue = model->B3SOIFDwwr;
            return(OK);
        case  B3SOIFD_MOD_WNFACTOR :
          value->rValue = model->B3SOIFDwnfactor;
            return(OK);
        case B3SOIFD_MOD_WDWG:
            value->rValue = model->B3SOIFDwdwg;
            return(OK);
        case B3SOIFD_MOD_WDWB:
            value->rValue = model->B3SOIFDwdwb;
            return(OK);
        case B3SOIFD_MOD_WVOFF:
            value->rValue = model->B3SOIFDwvoff;
            return(OK);
        case B3SOIFD_MOD_WETA0:
            value->rValue = model->B3SOIFDweta0;
            return(OK);
        case B3SOIFD_MOD_WETAB:
            value->rValue = model->B3SOIFDwetab;
            return(OK);
        case  B3SOIFD_MOD_WDSUB :
          value->rValue = model->B3SOIFDwdsub;
            return(OK);
        case  B3SOIFD_MOD_WCIT :
          value->rValue = model->B3SOIFDwcit;
            return(OK);
        case  B3SOIFD_MOD_WCDSC :
          value->rValue = model->B3SOIFDwcdsc;
            return(OK);
        case  B3SOIFD_MOD_WCDSCB :
          value->rValue = model->B3SOIFDwcdscb;
            return(OK);
        case  B3SOIFD_MOD_WCDSCD :
          value->rValue = model->B3SOIFDwcdscd;
            return(OK);
        case B3SOIFD_MOD_WPCLM:
            value->rValue = model->B3SOIFDwpclm;
            return(OK);
        case B3SOIFD_MOD_WPDIBL1:
            value->rValue = model->B3SOIFDwpdibl1;
            return(OK);
        case B3SOIFD_MOD_WPDIBL2:
            value->rValue = model->B3SOIFDwpdibl2;
            return(OK);
        case B3SOIFD_MOD_WPDIBLB:
            value->rValue = model->B3SOIFDwpdiblb;
            return(OK);
        case  B3SOIFD_MOD_WDROUT :
          value->rValue = model->B3SOIFDwdrout;
            return(OK);
        case B3SOIFD_MOD_WPVAG:
            value->rValue = model->B3SOIFDwpvag;
            return(OK);
        case B3SOIFD_MOD_WDELTA:
            value->rValue = model->B3SOIFDwdelta;
            return(OK);
        case B3SOIFD_MOD_WAII:
            value->rValue = model->B3SOIFDwaii;
            return(OK);
        case B3SOIFD_MOD_WBII:
            value->rValue = model->B3SOIFDwbii;
            return(OK);
        case B3SOIFD_MOD_WCII:
            value->rValue = model->B3SOIFDwcii;
            return(OK);
        case B3SOIFD_MOD_WDII:
            value->rValue = model->B3SOIFDwdii;
            return(OK);
        case B3SOIFD_MOD_WALPHA0:
            value->rValue = model->B3SOIFDwalpha0;
            return(OK);
        case B3SOIFD_MOD_WALPHA1:
            value->rValue = model->B3SOIFDwalpha1;
            return(OK);
        case B3SOIFD_MOD_WBETA0:
            value->rValue = model->B3SOIFDwbeta0;
            return(OK);
        case B3SOIFD_MOD_WAGIDL:
            value->rValue = model->B3SOIFDwagidl;
            return(OK);
        case B3SOIFD_MOD_WBGIDL:
            value->rValue = model->B3SOIFDwbgidl;
            return(OK);
        case B3SOIFD_MOD_WNGIDL:
            value->rValue = model->B3SOIFDwngidl;
            return(OK);
        case B3SOIFD_MOD_WNTUN:
            value->rValue = model->B3SOIFDwntun;
            return(OK);
        case B3SOIFD_MOD_WNDIODE:
            value->rValue = model->B3SOIFDwndiode;
            return(OK);
        case B3SOIFD_MOD_WISBJT:
            value->rValue = model->B3SOIFDwisbjt;
            return(OK);
        case B3SOIFD_MOD_WISDIF:
            value->rValue = model->B3SOIFDwisdif;
            return(OK);
        case B3SOIFD_MOD_WISREC:
            value->rValue = model->B3SOIFDwisrec;
            return(OK);
        case B3SOIFD_MOD_WISTUN:
            value->rValue = model->B3SOIFDwistun;
            return(OK);
        case B3SOIFD_MOD_WEDL:
            value->rValue = model->B3SOIFDwedl;
            return(OK);
        case B3SOIFD_MOD_WKBJT1:
            value->rValue = model->B3SOIFDwkbjt1;
            return(OK);
	/* CV Model */
        case B3SOIFD_MOD_WVSDFB:
            value->rValue = model->B3SOIFDwvsdfb;
            return(OK);
        case B3SOIFD_MOD_WVSDTH:
            value->rValue = model->B3SOIFDwvsdth;
            return(OK);
        /* Cross-term Dependence */
        case B3SOIFD_MOD_PNPEAK:
            value->rValue = model->B3SOIFDpnpeak;
            return(OK);
        case B3SOIFD_MOD_PNSUB:
            value->rValue = model->B3SOIFDpnsub;
            return(OK);
        case B3SOIFD_MOD_PNGATE:
            value->rValue = model->B3SOIFDpngate;
            return(OK);
        case B3SOIFD_MOD_PVTH0:
            value->rValue = model->B3SOIFDpvth0;
            return(OK);
        case  B3SOIFD_MOD_PK1:
          value->rValue = model->B3SOIFDpk1;
            return(OK);
        case  B3SOIFD_MOD_PK2:
          value->rValue = model->B3SOIFDpk2;
            return(OK);
        case  B3SOIFD_MOD_PK3:
          value->rValue = model->B3SOIFDpk3;
            return(OK);
        case  B3SOIFD_MOD_PK3B:
          value->rValue = model->B3SOIFDpk3b;
            return(OK);
        case  B3SOIFD_MOD_PVBSA:
          value->rValue = model->B3SOIFDpvbsa;
            return(OK);
        case  B3SOIFD_MOD_PDELP:
          value->rValue = model->B3SOIFDpdelp;
            return(OK);
        case  B3SOIFD_MOD_PKB1:
            value->rValue = model->B3SOIFDpkb1;
            return(OK);
        case  B3SOIFD_MOD_PKB3:
            value->rValue = model->B3SOIFDpkb3;
            return(OK);
        case  B3SOIFD_MOD_PDVBD0:
            value->rValue = model->B3SOIFDdvbd0;
            return(OK);
        case  B3SOIFD_MOD_PDVBD1:
            value->rValue = model->B3SOIFDdvbd1;
            return(OK);
        case  B3SOIFD_MOD_PW0:
          value->rValue = model->B3SOIFDpw0;
            return(OK);
        case  B3SOIFD_MOD_PNLX:
          value->rValue = model->B3SOIFDpnlx;
            return(OK);
        case  B3SOIFD_MOD_PDVT0 :
          value->rValue = model->B3SOIFDpdvt0;
            return(OK);
        case  B3SOIFD_MOD_PDVT1 :
          value->rValue = model->B3SOIFDpdvt1;
            return(OK);
        case  B3SOIFD_MOD_PDVT2 :
          value->rValue = model->B3SOIFDpdvt2;
            return(OK);
        case  B3SOIFD_MOD_PDVT0W :
          value->rValue = model->B3SOIFDpdvt0w;
            return(OK);
        case  B3SOIFD_MOD_PDVT1W :
          value->rValue = model->B3SOIFDpdvt1w;
            return(OK);
        case  B3SOIFD_MOD_PDVT2W :
          value->rValue = model->B3SOIFDpdvt2w;
            return(OK);
        case B3SOIFD_MOD_PU0:
            value->rValue = model->B3SOIFDpu0;
            return(OK);
        case B3SOIFD_MOD_PUA:
            value->rValue = model->B3SOIFDpua;
            return(OK);
        case B3SOIFD_MOD_PUB:
            value->rValue = model->B3SOIFDpub;
            return(OK);
        case B3SOIFD_MOD_PUC:
            value->rValue = model->B3SOIFDpuc;
            return(OK);
        case B3SOIFD_MOD_PVSAT:
            value->rValue = model->B3SOIFDpvsat;
            return(OK);
        case B3SOIFD_MOD_PA0:
            value->rValue = model->B3SOIFDpa0;
            return(OK);
        case B3SOIFD_MOD_PAGS:
            value->rValue = model->B3SOIFDpags;
            return(OK);
        case B3SOIFD_MOD_PB0:
            value->rValue = model->B3SOIFDpb0;
            return(OK);
        case B3SOIFD_MOD_PB1:
            value->rValue = model->B3SOIFDpb1;
            return(OK);
        case B3SOIFD_MOD_PKETA:
            value->rValue = model->B3SOIFDpketa;
            return(OK);
        case B3SOIFD_MOD_PABP:
            value->rValue = model->B3SOIFDpabp;
            return(OK);
        case B3SOIFD_MOD_PMXC:
            value->rValue = model->B3SOIFDpmxc;
            return(OK);
        case B3SOIFD_MOD_PADICE0:
            value->rValue = model->B3SOIFDpadice0;
            return(OK);
        case B3SOIFD_MOD_PA1:
            value->rValue = model->B3SOIFDpa1;
            return(OK);
        case B3SOIFD_MOD_PA2:
            value->rValue = model->B3SOIFDpa2;
            return(OK);
        case B3SOIFD_MOD_PRDSW:
            value->rValue = model->B3SOIFDprdsw;
            return(OK);
        case B3SOIFD_MOD_PPRWB:
            value->rValue = model->B3SOIFDpprwb;
            return(OK);
        case B3SOIFD_MOD_PPRWG:
            value->rValue = model->B3SOIFDpprwg;
            return(OK);
        case B3SOIFD_MOD_PWR:
            value->rValue = model->B3SOIFDpwr;
            return(OK);
        case  B3SOIFD_MOD_PNFACTOR :
          value->rValue = model->B3SOIFDpnfactor;
            return(OK);
        case B3SOIFD_MOD_PDWG:
            value->rValue = model->B3SOIFDpdwg;
            return(OK);
        case B3SOIFD_MOD_PDWB:
            value->rValue = model->B3SOIFDpdwb;
            return(OK);
        case B3SOIFD_MOD_PVOFF:
            value->rValue = model->B3SOIFDpvoff;
            return(OK);
        case B3SOIFD_MOD_PETA0:
            value->rValue = model->B3SOIFDpeta0;
            return(OK);
        case B3SOIFD_MOD_PETAB:
            value->rValue = model->B3SOIFDpetab;
            return(OK);
        case  B3SOIFD_MOD_PDSUB :
          value->rValue = model->B3SOIFDpdsub;
            return(OK);
        case  B3SOIFD_MOD_PCIT :
          value->rValue = model->B3SOIFDpcit;
            return(OK);
        case  B3SOIFD_MOD_PCDSC :
          value->rValue = model->B3SOIFDpcdsc;
            return(OK);
        case  B3SOIFD_MOD_PCDSCB :
          value->rValue = model->B3SOIFDpcdscb;
            return(OK);
        case  B3SOIFD_MOD_PCDSCD :
          value->rValue = model->B3SOIFDpcdscd;
            return(OK);
        case B3SOIFD_MOD_PPCLM:
            value->rValue = model->B3SOIFDppclm;
            return(OK);
        case B3SOIFD_MOD_PPDIBL1:
            value->rValue = model->B3SOIFDppdibl1;
            return(OK);
        case B3SOIFD_MOD_PPDIBL2:
            value->rValue = model->B3SOIFDppdibl2;
            return(OK);
        case B3SOIFD_MOD_PPDIBLB:
            value->rValue = model->B3SOIFDppdiblb;
            return(OK);
        case  B3SOIFD_MOD_PDROUT :
          value->rValue = model->B3SOIFDpdrout;
            return(OK);
        case B3SOIFD_MOD_PPVAG:
            value->rValue = model->B3SOIFDppvag;
            return(OK);
        case B3SOIFD_MOD_PDELTA:
            value->rValue = model->B3SOIFDpdelta;
            return(OK);
        case B3SOIFD_MOD_PAII:
            value->rValue = model->B3SOIFDpaii;
            return(OK);
        case B3SOIFD_MOD_PBII:
            value->rValue = model->B3SOIFDpbii;
            return(OK);
        case B3SOIFD_MOD_PCII:
            value->rValue = model->B3SOIFDpcii;
            return(OK);
        case B3SOIFD_MOD_PDII:
            value->rValue = model->B3SOIFDpdii;
            return(OK);
        case B3SOIFD_MOD_PALPHA0:
            value->rValue = model->B3SOIFDpalpha0;
            return(OK);
        case B3SOIFD_MOD_PALPHA1:
            value->rValue = model->B3SOIFDpalpha1;
            return(OK);
        case B3SOIFD_MOD_PBETA0:
            value->rValue = model->B3SOIFDpbeta0;
            return(OK);
        case B3SOIFD_MOD_PAGIDL:
            value->rValue = model->B3SOIFDpagidl;
            return(OK);
        case B3SOIFD_MOD_PBGIDL:
            value->rValue = model->B3SOIFDpbgidl;
            return(OK);
        case B3SOIFD_MOD_PNGIDL:
            value->rValue = model->B3SOIFDpngidl;
            return(OK);
        case B3SOIFD_MOD_PNTUN:
            value->rValue = model->B3SOIFDpntun;
            return(OK);
        case B3SOIFD_MOD_PNDIODE:
            value->rValue = model->B3SOIFDpndiode;
            return(OK);
        case B3SOIFD_MOD_PISBJT:
            value->rValue = model->B3SOIFDpisbjt;
            return(OK);
        case B3SOIFD_MOD_PISDIF:
            value->rValue = model->B3SOIFDpisdif;
            return(OK);
        case B3SOIFD_MOD_PISREC:
            value->rValue = model->B3SOIFDpisrec;
            return(OK);
        case B3SOIFD_MOD_PISTUN:
            value->rValue = model->B3SOIFDpistun;
            return(OK);
        case B3SOIFD_MOD_PEDL:
            value->rValue = model->B3SOIFDpedl;
            return(OK);
        case B3SOIFD_MOD_PKBJT1:
            value->rValue = model->B3SOIFDpkbjt1;
            return(OK);
	/* CV Model */
        case B3SOIFD_MOD_PVSDFB:
            value->rValue = model->B3SOIFDpvsdfb;
            return(OK);
        case B3SOIFD_MOD_PVSDTH:
            value->rValue = model->B3SOIFDpvsdth;
            return(OK);
/* Added for binning - END */

        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}



