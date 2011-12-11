/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
Modified by Paolo Nenzi 2002
File: b3soifdcheck.c          98/5/01
**********/

/*
 * Revision 2.1  99/9/27 Pin Su 
 * BSIMFD2.1 release
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "b3soifddef.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/devdefs.h"
#include "ngspice/suffix.h"

int
B3SOIFDcheckModel(B3SOIFDmodel *model, B3SOIFDinstance *here, CKTcircuit *ckt)
{
struct b3soifdSizeDependParam *pParam;
int Fatal_Flag = 0;
FILE *fplog;

    NG_IGNORE(ckt);
    
    if ((fplog = fopen("b3soifdv2check.log", "w")) != NULL)
    {   pParam = here->pParam;
	fprintf(fplog, "B3SOI (FD) Version 2.1 Parameter Check\n");
	fprintf(fplog, "Model = %s\n", model->B3SOIFDmodName);
	fprintf(fplog, "W = %g, L = %g M = %g\n", here->B3SOIFDw, 
	        here->B3SOIFDl, here->B3SOIFDm);
   	    

            if (pParam->B3SOIFDnlx < -pParam->B3SOIFDleff)
	    {   fprintf(fplog, "Fatal: Nlx = %g is less than -Leff.\n",
			pParam->B3SOIFDnlx);
	        printf("Fatal: Nlx = %g is less than -Leff.\n",
			pParam->B3SOIFDnlx);
		Fatal_Flag = 1;
            }

	if (model->B3SOIFDtox <= 0.0)
	{   fprintf(fplog, "Fatal: Tox = %g is not positive.\n",
		    model->B3SOIFDtox);
	    printf("Fatal: Tox = %g is not positive.\n", model->B3SOIFDtox);
	    Fatal_Flag = 1;
	}

	if (model->B3SOIFDtbox <= 0.0)
	{   fprintf(fplog, "Fatal: Tbox = %g is not positive.\n",
		    model->B3SOIFDtbox);
	    printf("Fatal: Tbox = %g is not positive.\n", model->B3SOIFDtbox);
	    Fatal_Flag = 1;
	}

	if (pParam->B3SOIFDnpeak <= 0.0)
	{   fprintf(fplog, "Fatal: Nch = %g is not positive.\n",
		    pParam->B3SOIFDnpeak);
	    printf("Fatal: Nch = %g is not positive.\n",
		   pParam->B3SOIFDnpeak);
	    Fatal_Flag = 1;
	}
	if (pParam->B3SOIFDngate < 0.0)
	{   fprintf(fplog, "Fatal: Ngate = %g is not positive.\n",
		    pParam->B3SOIFDngate);
	    printf("Fatal: Ngate = %g Ngate is not positive.\n",
		   pParam->B3SOIFDngate);
	    Fatal_Flag = 1;
	}
	if (pParam->B3SOIFDngate > 1.e25)
	{   fprintf(fplog, "Fatal: Ngate = %g is too high.\n",
		    pParam->B3SOIFDngate);
	    printf("Fatal: Ngate = %g Ngate is too high\n",
		   pParam->B3SOIFDngate);
	    Fatal_Flag = 1;
	}

	if (model->B3SOIFDdvbd1 < 0.0)
	{   fprintf(fplog, "Fatal: Dvbd1 = %g is negative.\n",
		    model->B3SOIFDdvbd1);   
	    printf("Fatal: Dvbd1 = %g is negative.\n", model->B3SOIFDdvbd1);   
	    Fatal_Flag = 1;
	}

	if (pParam->B3SOIFDdvt1 < 0.0)
	{   fprintf(fplog, "Fatal: Dvt1 = %g is negative.\n",
		    pParam->B3SOIFDdvt1);   
	    printf("Fatal: Dvt1 = %g is negative.\n", pParam->B3SOIFDdvt1);   
	    Fatal_Flag = 1;
	}
	    
	if (pParam->B3SOIFDdvt1w < 0.0)
	{   fprintf(fplog, "Fatal: Dvt1w = %g is negative.\n",
		    pParam->B3SOIFDdvt1w);
	    printf("Fatal: Dvt1w = %g is negative.\n", pParam->B3SOIFDdvt1w);
	    Fatal_Flag = 1;
	}
	    
	if (pParam->B3SOIFDw0 == -pParam->B3SOIFDweff)
	{   fprintf(fplog, "Fatal: (W0 + Weff) = 0 cauing divided-by-zero.\n");
	    printf("Fatal: (W0 + Weff) = 0 cauing divided-by-zero.\n");
	    Fatal_Flag = 1;
        }   

	if (pParam->B3SOIFDdsub < 0.0)
	{   fprintf(fplog, "Fatal: Dsub = %g is negative.\n", pParam->B3SOIFDdsub);
	    printf("Fatal: Dsub = %g is negative.\n", pParam->B3SOIFDdsub);
	    Fatal_Flag = 1;
	}
	if (pParam->B3SOIFDb1 == -pParam->B3SOIFDweff)
	{   fprintf(fplog, "Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n");
	    printf("Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n");
	    Fatal_Flag = 1;
        }  
        if (pParam->B3SOIFDu0temp <= 0.0)
	{   fprintf(fplog, "Fatal: u0 at current temperature = %g is not positive.\n", pParam->B3SOIFDu0temp);
	    printf("Fatal: u0 at current temperature = %g is not positive.\n",
		   pParam->B3SOIFDu0temp);
	    Fatal_Flag = 1;
        }
    
/* Check delta parameter */      
        if (pParam->B3SOIFDdelta < 0.0)
	{   fprintf(fplog, "Fatal: Delta = %g is less than zero.\n",
		    pParam->B3SOIFDdelta);
	    printf("Fatal: Delta = %g is less than zero.\n", pParam->B3SOIFDdelta);
	    Fatal_Flag = 1;
        }      

	if (pParam->B3SOIFDvsattemp <= 0.0)
	{   fprintf(fplog, "Fatal: Vsat at current temperature = %g is not positive.\n", pParam->B3SOIFDvsattemp);
	    printf("Fatal: Vsat at current temperature = %g is not positive.\n",
		   pParam->B3SOIFDvsattemp);
	    Fatal_Flag = 1;
	}
/* Check Rout parameters */
	if (pParam->B3SOIFDpclm <= 0.0)
	{   fprintf(fplog, "Fatal: Pclm = %g is not positive.\n", pParam->B3SOIFDpclm);
	    printf("Fatal: Pclm = %g is not positive.\n", pParam->B3SOIFDpclm);
	    Fatal_Flag = 1;
	}

	if (pParam->B3SOIFDdrout < 0.0)
	{   fprintf(fplog, "Fatal: Drout = %g is negative.\n", pParam->B3SOIFDdrout);
	    printf("Fatal: Drout = %g is negative.\n", pParam->B3SOIFDdrout);
	    Fatal_Flag = 1;
	}
      if ( model->B3SOIFDunitLengthGateSidewallJctCap > 0.0)
      {
	if (here->B3SOIFDdrainPerimeter < pParam->B3SOIFDweff)
	{   fprintf(fplog, "Warning: Pd = %g is less than W.\n",
		    here->B3SOIFDdrainPerimeter);
	   printf("Warning: Pd = %g is less than W.\n",
		    here->B3SOIFDdrainPerimeter);
            here->B3SOIFDdrainPerimeter =pParam->B3SOIFDweff; 
	}
	if (here->B3SOIFDsourcePerimeter < pParam->B3SOIFDweff)
	{   fprintf(fplog, "Warning: Ps = %g is less than W.\n",
		    here->B3SOIFDsourcePerimeter);
	   printf("Warning: Ps = %g is less than W.\n",
		    here->B3SOIFDsourcePerimeter);
            here->B3SOIFDsourcePerimeter =pParam->B3SOIFDweff;
	}
      }
/* Check capacitance parameters */
        if (pParam->B3SOIFDclc < 0.0)
	{   fprintf(fplog, "Fatal: Clc = %g is negative.\n", pParam->B3SOIFDclc);
	    printf("Fatal: Clc = %g is negative.\n", pParam->B3SOIFDclc);
	    Fatal_Flag = 1;
        }      
      if (model->B3SOIFDparamChk ==1)
      {
/* Check L and W parameters */ 
	if (pParam->B3SOIFDleff <= 5.0e-8)
	{   fprintf(fplog, "Warning: Leff = %g may be too small.\n",
	            pParam->B3SOIFDleff);
	    printf("Warning: Leff = %g may be too small.\n",
		    pParam->B3SOIFDleff);
	}    
	
	if (pParam->B3SOIFDleffCV <= 5.0e-8)
	{   fprintf(fplog, "Warning: Leff for CV = %g may be too small.\n",
		    pParam->B3SOIFDleffCV);
	    printf("Warning: Leff for CV = %g may be too small.\n",
		   pParam->B3SOIFDleffCV);
	}  
	
        if (pParam->B3SOIFDweff <= 1.0e-7)
	{   fprintf(fplog, "Warning: Weff = %g may be too small.\n",
		    pParam->B3SOIFDweff);
	    printf("Warning: Weff = %g may be too small.\n",
		   pParam->B3SOIFDweff);
	}             
	
	if (pParam->B3SOIFDweffCV <= 1.0e-7)
	{   fprintf(fplog, "Warning: Weff for CV = %g may be too small.\n",
		    pParam->B3SOIFDweffCV);
	    printf("Warning: Weff for CV = %g may be too small.\n",
		   pParam->B3SOIFDweffCV);
	}        
	
/* Check threshold voltage parameters */
	if (pParam->B3SOIFDnlx < 0.0)
	{   fprintf(fplog, "Warning: Nlx = %g is negative.\n", pParam->B3SOIFDnlx);
	   printf("Warning: Nlx = %g is negative.\n", pParam->B3SOIFDnlx);
        }
	if (model->B3SOIFDtox < 1.0e-9)
	{   fprintf(fplog, "Warning: Tox = %g is less than 10A.\n",
	            model->B3SOIFDtox);
	    printf("Warning: Tox = %g is less than 10A.\n", model->B3SOIFDtox);
        }

        if (pParam->B3SOIFDnpeak <= 1.0e15)
	{   fprintf(fplog, "Warning: Nch = %g may be too small.\n",
	            pParam->B3SOIFDnpeak);
	    printf("Warning: Nch = %g may be too small.\n",
	           pParam->B3SOIFDnpeak);
	}
	else if (pParam->B3SOIFDnpeak >= 1.0e21)
	{   fprintf(fplog, "Warning: Nch = %g may be too large.\n",
	            pParam->B3SOIFDnpeak);
	    printf("Warning: Nch = %g may be too large.\n",
	           pParam->B3SOIFDnpeak);
	}

	if (fabs(pParam->B3SOIFDnsub) >= 1.0e21)
	{   fprintf(fplog, "Warning: Nsub = %g may be too large.\n",
	            pParam->B3SOIFDnsub);
	    printf("Warning: Nsub = %g may be too large.\n",
	           pParam->B3SOIFDnsub);
	}

	if ((pParam->B3SOIFDngate > 0.0) &&
	    (pParam->B3SOIFDngate <= 1.e18))
	{   fprintf(fplog, "Warning: Ngate = %g is less than 1.E18cm^-3.\n",
	            pParam->B3SOIFDngate);
	    printf("Warning: Ngate = %g is less than 1.E18cm^-3.\n",
	           pParam->B3SOIFDngate);
	}

        if (model->B3SOIFDdvbd0 < 0.0)
	{   fprintf(fplog, "Warning: Dvbd0 = %g is negative.\n",
		    model->B3SOIFDdvbd0);   
	    printf("Warning: Dvbd0 = %g is negative.\n", model->B3SOIFDdvbd0);   
	}
       
        if (pParam->B3SOIFDdvt0 < 0.0)
	{   fprintf(fplog, "Warning: Dvt0 = %g is negative.\n",
		    pParam->B3SOIFDdvt0);   
	    printf("Warning: Dvt0 = %g is negative.\n", pParam->B3SOIFDdvt0);   
	}
	    
	if (fabs(1.0e-6 / (pParam->B3SOIFDw0 + pParam->B3SOIFDweff)) > 10.0)
	{   fprintf(fplog, "Warning: (W0 + Weff) may be too small.\n");
	    printf("Warning: (W0 + Weff) may be too small.\n");
        }

/* Check subthreshold parameters */
	if (pParam->B3SOIFDnfactor < 0.0)
	{   fprintf(fplog, "Warning: Nfactor = %g is negative.\n",
		    pParam->B3SOIFDnfactor);
	    printf("Warning: Nfactor = %g is negative.\n", pParam->B3SOIFDnfactor);
	}
	if (model->B3SOIFDkb3 < 0.0)
	{   fprintf(fplog, "Warning: Kb3 = %g is negative.\n",
		    model->B3SOIFDkb3);
	    printf("Warning: Kb3 = %g is negative.\n", model->B3SOIFDkb3);
	}

	if (pParam->B3SOIFDcdsc < 0.0)
	{   fprintf(fplog, "Warning: Cdsc = %g is negative.\n",
		    pParam->B3SOIFDcdsc);
	    printf("Warning: Cdsc = %g is negative.\n", pParam->B3SOIFDcdsc);
	}
	if (pParam->B3SOIFDcdscd < 0.0)
	{   fprintf(fplog, "Warning: Cdscd = %g is negative.\n",
		    pParam->B3SOIFDcdscd);
	    printf("Warning: Cdscd = %g is negative.\n", pParam->B3SOIFDcdscd);
	}
/* Check DIBL parameters */
	if (pParam->B3SOIFDeta0 < 0.0)
	{   fprintf(fplog, "Warning: Eta0 = %g is negative.\n",
		    pParam->B3SOIFDeta0); 
	    printf("Warning: Eta0 = %g is negative.\n", pParam->B3SOIFDeta0); 
	}
	      
/* Check Abulk parameters */	    
        if (fabs(1.0e-6 / (pParam->B3SOIFDb1 + pParam->B3SOIFDweff)) > 10.0)
       	{   fprintf(fplog, "Warning: (B1 + Weff) may be too small.\n");
       	    printf("Warning: (B1 + Weff) may be too small.\n");
        }    

	if (model->B3SOIFDadice0 > 1.0)
	{   fprintf(fplog, "Warning: Adice0 = %g should be smaller than 1.\n",
		    model->B3SOIFDadice0); 
	    printf("Warning: Adice0 = %g should be smaller than 1.\n", model->B3SOIFDadice0); 
	}

	if (model->B3SOIFDabp < 0.2)
	{   fprintf(fplog, "Warning: Abp = %g is too small.\n",
		    model->B3SOIFDabp); 
	    printf("Warning: Abp = %g is too small.\n", model->B3SOIFDabp); 
	}
    
	if ((model->B3SOIFDmxc < -1.0) || (model->B3SOIFDmxc > 1.0))
	{   fprintf(fplog, "Warning: Mxc = %g should be within (-1, 1).\n",
		    model->B3SOIFDmxc); 
	    printf("Warning: Mxc = %g should be within (-1, 1).\n", model->B3SOIFDmxc); 
	}

/* Check Saturation parameters */
     	if (pParam->B3SOIFDa2 < 0.01)
	{   fprintf(fplog, "Warning: A2 = %g is too small. Set to 0.01.\n", pParam->B3SOIFDa2);
	    printf("Warning: A2 = %g is too small. Set to 0.01.\n",
		   pParam->B3SOIFDa2);
	    pParam->B3SOIFDa2 = 0.01;
	}
	else if (pParam->B3SOIFDa2 > 1.0)
	{   fprintf(fplog, "Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
		    pParam->B3SOIFDa2);
	    printf("Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
		   pParam->B3SOIFDa2);
	    pParam->B3SOIFDa2 = 1.0;
	    pParam->B3SOIFDa1 = 0.0;

	}

	if (pParam->B3SOIFDrdsw < 0.0)
	{   fprintf(fplog, "Warning: Rdsw = %g is negative. Set to zero.\n",
		    pParam->B3SOIFDrdsw);
	    printf("Warning: Rdsw = %g is negative. Set to zero.\n",
		   pParam->B3SOIFDrdsw);
	    pParam->B3SOIFDrdsw = 0.0;
	    pParam->B3SOIFDrds0 = 0.0;
	}
	else if ((pParam->B3SOIFDrds0 > 0.0) && (pParam->B3SOIFDrds0 < 0.001))
	{   fprintf(fplog, "Warning: Rds at current temperature = %g is less than 0.001 ohm. Set to zero.\n",
		    pParam->B3SOIFDrds0);
	    printf("Warning: Rds at current temperature = %g is less than 0.001 ohm. Set to zero.\n",
		   pParam->B3SOIFDrds0);
	    pParam->B3SOIFDrds0 = 0.0;
	}
	 if (pParam->B3SOIFDvsattemp < 1.0e3)
	{   fprintf(fplog, "Warning: Vsat at current temperature = %g may be too small.\n", pParam->B3SOIFDvsattemp);
	   printf("Warning: Vsat at current temperature = %g may be too small.\n", pParam->B3SOIFDvsattemp);
	}

	if (pParam->B3SOIFDpdibl1 < 0.0)
	{   fprintf(fplog, "Warning: Pdibl1 = %g is negative.\n",
		    pParam->B3SOIFDpdibl1);
	    printf("Warning: Pdibl1 = %g is negative.\n", pParam->B3SOIFDpdibl1);
	}
	if (pParam->B3SOIFDpdibl2 < 0.0)
	{   fprintf(fplog, "Warning: Pdibl2 = %g is negative.\n",
		    pParam->B3SOIFDpdibl2);
	    printf("Warning: Pdibl2 = %g is negative.\n", pParam->B3SOIFDpdibl2);
	}
/* Check overlap capacitance parameters */
        if (model->B3SOIFDcgdo < 0.0)
	{   fprintf(fplog, "Warning: cgdo = %g is negative. Set to zero.\n", model->B3SOIFDcgdo);
	    printf("Warning: cgdo = %g is negative. Set to zero.\n", model->B3SOIFDcgdo);
	    model->B3SOIFDcgdo = 0.0;
        }      
        if (model->B3SOIFDcgso < 0.0)
	{   fprintf(fplog, "Warning: cgso = %g is negative. Set to zero.\n", model->B3SOIFDcgso);
	    printf("Warning: cgso = %g is negative. Set to zero.\n", model->B3SOIFDcgso);
	    model->B3SOIFDcgso = 0.0;
        }      
        if (model->B3SOIFDcgeo < 0.0)
	{   fprintf(fplog, "Warning: cgeo = %g is negative. Set to zero.\n", model->B3SOIFDcgeo);
	    printf("Warning: cgeo = %g is negative. Set to zero.\n", model->B3SOIFDcgeo);
	    model->B3SOIFDcgeo = 0.0;
        }

	if (model->B3SOIFDntun < 0.0)
	{   fprintf(fplog, "Warning: Ntun = %g is negative.\n",
		    model->B3SOIFDntun); 
	    printf("Warning: Ntun = %g is negative.\n", model->B3SOIFDntun); 
	}

	if (model->B3SOIFDndiode < 0.0)
	{   fprintf(fplog, "Warning: Ndiode = %g is negative.\n",
		    model->B3SOIFDndiode); 
	    printf("Warning: Ndiode = %g is negative.\n", model->B3SOIFDndiode); 
	}

	if (model->B3SOIFDisbjt < 0.0)
	{   fprintf(fplog, "Warning: Isbjt = %g is negative.\n",
		    model->B3SOIFDisbjt); 
	    printf("Warning: Isbjt = %g is negative.\n", model->B3SOIFDisbjt); 
	}

	if (model->B3SOIFDisdif < 0.0)
	{   fprintf(fplog, "Warning: Isdif = %g is negative.\n",
		    model->B3SOIFDisdif); 
	    printf("Warning: Isdif = %g is negative.\n", model->B3SOIFDisdif); 
	}

	if (model->B3SOIFDisrec < 0.0)
	{   fprintf(fplog, "Warning: Isrec = %g is negative.\n",
		    model->B3SOIFDisrec); 
	    printf("Warning: Isrec = %g is negative.\n", model->B3SOIFDisrec); 
	}

	if (model->B3SOIFDistun < 0.0)
	{   fprintf(fplog, "Warning: Istun = %g is negative.\n",
		    model->B3SOIFDistun); 
	    printf("Warning: Istun = %g is negative.\n", model->B3SOIFDistun); 
	}

	if (model->B3SOIFDedl < 0.0)
	{   fprintf(fplog, "Warning: Edl = %g is negative.\n",
		    model->B3SOIFDedl); 
	    printf("Warning: Edl = %g is negative.\n", model->B3SOIFDedl); 
	}

	if (model->B3SOIFDkbjt1 < 0.0)
	{   fprintf(fplog, "Warning: Kbjt1 = %g is negative.\n",
		    model->B3SOIFDkbjt1); 
	    printf("Warning: kbjt1 = %g is negative.\n", model->B3SOIFDkbjt1); 
	}

	if (model->B3SOIFDtt < 0.0)
	{   fprintf(fplog, "Warning: Tt = %g is negative.\n",
		    model->B3SOIFDtt); 
	    printf("Warning: Tt = %g is negative.\n", model->B3SOIFDtt); 
	}

	if (model->B3SOIFDcsdmin < 0.0)
	{   fprintf(fplog, "Warning: Csdmin = %g is negative.\n",
		    model->B3SOIFDcsdmin); 
	    printf("Warning: Csdmin = %g is negative.\n", model->B3SOIFDcsdmin); 
	}

	if (model->B3SOIFDcsdesw < 0.0)
	{   fprintf(fplog, "Warning: Csdesw = %g is negative.\n",
		    model->B3SOIFDcsdesw); 
	    printf("Warning: Csdesw = %g is negative.\n", model->B3SOIFDcsdesw); 
	}

	if ((model->B3SOIFDasd < 0.0) || (model->B3SOIFDmxc > 1.0))
	{   fprintf(fplog, "Warning: Asd = %g should be within (0, 1).\n",
		    model->B3SOIFDasd); 
	    printf("Warning: Asd = %g should be within (0, 1).\n", model->B3SOIFDasd); 
	}

	if (model->B3SOIFDrth0 < 0.0)
	{   fprintf(fplog, "Warning: Rth0 = %g is negative.\n",
		    model->B3SOIFDrth0); 
	    printf("Warning: Rth0 = %g is negative.\n", model->B3SOIFDrth0); 
	}

	if (model->B3SOIFDcth0 < 0.0)
	{   fprintf(fplog, "Warning: Cth0 = %g is negative.\n",
		    model->B3SOIFDcth0); 
	    printf("Warning: Cth0 = %g is negative.\n", model->B3SOIFDcth0); 
	}

	if (model->B3SOIFDrbody < 0.0)
	{   fprintf(fplog, "Warning: Rbody = %g is negative.\n",
		    model->B3SOIFDrbody); 
	    printf("Warning: Rbody = %g is negative.\n", model->B3SOIFDrbody); 
	}

	if (model->B3SOIFDrbsh < 0.0)
	{   fprintf(fplog, "Warning: Rbsh = %g is negative.\n",
		    model->B3SOIFDrbsh); 
	    printf("Warning: Rbsh = %g is negative.\n", model->B3SOIFDrbsh); 
	}

	if (model->B3SOIFDxj > model->B3SOIFDtsi)
	{   fprintf(fplog, "Warning: Xj = %g is thicker than Tsi = %g.\n",
		    model->B3SOIFDxj, model->B3SOIFDtsi); 
	    printf("Warning: Xj = %g is thicker than Tsi = %g.\n", 
		    model->B3SOIFDxj, model->B3SOIFDtsi); 
	}

        if (model->B3SOIFDcapMod < 2)
	{   fprintf(fplog, "Warning: capMod < 2 is not supported by BSIM3SOI.\n");
	    printf("Warning: Warning: capMod < 2 is not supported by BSIM3SOI.\n");
	}

	if (model->B3SOIFDcii > 2.0)
	{   fprintf(fplog, "Warning: Cii = %g is larger than 2.0.\n", model->B3SOIFDcii);
	    printf("Warning: Cii = %g is larger than 2.0.\n", model->B3SOIFDcii);
	}

	if (model->B3SOIFDdii > 1.5)
	{   fprintf(fplog, "Warning: Dii = %g is larger than 1.5.\n", model->B3SOIFDcii);
	    printf("Warning: Dii = %g is too larger than 1.5.\n", model->B3SOIFDcii);
	}

     }/* loop for the parameter check for warning messages */      
	fclose(fplog);
    }
    else
    {   fprintf(stderr, "Warning: Can't open log file. Parameter checking skipped.\n");
    }

    return(Fatal_Flag);
}

