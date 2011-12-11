/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soiddcheck.c          98/5/01
Modified by Paolo Nenzi 2002
**********/

/*
 * Revision 2.1  99/9/27 Pin Su 
 * BSIMDD2.1 release
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "b3soidddef.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/devdefs.h"
#include "ngspice/suffix.h"

int
B3SOIDDcheckModel(B3SOIDDmodel *model, B3SOIDDinstance *here, CKTcircuit *ckt)
{
struct b3soiddSizeDependParam *pParam;
int Fatal_Flag = 0;
FILE *fplog;
    
    NG_IGNORE(ckt);

    if ((fplog = fopen("b3soiddv2check.log", "w")) != NULL)
    {   pParam = here->pParam;
	fprintf(fplog, "B3SOI(DD)V2.1 Parameter Check\n");
	fprintf(fplog, "Model = %s\n", model->B3SOIDDmodName);
	fprintf(fplog, "W = %g, L = %g\n, M = %g\n", here->B3SOIDDw, 
	        here->B3SOIDDl, here->B3SOIDDm);
   	    

            if (pParam->B3SOIDDnlx < -pParam->B3SOIDDleff)
	    {   fprintf(fplog, "Fatal: Nlx = %g is less than -Leff.\n",
			pParam->B3SOIDDnlx);
	        printf("Fatal: Nlx = %g is less than -Leff.\n",
			pParam->B3SOIDDnlx);
		Fatal_Flag = 1;
            }

	if (model->B3SOIDDtox <= 0.0)
	{   fprintf(fplog, "Fatal: Tox = %g is not positive.\n",
		    model->B3SOIDDtox);
	    printf("Fatal: Tox = %g is not positive.\n", model->B3SOIDDtox);
	    Fatal_Flag = 1;
	}

	if (model->B3SOIDDtbox <= 0.0)
	{   fprintf(fplog, "Fatal: Tbox = %g is not positive.\n",
		    model->B3SOIDDtbox);
	    printf("Fatal: Tbox = %g is not positive.\n", model->B3SOIDDtbox);
	    Fatal_Flag = 1;
	}

	if (pParam->B3SOIDDnpeak <= 0.0)
	{   fprintf(fplog, "Fatal: Nch = %g is not positive.\n",
		    pParam->B3SOIDDnpeak);
	    printf("Fatal: Nch = %g is not positive.\n",
		   pParam->B3SOIDDnpeak);
	    Fatal_Flag = 1;
	}
	if (pParam->B3SOIDDngate < 0.0)
	{   fprintf(fplog, "Fatal: Ngate = %g is not positive.\n",
		    pParam->B3SOIDDngate);
	    printf("Fatal: Ngate = %g Ngate is not positive.\n",
		   pParam->B3SOIDDngate);
	    Fatal_Flag = 1;
	}
	if (pParam->B3SOIDDngate > 1.e25)
	{   fprintf(fplog, "Fatal: Ngate = %g is too high.\n",
		    pParam->B3SOIDDngate);
	    printf("Fatal: Ngate = %g Ngate is too high\n",
		   pParam->B3SOIDDngate);
	    Fatal_Flag = 1;
	}

	if (model->B3SOIDDdvbd1 < 0.0)
	{   fprintf(fplog, "Fatal: Dvbd1 = %g is negative.\n",
		    model->B3SOIDDdvbd1);   
	    printf("Fatal: Dvbd1 = %g is negative.\n", model->B3SOIDDdvbd1);   
	    Fatal_Flag = 1;
	}

	if (pParam->B3SOIDDdvt1 < 0.0)
	{   fprintf(fplog, "Fatal: Dvt1 = %g is negative.\n",
		    pParam->B3SOIDDdvt1);   
	    printf("Fatal: Dvt1 = %g is negative.\n", pParam->B3SOIDDdvt1);   
	    Fatal_Flag = 1;
	}
	    
	if (pParam->B3SOIDDdvt1w < 0.0)
	{   fprintf(fplog, "Fatal: Dvt1w = %g is negative.\n",
		    pParam->B3SOIDDdvt1w);
	    printf("Fatal: Dvt1w = %g is negative.\n", pParam->B3SOIDDdvt1w);
	    Fatal_Flag = 1;
	}
	    
	if (pParam->B3SOIDDw0 == -pParam->B3SOIDDweff)
	{   fprintf(fplog, "Fatal: (W0 + Weff) = 0 cauing divided-by-zero.\n");
	    printf("Fatal: (W0 + Weff) = 0 cauing divided-by-zero.\n");
	    Fatal_Flag = 1;
        }   

	if (pParam->B3SOIDDdsub < 0.0)
	{   fprintf(fplog, "Fatal: Dsub = %g is negative.\n", pParam->B3SOIDDdsub);
	    printf("Fatal: Dsub = %g is negative.\n", pParam->B3SOIDDdsub);
	    Fatal_Flag = 1;
	}
	if (pParam->B3SOIDDb1 == -pParam->B3SOIDDweff)
	{   fprintf(fplog, "Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n");
	    printf("Fatal: (B1 + Weff) = 0 causing divided-by-zero.\n");
	    Fatal_Flag = 1;
        }  
        if (pParam->B3SOIDDu0temp <= 0.0)
	{   fprintf(fplog, "Fatal: u0 at current temperature = %g is not positive.\n", pParam->B3SOIDDu0temp);
	    printf("Fatal: u0 at current temperature = %g is not positive.\n",
		   pParam->B3SOIDDu0temp);
	    Fatal_Flag = 1;
        }
    
/* Check delta parameter */      
        if (pParam->B3SOIDDdelta < 0.0)
	{   fprintf(fplog, "Fatal: Delta = %g is less than zero.\n",
		    pParam->B3SOIDDdelta);
	    printf("Fatal: Delta = %g is less than zero.\n", pParam->B3SOIDDdelta);
	    Fatal_Flag = 1;
        }      

	if (pParam->B3SOIDDvsattemp <= 0.0)
	{   fprintf(fplog, "Fatal: Vsat at current temperature = %g is not positive.\n", pParam->B3SOIDDvsattemp);
	    printf("Fatal: Vsat at current temperature = %g is not positive.\n",
		   pParam->B3SOIDDvsattemp);
	    Fatal_Flag = 1;
	}
/* Check Rout parameters */
	if (pParam->B3SOIDDpclm <= 0.0)
	{   fprintf(fplog, "Fatal: Pclm = %g is not positive.\n", pParam->B3SOIDDpclm);
	    printf("Fatal: Pclm = %g is not positive.\n", pParam->B3SOIDDpclm);
	    Fatal_Flag = 1;
	}

	if (pParam->B3SOIDDdrout < 0.0)
	{   fprintf(fplog, "Fatal: Drout = %g is negative.\n", pParam->B3SOIDDdrout);
	    printf("Fatal: Drout = %g is negative.\n", pParam->B3SOIDDdrout);
	    Fatal_Flag = 1;
	}
      if ( model->B3SOIDDunitLengthGateSidewallJctCap > 0.0)
      {
	if (here->B3SOIDDdrainPerimeter < pParam->B3SOIDDweff)
	{   fprintf(fplog, "Warning: Pd = %g is less than W.\n",
		    here->B3SOIDDdrainPerimeter);
	   printf("Warning: Pd = %g is less than W.\n",
		    here->B3SOIDDdrainPerimeter);
            here->B3SOIDDdrainPerimeter =pParam->B3SOIDDweff; 
	}
	if (here->B3SOIDDsourcePerimeter < pParam->B3SOIDDweff)
	{   fprintf(fplog, "Warning: Ps = %g is less than W.\n",
		    here->B3SOIDDsourcePerimeter);
	   printf("Warning: Ps = %g is less than W.\n",
		    here->B3SOIDDsourcePerimeter);
            here->B3SOIDDsourcePerimeter =pParam->B3SOIDDweff;
	}
      }
/* Check capacitance parameters */
        if (pParam->B3SOIDDclc < 0.0)
	{   fprintf(fplog, "Fatal: Clc = %g is negative.\n", pParam->B3SOIDDclc);
	    printf("Fatal: Clc = %g is negative.\n", pParam->B3SOIDDclc);
	    Fatal_Flag = 1;
        }      
      if (model->B3SOIDDparamChk ==1)
      {
/* Check L and W parameters */ 
	if (pParam->B3SOIDDleff <= 5.0e-8)
	{   fprintf(fplog, "Warning: Leff = %g may be too small.\n",
	            pParam->B3SOIDDleff);
	    printf("Warning: Leff = %g may be too small.\n",
		    pParam->B3SOIDDleff);
	}    
	
	if (pParam->B3SOIDDleffCV <= 5.0e-8)
	{   fprintf(fplog, "Warning: Leff for CV = %g may be too small.\n",
		    pParam->B3SOIDDleffCV);
	    printf("Warning: Leff for CV = %g may be too small.\n",
		   pParam->B3SOIDDleffCV);
	}  
	
        if (pParam->B3SOIDDweff <= 1.0e-7)
	{   fprintf(fplog, "Warning: Weff = %g may be too small.\n",
		    pParam->B3SOIDDweff);
	    printf("Warning: Weff = %g may be too small.\n",
		   pParam->B3SOIDDweff);
	}             
	
	if (pParam->B3SOIDDweffCV <= 1.0e-7)
	{   fprintf(fplog, "Warning: Weff for CV = %g may be too small.\n",
		    pParam->B3SOIDDweffCV);
	    printf("Warning: Weff for CV = %g may be too small.\n",
		   pParam->B3SOIDDweffCV);
	}        
	
/* Check threshold voltage parameters */
	if (pParam->B3SOIDDnlx < 0.0)
	{   fprintf(fplog, "Warning: Nlx = %g is negative.\n", pParam->B3SOIDDnlx);
	   printf("Warning: Nlx = %g is negative.\n", pParam->B3SOIDDnlx);
        }
	if (model->B3SOIDDtox < 1.0e-9)
	{   fprintf(fplog, "Warning: Tox = %g is less than 10A.\n",
	            model->B3SOIDDtox);
	    printf("Warning: Tox = %g is less than 10A.\n", model->B3SOIDDtox);
        }

        if (pParam->B3SOIDDnpeak <= 1.0e15)
	{   fprintf(fplog, "Warning: Nch = %g may be too small.\n",
	            pParam->B3SOIDDnpeak);
	    printf("Warning: Nch = %g may be too small.\n",
	           pParam->B3SOIDDnpeak);
	}
	else if (pParam->B3SOIDDnpeak >= 1.0e21)
	{   fprintf(fplog, "Warning: Nch = %g may be too large.\n",
	            pParam->B3SOIDDnpeak);
	    printf("Warning: Nch = %g may be too large.\n",
	           pParam->B3SOIDDnpeak);
	}

	if (fabs(pParam->B3SOIDDnsub) >= 1.0e21)
	{   fprintf(fplog, "Warning: Nsub = %g may be too large.\n",
	            pParam->B3SOIDDnsub);
	    printf("Warning: Nsub = %g may be too large.\n",
	           pParam->B3SOIDDnsub);
	}

	if ((pParam->B3SOIDDngate > 0.0) &&
	    (pParam->B3SOIDDngate <= 1.e18))
	{   fprintf(fplog, "Warning: Ngate = %g is less than 1.E18cm^-3.\n",
	            pParam->B3SOIDDngate);
	    printf("Warning: Ngate = %g is less than 1.E18cm^-3.\n",
	           pParam->B3SOIDDngate);
	}

        if (model->B3SOIDDdvbd0 < 0.0)
	{   fprintf(fplog, "Warning: Dvbd0 = %g is negative.\n",
		    model->B3SOIDDdvbd0);   
	    printf("Warning: Dvbd0 = %g is negative.\n", model->B3SOIDDdvbd0);   
	}
       
        if (pParam->B3SOIDDdvt0 < 0.0)
	{   fprintf(fplog, "Warning: Dvt0 = %g is negative.\n",
		    pParam->B3SOIDDdvt0);   
	    printf("Warning: Dvt0 = %g is negative.\n", pParam->B3SOIDDdvt0);   
	}
	    
	if (fabs(1.0e-6 / (pParam->B3SOIDDw0 + pParam->B3SOIDDweff)) > 10.0)
	{   fprintf(fplog, "Warning: (W0 + Weff) may be too small.\n");
	    printf("Warning: (W0 + Weff) may be too small.\n");
        }

/* Check subthreshold parameters */
	if (pParam->B3SOIDDnfactor < 0.0)
	{   fprintf(fplog, "Warning: Nfactor = %g is negative.\n",
		    pParam->B3SOIDDnfactor);
	    printf("Warning: Nfactor = %g is negative.\n", pParam->B3SOIDDnfactor);
	}
	if (model->B3SOIDDkb3 < 0.0)
	{   fprintf(fplog, "Warning: Kb3 = %g is negative.\n",
		    model->B3SOIDDkb3);
	    printf("Warning: Kb3 = %g is negative.\n", model->B3SOIDDkb3);
	}

	if (pParam->B3SOIDDcdsc < 0.0)
	{   fprintf(fplog, "Warning: Cdsc = %g is negative.\n",
		    pParam->B3SOIDDcdsc);
	    printf("Warning: Cdsc = %g is negative.\n", pParam->B3SOIDDcdsc);
	}
	if (pParam->B3SOIDDcdscd < 0.0)
	{   fprintf(fplog, "Warning: Cdscd = %g is negative.\n",
		    pParam->B3SOIDDcdscd);
	    printf("Warning: Cdscd = %g is negative.\n", pParam->B3SOIDDcdscd);
	}
/* Check DIBL parameters */
	if (pParam->B3SOIDDeta0 < 0.0)
	{   fprintf(fplog, "Warning: Eta0 = %g is negative.\n",
		    pParam->B3SOIDDeta0); 
	    printf("Warning: Eta0 = %g is negative.\n", pParam->B3SOIDDeta0); 
	}
	      
/* Check Abulk parameters */	    
        if (fabs(1.0e-6 / (pParam->B3SOIDDb1 + pParam->B3SOIDDweff)) > 10.0)
       	{   fprintf(fplog, "Warning: (B1 + Weff) may be too small.\n");
       	    printf("Warning: (B1 + Weff) may be too small.\n");
        }    

	if (model->B3SOIDDadice0 > 1.0)
	{   fprintf(fplog, "Warning: Adice0 = %g should be smaller than 1.\n",
		    model->B3SOIDDadice0); 
	    printf("Warning: Adice0 = %g should be smaller than 1.\n", model->B3SOIDDadice0); 
	}

	if (model->B3SOIDDabp < 0.2)
	{   fprintf(fplog, "Warning: Abp = %g is too small.\n",
		    model->B3SOIDDabp); 
	    printf("Warning: Abp = %g is too small.\n", model->B3SOIDDabp); 
	}
    
	if ((model->B3SOIDDmxc < -1.0) || (model->B3SOIDDmxc > 1.0))
	{   fprintf(fplog, "Warning: Mxc = %g should be within (-1, 1).\n",
		    model->B3SOIDDmxc); 
	    printf("Warning: Mxc = %g should be within (-1, 1).\n", model->B3SOIDDmxc); 
	}

/* Check Saturation parameters */
     	if (pParam->B3SOIDDa2 < 0.01)
	{   fprintf(fplog, "Warning: A2 = %g is too small. Set to 0.01.\n", pParam->B3SOIDDa2);
	    printf("Warning: A2 = %g is too small. Set to 0.01.\n",
		   pParam->B3SOIDDa2);
	    pParam->B3SOIDDa2 = 0.01;
	}
	else if (pParam->B3SOIDDa2 > 1.0)
	{   fprintf(fplog, "Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
		    pParam->B3SOIDDa2);
	    printf("Warning: A2 = %g is larger than 1. A2 is set to 1 and A1 is set to 0.\n",
		   pParam->B3SOIDDa2);
	    pParam->B3SOIDDa2 = 1.0;
	    pParam->B3SOIDDa1 = 0.0;

	}

	if (pParam->B3SOIDDrdsw < 0.0)
	{   fprintf(fplog, "Warning: Rdsw = %g is negative. Set to zero.\n",
		    pParam->B3SOIDDrdsw);
	    printf("Warning: Rdsw = %g is negative. Set to zero.\n",
		   pParam->B3SOIDDrdsw);
	    pParam->B3SOIDDrdsw = 0.0;
	    pParam->B3SOIDDrds0 = 0.0;
	}
	else if ((pParam->B3SOIDDrds0 > 0.0) && (pParam->B3SOIDDrds0 < 0.001))
	{   fprintf(fplog, "Warning: Rds at current temperature = %g is less than 0.001 ohm. Set to zero.\n",
		    pParam->B3SOIDDrds0);
	    printf("Warning: Rds at current temperature = %g is less than 0.001 ohm. Set to zero.\n",
		   pParam->B3SOIDDrds0);
	    pParam->B3SOIDDrds0 = 0.0;
	}
	 if (pParam->B3SOIDDvsattemp < 1.0e3)
	{   fprintf(fplog, "Warning: Vsat at current temperature = %g may be too small.\n", pParam->B3SOIDDvsattemp);
	   printf("Warning: Vsat at current temperature = %g may be too small.\n", pParam->B3SOIDDvsattemp);
	}

	if (pParam->B3SOIDDpdibl1 < 0.0)
	{   fprintf(fplog, "Warning: Pdibl1 = %g is negative.\n",
		    pParam->B3SOIDDpdibl1);
	    printf("Warning: Pdibl1 = %g is negative.\n", pParam->B3SOIDDpdibl1);
	}
	if (pParam->B3SOIDDpdibl2 < 0.0)
	{   fprintf(fplog, "Warning: Pdibl2 = %g is negative.\n",
		    pParam->B3SOIDDpdibl2);
	    printf("Warning: Pdibl2 = %g is negative.\n", pParam->B3SOIDDpdibl2);
	}
/* Check overlap capacitance parameters */
        if (model->B3SOIDDcgdo < 0.0)
	{   fprintf(fplog, "Warning: cgdo = %g is negative. Set to zero.\n", model->B3SOIDDcgdo);
	    printf("Warning: cgdo = %g is negative. Set to zero.\n", model->B3SOIDDcgdo);
	    model->B3SOIDDcgdo = 0.0;
        }      
        if (model->B3SOIDDcgso < 0.0)
	{   fprintf(fplog, "Warning: cgso = %g is negative. Set to zero.\n", model->B3SOIDDcgso);
	    printf("Warning: cgso = %g is negative. Set to zero.\n", model->B3SOIDDcgso);
	    model->B3SOIDDcgso = 0.0;
        }      
        if (model->B3SOIDDcgeo < 0.0)
	{   fprintf(fplog, "Warning: cgeo = %g is negative. Set to zero.\n", model->B3SOIDDcgeo);
	    printf("Warning: cgeo = %g is negative. Set to zero.\n", model->B3SOIDDcgeo);
	    model->B3SOIDDcgeo = 0.0;
        }

	if (model->B3SOIDDntun < 0.0)
	{   fprintf(fplog, "Warning: Ntun = %g is negative.\n",
		    model->B3SOIDDntun); 
	    printf("Warning: Ntun = %g is negative.\n", model->B3SOIDDntun); 
	}

	if (model->B3SOIDDndiode < 0.0)
	{   fprintf(fplog, "Warning: Ndiode = %g is negative.\n",
		    model->B3SOIDDndiode); 
	    printf("Warning: Ndiode = %g is negative.\n", model->B3SOIDDndiode); 
	}

	if (model->B3SOIDDisbjt < 0.0)
	{   fprintf(fplog, "Warning: Isbjt = %g is negative.\n",
		    model->B3SOIDDisbjt); 
	    printf("Warning: Isbjt = %g is negative.\n", model->B3SOIDDisbjt); 
	}

	if (model->B3SOIDDisdif < 0.0)
	{   fprintf(fplog, "Warning: Isdif = %g is negative.\n",
		    model->B3SOIDDisdif); 
	    printf("Warning: Isdif = %g is negative.\n", model->B3SOIDDisdif); 
	}

	if (model->B3SOIDDisrec < 0.0)
	{   fprintf(fplog, "Warning: Isrec = %g is negative.\n",
		    model->B3SOIDDisrec); 
	    printf("Warning: Isrec = %g is negative.\n", model->B3SOIDDisrec); 
	}

	if (model->B3SOIDDistun < 0.0)
	{   fprintf(fplog, "Warning: Istun = %g is negative.\n",
		    model->B3SOIDDistun); 
	    printf("Warning: Istun = %g is negative.\n", model->B3SOIDDistun); 
	}

	if (model->B3SOIDDedl < 0.0)
	{   fprintf(fplog, "Warning: Edl = %g is negative.\n",
		    model->B3SOIDDedl); 
	    printf("Warning: Edl = %g is negative.\n", model->B3SOIDDedl); 
	}

	if (model->B3SOIDDkbjt1 < 0.0)
	{   fprintf(fplog, "Warning: Kbjt1 = %g is negative.\n",
		    model->B3SOIDDkbjt1); 
	    printf("Warning: kbjt1 = %g is negative.\n", model->B3SOIDDkbjt1); 
	}

	if (model->B3SOIDDtt < 0.0)
	{   fprintf(fplog, "Warning: Tt = %g is negative.\n",
		    model->B3SOIDDtt); 
	    printf("Warning: Tt = %g is negative.\n", model->B3SOIDDtt); 
	}

	if (model->B3SOIDDcsdmin < 0.0)
	{   fprintf(fplog, "Warning: Csdmin = %g is negative.\n",
		    model->B3SOIDDcsdmin); 
	    printf("Warning: Csdmin = %g is negative.\n", model->B3SOIDDcsdmin); 
	}

	if (model->B3SOIDDcsdesw < 0.0)
	{   fprintf(fplog, "Warning: Csdesw = %g is negative.\n",
		    model->B3SOIDDcsdesw); 
	    printf("Warning: Csdesw = %g is negative.\n", model->B3SOIDDcsdesw); 
	}

	if ((model->B3SOIDDasd < 0.0) || (model->B3SOIDDmxc > 1.0))
	{   fprintf(fplog, "Warning: Asd = %g should be within (0, 1).\n",
		    model->B3SOIDDasd); 
	    printf("Warning: Asd = %g should be within (0, 1).\n", model->B3SOIDDasd); 
	}

	if (model->B3SOIDDrth0 < 0.0)
	{   fprintf(fplog, "Warning: Rth0 = %g is negative.\n",
		    model->B3SOIDDrth0); 
	    printf("Warning: Rth0 = %g is negative.\n", model->B3SOIDDrth0); 
	}

	if (model->B3SOIDDcth0 < 0.0)
	{   fprintf(fplog, "Warning: Cth0 = %g is negative.\n",
		    model->B3SOIDDcth0); 
	    printf("Warning: Cth0 = %g is negative.\n", model->B3SOIDDcth0); 
	}

	if (model->B3SOIDDrbody < 0.0)
	{   fprintf(fplog, "Warning: Rbody = %g is negative.\n",
		    model->B3SOIDDrbody); 
	    printf("Warning: Rbody = %g is negative.\n", model->B3SOIDDrbody); 
	}

	if (model->B3SOIDDrbsh < 0.0)
	{   fprintf(fplog, "Warning: Rbsh = %g is negative.\n",
		    model->B3SOIDDrbsh); 
	    printf("Warning: Rbsh = %g is negative.\n", model->B3SOIDDrbsh); 
	}

	if (model->B3SOIDDxj > model->B3SOIDDtsi)
	{   fprintf(fplog, "Warning: Xj = %g is thicker than Tsi = %g.\n",
		    model->B3SOIDDxj, model->B3SOIDDtsi); 
	    printf("Warning: Xj = %g is thicker than Tsi = %g.\n", 
		    model->B3SOIDDxj, model->B3SOIDDtsi); 
	}

        if (model->B3SOIDDcapMod < 2)
	{   fprintf(fplog, "Warning: capMod < 2 is not supported by BSIM3SOI.\n");
	    printf("Warning: Warning: capMod < 2 is not supported by BSIM3SOI.\n");
	}

	if (model->B3SOIDDcii > 2.0)
	{   fprintf(fplog, "Warning: Cii = %g is larger than 2.0.\n", model->B3SOIDDcii);
	    printf("Warning: Cii = %g is larger than 2.0.\n", model->B3SOIDDcii);
	}

	if (model->B3SOIDDdii > 1.5)
	{   fprintf(fplog, "Warning: Dii = %g is larger than 1.5.\n", model->B3SOIDDcii);
	    printf("Warning: Dii = %g is too larger than 1.5.\n", model->B3SOIDDcii);
	}

     }/* loop for the parameter check for warning messages */      
	fclose(fplog);
    }
    else
    {   fprintf(stderr, "Warning: Can't open log file. Parameter checking skipped.\n");
    }

    return(Fatal_Flag);
}

