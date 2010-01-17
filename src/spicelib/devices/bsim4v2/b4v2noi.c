/**** BSIM4.2.1, Released by Xuemei Xi 10/05/2001 ****/

/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * File: b4noi.c of BSIM4.2.1.
 * Author: 2000 Weidong Liu
 * Authors: Xuemei Xi, Kanyu M. Cao, Hui Wan, Mansun Chan, Chenming Hu.
 * Project Director: Prof. Chenming Hu.

 * Modified by Xuemei Xi, 10/05/2001.
 **********/

#include "ngspice.h"
#include "bsim4v2def.h"
#include "cktdefs.h"
#include "iferrmsg.h"
#include "noisedef.h"
#include "const.h"


extern void   NevalSrc (double *noise, double *lnNoise, CKTcircuit *ckt, int type, int node1, int node2, double param);
extern double Nintegrate (double noizDens, double lnNdens, double lnNlstDens, Ndata *data);

/*
 * WDL: 1/f noise model has been smoothed out and enhanced with
 * bulk charge effect as well as physical N* equ. and necessary
 * conversion into the SI unit system.
 */

double
BSIM4v2Eval1ovFNoise(
double Vds,
BSIM4v2model *model,
BSIM4v2instance *here,
double freq, double temp)
{
struct bsim4SizeDependParam *pParam;
double cd, esat, DelClm, EffFreq, N0, Nl;
double T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, Ssi;

    pParam = here->pParam;
    cd = fabs(here->BSIM4v2cd);
    esat = 2.0 * pParam->BSIM4v2vsattemp / here->BSIM4v2ueff;
    if(model->BSIM4v2em<=0.0) DelClm = 0.0; /* flicker noise modified -JX  */
    else {
    	T0 = ((((Vds - here->BSIM4v2Vdseff) / pParam->BSIM4v2litl)
       		+ model->BSIM4v2em) / esat);
    	DelClm = pParam->BSIM4v2litl * log (MAX(T0, N_MINLOG));
    }
    EffFreq = pow(freq, model->BSIM4v2ef);
    T1 = CHARGE * CHARGE * CONSTboltz * cd * temp * here->BSIM4v2ueff;
    T2 = 1.0e10 * EffFreq * here->BSIM4v2Abulk * model->BSIM4v2coxe
       * pParam->BSIM4v2leff * pParam->BSIM4v2leff;
    N0 = model->BSIM4v2coxe * here->BSIM4v2Vgsteff / CHARGE;
    Nl = model->BSIM4v2coxe * here->BSIM4v2Vgsteff
       * (1.0 - here->BSIM4v2AbovVgst2Vtm * here->BSIM4v2Vdseff) / CHARGE;

    T3 = model->BSIM4v2oxideTrapDensityA
       * log(MAX(((N0 + here->BSIM4v2nstar) / (Nl + here->BSIM4v2nstar)), N_MINLOG));
    T4 = model->BSIM4v2oxideTrapDensityB * (N0 - Nl);
    T5 = model->BSIM4v2oxideTrapDensityC * 0.5 * (N0 * N0 - Nl * Nl);

    T6 = CONSTboltz * temp * cd * cd;
    T7 = 1.0e10 * EffFreq * pParam->BSIM4v2leff
       * pParam->BSIM4v2leff * pParam->BSIM4v2weff;
    T8 = model->BSIM4v2oxideTrapDensityA + model->BSIM4v2oxideTrapDensityB * Nl
       + model->BSIM4v2oxideTrapDensityC * Nl * Nl;
    T9 = (Nl + here->BSIM4v2nstar) * (Nl + here->BSIM4v2nstar);
    Ssi = T1 / T2 * (T3 + T4 + T5) + T6 / T7 * DelClm * T8 / T9;
    return Ssi;
}


int
BSIM4v2noise (
int mode, int operation,
GENmodel *inModel,
CKTcircuit *ckt,
Ndata *data,
double *OnDens)
{
BSIM4v2model *model = (BSIM4v2model *)inModel;
BSIM4v2instance *here;
struct bsim4SizeDependParam *pParam;
char name[N_MXVLNTH];
double tempOnoise;
double tempInoise;
double noizDens[BSIM4v2NSRCS];
double lnNdens[BSIM4v2NSRCS];


double T0, T1, T2, T5, T10, T11;
double Vds, Ssi, Swi;
double tmp=0.0, gdpr, gspr, npart_theta=0.0, npart_beta=0.0, igsquare;

double m;

int i;

    /* define the names of the noise sources */
    static char *BSIM4v2nNames[BSIM4v2NSRCS] =
    {   /* Note that we have to keep the order */
	".rd",              /* noise due to rd */
	".rs",              /* noise due to rs */
        ".rg",              /* noise due to rgeltd */
        ".rbps",            /* noise due to rbps */
        ".rbpd",            /* noise due to rbpd */
        ".rbpb",            /* noise due to rbpb */
        ".rbsb",            /* noise due to rbsb */
        ".rbdb",            /* noise due to rbdb */
	".id",              /* noise due to id */
	".1overf",          /* flicker (1/f) noise */
        ".igs",             /* shot noise due to IGS */
        ".igd",             /* shot noise due to IGD */
        ".igb",             /* shot noise due to IGB */
	""                  /* total transistor noise */
    };

    for (; model != NULL; model = model->BSIM4v2nextModel)
    {    for (here = model->BSIM4v2instances; here != NULL;
	      here = here->BSIM4v2nextInstance)
	 {    pParam = here->pParam;
	      switch (operation)
	      {  case N_OPEN:
		     /* see if we have to to produce a summary report */
		     /* if so, name all the noise generators */

		      if (((NOISEAN*)ckt->CKTcurJob)->NStpsSm != 0)
		      {   switch (mode)
			  {  case N_DENS:
			          for (i = 0; i < BSIM4v2NSRCS; i++)
				  {    (void) sprintf(name, "onoise.%s%s",
					              here->BSIM4v2name,
						      BSIM4v2nNames[i]);
                                       data->namelist = (IFuid *) trealloc(
					     (char *) data->namelist,
					     (data->numPlots + 1)
					     * sizeof(IFuid));
                                       if (!data->namelist)
					   return(E_NOMEM);
		                       (*(SPfrontEnd->IFnewUid)) (ckt,
			                  &(data->namelist[data->numPlots++]),
			                  (IFuid) NULL, name, UID_OTHER,
					  (void **) NULL);
				       /* we've added one more plot */
			          }
			          break;
		             case INT_NOIZ:
			          for (i = 0; i < BSIM4v2NSRCS; i++)
				  {    (void) sprintf(name, "onoise_total.%s%s",
						      here->BSIM4v2name,
						      BSIM4v2nNames[i]);
                                       data->namelist = (IFuid *) trealloc(
					     (char *) data->namelist,
					     (data->numPlots + 1)
					     * sizeof(IFuid));
                                       if (!data->namelist)
					   return(E_NOMEM);
		                       (*(SPfrontEnd->IFnewUid)) (ckt,
			                  &(data->namelist[data->numPlots++]),
			                  (IFuid) NULL, name, UID_OTHER,
					  (void **) NULL);
				       /* we've added one more plot */

			               (void) sprintf(name, "inoise_total.%s%s",
						      here->BSIM4v2name,
						      BSIM4v2nNames[i]);
                                       data->namelist = (IFuid *) trealloc(
					     (char *) data->namelist,
					     (data->numPlots + 1)
					     * sizeof(IFuid));
                                       if (!data->namelist)
					   return(E_NOMEM);
		                       (*(SPfrontEnd->IFnewUid)) (ckt,
			                  &(data->namelist[data->numPlots++]),
			                  (IFuid) NULL, name, UID_OTHER,
					  (void **)NULL);
				       /* we've added one more plot */
			          }
			          break;
		          }
		      }
		      break;
	         case N_CALC:
		      m = here->BSIM4v2m;
		      switch (mode)
		      {  case N_DENS:
			      if (model->BSIM4v2tnoiMod == 0)
			      {   if (model->BSIM4v2rdsMod == 0)
				  {   gspr = here->BSIM4v2sourceConductance;
                                      gdpr = here->BSIM4v2drainConductance;
				      if (here->BSIM4v2grdsw > 0.0)
				          tmp = 1.0 / here->BSIM4v2grdsw; /* tmp used below */ 
				      else
					  tmp = 0.0;
				  }
				  else
				  {   gspr = here->BSIM4v2gstot;
                                      gdpr = here->BSIM4v2gdtot;
                                      tmp = 0.0;
				  }
			      }
			      else
			      {   T5 = here->BSIM4v2Vgsteff / here->BSIM4v2EsatL;
				  T5 *= T5;
				  npart_beta = 0.577 * (1.0 + T5
					     * model->BSIM4v2tnoia * pParam->BSIM4v2leff);
				  npart_theta = 0.37 * (1.0 + T5
                                              * model->BSIM4v2tnoib * pParam->BSIM4v2leff);

				  if (model->BSIM4v2rdsMod == 0)
                                  {   gspr = here->BSIM4v2sourceConductance;
                                      gdpr = here->BSIM4v2drainConductance;
                                  }
                                  else
                                  {   gspr = here->BSIM4v2gstot;
                                      gdpr = here->BSIM4v2gdtot;
                                  }

				  if ((*(ckt->CKTstates[0] + here->BSIM4v2vds)) >= 0.0)
			              gspr = gspr * (1.0 + npart_theta * npart_theta * gspr
				 	   / here->BSIM4v2IdovVds);
				  else
				      gdpr = gdpr * (1.0 + npart_theta * npart_theta * gdpr
                                           / here->BSIM4v2IdovVds);
			      } 

		              NevalSrc(&noizDens[BSIM4v2RDNOIZ],
				       &lnNdens[BSIM4v2RDNOIZ], ckt, THERMNOISE,
				       here->BSIM4v2dNodePrime, here->BSIM4v2dNode,
				       gdpr * m);

		              NevalSrc(&noizDens[BSIM4v2RSNOIZ],
				       &lnNdens[BSIM4v2RSNOIZ], ckt, THERMNOISE,
				       here->BSIM4v2sNodePrime, here->BSIM4v2sNode,
				       gspr * m);


			      if ((here->BSIM4v2rgateMod == 1) || (here->BSIM4v2rgateMod == 2))
			      {   NevalSrc(&noizDens[BSIM4v2RGNOIZ],
                                       &lnNdens[BSIM4v2RGNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v2gNodePrime, here->BSIM4v2gNodeExt,
                                       here->BSIM4v2grgeltd * m);
			      }
			      else if (here->BSIM4v2rgateMod == 3)
			      {   NevalSrc(&noizDens[BSIM4v2RGNOIZ],
                                       &lnNdens[BSIM4v2RGNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v2gNodeMid, here->BSIM4v2gNodeExt,
                                       here->BSIM4v2grgeltd * m);
			      }
			      else
			      {    noizDens[BSIM4v2RGNOIZ] = 0.0;
                                   lnNdens[BSIM4v2RGNOIZ] =
                                          log(MAX(noizDens[BSIM4v2RGNOIZ], N_MINLOG));
			      }


                              if (here->BSIM4v2rbodyMod)
                              {   NevalSrc(&noizDens[BSIM4v2RBPSNOIZ],
                                       &lnNdens[BSIM4v2RBPSNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v2bNodePrime, here->BSIM4v2sbNode,
                                       here->BSIM4v2grbps * m);
                                  NevalSrc(&noizDens[BSIM4v2RBPDNOIZ],
                                       &lnNdens[BSIM4v2RBPDNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v2bNodePrime, here->BSIM4v2dbNode,
                                       here->BSIM4v2grbpd * m);
                                  NevalSrc(&noizDens[BSIM4v2RBPBNOIZ],
                                       &lnNdens[BSIM4v2RBPBNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v2bNodePrime, here->BSIM4v2bNode,
                                       here->BSIM4v2grbpb * m);
                                  NevalSrc(&noizDens[BSIM4v2RBSBNOIZ],
                                       &lnNdens[BSIM4v2RBSBNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v2bNode, here->BSIM4v2sbNode,
                                       here->BSIM4v2grbsb * m);
                                  NevalSrc(&noizDens[BSIM4v2RBDBNOIZ],
                                       &lnNdens[BSIM4v2RBDBNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v2bNode, here->BSIM4v2dbNode,
                                       here->BSIM4v2grbdb * m);
                              }
                              else
                              {   noizDens[BSIM4v2RBPSNOIZ] = noizDens[BSIM4v2RBPDNOIZ] = 0.0;   
                                  noizDens[BSIM4v2RBPBNOIZ] = 0.0;
                                  noizDens[BSIM4v2RBSBNOIZ] = noizDens[BSIM4v2RBDBNOIZ] = 0.0;
                                  lnNdens[BSIM4v2RBPSNOIZ] =
                                          log(MAX(noizDens[BSIM4v2RBPSNOIZ], N_MINLOG));
                                  lnNdens[BSIM4v2RBPDNOIZ] =
                                          log(MAX(noizDens[BSIM4v2RBPDNOIZ], N_MINLOG));
                                  lnNdens[BSIM4v2RBPBNOIZ] =
                                          log(MAX(noizDens[BSIM4v2RBPBNOIZ], N_MINLOG));
                                  lnNdens[BSIM4v2RBSBNOIZ] =
                                          log(MAX(noizDens[BSIM4v2RBSBNOIZ], N_MINLOG));
                                  lnNdens[BSIM4v2RBDBNOIZ] =
                                          log(MAX(noizDens[BSIM4v2RBDBNOIZ], N_MINLOG));
                              }


                              switch(model->BSIM4v2tnoiMod)
			      {  case 0:
				      T0 = m * here->BSIM4v2ueff * fabs(here->BSIM4v2qinv);
				      T1 = T0 * tmp + pParam->BSIM4v2leff
                                         * pParam->BSIM4v2leff;
		                      NevalSrc(&noizDens[BSIM4v2IDNOIZ],
				               &lnNdens[BSIM4v2IDNOIZ], ckt,
					       THERMNOISE, here->BSIM4v2dNodePrime,
                                               here->BSIM4v2sNodePrime,
					       (T0 / T1) * model->BSIM4v2ntnoi);
				      break;
				 case 1:
				      T0 = m * (here->BSIM4v2gm + here->BSIM4v2gmbs + here->BSIM4v2gds);
				      T0 *= T0;
				      igsquare = npart_theta * npart_theta * T0 / here->BSIM4v2IdovVds;
				      T1 = npart_beta * (here->BSIM4v2gm
					 + here->BSIM4v2gmbs) + here->BSIM4v2gds;
				      T2 = T1 * T1 / here->BSIM4v2IdovVds;
                                      NevalSrc(&noizDens[BSIM4v2IDNOIZ],
                                               &lnNdens[BSIM4v2IDNOIZ], ckt,
                                               THERMNOISE, here->BSIM4v2dNodePrime,
                                               here->BSIM4v2sNodePrime, (T2 - igsquare));
                                      break;
			      }

		              NevalSrc(&noizDens[BSIM4v2FLNOIZ], (double*) NULL,
				       ckt, N_GAIN, here->BSIM4v2dNodePrime,
				       here->BSIM4v2sNodePrime, (double) 0.0);

                              switch(model->BSIM4v2fnoiMod)
			      {  case 0:
			              noizDens[BSIM4v2FLNOIZ] *= m * model->BSIM4v2kf
					    * exp(model->BSIM4v2af
					    * log(MAX(fabs(here->BSIM4v2cd),
					    N_MINLOG)))
					    / (pow(data->freq, model->BSIM4v2ef)
					    * pParam->BSIM4v2leff
				            * pParam->BSIM4v2leff
					    * model->BSIM4v2coxe);
				      break;
			         case 1:
		                      Vds = *(ckt->CKTstates[0] + here->BSIM4v2vds);
			              if (Vds < 0.0)
			                  Vds = -Vds;

                                      Ssi = BSIM4v2Eval1ovFNoise(Vds, model, here,
                                          data->freq, ckt->CKTtemp);
                                      T10 = model->BSIM4v2oxideTrapDensityA
                                          * CONSTboltz * ckt->CKTtemp;
                                      T11 = pParam->BSIM4v2weff * pParam->BSIM4v2leff
                                          * pow(data->freq, model->BSIM4v2ef) * 1.0e10
					  * here->BSIM4v2nstar * here->BSIM4v2nstar;
                                      Swi = T10 / T11 * here->BSIM4v2cd
                                          * here->BSIM4v2cd;
                                      T1 = Swi + Ssi;
                                      if (T1 > 0.0)
                                          noizDens[BSIM4v2FLNOIZ] *= m * (Ssi * Swi) / T1;
                                      else
                                          noizDens[BSIM4v2FLNOIZ] *= 0.0;
				      break;
			      }

		              lnNdens[BSIM4v2FLNOIZ] =
				     log(MAX(noizDens[BSIM4v2FLNOIZ], N_MINLOG));


                              NevalSrc(&noizDens[BSIM4v2IGSNOIZ],
                                   &lnNdens[BSIM4v2IGSNOIZ], ckt, SHOTNOISE,
                                   here->BSIM4v2gNodePrime, here->BSIM4v2sNodePrime,
                                   m * (here->BSIM4v2Igs + here->BSIM4v2Igcs));
                              NevalSrc(&noizDens[BSIM4v2IGDNOIZ],
                                   &lnNdens[BSIM4v2IGDNOIZ], ckt, SHOTNOISE,
                                   here->BSIM4v2gNodePrime, here->BSIM4v2dNodePrime,
                                   m * (here->BSIM4v2Igd + here->BSIM4v2Igcd));

                              NevalSrc(&noizDens[BSIM4v2IGBNOIZ],
                                   &lnNdens[BSIM4v2IGBNOIZ], ckt, SHOTNOISE,
                                   here->BSIM4v2gNodePrime, here->BSIM4v2bNodePrime,
                                   m * here->BSIM4v2Igb);


		              noizDens[BSIM4v2TOTNOIZ] = noizDens[BSIM4v2RDNOIZ]
				     + noizDens[BSIM4v2RSNOIZ] + noizDens[BSIM4v2RGNOIZ]
				     + noizDens[BSIM4v2RBPSNOIZ] + noizDens[BSIM4v2RBPDNOIZ]
				     + noizDens[BSIM4v2RBPBNOIZ]
				     + noizDens[BSIM4v2RBSBNOIZ] + noizDens[BSIM4v2RBDBNOIZ]
				     + noizDens[BSIM4v2IDNOIZ] + noizDens[BSIM4v2FLNOIZ]
                                     + noizDens[BSIM4v2IGSNOIZ] + noizDens[BSIM4v2IGDNOIZ]
                                     + noizDens[BSIM4v2IGBNOIZ];
		              lnNdens[BSIM4v2TOTNOIZ] = 
				     log(MAX(noizDens[BSIM4v2TOTNOIZ], N_MINLOG));

		              *OnDens += noizDens[BSIM4v2TOTNOIZ];

		              if (data->delFreq == 0.0)
			      {   /* if we haven't done any previous 
				     integration, we need to initialize our
				     "history" variables.
				    */

			          for (i = 0; i < BSIM4v2NSRCS; i++)
				  {    here->BSIM4v2nVar[LNLSTDENS][i] =
					     lnNdens[i];
			          }

			          /* clear out our integration variables
				     if it's the first pass
				   */
			          if (data->freq ==
				      ((NOISEAN*) ckt->CKTcurJob)->NstartFreq)
				  {   for (i = 0; i < BSIM4v2NSRCS; i++)
				      {    here->BSIM4v2nVar[OUTNOIZ][i] = 0.0;
				           here->BSIM4v2nVar[INNOIZ][i] = 0.0;
			              }
			          }
		              }
			      else
			      {   /* data->delFreq != 0.0,
				     we have to integrate.
				   */
			          for (i = 0; i < BSIM4v2NSRCS; i++)
				  {    if (i != BSIM4v2TOTNOIZ)
				       {   tempOnoise = Nintegrate(noizDens[i],
						lnNdens[i],
				                here->BSIM4v2nVar[LNLSTDENS][i],
						data);
				           tempInoise = Nintegrate(noizDens[i]
						* data->GainSqInv, lnNdens[i]
						+ data->lnGainInv,
				                here->BSIM4v2nVar[LNLSTDENS][i]
						+ data->lnGainInv, data);
				           here->BSIM4v2nVar[LNLSTDENS][i] =
						lnNdens[i];
				           data->outNoiz += tempOnoise;
				           data->inNoise += tempInoise;
				           if (((NOISEAN*)
					       ckt->CKTcurJob)->NStpsSm != 0)
					   {   here->BSIM4v2nVar[OUTNOIZ][i]
						     += tempOnoise;
				               here->BSIM4v2nVar[OUTNOIZ][BSIM4v2TOTNOIZ]
						     += tempOnoise;
				               here->BSIM4v2nVar[INNOIZ][i]
						     += tempInoise;
				               here->BSIM4v2nVar[INNOIZ][BSIM4v2TOTNOIZ]
						     += tempInoise;
                                           }
			               }
			          }
		              }
		              if (data->prtSummary)
			      {   for (i = 0; i < BSIM4v2NSRCS; i++)
				  {    /* print a summary report */
			               data->outpVector[data->outNumber++]
					     = noizDens[i];
			          }
		              }
		              break;
		         case INT_NOIZ:
			      /* already calculated, just output */
		              if (((NOISEAN*)ckt->CKTcurJob)->NStpsSm != 0)
			      {   for (i = 0; i < BSIM4v2NSRCS; i++)
				  {    data->outpVector[data->outNumber++]
					     = here->BSIM4v2nVar[OUTNOIZ][i];
			               data->outpVector[data->outNumber++]
					     = here->BSIM4v2nVar[INNOIZ][i];
			          }
		              }
		              break;
		      }
		      break;
	         case N_CLOSE:
		      /* do nothing, the main calling routine will close */
		      return (OK);
		      break;   /* the plots */
	      }       /* switch (operation) */
	 }    /* for here */
    }    /* for model */

    return(OK);
}
