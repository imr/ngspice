/**** BSIM4.4.0  Released by Xuemei (Jane) Xi 03/04/2004 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b4noi.c of BSIM4.4.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 * Modified by Xuemei Xi, 11/15/2002.
 * Modified by Xuemei Xi, 05/09/2003.
 * Modified by Xuemei Xi, 03/04/2004.
 **********/

#include "ngspice/ngspice.h"
#include "bsim4v4def.h"
#include "ngspice/cktdefs.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/noisedef.h"
#include "ngspice/suffix.h"
#include "ngspice/const.h"


/*
 * WDL: 1/f noise model has been smoothed out and enhanced with
 * bulk charge effect as well as physical N* equ. and necessary
 * conversion into the SI unit system.
 */

static double
Eval1ovFNoise(Vds, model, here, freq, temp)
double Vds, freq, temp;
BSIM4v4model *model;
BSIM4v4instance *here;
{
struct bsim4v4SizeDependParam *pParam;
double cd, esat, DelClm, EffFreq, N0, Nl, Leff, Leffsq;
double T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, Ssi;

    pParam = here->pParam;
    cd = fabs(here->BSIM4v4cd);
    Leff = pParam->BSIM4v4leff - 2.0 * model->BSIM4v4lintnoi;
    Leffsq = Leff * Leff;
    esat = 2.0 * here->BSIM4v4vsattemp / here->BSIM4v4ueff;
    if(model->BSIM4v4em<=0.0) DelClm = 0.0; /* flicker noise modified -JX  */
    else {
    	T0 = ((((Vds - here->BSIM4v4Vdseff) / pParam->BSIM4v4litl)
       		+ model->BSIM4v4em) / esat);
    	DelClm = pParam->BSIM4v4litl * log (MAX(T0, N_MINLOG));
    }
    EffFreq = pow(freq, model->BSIM4v4ef);
    T1 = CHARGE * CHARGE * CONSTboltz * cd * temp * here->BSIM4v4ueff;
    T2 = 1.0e10 * EffFreq * here->BSIM4v4Abulk * model->BSIM4v4coxe * Leffsq;
    N0 = model->BSIM4v4coxe * here->BSIM4v4Vgsteff / CHARGE;
    Nl = model->BSIM4v4coxe * here->BSIM4v4Vgsteff
       * (1.0 - here->BSIM4v4AbovVgst2Vtm * here->BSIM4v4Vdseff) / CHARGE;

    T3 = model->BSIM4v4oxideTrapDensityA
       * log(MAX(((N0 + here->BSIM4v4nstar) / (Nl + here->BSIM4v4nstar)), N_MINLOG));
    T4 = model->BSIM4v4oxideTrapDensityB * (N0 - Nl);
    T5 = model->BSIM4v4oxideTrapDensityC * 0.5 * (N0 * N0 - Nl * Nl);

    T6 = CONSTboltz * temp * cd * cd;
    T7 = 1.0e10 * EffFreq * Leffsq * pParam->BSIM4v4weff;
    T8 = model->BSIM4v4oxideTrapDensityA + model->BSIM4v4oxideTrapDensityB * Nl
       + model->BSIM4v4oxideTrapDensityC * Nl * Nl;
    T9 = (Nl + here->BSIM4v4nstar) * (Nl + here->BSIM4v4nstar);
    Ssi = T1 / T2 * (T3 + T4 + T5) + T6 / T7 * DelClm * T8 / T9;
    return Ssi;
}


int
BSIM4v4noise (mode, operation, inModel, ckt, data, OnDens)
int mode, operation;
GENmodel *inModel;
CKTcircuit *ckt;
register Ndata *data;
double *OnDens;
{
#define job ((NOISEAN*)ckt->CKTcurJob)

register BSIM4v4model *model = (BSIM4v4model *)inModel;
register BSIM4v4instance *here;
struct bsim4v4SizeDependParam *pParam;
char name[N_MXVLNTH];
double tempOnoise;
double tempInoise;
double noizDens[BSIM4v4NSRCS];
double lnNdens[BSIM4v4NSRCS];

double N0, Nl;
double T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13;
double Vds, n, ExpArg, Ssi, Swi;
double tmp, gdpr, gspr, npart_theta, npart_beta, igsquare;

int error, i;

    /* define the names of the noise sources */
    static char *BSIM4v4nNames[BSIM4v4NSRCS] =
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

    for (; model != NULL; model = model->BSIM4v4nextModel)
    {    for (here = model->BSIM4v4instances; here != NULL;
	      here = here->BSIM4v4nextInstance)
	 {    pParam = here->pParam;
	      switch (operation)
	      {  case N_OPEN:
		     /* see if we have to to produce a summary report */
		     /* if so, name all the noise generators */

		      if (job->NStpsSm != 0)
		      {   switch (mode)
			  {  case N_DENS:
			          for (i = 0; i < BSIM4v4NSRCS; i++)
				  {    (void) sprintf(name, "onoise.%s%s",
					              here->BSIM4v4name,
						      BSIM4v4nNames[i]);
                                       data->namelist = TREALLOC(IFuid, data->namelist, data->numPlots + 1);
                                       if (!data->namelist)
					   return(E_NOMEM);
		                       SPfrontEnd->IFnewUid (ckt,
			                  &(data->namelist[data->numPlots++]),
			                  NULL, name, UID_OTHER,
					  NULL);
				       /* we've added one more plot */
			          }
			          break;
		             case INT_NOIZ:
			          for (i = 0; i < BSIM4v4NSRCS; i++)
				  {    (void) sprintf(name, "onoise_total.%s%s",
						      here->BSIM4v4name,
						      BSIM4v4nNames[i]);
                                       data->namelist = TREALLOC(IFuid, data->namelist, data->numPlots + 1);
                                       if (!data->namelist)
					   return(E_NOMEM);
		                       SPfrontEnd->IFnewUid (ckt,
			                  &(data->namelist[data->numPlots++]),
			                  NULL, name, UID_OTHER,
					  NULL);
				       /* we've added one more plot */

			               (void) sprintf(name, "inoise_total.%s%s",
						      here->BSIM4v4name,
						      BSIM4v4nNames[i]);
                                       data->namelist = TREALLOC(IFuid, data->namelist, data->numPlots + 1);
                                       if (!data->namelist)
					   return(E_NOMEM);
		                       SPfrontEnd->IFnewUid (ckt,
			                  &(data->namelist[data->numPlots++]),
			                  NULL, name, UID_OTHER,
					  NULL);
				       /* we've added one more plot */
			          }
			          break;
		          }
		      }
		      break;
	         case N_CALC:
		      switch (mode)
		      {  case N_DENS:
			      if (model->BSIM4v4tnoiMod == 0)
			      {   if (model->BSIM4v4rdsMod == 0)
				  {   gspr = here->BSIM4v4sourceConductance;
                                      gdpr = here->BSIM4v4drainConductance;
				      if (here->BSIM4v4grdsw > 0.0)
				          tmp = 1.0 / here->BSIM4v4grdsw; /* tmp used below */
				      else
					  tmp = 0.0;
				  }
				  else
				  {   gspr = here->BSIM4v4gstot;
                                      gdpr = here->BSIM4v4gdtot;
                                      tmp = 0.0;
				  }
			      }
			      else
			      {   T5 = here->BSIM4v4Vgsteff / here->BSIM4v4EsatL;
				  T5 *= T5;
				  npart_beta = model->BSIM4v4rnoia * (1.0 + T5
					     * model->BSIM4v4tnoia * pParam->BSIM4v4leff);
				  npart_theta = model->BSIM4v4rnoib * (1.0 + T5
                                              * model->BSIM4v4tnoib * pParam->BSIM4v4leff);

				  if (model->BSIM4v4rdsMod == 0)
                                  {   gspr = here->BSIM4v4sourceConductance;
                                      gdpr = here->BSIM4v4drainConductance;
                                  }
                                  else
                                  {   gspr = here->BSIM4v4gstot;
                                      gdpr = here->BSIM4v4gdtot;
                                  }

				  if ((*(ckt->CKTstates[0] + here->BSIM4v4vds)) >= 0.0)
			              gspr = gspr / (1.0 + npart_theta * npart_theta * gspr
				 	   / here->BSIM4v4IdovVds);  /* bugfix */
				  else
				      gdpr = gdpr / (1.0 + npart_theta * npart_theta * gdpr
                                           / here->BSIM4v4IdovVds);
			      }

		              NevalSrc(&noizDens[BSIM4v4RDNOIZ],
				       &lnNdens[BSIM4v4RDNOIZ], ckt, THERMNOISE,
				       here->BSIM4v4dNodePrime, here->BSIM4v4dNode,
				       gdpr);

		              NevalSrc(&noizDens[BSIM4v4RSNOIZ],
				       &lnNdens[BSIM4v4RSNOIZ], ckt, THERMNOISE,
				       here->BSIM4v4sNodePrime, here->BSIM4v4sNode,
				       gspr);


			      if ((here->BSIM4v4rgateMod == 1) || (here->BSIM4v4rgateMod == 2))
			      {   NevalSrc(&noizDens[BSIM4v4RGNOIZ],
                                       &lnNdens[BSIM4v4RGNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v4gNodePrime, here->BSIM4v4gNodeExt,
                                       here->BSIM4v4grgeltd);
			      }
			      else if (here->BSIM4v4rgateMod == 3)
			      {   NevalSrc(&noizDens[BSIM4v4RGNOIZ],
                                       &lnNdens[BSIM4v4RGNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v4gNodeMid, here->BSIM4v4gNodeExt,
                                       here->BSIM4v4grgeltd);
			      }
			      else
			      {    noizDens[BSIM4v4RGNOIZ] = 0.0;
                                   lnNdens[BSIM4v4RGNOIZ] =
                                          log(MAX(noizDens[BSIM4v4RGNOIZ], N_MINLOG));
			      }


                              if (here->BSIM4v4rbodyMod)
                              {   NevalSrc(&noizDens[BSIM4v4RBPSNOIZ],
                                       &lnNdens[BSIM4v4RBPSNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v4bNodePrime, here->BSIM4v4sbNode,
                                       here->BSIM4v4grbps);
                                  NevalSrc(&noizDens[BSIM4v4RBPDNOIZ],
                                       &lnNdens[BSIM4v4RBPDNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v4bNodePrime, here->BSIM4v4dbNode,
                                       here->BSIM4v4grbpd);
                                  NevalSrc(&noizDens[BSIM4v4RBPBNOIZ],
                                       &lnNdens[BSIM4v4RBPBNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v4bNodePrime, here->BSIM4v4bNode,
                                       here->BSIM4v4grbpb);
                                  NevalSrc(&noizDens[BSIM4v4RBSBNOIZ],
                                       &lnNdens[BSIM4v4RBSBNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v4bNode, here->BSIM4v4sbNode,
                                       here->BSIM4v4grbsb);
                                  NevalSrc(&noizDens[BSIM4v4RBDBNOIZ],
                                       &lnNdens[BSIM4v4RBDBNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v4bNode, here->BSIM4v4dbNode,
                                       here->BSIM4v4grbdb);
                              }
                              else
                              {   noizDens[BSIM4v4RBPSNOIZ] = noizDens[BSIM4v4RBPDNOIZ] = 0.0;
                                  noizDens[BSIM4v4RBPBNOIZ] = 0.0;
                                  noizDens[BSIM4v4RBSBNOIZ] = noizDens[BSIM4v4RBDBNOIZ] = 0.0;
                                  lnNdens[BSIM4v4RBPSNOIZ] =
                                          log(MAX(noizDens[BSIM4v4RBPSNOIZ], N_MINLOG));
                                  lnNdens[BSIM4v4RBPDNOIZ] =
                                          log(MAX(noizDens[BSIM4v4RBPDNOIZ], N_MINLOG));
                                  lnNdens[BSIM4v4RBPBNOIZ] =
                                          log(MAX(noizDens[BSIM4v4RBPBNOIZ], N_MINLOG));
                                  lnNdens[BSIM4v4RBSBNOIZ] =
                                          log(MAX(noizDens[BSIM4v4RBSBNOIZ], N_MINLOG));
                                  lnNdens[BSIM4v4RBDBNOIZ] =
                                          log(MAX(noizDens[BSIM4v4RBDBNOIZ], N_MINLOG));
                              }


                              switch(model->BSIM4v4tnoiMod)
			      {  case 0:
				      T0 = here->BSIM4v4ueff * fabs(here->BSIM4v4qinv);
				      T1 = T0 * tmp + pParam->BSIM4v4leff
                                         * pParam->BSIM4v4leff;
		                      NevalSrc(&noizDens[BSIM4v4IDNOIZ],
				               &lnNdens[BSIM4v4IDNOIZ], ckt,
					       THERMNOISE, here->BSIM4v4dNodePrime,
                                               here->BSIM4v4sNodePrime,
					       (T0 / T1) * model->BSIM4v4ntnoi);
				      break;
				 case 1:
				      T0 = here->BSIM4v4gm + here->BSIM4v4gmbs + here->BSIM4v4gds;
				      T0 *= T0;
				      igsquare = npart_theta * npart_theta * T0 / here->BSIM4v4IdovVds;
				      T1 = npart_beta * (here->BSIM4v4gm
					 + here->BSIM4v4gmbs) + here->BSIM4v4gds;
				      T2 = T1 * T1 / here->BSIM4v4IdovVds;
                                      NevalSrc(&noizDens[BSIM4v4IDNOIZ],
                                               &lnNdens[BSIM4v4IDNOIZ], ckt,
                                               THERMNOISE, here->BSIM4v4dNodePrime,
                                               here->BSIM4v4sNodePrime, (T2 - igsquare));
                                      break;
			      }

		              NevalSrc(&noizDens[BSIM4v4FLNOIZ], NULL,
				       ckt, N_GAIN, here->BSIM4v4dNodePrime,
				       here->BSIM4v4sNodePrime, (double) 0.0);

                              switch(model->BSIM4v4fnoiMod)
			      {  case 0:
			              noizDens[BSIM4v4FLNOIZ] *= model->BSIM4v4kf
					    * exp(model->BSIM4v4af
					    * log(MAX(fabs(here->BSIM4v4cd),
					    N_MINLOG)))
					    / (pow(data->freq, model->BSIM4v4ef)
					    * pParam->BSIM4v4leff
				            * pParam->BSIM4v4leff
					    * model->BSIM4v4coxe);
				      break;
			         case 1:
		                      Vds = *(ckt->CKTstates[0] + here->BSIM4v4vds);
			              if (Vds < 0.0)
			                  Vds = -Vds;

                                      Ssi = Eval1ovFNoise(Vds, model, here,
                                          data->freq, ckt->CKTtemp);
                                      T10 = model->BSIM4v4oxideTrapDensityA
                                          * CONSTboltz * ckt->CKTtemp;
                                      T11 = pParam->BSIM4v4weff * pParam->BSIM4v4leff
                                          * pow(data->freq, model->BSIM4v4ef) * 1.0e10
					  * here->BSIM4v4nstar * here->BSIM4v4nstar;
                                      Swi = T10 / T11 * here->BSIM4v4cd
                                          * here->BSIM4v4cd;
                                      T1 = Swi + Ssi;
                                      if (T1 > 0.0)
                                          noizDens[BSIM4v4FLNOIZ] *= (Ssi * Swi) / T1;
                                      else
                                          noizDens[BSIM4v4FLNOIZ] *= 0.0;
				      break;
			      }

		              lnNdens[BSIM4v4FLNOIZ] =
				     log(MAX(noizDens[BSIM4v4FLNOIZ], N_MINLOG));


                       	if(here->BSIM4v4mode >= 0) {  /* bugfix  */
                              NevalSrc(&noizDens[BSIM4v4IGSNOIZ],
                                   &lnNdens[BSIM4v4IGSNOIZ], ckt, SHOTNOISE,
                                   here->BSIM4v4gNodePrime, here->BSIM4v4sNodePrime,
                                   (here->BSIM4v4Igs + here->BSIM4v4Igcs));
                              NevalSrc(&noizDens[BSIM4v4IGDNOIZ],
                                   &lnNdens[BSIM4v4IGDNOIZ], ckt, SHOTNOISE,
                                   here->BSIM4v4gNodePrime, here->BSIM4v4dNodePrime,
                                   (here->BSIM4v4Igd + here->BSIM4v4Igcd));
			} else {
                              NevalSrc(&noizDens[BSIM4v4IGSNOIZ],
                                   &lnNdens[BSIM4v4IGSNOIZ], ckt, SHOTNOISE,
                                   here->BSIM4v4gNodePrime, here->BSIM4v4sNodePrime,
                                   (here->BSIM4v4Igs + here->BSIM4v4Igcd));
                              NevalSrc(&noizDens[BSIM4v4IGDNOIZ],
                                   &lnNdens[BSIM4v4IGDNOIZ], ckt, SHOTNOISE,
                                   here->BSIM4v4gNodePrime, here->BSIM4v4dNodePrime,
                                   (here->BSIM4v4Igd + here->BSIM4v4Igcs));
                        }
                              NevalSrc(&noizDens[BSIM4v4IGBNOIZ],
                                   &lnNdens[BSIM4v4IGBNOIZ], ckt, SHOTNOISE,
                                   here->BSIM4v4gNodePrime, here->BSIM4v4bNodePrime,
                                   here->BSIM4v4Igb);


		              noizDens[BSIM4v4TOTNOIZ] = noizDens[BSIM4v4RDNOIZ]
				     + noizDens[BSIM4v4RSNOIZ] + noizDens[BSIM4v4RGNOIZ]
				     + noizDens[BSIM4v4RBPSNOIZ] + noizDens[BSIM4v4RBPDNOIZ]
				     + noizDens[BSIM4v4RBPBNOIZ]
				     + noizDens[BSIM4v4RBSBNOIZ] + noizDens[BSIM4v4RBDBNOIZ]
				     + noizDens[BSIM4v4IDNOIZ] + noizDens[BSIM4v4FLNOIZ]
                                     + noizDens[BSIM4v4IGSNOIZ] + noizDens[BSIM4v4IGDNOIZ]
                                     + noizDens[BSIM4v4IGBNOIZ];
		              lnNdens[BSIM4v4TOTNOIZ] =
				     log(MAX(noizDens[BSIM4v4TOTNOIZ], N_MINLOG));

		              *OnDens += noizDens[BSIM4v4TOTNOIZ];

		              if (data->delFreq == 0.0)
			      {   /* if we haven't done any previous
				     integration, we need to initialize our
				     "history" variables.
				    */

			          for (i = 0; i < BSIM4v4NSRCS; i++)
				  {    here->BSIM4v4nVar[LNLSTDENS][i] =
					     lnNdens[i];
			          }

			          /* clear out our integration variables
				     if it's the first pass
				   */
			          if (data->freq ==
				      job->NstartFreq)
				  {   for (i = 0; i < BSIM4v4NSRCS; i++)
				      {    here->BSIM4v4nVar[OUTNOIZ][i] = 0.0;
				           here->BSIM4v4nVar[INNOIZ][i] = 0.0;
			              }
			          }
		              }
			      else
			      {   /* data->delFreq != 0.0,
				     we have to integrate.
				   */
			          for (i = 0; i < BSIM4v4NSRCS; i++)
				  {    if (i != BSIM4v4TOTNOIZ)
				       {   tempOnoise = Nintegrate(noizDens[i],
						lnNdens[i],
				                here->BSIM4v4nVar[LNLSTDENS][i],
						data);
				           tempInoise = Nintegrate(noizDens[i]
						* data->GainSqInv, lnNdens[i]
						+ data->lnGainInv,
				                here->BSIM4v4nVar[LNLSTDENS][i]
						+ data->lnGainInv, data);
				           here->BSIM4v4nVar[LNLSTDENS][i] =
						lnNdens[i];
				           data->outNoiz += tempOnoise;
				           data->inNoise += tempInoise;
				           if (job->NStpsSm != 0)
					   {   here->BSIM4v4nVar[OUTNOIZ][i]
						     += tempOnoise;
				               here->BSIM4v4nVar[OUTNOIZ][BSIM4v4TOTNOIZ]
						     += tempOnoise;
				               here->BSIM4v4nVar[INNOIZ][i]
						     += tempInoise;
				               here->BSIM4v4nVar[INNOIZ][BSIM4v4TOTNOIZ]
						     += tempInoise;
                                           }
			               }
			          }
		              }
		              if (data->prtSummary)
			      {   for (i = 0; i < BSIM4v4NSRCS; i++)
				  {    /* print a summary report */
			               data->outpVector[data->outNumber++]
					     = noizDens[i];
			          }
		              }
		              break;
		         case INT_NOIZ:
			      /* already calculated, just output */
		              if (job->NStpsSm != 0)
			      {   for (i = 0; i < BSIM4v4NSRCS; i++)
				  {    data->outpVector[data->outNumber++]
					     = here->BSIM4v4nVar[OUTNOIZ][i];
			               data->outpVector[data->outNumber++]
					     = here->BSIM4v4nVar[INNOIZ][i];
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
