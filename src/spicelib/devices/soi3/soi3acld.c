/**********
STAG version 2.6
Copyright 2000 owned by the United Kingdom Secretary of State for Defence
acting through the Defence Evaluation and Research Agency.
Developed by :     Jim Benson,
                   Department of Electronics and Computer Science,
                   University of Southampton,
                   United Kingdom.
With help from :   Nele D'Halleweyn, Bill Redman-White, and Craig Easson.

Based on STAG version 2.1
Developed by :     Mike Lee,
With help from :   Bernard Tenbroek, Bill Redman-White, Mike Uren, Chris Edwards
                   and John Bunyan.
Acknowledgements : Rupert Howes and Pete Mole.
**********/

/* Modified: 2001 Paolo Nenzi */

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "soi3defs.h"
#include "sperror.h"
#include "suffix.h"


int
SOI3acLoad(inModel,ckt)
    GENmodel *inModel;
    CKTcircuit *ckt;
{
    SOI3model *model = (SOI3model*)inModel;
    SOI3instance *here;
    int xnrm;
    int xrev;

    double cgfgf,cgfd,cgfs,cgfdeltaT;
    double cdgf,cdd,cds,cddeltaT;
    double csgf,csd,css,csdeltaT;
    double cbgf,cbd,cbs,cbdeltaT,cbgb;
    double cgbgb,cgbsb,cgbdb;

    double xcgfgf,xcgfd,xcgfs,xcgfdeltaT;
    double xcdgf,xcdd,xcds,xcddeltaT;
    double xcsgf,xcsd,xcss,xcsdeltaT;
    double xcbgf,xcbd,xcbs,xcbdeltaT,xcbgb;
    double xcgbgb,xcgbsb,xcgbdb;

    double capbd,capbs; /* diode capacitances */

    double xcBJTbsbs,xcBJTbsdeltaT;
    double xcBJTbdbd,xcBJTbddeltaT;

    double rtargs[5];
    double grt[5];
    double ctargs[5];
    double xct[5]; /* 1/reactance of thermal cap */
    int tnodeindex;

    double cgfb0;
    double cgfd0;
    double cgfs0;

    double cgbb0;
    double cgbd0;
    double cgbs0;

    double xcgbd0,xcgbs0;

    double EffectiveLength;

    double omega;

    omega = ckt->CKTomega;

    for( ; model != NULL; model = model->SOI3nextModel)
    {
        for(here = model->SOI3instances; here!= NULL;
                here = here->SOI3nextInstance)
        {
        
            if (here->SOI3mode < 0)
            {
                xnrm=0;
                xrev=1;
            }
            else
            {
                xnrm=1;
                xrev=0;
            }

            EffectiveLength=here->SOI3l - 2*model->SOI3latDiff;
            cgfs0 = model->SOI3frontGateSourceOverlapCapFactor * here->SOI3w;
            cgfd0 = model->SOI3frontGateDrainOverlapCapFactor * here->SOI3w;
            cgfb0 = model->SOI3frontGateBulkOverlapCapFactor * EffectiveLength;

            /* JimB - can use basic device geometry to calculate source/back gate  */
            /* and drain/back gate capacitances to a first approximation.  Use this*/
            /* default model if capacitance factors aren't given in model netlist. */

        		if(model->SOI3backGateSourceOverlapCapFactorGiven)
            {
               cgbs0 = model->SOI3backGateSourceOverlapCapFactor * here->SOI3w;
        		}
            else
            {
               /* As a basic circuit designers approximation, length of drain and  */
               /* source regions often taken to have length of twice the minimum   */
               /* feature size for that technology. */
               cgbs0 = 2*1e-6*model->SOI3minimumFeatureSize * here->SOI3w
               			* model->SOI3backOxideCapFactor;
            }
        		if(model->SOI3backGateDrainOverlapCapFactorGiven)
            {
               cgbd0 = model->SOI3backGateDrainOverlapCapFactor * here->SOI3w;
        		}
            else
            {
               /* As a basic circuit designer's approximation, length of drain and */
               /* source regions often taken to have length of twice the minimum   */
               /* feature size for that technology. */
               cgbd0 = 2*1e-6*model->SOI3minimumFeatureSize * here->SOI3w
               			* model->SOI3backOxideCapFactor;
            }
            cgbb0 = model->SOI3backGateBulkOverlapCapFactor * EffectiveLength;

            capbd = here->SOI3capbd;
            capbs = here->SOI3capbs;

            cgfgf = *(ckt->CKTstate0 + here->SOI3cgfgf);
            cgfd  = *(ckt->CKTstate0 + here->SOI3cgfd);
            cgfs  = *(ckt->CKTstate0 + here->SOI3cgfs);
            cgfdeltaT  = *(ckt->CKTstate0 + here->SOI3cgfdeltaT);
            csgf = *(ckt->CKTstate0 + here->SOI3csgf);
            csd  = *(ckt->CKTstate0 + here->SOI3csd);
            css  = *(ckt->CKTstate0 + here->SOI3css);
            csdeltaT  = *(ckt->CKTstate0 + here->SOI3csdeltaT);
            cdgf = *(ckt->CKTstate0 + here->SOI3cdgf);
            cdd  = *(ckt->CKTstate0 + here->SOI3cdd);
            cds  = *(ckt->CKTstate0 + here->SOI3cds);
            cddeltaT  = *(ckt->CKTstate0 + here->SOI3cddeltaT);
            cgbgb = *(ckt->CKTstate0 + here->SOI3cgbgb);
            cgbsb = *(ckt->CKTstate0 + here->SOI3cgbsb);
            cgbdb = *(ckt->CKTstate0 + here->SOI3cgbdb);
            cbgf = -(cgfgf + cdgf + csgf);
            cbd = -(cgfd + cdd + csd + cgbdb);
            cbs = -(cgfs + cds + css + cgbsb);
            cbdeltaT = -(cgfdeltaT + cddeltaT + csdeltaT);
            cbgb = -cgbgb;

            xcgfgf = (cgfgf + cgfd0 + cgfs0 + cgfb0) * omega;
            xcgfd  = (cgfd - cgfd0) * omega;
            xcgfs  = (cgfs - cgfs0) * omega;
            xcgfdeltaT  = cgfdeltaT * omega;
            xcsgf = (csgf - cgfs0) * omega;
            xcsd  = csd * omega;
            xcss  = (css + capbs + cgfs0) * omega;
            xcsdeltaT  = csdeltaT * omega;
            xcdgf = (cdgf - cgfd0) * omega;
            xcdd  = (cdd + capbd + cgfd0) * omega;
            xcds  = cds * omega;
            xcddeltaT  = cddeltaT * omega;
            xcgbgb = (cgbgb + cgbb0 + cgbd0 + cgbs0) * omega;
            xcgbsb = (cgbsb - cgbs0) * omega;
            xcgbdb = -cgbd0 * omega;
            xcbgf = (cbgf - cgfb0) * omega;
            xcbd = (cbd - capbd) * omega;
            xcbs = (cbs - capbs) * omega;
            xcbdeltaT = cbdeltaT * omega;
            xcbgb = cbgb * omega;

            xcgbs0 = cgbs0 * omega;
            xcgbd0 = cgbd0 * omega;

            xcBJTbsbs = *(ckt->CKTstate0 + here->SOI3cBJTbsbs) * omega;
            xcBJTbsdeltaT = *(ckt->CKTstate0 + here->SOI3cBJTbsdeltaT) * omega;
            xcBJTbdbd = *(ckt->CKTstate0 + here->SOI3cBJTbdbd) * omega;
            xcBJTbddeltaT = *(ckt->CKTstate0 + here->SOI3cBJTbddeltaT) * omega;

            /* JimB - 15/9/99 */
            /* Code for multiple thermal time constants.  Start by moving all */
            /* rt and ct constants into arrays. */
            rtargs[0]=here->SOI3rt;
            rtargs[1]=here->SOI3rt1;
            rtargs[2]=here->SOI3rt2;
            rtargs[3]=here->SOI3rt3;
            rtargs[4]=here->SOI3rt4;

            ctargs[0]=here->SOI3ct;
            ctargs[1]=here->SOI3ct1;
            ctargs[2]=here->SOI3ct2;
            ctargs[3]=here->SOI3ct3;
            ctargs[4]=here->SOI3ct4;

            /* Set all admittance components to zero. */
            grt[0]=grt[1]=grt[2]=grt[3]=grt[4]=0.0;
            xct[0]=xct[1]=xct[2]=xct[3]=xct[4]=0.0;
            /* Now calculate conductances and susceptances from rt and ct. */
            /* Don't need to worry about divide by zero when calculating */
            /* grt components, as soi3setup() only creates a thermal node */
            /* if corresponding rt is greater than zero. */
            for(tnodeindex=0;tnodeindex<here->SOI3numThermalNodes;tnodeindex++)
            {
				   xct[tnodeindex] = ctargs[tnodeindex] * ckt->CKTomega;
               grt[tnodeindex] = 1/rtargs[tnodeindex];
            }
            /* End JimB */

            /*
             *    load matrix
             */

            *(here->SOI3GF_gfPtr + 1) += xcgfgf;
            *(here->SOI3GB_gbPtr + 1) += xcgbgb;
            *(here->SOI3B_bPtr + 1) += -(xcbgf+xcbd+xcbs+xcbgb)
                                       +xcBJTbsbs+xcBJTbdbd;

            *(here->SOI3DP_dpPtr + 1) += xcdd+xcgbd0+xcBJTbdbd;
            *(here->SOI3SP_spPtr + 1) += xcss+xcgbs0+xcBJTbsbs;

            *(here->SOI3GF_dpPtr + 1) += xcgfd;
            *(here->SOI3GF_spPtr + 1) += xcgfs;
            *(here->SOI3GF_bPtr + 1)  -= (xcgfgf + xcgfd + xcgfs);

            *(here->SOI3GB_dpPtr + 1) += xcgbdb;
            *(here->SOI3GB_spPtr + 1) += xcgbsb;
            *(here->SOI3GB_bPtr + 1)  -= (xcgbgb + xcgbdb + xcgbsb);

            *(here->SOI3B_gfPtr + 1) += xcbgf;
            *(here->SOI3B_gbPtr + 1) += xcbgb;
            *(here->SOI3B_dpPtr + 1) += xcbd-xcBJTbdbd;
            *(here->SOI3B_spPtr + 1) += xcbs-xcBJTbsbs;

            *(here->SOI3DP_gfPtr + 1) += xcdgf;
            *(here->SOI3DP_gbPtr + 1) += -xcgbd0;
            *(here->SOI3DP_bPtr + 1) += -(xcdgf + xcdd + xcds + xcBJTbdbd);
            *(here->SOI3DP_spPtr + 1) += xcds;

            *(here->SOI3SP_gfPtr + 1) += xcsgf;
            *(here->SOI3SP_gbPtr + 1) += -xcgbs0;
            *(here->SOI3SP_bPtr + 1) += -(xcsgf + xcsd + xcss + xcBJTbsbs);
            *(here->SOI3SP_dpPtr + 1) += xcsd;
        
/* if no thermal behaviour specified, then put in zero valued indpt. voltage source
   between TOUT and ground */
            if (here->SOI3rt==0)
            {
            	*(here->SOI3TOUT_ibrPtr + 1) += 1.0;
   				*(here->SOI3IBR_toutPtr + 1) += 1.0;
            	*(ckt->CKTirhs + (here->SOI3branch)) = 0;
            }
            else
            {
            	*(here->SOI3TOUT_toutPtr + 1) += xct[0];
            	if (here->SOI3numThermalNodes > 1)
              	{
               	*(here->SOI3TOUT_tout1Ptr + 1) += -xct[0];
               	*(here->SOI3TOUT1_toutPtr + 1) += -xct[0];
               	*(here->SOI3TOUT1_tout1Ptr + 1) += xct[0]+xct[1];
              	}
              	if (here->SOI3numThermalNodes > 2)
              	{
               	*(here->SOI3TOUT1_tout2Ptr + 1) += -xct[1];
               	*(here->SOI3TOUT2_tout1Ptr + 1) += -xct[1];
               	*(here->SOI3TOUT2_tout2Ptr + 1) += xct[1]+xct[2];
              	}
            	if (here->SOI3numThermalNodes > 3)
              	{
               	*(here->SOI3TOUT2_tout3Ptr + 1) += -xct[2];
               	*(here->SOI3TOUT3_tout2Ptr + 1) += -xct[2];
               	*(here->SOI3TOUT3_tout3Ptr + 1) += xct[2]+xct[3];
              	}
              	if (here->SOI3numThermalNodes > 4)
              	{
               	*(here->SOI3TOUT3_tout4Ptr + 1) += -xct[3];
               	*(here->SOI3TOUT4_tout3Ptr + 1) += -xct[3];
               	*(here->SOI3TOUT4_tout4Ptr + 1) += xct[3]+xct[4];
              	}
            	*(here->SOI3GF_toutPtr + 1) += xcgfdeltaT*model->SOI3type;
            	*(here->SOI3DP_toutPtr + 1) += (xcddeltaT - xcBJTbddeltaT)*model->SOI3type;
            	*(here->SOI3SP_toutPtr + 1) += (xcsdeltaT - xcBJTbsdeltaT)*model->SOI3type;
            	*(here->SOI3B_toutPtr + 1) += model->SOI3type*
                                            (xcbdeltaT + xcBJTbsdeltaT + xcBJTbddeltaT);
            }


            /* and now real part */
            *(here->SOI3D_dPtr) += (here->SOI3drainConductance);
            *(here->SOI3S_sPtr) += (here->SOI3sourceConductance);
            *(here->SOI3B_bPtr) += (here->SOI3gbd+here->SOI3gbs -
                                   here->SOI3gMmbs
                                   - here->SOI3gBJTdb_bs - here->SOI3gBJTsb_bd);

            *(here->SOI3DP_dpPtr) +=
                    (here->SOI3drainConductance+here->SOI3gds+
                    here->SOI3gbd+xrev*(here->SOI3gmf+here->SOI3gmbs+
                    here->SOI3gmb)+xnrm*here->SOI3gMd);
            *(here->SOI3SP_spPtr) += 
                    (here->SOI3sourceConductance+here->SOI3gds+
                    here->SOI3gbs+xnrm*(here->SOI3gmf+here->SOI3gmbs+
                    here->SOI3gmb)+xrev*here->SOI3gMd);

            *(here->SOI3D_dpPtr) += (-here->SOI3drainConductance);

            *(here->SOI3S_spPtr) += (-here->SOI3sourceConductance);
            *(here->SOI3B_gfPtr) += -here->SOI3gMmf;
            *(here->SOI3B_gbPtr) += -(here->SOI3gMmb);
            *(here->SOI3B_dpPtr) += -(here->SOI3gbd) + here->SOI3gBJTsb_bd +
                                   xrev*(here->SOI3gMmf+here->SOI3gMmb+
                                         here->SOI3gMmbs+here->SOI3gMd) -
                                   xnrm*here->SOI3gMd;
            *(here->SOI3B_spPtr) += -(here->SOI3gbs) + here->SOI3gBJTdb_bs +
                                   xnrm*(here->SOI3gMmf+here->SOI3gMmb+
                                         here->SOI3gMmbs+here->SOI3gMd) -
                                   xrev*here->SOI3gMd;
            *(here->SOI3DP_dPtr) += (-here->SOI3drainConductance);
            *(here->SOI3SP_sPtr) += (-here->SOI3sourceConductance);

            *(here->SOI3DP_gfPtr) += ((xnrm-xrev)*here->SOI3gmf +
                                      xnrm*here->SOI3gMmf);
            *(here->SOI3DP_gbPtr) += ((xnrm-xrev)*here->SOI3gmb +
                                      xnrm*here->SOI3gMmb);
            *(here->SOI3DP_bPtr) += (-here->SOI3gbd + here->SOI3gBJTdb_bs
                                     +(xnrm-xrev)*here->SOI3gmbs+
                                      xnrm*here->SOI3gMmbs);
            *(here->SOI3DP_spPtr) += (-here->SOI3gds - here->SOI3gBJTdb_bs
                                      -xnrm*(here->SOI3gmf+here->SOI3gmb+here->SOI3gmbs +
                   here->SOI3gMmf+here->SOI3gMmb+here->SOI3gMmbs+here->SOI3gMd));

            *(here->SOI3SP_gfPtr) += (-(xnrm-xrev)*here->SOI3gmf+
                                      xrev*here->SOI3gMmf);
            *(here->SOI3SP_gbPtr) += (-(xnrm-xrev)*here->SOI3gmb+
                                      xrev*here->SOI3gMmb);
            *(here->SOI3SP_bPtr) += (-here->SOI3gbs + here->SOI3gBJTsb_bd
                                     -(xnrm-xrev)*here->SOI3gmbs+
                                      xrev*here->SOI3gMmbs);
            *(here->SOI3SP_dpPtr) += (-here->SOI3gds - here->SOI3gBJTsb_bd
                                      -xrev*(here->SOI3gmf+here->SOI3gmb+here->SOI3gmbs+
                   here->SOI3gMmf+here->SOI3gMmb+here->SOI3gMmbs+here->SOI3gMd));
        
/* if no thermal behaviour specified, then put in zero valued indpt. voltage source
   between TOUT and ground */
            if (here->SOI3rt==0)
            {
              *(here->SOI3TOUT_ibrPtr) += 1.0;
              *(here->SOI3IBR_toutPtr) += 1.0;
              *(ckt->CKTrhs + (here->SOI3branch)) = 0;
            }
            else
            {
            	*(here->SOI3TOUT_toutPtr) += -(here->SOI3gPdT)+grt[0];
            	if (here->SOI3numThermalNodes > 1)
              	{
                	*(here->SOI3TOUT_tout1Ptr) += -grt[0];
                	*(here->SOI3TOUT1_toutPtr) += -grt[0];
                	*(here->SOI3TOUT1_tout1Ptr) += grt[0]+grt[1];
              	}
              	if (here->SOI3numThermalNodes > 2)
              	{
                 	*(here->SOI3TOUT1_tout2Ptr) += -grt[1];
                	*(here->SOI3TOUT2_tout1Ptr) += -grt[1];
                	*(here->SOI3TOUT2_tout2Ptr) += grt[1]+grt[2];
              	}
      			if (here->SOI3numThermalNodes > 3)
              	{
                	*(here->SOI3TOUT2_tout3Ptr) += -grt[2];
                	*(here->SOI3TOUT3_tout2Ptr) += -grt[2];
                	*(here->SOI3TOUT3_tout3Ptr) += grt[2]+grt[3];
              	}
              	if (here->SOI3numThermalNodes > 4)
              	{
                 	*(here->SOI3TOUT3_tout4Ptr) += -grt[3];
                	*(here->SOI3TOUT4_tout3Ptr) += -grt[3];
                	*(here->SOI3TOUT4_tout4Ptr) += grt[3]+grt[4];
              	}

              *(here->SOI3TOUT_dpPtr) += xnrm*(-(here->SOI3gPds*model->SOI3type))
                                        +xrev*(here->SOI3gPds+here->SOI3gPmf+
                                               here->SOI3gPmb+here->SOI3gPmbs)*
                                               model->SOI3type;
              *(here->SOI3TOUT_gfPtr) += -(here->SOI3gPmf*model->SOI3type);
              *(here->SOI3TOUT_gbPtr) += -(here->SOI3gPmb*model->SOI3type);
              *(here->SOI3TOUT_bPtr) += -(here->SOI3gPmbs*model->SOI3type);
              *(here->SOI3TOUT_spPtr) += xnrm*(here->SOI3gPds+here->SOI3gPmf+
                                          here->SOI3gPmb+here->SOI3gPmbs)*model->SOI3type
                                        +xrev*(-(here->SOI3gPds*model->SOI3type));

              *(here->SOI3DP_toutPtr) += (xnrm-xrev)*here->SOI3gt*model->SOI3type;
              *(here->SOI3SP_toutPtr) += (xrev-xnrm)*here->SOI3gt*model->SOI3type;
/* need to mult by type in above as conductances will be used with exterior voltages
  which will be -ve for PMOS except for gPdT */
/* now for thermal influence on impact ionisation current and tranisent stuff */
              *(here->SOI3DP_toutPtr) += (xnrm*here->SOI3gMdeltaT -
                                          here->SOI3gbdT + here->SOI3gBJTdb_deltaT)*model->SOI3type;
              *(here->SOI3SP_toutPtr) += (xrev*here->SOI3gMdeltaT -
                                          here->SOI3gbsT + here->SOI3gBJTsb_deltaT)*model->SOI3type;
              *(here->SOI3B_toutPtr) -= (here->SOI3gMdeltaT - here->SOI3gbsT -
                                         here->SOI3gbdT + here->SOI3gBJTdb_deltaT +
                                         here->SOI3gBJTsb_deltaT)*model->SOI3type;
            }
        }
    }
    return(OK);
}
