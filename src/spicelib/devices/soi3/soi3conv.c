/**********
STAG version 2.7
Copyright 2000 owned by the United Kingdom Secretary of State for Defence
acting through the Defence Evaluation and Research Agency.
Developed by :     Jim Benson,
                   Department of Electronics and Computer Science,
                   University of Southampton,
                   United Kingdom.
With help from :   Nele D'Halleweyn, Ketan Mistry, Bill Redman-White, and Craig Easson.

Based on STAG version 2.1
Developed by :     Mike Lee,
With help from :   Bernard Tenbroek, Bill Redman-White, Mike Uren, Chris Edwards
                   and John Bunyan.
Acknowledgements : Rupert Howes and Pete Mole.
**********/

/********** 
Modified by Paolo Nenzi 2002
ngspice integration
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "soi3defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
SOI3convTest(GENmodel *inModel, CKTcircuit *ckt)
{
    SOI3model *model = (SOI3model*)inModel;
    SOI3instance *here;
    double delvbs;
    double delvbd;
    double delvgfs;
    double delvgbs;
    double delvds;
    double delvgfd;
    double delvgbd;
    double deldeltaT;
    double ibhat;
    double idhat;
    double iPthat;
    double vbs;
    double vbd;
    double vgfs;
    double vgbs;
    double vds;
    double deltaT;
    double vgfd;
    double vgbd;
    double vgfdo;
    double vgbdo;
    double tol;
    /* debug stuff */
    /*    FILE *fp,*fopen(); */

    for( ; model != NULL; model = SOI3nextModel(model)) {
        for(here = SOI3instances(model); here!= NULL;
                here = SOI3nextInstance(here)) {

            vbs = model->SOI3type * (
                *(ckt->CKTrhs+here->SOI3bNode) -
                *(ckt->CKTrhs+here->SOI3sNodePrime));
            vgfs = model->SOI3type * (
                *(ckt->CKTrhs+here->SOI3gfNode) -
                *(ckt->CKTrhs+here->SOI3sNodePrime));
            vgbs = model->SOI3type * (
                *(ckt->CKTrhs+here->SOI3gbNode) -
                *(ckt->CKTrhs+here->SOI3sNodePrime));
            vds = model->SOI3type * (
                *(ckt->CKTrhs+here->SOI3dNodePrime) -
                *(ckt->CKTrhs+here->SOI3sNodePrime));
         deltaT = MAX(0,*(ckt->CKTrhs+here->SOI3toutNode));
              /* voltage deltaT is V(tout) wrt thermal ground */
            vbd=vbs-vds;
            vgfd=vgfs-vds;
            vgbd=vgbs-vds;
            vgfdo = *(ckt->CKTstate0 + here->SOI3vgfs) -
                *(ckt->CKTstate0 + here->SOI3vds);
            vgbdo = *(ckt->CKTstate0 + here->SOI3vgbs) -
                *(ckt->CKTstate0 + here->SOI3vds);
            delvbs = vbs - *(ckt->CKTstate0 + here->SOI3vbs);
            delvbd = vbd - *(ckt->CKTstate0 + here->SOI3vbd);
            delvgfs = vgfs - *(ckt->CKTstate0 + here->SOI3vgfs);
            delvgbs = vgbs - *(ckt->CKTstate0 + here->SOI3vgbs);
            delvds = vds - *(ckt->CKTstate0 + here->SOI3vds);
            delvgfd = vgfd-vgfdo;
            delvgbd = vgbd-vgbdo;
            deldeltaT = deltaT - *(ckt->CKTstate0 + here->SOI3deltaT);

            /* these are needed for convergence testing */

                if (here->SOI3mode >= 0) {  /* normal */
                    idhat=
                        here->SOI3id-
                        here->SOI3gbd * delvbd -
                        here->SOI3gbdT * deldeltaT +  /* for -ibd bit of id */
                        (here->SOI3gmbs +
                        here->SOI3gMmbs) * delvbs +
                        (here->SOI3gmf +
                        here->SOI3gMmf) * delvgfs +
                        (here->SOI3gmb +
                        here->SOI3gMmb) * delvgbs +
                        (here->SOI3gds +
                        here->SOI3gMd) * delvds +
                        (here->SOI3gt +
                        here->SOI3gMdeltaT) * deldeltaT +
                        here->SOI3gBJTdb_bs * delvbs +
                        here->SOI3gBJTdb_deltaT * deldeltaT;
                    ibhat=
                        here->SOI3ibs +
                        here->SOI3ibd +
                        here->SOI3gbd * delvbd +
                        here->SOI3gbdT * deldeltaT +
                        here->SOI3gbs * delvbs +
                        here->SOI3gbsT * deldeltaT -
                        here->SOI3iMdb -
                        here->SOI3gMmbs * delvbs -
                        (here->SOI3gMmf)* delvgfs -
                        (here->SOI3gMmb)* delvgbs -
                        here->SOI3gMd * delvds -
                        here->SOI3gMdeltaT * deldeltaT -
                        here->SOI3iBJTsb -
                        here->SOI3gBJTsb_bd * delvbd -
                        here->SOI3gBJTsb_deltaT * deldeltaT -
                        here->SOI3iBJTdb -
                        here->SOI3gBJTdb_bs * delvbs -
                        here->SOI3gBJTdb_deltaT * deldeltaT;
                } else {                   /* A over T */
                    idhat=
                        here->SOI3id -
                        ( here->SOI3gbd +
                        here->SOI3gmbs) * delvbd -
                        (here->SOI3gmf) * delvgfd -
                        (here->SOI3gmb) * delvgbd +
                        (here->SOI3gds) * delvds -
                        (here->SOI3gt +
                         here->SOI3gbdT) * deldeltaT +
                        here->SOI3gBJTdb_bs * delvbs +
                        here->SOI3gBJTdb_deltaT * deldeltaT;
                    ibhat=
                        here->SOI3ibs +
                        here->SOI3ibd +
                        here->SOI3gbd * delvbd +
                        here->SOI3gbdT * deldeltaT +
                        here->SOI3gbs * delvbs  +
                        here->SOI3gbsT * deldeltaT -
                        here->SOI3iMsb -
                        here->SOI3gMmbs * delvbd -
                        here->SOI3gMmf * delvgfd -
                        here->SOI3gMmb * delvgbd +
                        here->SOI3gMd * delvds -   /* gMd should go with vsd */
                        here->SOI3gMdeltaT * deldeltaT -
                        here->SOI3iBJTsb -
                        here->SOI3gBJTsb_bd * delvbd -
                        here->SOI3gBJTsb_deltaT * deldeltaT -
                        here->SOI3iBJTdb -
                        here->SOI3gBJTdb_bs * delvbs -
                        here->SOI3gBJTdb_deltaT * deldeltaT;
                }
                iPthat =
                        here->SOI3iPt +
                        here->SOI3gPmbs * delvbs +
                        here->SOI3gPmf  * delvgfs +
                        here->SOI3gPmb  * delvgbs +
                        here->SOI3gPds  * delvds * here->SOI3mode +
                        here->SOI3gPdT  * deldeltaT;
            /*
             *  check convergence
             */
            tol=ckt->CKTreltol*MAX(fabs(idhat),fabs(here->SOI3id))+
                    ckt->CKTabstol;
            if (fabs(idhat-here->SOI3id) >= tol) {
                ckt->CKTnoncon++;

		/* JimB - Remove line containing ckt->CKTtroubleElt for the */
                /* Simetrix DLL version - element removed from ckt structure */
                ckt->CKTtroubleElt = (GENinstance *) here;

                return(OK); /* no reason to continue, we haven't converged */
            } else {
                tol=ckt->CKTreltol*
                        MAX(fabs(ibhat),fabs(here->SOI3ibs+here->SOI3ibd
                        - here->SOI3iMdb - here->SOI3iMsb
                        - here->SOI3iBJTdb - here->SOI3iBJTsb))+
                        ckt->CKTabstol;
                if (fabs(ibhat-(here->SOI3ibs+here->SOI3ibd
                        - here->SOI3iMdb - here->SOI3iMsb
                        - here->SOI3iBJTdb - here->SOI3iBJTsb)) > tol) {
                    ckt->CKTnoncon++;

		/* JimB - Remove line containing ckt->CKTtroubleElt for the */
		/* Simetrix DLL version - element removed from ckt structure */
                    ckt->CKTtroubleElt = (GENinstance *) here;
                    return(OK); /* no reason to continue,we haven't converged*/
                } else {
                  tol=ckt->CKTreltol*MAX(fabs(iPthat),
                         fabs(here->SOI3iPt))+ckt->CKTabstol;
                    if (fabs(iPthat-here->SOI3iPt) >= tol) {
                      ckt->CKTnoncon++;

		   /* JimB - Remove line containing ckt->CKTtroubleElt for the */
                      /* Simetrix DLL version - element removed from ckt structure */
                      ckt->CKTtroubleElt = (GENinstance *) here;
                      return(OK); /* no reason to continue,we haven't converged*/
                    }
                }
            }
/* debug stuff */
/*            fp=fopen("level3.dat","a");
            fprintf(fp,"%2.3f %2.3f %.15e %.15e %.15e %.15e %.15e %.15e %.15e\n",
                    vgfs-vbs,vds,ckt->CKTtime,here->SOI3debug1,here->SOI3debug2,here->SOI3debug3,
                    here->SOI3debug4,here->SOI3debug5,here->SOI3debug6);
            fclose(fp);
*/
        }
    }
    return(OK);
}
