/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Hong J. Park, Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim2def.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/devdefs.h"
#include "ngspice/suffix.h"

int
B2load(GENmodel *inModel, CKTcircuit *ckt)

        /* actually load the current value into the 
         * sparse matrix previously provided 
         */
{
    B2model *model = (B2model*)inModel;
    B2instance *here;
    double DrainSatCurrent;
    double EffectiveLength;
    double GateBulkOverlapCap;
    double GateDrainOverlapCap;
    double GateSourceOverlapCap;
    double SourceSatCurrent;
    double DrainArea;
    double SourceArea;
    double DrainPerimeter;
    double SourcePerimeter;
    double arg;
    double capbd = 0.0;
    double capbs = 0.0;
    double cbd;
    double cbhat;
    double cbs;
    double cd;
    double cdrain;
    double cdhat;
    double cdreq;
    double ceq;
    double ceqbd;
    double ceqbs;
    double ceqqb;
    double ceqqd;
    double ceqqg;
    double czbd;
    double czbdsw;
    double czbs;
    double czbssw;
    double delvbd;
    double delvbs;
    double delvds;
    double delvgd;
    double delvgs;
    double evbd;
    double evbs;
    double gbd;
    double gbs;
    double gcbdb;
    double gcbgb;
    double gcbsb;
    double gcddb;
    double gcdgb;
    double gcdsb;
    double gcgdb;
    double gcggb;
    double gcgsb;
    double gcsdb;
    double gcsgb;
    double gcssb;
    double gds;
    double geq;
    double gm;
    double gmbs;
    double sarg;
    double sargsw;
    double vbd;
    double vbs;
    double vcrit;
    double vds;
    double vdsat;
    double vgb;
    double vgd;
    double vgdo;
    double vgs;
    double von;
#ifndef PREDICTOR
    double xfact;
#endif
    double xnrm;
    double xrev;
    int Check;
    double cgdb;
    double cgsb;
    double cbdb;
    double cdgb = 0.0;
    double cddb = 0.0;
    double cdsb = 0.0;
    double cggb;
    double cbgb;
    double cbsb;
    double csgb = 0.0;
    double cssb = 0.0;
    double csdb = 0.0;
    double PhiB;
    double PhiBSW;
    double MJ;
    double MJSW;
    double argsw;
    double qgate;
    double qbulk;
    double qdrn;
    double qsrc;
    double cqgate;
    double cqbulk;
    double cqdrn;
    double vt0;
    double args[8];
    int    ByPass;
#ifndef NOBYPASS    
    double tempv;
#endif /*NOBYPASS*/    
    int error;

    double m;


    /*  loop through all the B2 device models */
    for( ; model != NULL; model = B2nextModel(model)) {

        /* loop through all the instances of the model */
        for (here = B2instances(model); here != NULL ;
                here=B2nextInstance(here)) {
        
            EffectiveLength=here->B2l - model->B2deltaL * 1.e-6;/* m */
            DrainArea = here->B2drainArea;
            SourceArea = here->B2sourceArea;
            DrainPerimeter = here->B2drainPerimeter;
            SourcePerimeter = here->B2sourcePerimeter;
            if( (DrainSatCurrent=DrainArea*model->B2jctSatCurDensity) 
                    < 1e-15){
                DrainSatCurrent = 1.0e-15;
            }
            if( (SourceSatCurrent=SourceArea*model->B2jctSatCurDensity)
                    <1.0e-15){
                SourceSatCurrent = 1.0e-15;
            }
            GateSourceOverlapCap = model->B2gateSourceOverlapCap *here->B2w;
            GateDrainOverlapCap = model->B2gateDrainOverlapCap * here->B2w;
            GateBulkOverlapCap = model->B2gateBulkOverlapCap *EffectiveLength;
            von = model->B2type * here->B2von;
            vdsat = model->B2type * here->B2vdsat;
            vt0 = model->B2type * here->pParam->B2vt0;

            Check=1;
            ByPass = 0;
            if((ckt->CKTmode & MODEINITSMSIG)) {
                vbs= *(ckt->CKTstate0 + here->B2vbs);
                vgs= *(ckt->CKTstate0 + here->B2vgs);
                vds= *(ckt->CKTstate0 + here->B2vds);
            } else if ((ckt->CKTmode & MODEINITTRAN)) {
                vbs= *(ckt->CKTstate1 + here->B2vbs);
                vgs= *(ckt->CKTstate1 + here->B2vgs);
                vds= *(ckt->CKTstate1 + here->B2vds);
            } else if((ckt->CKTmode & MODEINITJCT) && !here->B2off) {
                vds= model->B2type * here->B2icVDS;
                vgs= model->B2type * here->B2icVGS;
                vbs= model->B2type * here->B2icVBS;
                if((vds==0) && (vgs==0) && (vbs==0) && 
                        ((ckt->CKTmode & 
                        (MODETRAN|MODEAC|MODEDCOP|MODEDCTRANCURVE)) ||
                        (!(ckt->CKTmode & MODEUIC)))) {
                    vbs = -1;
                    vgs = vt0;
                    vds = 0;
                }
            } else if((ckt->CKTmode & (MODEINITJCT | MODEINITFIX) ) && 
                    (here->B2off)) {
                vbs=vgs=vds=0;
            } else {
#ifndef PREDICTOR
                if((ckt->CKTmode & MODEINITPRED)) {
                    xfact=ckt->CKTdelta/ckt->CKTdeltaOld[1];
                    *(ckt->CKTstate0 + here->B2vbs) = 
                            *(ckt->CKTstate1 + here->B2vbs);
                    vbs = (1+xfact)* (*(ckt->CKTstate1 + here->B2vbs))
                            -(xfact * (*(ckt->CKTstate2 + here->B2vbs)));
                    *(ckt->CKTstate0 + here->B2vgs) = 
                            *(ckt->CKTstate1 + here->B2vgs);
                    vgs = (1+xfact)* (*(ckt->CKTstate1 + here->B2vgs))
                            -(xfact * (*(ckt->CKTstate2 + here->B2vgs)));
                    *(ckt->CKTstate0 + here->B2vds) = 
                            *(ckt->CKTstate1 + here->B2vds);
                    vds = (1+xfact)* (*(ckt->CKTstate1 + here->B2vds))
                            -(xfact * (*(ckt->CKTstate2 + here->B2vds)));
                    *(ckt->CKTstate0 + here->B2vbd) = 
                            *(ckt->CKTstate0 + here->B2vbs)-
                            *(ckt->CKTstate0 + here->B2vds);
                    *(ckt->CKTstate0 + here->B2cd) = 
                            *(ckt->CKTstate1 + here->B2cd);
                    *(ckt->CKTstate0 + here->B2cbs) = 
                            *(ckt->CKTstate1 + here->B2cbs);
                    *(ckt->CKTstate0 + here->B2cbd) = 
                            *(ckt->CKTstate1 + here->B2cbd);
                    *(ckt->CKTstate0 + here->B2gm) = 
                            *(ckt->CKTstate1 + here->B2gm);
                    *(ckt->CKTstate0 + here->B2gds) = 
                            *(ckt->CKTstate1 + here->B2gds);
                    *(ckt->CKTstate0 + here->B2gmbs) = 
                            *(ckt->CKTstate1 + here->B2gmbs);
                    *(ckt->CKTstate0 + here->B2gbd) = 
                            *(ckt->CKTstate1 + here->B2gbd);
                    *(ckt->CKTstate0 + here->B2gbs) = 
                            *(ckt->CKTstate1 + here->B2gbs);
                    *(ckt->CKTstate0 + here->B2cggb) = 
                            *(ckt->CKTstate1 + here->B2cggb);
                    *(ckt->CKTstate0 + here->B2cbgb) = 
                            *(ckt->CKTstate1 + here->B2cbgb);
                    *(ckt->CKTstate0 + here->B2cbsb) = 
                            *(ckt->CKTstate1 + here->B2cbsb);
                    *(ckt->CKTstate0 + here->B2cgdb) = 
                            *(ckt->CKTstate1 + here->B2cgdb);
                    *(ckt->CKTstate0 + here->B2cgsb) = 
                            *(ckt->CKTstate1 + here->B2cgsb);
                    *(ckt->CKTstate0 + here->B2cbdb) = 
                            *(ckt->CKTstate1 + here->B2cbdb);
                    *(ckt->CKTstate0 + here->B2cdgb) = 
                            *(ckt->CKTstate1 + here->B2cdgb);
                    *(ckt->CKTstate0 + here->B2cddb) = 
                            *(ckt->CKTstate1 + here->B2cddb);
                    *(ckt->CKTstate0 + here->B2cdsb) = 
                            *(ckt->CKTstate1 + here->B2cdsb);
                } else {
#endif /* PREDICTOR */
                    vbs = model->B2type * ( 
                        *(ckt->CKTrhsOld+here->B2bNode) -
                        *(ckt->CKTrhsOld+here->B2sNodePrime));
                    vgs = model->B2type * ( 
                        *(ckt->CKTrhsOld+here->B2gNode) -
                        *(ckt->CKTrhsOld+here->B2sNodePrime));
                    vds = model->B2type * ( 
                        *(ckt->CKTrhsOld+here->B2dNodePrime) -
                        *(ckt->CKTrhsOld+here->B2sNodePrime));
#ifndef PREDICTOR
                }
#endif /* PREDICTOR */
                vbd=vbs-vds;
                vgd=vgs-vds;
                vgdo = *(ckt->CKTstate0 + here->B2vgs) - 
                    *(ckt->CKTstate0 + here->B2vds);
                delvbs = vbs - *(ckt->CKTstate0 + here->B2vbs);
                delvbd = vbd - *(ckt->CKTstate0 + here->B2vbd);
                delvgs = vgs - *(ckt->CKTstate0 + here->B2vgs);
                delvds = vds - *(ckt->CKTstate0 + here->B2vds);
                delvgd = vgd-vgdo;

                if (here->B2mode >= 0) {
                    cdhat=
                        *(ckt->CKTstate0 + here->B2cd) -
                        *(ckt->CKTstate0 + here->B2gbd) * delvbd +
                        *(ckt->CKTstate0 + here->B2gmbs) * delvbs +
                        *(ckt->CKTstate0 + here->B2gm) * delvgs + 
                        *(ckt->CKTstate0 + here->B2gds) * delvds ;
                } else {
                    cdhat=
                        *(ckt->CKTstate0 + here->B2cd) -
                        ( *(ckt->CKTstate0 + here->B2gbd) -
                          *(ckt->CKTstate0 + here->B2gmbs)) * delvbd -
                        *(ckt->CKTstate0 + here->B2gm) * delvgd +
                        *(ckt->CKTstate0 + here->B2gds) * delvds;
                }
                cbhat=
                    *(ckt->CKTstate0 + here->B2cbs) +
                    *(ckt->CKTstate0 + here->B2cbd) +
                    *(ckt->CKTstate0 + here->B2gbd) * delvbd +
                    *(ckt->CKTstate0 + here->B2gbs) * delvbs ;

#ifndef NOBYPASS
                    /* now lets see if we can bypass (ugh) */

                /* following should be one big if connected by && all over
                 * the place, but some C compilers can't handle that, so
                 * we split it up here to let them digest it in stages
                 */
                tempv = MAX(fabs(cbhat),fabs(*(ckt->CKTstate0 + here->B2cbs)
                        + *(ckt->CKTstate0 + here->B2cbd)))+ckt->CKTabstol;
                if((!(ckt->CKTmode & MODEINITPRED)) && (ckt->CKTbypass) )
                if( (fabs(delvbs) < (ckt->CKTreltol * MAX(fabs(vbs),
                        fabs(*(ckt->CKTstate0+here->B2vbs)))+
                        ckt->CKTvoltTol)) )
                if ( (fabs(delvbd) < (ckt->CKTreltol * MAX(fabs(vbd),
                        fabs(*(ckt->CKTstate0+here->B2vbd)))+
                        ckt->CKTvoltTol)) )
                if( (fabs(delvgs) < (ckt->CKTreltol * MAX(fabs(vgs),
                        fabs(*(ckt->CKTstate0+here->B2vgs)))+
                        ckt->CKTvoltTol)))
                if ( (fabs(delvds) < (ckt->CKTreltol * MAX(fabs(vds),
                        fabs(*(ckt->CKTstate0+here->B2vds)))+
                        ckt->CKTvoltTol)) )
                if( (fabs(cdhat- *(ckt->CKTstate0 + here->B2cd)) <
                        ckt->CKTreltol * MAX(fabs(cdhat),fabs(*(ckt->CKTstate0 +
                        here->B2cd))) + ckt->CKTabstol) )
                if ( (fabs(cbhat-(*(ckt->CKTstate0 + here->B2cbs) +
                        *(ckt->CKTstate0 + here->B2cbd))) < ckt->CKTreltol *
                        tempv)) {
                    /* bypass code */
                    vbs = *(ckt->CKTstate0 + here->B2vbs);
                    vbd = *(ckt->CKTstate0 + here->B2vbd);
                    vgs = *(ckt->CKTstate0 + here->B2vgs);
                    vds = *(ckt->CKTstate0 + here->B2vds);
                    vgd = vgs - vds;
                    vgb = vgs - vbs;
                    cd = *(ckt->CKTstate0 + here->B2cd);
                    cbs = *(ckt->CKTstate0 + here->B2cbs);
                    cbd = *(ckt->CKTstate0 + here->B2cbd);
                    cdrain = here->B2mode * (cd + cbd);
                    gm = *(ckt->CKTstate0 + here->B2gm);
                    gds = *(ckt->CKTstate0 + here->B2gds);
                    gmbs = *(ckt->CKTstate0 + here->B2gmbs);
                    gbd = *(ckt->CKTstate0 + here->B2gbd);
                    gbs = *(ckt->CKTstate0 + here->B2gbs);
                    if((ckt->CKTmode & (MODETRAN | MODEAC)) || 
                            ((ckt->CKTmode & MODETRANOP) && 
                            (ckt->CKTmode & MODEUIC))) {
                        cggb = *(ckt->CKTstate0 + here->B2cggb);
                        cgdb = *(ckt->CKTstate0 + here->B2cgdb);
                        cgsb = *(ckt->CKTstate0 + here->B2cgsb);
                        cbgb = *(ckt->CKTstate0 + here->B2cbgb);
                        cbdb = *(ckt->CKTstate0 + here->B2cbdb);
                        cbsb = *(ckt->CKTstate0 + here->B2cbsb);
                        cdgb = *(ckt->CKTstate0 + here->B2cdgb);
                        cddb = *(ckt->CKTstate0 + here->B2cddb);
                        cdsb = *(ckt->CKTstate0 + here->B2cdsb);
                        capbs = *(ckt->CKTstate0 + here->B2capbs);
                        capbd = *(ckt->CKTstate0 + here->B2capbd);
                        ByPass = 1;
                        goto line755;
                    } else {
                        goto line850;
                    }
                }
#endif /*NOBYPASS*/

                von = model->B2type * here->B2von;
                if(*(ckt->CKTstate0 + here->B2vds) >=0) {
                    vgs = DEVfetlim(vgs,*(ckt->CKTstate0 + here->B2vgs)
                            ,von);
                    vds = vgs - vgd;
                    vds = DEVlimvds(vds,*(ckt->CKTstate0 + here->B2vds));
                    vgd = vgs - vds;
                } else {
                    vgd = DEVfetlim(vgd,vgdo,von);
                    vds = vgs - vgd;
                    vds = -DEVlimvds(-vds,-(*(ckt->CKTstate0 + 
                            here->B2vds)));
                    vgs = vgd + vds;
                }
                if(vds >= 0) {
                    vcrit = CONSTvt0 *log(CONSTvt0/(CONSTroot2*SourceSatCurrent));
                    vbs = DEVpnjlim(vbs,*(ckt->CKTstate0 + here->B2vbs),
                            CONSTvt0,vcrit,&Check); /* B2 test */
                    vbd = vbs-vds;
                } else {
                    vcrit = CONSTvt0 * log(CONSTvt0/(CONSTroot2*DrainSatCurrent));
                    vbd = DEVpnjlim(vbd,*(ckt->CKTstate0 + here->B2vbd),
                            CONSTvt0,vcrit,&Check); /* B2 test*/
                    vbs = vbd + vds;
                }
            } 

             /* determine DC current and derivatives */
            vbd = vbs - vds;
            vgd = vgs - vds;
            vgb = vgs - vbs;


            if(vbs <= 0.0 ) {
                gbs = SourceSatCurrent / CONSTvt0 + ckt->CKTgmin;
                cbs = gbs * vbs ;
            } else {
                evbs = exp(vbs/CONSTvt0);
                gbs = SourceSatCurrent*evbs/CONSTvt0 + ckt->CKTgmin;
                cbs = SourceSatCurrent * (evbs-1) + ckt->CKTgmin * vbs ;
            }
            if(vbd <= 0.0) {
                gbd = DrainSatCurrent / CONSTvt0 + ckt->CKTgmin;
                cbd = gbd * vbd ;
            } else {
                evbd = exp(vbd/CONSTvt0);
                gbd = DrainSatCurrent*evbd/CONSTvt0 +ckt->CKTgmin;
                cbd = DrainSatCurrent *(evbd-1)+ckt->CKTgmin*vbd;
            }
            /* line 400 */
            if(vds >= 0) {
                /* normal mode */
                here->B2mode = 1;
            } else {
                /* inverse mode */
                here->B2mode = -1;
            }
            /* call B2evaluate to calculate drain current and its 
             * derivatives and charge and capacitances related to gate
             * drain, and bulk
             */
           if( vds >= 0 )  {
                B2evaluate(vds,vbs,vgs,here,model,&gm,&gds,&gmbs,&qgate,
                    &qbulk,&qdrn,&cggb,&cgdb,&cgsb,&cbgb,&cbdb,&cbsb,&cdgb,
                    &cddb,&cdsb,&cdrain,&von,&vdsat,ckt);
            } else {
                B2evaluate(-vds,vbd,vgd,here,model,&gm,&gds,&gmbs,&qgate,
                    &qbulk,&qsrc,&cggb,&cgsb,&cgdb,&cbgb,&cbsb,&cbdb,&csgb,
                    &cssb,&csdb,&cdrain,&von,&vdsat,ckt);
            }
          
            here->B2von = model->B2type * von;
            here->B2vdsat = model->B2type * vdsat;  

        

            /*
             *  COMPUTE EQUIVALENT DRAIN CURRENT SOURCE
             */
            cd=here->B2mode * cdrain - cbd;
            if ((ckt->CKTmode & (MODETRAN | MODEAC | MODEINITSMSIG)) ||
                    ((ckt->CKTmode & MODETRANOP ) && 
                    (ckt->CKTmode & MODEUIC))) {
                /*
                 *  charge storage elements
                 *
                 *   bulk-drain and bulk-source depletion capacitances
                 *  czbd : zero bias drain junction capacitance
                 *  czbs : zero bias source junction capacitance
                 * czbdsw:zero bias drain junction sidewall capacitance
                 * czbssw:zero bias source junction sidewall capacitance
                 */

                czbd  = model->B2unitAreaJctCap * DrainArea;
                czbs  = model->B2unitAreaJctCap * SourceArea;
                czbdsw= model->B2unitLengthSidewallJctCap * DrainPerimeter;
                czbssw= model->B2unitLengthSidewallJctCap * SourcePerimeter;
                PhiB = model->B2bulkJctPotential;
                PhiBSW = model->B2sidewallJctPotential;
                MJ = model->B2bulkJctBotGradingCoeff;
                MJSW = model->B2bulkJctSideGradingCoeff;

                /* Source Bulk Junction */
                if( vbs < 0 ) {  
                    arg = 1 - vbs / PhiB;
                    argsw = 1 - vbs / PhiBSW;
                    sarg = exp(-MJ*log(arg));
                    sargsw = exp(-MJSW*log(argsw));
                    *(ckt->CKTstate0 + here->B2qbs) =
                        PhiB * czbs * (1-arg*sarg)/(1-MJ) + PhiBSW * 
                    czbssw * (1-argsw*sargsw)/(1-MJSW);
                    capbs = czbs * sarg + czbssw * sargsw ;
                } else {  
                    *(ckt->CKTstate0+here->B2qbs) =
                        vbs*(czbs+czbssw)+ vbs*vbs*(czbs*MJ*0.5/PhiB 
                        + czbssw * MJSW * 0.5/PhiBSW);
                    capbs = czbs + czbssw + vbs *(czbs*MJ/PhiB+
                        czbssw * MJSW / PhiBSW );
                }

                /* Drain Bulk Junction */
                if( vbd < 0 ) {  
                    arg = 1 - vbd / PhiB;
                    argsw = 1 - vbd / PhiBSW;
                    sarg = exp(-MJ*log(arg));
                    sargsw = exp(-MJSW*log(argsw));
                    *(ckt->CKTstate0 + here->B2qbd) =
                        PhiB * czbd * (1-arg*sarg)/(1-MJ) + PhiBSW * 
                    czbdsw * (1-argsw*sargsw)/(1-MJSW);
                    capbd = czbd * sarg + czbdsw * sargsw ;
                } else {  
                    *(ckt->CKTstate0+here->B2qbd) =
                        vbd*(czbd+czbdsw)+ vbd*vbd*(czbd*MJ*0.5/PhiB 
                        + czbdsw * MJSW * 0.5/PhiBSW);
                    capbd = czbd + czbdsw + vbd *(czbd*MJ/PhiB+
                        czbdsw * MJSW / PhiBSW );
                }

            }




            /*
             *  check convergence
             */
            if ( (here->B2off == 0)  || (!(ckt->CKTmode & MODEINITFIX)) ){
                if (Check == 1) {
                    ckt->CKTnoncon++;
		    ckt->CKTtroubleElt = (GENinstance *) here;
                }
            }
            *(ckt->CKTstate0 + here->B2vbs) = vbs;
            *(ckt->CKTstate0 + here->B2vbd) = vbd;
            *(ckt->CKTstate0 + here->B2vgs) = vgs;
            *(ckt->CKTstate0 + here->B2vds) = vds;
            *(ckt->CKTstate0 + here->B2cd) = cd;
            *(ckt->CKTstate0 + here->B2cbs) = cbs;
            *(ckt->CKTstate0 + here->B2cbd) = cbd;
            *(ckt->CKTstate0 + here->B2gm) = gm;
            *(ckt->CKTstate0 + here->B2gds) = gds;
            *(ckt->CKTstate0 + here->B2gmbs) = gmbs;
            *(ckt->CKTstate0 + here->B2gbd) = gbd;
            *(ckt->CKTstate0 + here->B2gbs) = gbs;

            *(ckt->CKTstate0 + here->B2cggb) = cggb;
            *(ckt->CKTstate0 + here->B2cgdb) = cgdb;
            *(ckt->CKTstate0 + here->B2cgsb) = cgsb;

            *(ckt->CKTstate0 + here->B2cbgb) = cbgb;
            *(ckt->CKTstate0 + here->B2cbdb) = cbdb;
            *(ckt->CKTstate0 + here->B2cbsb) = cbsb;

            *(ckt->CKTstate0 + here->B2cdgb) = cdgb;
            *(ckt->CKTstate0 + here->B2cddb) = cddb;
            *(ckt->CKTstate0 + here->B2cdsb) = cdsb;

            *(ckt->CKTstate0 + here->B2capbs) = capbs;
            *(ckt->CKTstate0 + here->B2capbd) = capbd;

           /* bulk and channel charge plus overlaps */

            if((!(ckt->CKTmode & (MODETRAN | MODEAC))) &&
                    ((!(ckt->CKTmode & MODETRANOP)) ||
                    (!(ckt->CKTmode & MODEUIC)))  && (!(ckt->CKTmode 
                    &  MODEINITSMSIG))) goto line850; 
#ifndef NOBYPASS         
line755:
#endif
            if( here->B2mode > 0 ) {

		args[0] = GateDrainOverlapCap;
		args[1] = GateSourceOverlapCap;
		args[2] = GateBulkOverlapCap;
		args[3] = capbd;
		args[4] = capbs;
		args[5] = cggb;
		args[6] = cgdb;
		args[7] = cgsb;

                B2mosCap(ckt,vgd,vgs,vgb,
		    args,
		    /*
		    GateDrainOverlapCap,
		    GateSourceOverlapCap,GateBulkOverlapCap,
		    capbd,capbs,cggb,cgdb,cgsb,
		    */
		    cbgb,cbdb,cbsb,cdgb,cddb,cdsb
		    ,&gcggb,&gcgdb,&gcgsb,&gcbgb,&gcbdb,&gcbsb,&gcdgb
		    ,&gcddb,&gcdsb,&gcsgb,&gcsdb,&gcssb,&qgate,&qbulk
		    ,&qdrn,&qsrc);
            } else {

		args[0] = GateSourceOverlapCap;
		args[1] = GateDrainOverlapCap;
		args[2] = GateBulkOverlapCap;
		args[3] = capbs;
		args[4] = capbd;
		args[5] = cggb;
		args[6] = cgsb;
		args[7] = cgdb;

                B2mosCap(ckt,vgs,vgd,vgb,args,
		    /*
		    GateSourceOverlapCap,
                    GateDrainOverlapCap,GateBulkOverlapCap,
		    capbs,capbd,cggb,cgsb,cgdb,
		    */
		    cbgb,cbsb,cbdb,csgb,cssb,csdb
                    ,&gcggb,&gcgsb,&gcgdb,&gcbgb,&gcbsb,&gcbdb,&gcsgb
                    ,&gcssb,&gcsdb,&gcdgb,&gcdsb,&gcddb,&qgate,&qbulk
                    ,&qsrc,&qdrn);
            }
             
            if(ByPass) goto line860;
            *(ckt->CKTstate0 + here->B2qg) = qgate;
            *(ckt->CKTstate0 + here->B2qd) = qdrn -  
                    *(ckt->CKTstate0 + here->B2qbd);
            *(ckt->CKTstate0 + here->B2qb) = qbulk +  
                    *(ckt->CKTstate0 + here->B2qbd) +  
                    *(ckt->CKTstate0 + here->B2qbs); 

            /* store small signal parameters */
            if((!(ckt->CKTmode & (MODEAC | MODETRAN))) &&
                    (ckt->CKTmode & MODETRANOP ) && (ckt->CKTmode &
                    MODEUIC ))   goto line850;
            if(ckt->CKTmode & MODEINITSMSIG ) {  
                *(ckt->CKTstate0+here->B2cggb) = cggb;
                *(ckt->CKTstate0+here->B2cgdb) = cgdb;
                *(ckt->CKTstate0+here->B2cgsb) = cgsb;
                *(ckt->CKTstate0+here->B2cbgb) = cbgb;
                *(ckt->CKTstate0+here->B2cbdb) = cbdb;
                *(ckt->CKTstate0+here->B2cbsb) = cbsb;
                *(ckt->CKTstate0+here->B2cdgb) = cdgb;
                *(ckt->CKTstate0+here->B2cddb) = cddb;
                *(ckt->CKTstate0+here->B2cdsb) = cdsb;     
                *(ckt->CKTstate0+here->B2capbd) = capbd;
                *(ckt->CKTstate0+here->B2capbs) = capbs;

                goto line1000;
            }
       
            if(ckt->CKTmode & MODEINITTRAN ) { 
                *(ckt->CKTstate1+here->B2qb) =
                    *(ckt->CKTstate0+here->B2qb) ;
                *(ckt->CKTstate1+here->B2qg) =
                    *(ckt->CKTstate0+here->B2qg) ;
                *(ckt->CKTstate1+here->B2qd) =
                    *(ckt->CKTstate0+here->B2qd) ;
            }
       
       
            error = NIintegrate(ckt,&geq,&ceq,0.0,here->B2qb);
            if(error) return(error);
            error = NIintegrate(ckt,&geq,&ceq,0.0,here->B2qg);
            if(error) return(error);
            error = NIintegrate(ckt,&geq,&ceq,0.0,here->B2qd);
            if(error) return(error);
      
            goto line860;

line850:
            /* initialize to zero charge conductance and current */
            ceqqg = ceqqb = ceqqd = 0.0;
            gcdgb = gcddb = gcdsb = 0.0;
            gcsgb = gcsdb = gcssb = 0.0;
            gcggb = gcgdb = gcgsb = 0.0;
            gcbgb = gcbdb = gcbsb = 0.0;
            goto line900;
            
line860:
            /* evaluate equivalent charge current */
            cqgate = *(ckt->CKTstate0 + here->B2iqg);
            cqbulk = *(ckt->CKTstate0 + here->B2iqb);
            cqdrn = *(ckt->CKTstate0 + here->B2iqd);
            ceqqg = cqgate - gcggb * vgb + gcgdb * vbd + gcgsb * vbs;
            ceqqb = cqbulk - gcbgb * vgb + gcbdb * vbd + gcbsb * vbs;
            ceqqd = cqdrn - gcdgb * vgb + gcddb * vbd + gcdsb * vbs;

            if(ckt->CKTmode & MODEINITTRAN ) {  
                *(ckt->CKTstate1 + here->B2iqb) =  
                    *(ckt->CKTstate0 + here->B2iqb);
                *(ckt->CKTstate1 + here->B2iqg) =  
                    *(ckt->CKTstate0 + here->B2iqg);
                *(ckt->CKTstate1 + here->B2iqd) =  
                    *(ckt->CKTstate0 + here->B2iqd);
            }

            /*
             *  load current vector
             */
line900:

            m = here->B2m;
   
            ceqbs = model->B2type * (cbs-(gbs-ckt->CKTgmin)*vbs);
            ceqbd = model->B2type * (cbd-(gbd-ckt->CKTgmin)*vbd);
     
            ceqqg = model->B2type * ceqqg;
            ceqqb = model->B2type * ceqqb;
            ceqqd =  model->B2type * ceqqd;
            if (here->B2mode >= 0) {
                xnrm=1;
                xrev=0;
                cdreq=model->B2type*(cdrain-gds*vds-gm*vgs-gmbs*vbs);
            } else {
                xnrm=0;
                xrev=1;
                cdreq = -(model->B2type)*(cdrain+gds*vds-gm*vgd-gmbs*vbd);
            }

            *(ckt->CKTrhs + here->B2gNode) -= m * (ceqqg);
            *(ckt->CKTrhs + here->B2bNode) -= m * (ceqbs+ceqbd+ceqqb);
            *(ckt->CKTrhs + here->B2dNodePrime) +=
                    m * (ceqbd-cdreq-ceqqd);
            *(ckt->CKTrhs + here->B2sNodePrime) += 
                    m * (cdreq+ceqbs+ceqqg+ceqqb+ceqqd);

            /*
             *  load y matrix
             */

            *(here->B2DdPtr) += m * (here->B2drainConductance);
            *(here->B2GgPtr) += m * (gcggb);
            *(here->B2SsPtr) += m * (here->B2sourceConductance);
            *(here->B2BbPtr) += m * (gbd+gbs-gcbgb-gcbdb-gcbsb);
            *(here->B2DPdpPtr) += 
                m * (here->B2drainConductance+gds+gbd+xrev*(gm+gmbs)+gcddb);
            *(here->B2SPspPtr) += 
                m * (here->B2sourceConductance+gds+gbs+xnrm*(gm+gmbs)+gcssb);
            *(here->B2DdpPtr) += m * (-here->B2drainConductance);
            *(here->B2GbPtr) += m * (-gcggb-gcgdb-gcgsb);
            *(here->B2GdpPtr) += m * (gcgdb);
            *(here->B2GspPtr) += m * (gcgsb);
            *(here->B2SspPtr) += m * (-here->B2sourceConductance);
            *(here->B2BgPtr) += m * (gcbgb);
            *(here->B2BdpPtr) += m * (-gbd+gcbdb);
            *(here->B2BspPtr) += m * (-gbs+gcbsb);
            *(here->B2DPdPtr) += m * (-here->B2drainConductance);
            *(here->B2DPgPtr) += m * ((xnrm-xrev)*gm+gcdgb);
            *(here->B2DPbPtr) += m * (-gbd+(xnrm-xrev)*gmbs-gcdgb-gcddb-gcdsb);
            *(here->B2DPspPtr) += m * (-gds-xnrm*(gm+gmbs)+gcdsb);
            *(here->B2SPgPtr) += m * (-(xnrm-xrev)*gm+gcsgb);
            *(here->B2SPsPtr) += m * (-here->B2sourceConductance);
            *(here->B2SPbPtr) += m * (-gbs-(xnrm-xrev)*gmbs-gcsgb-gcsdb-gcssb);
            *(here->B2SPdpPtr) += m * (-gds-xrev*(gm+gmbs)+gcsdb);


line1000:  ;

        }   /* End of Mosfet Instance */

    }       /* End of Model Instance */
    return(OK);
}

