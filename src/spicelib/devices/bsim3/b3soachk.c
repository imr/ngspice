/**********
Copyright 2013 Dietmar Warning. All rights reserved.
Author:   2013 Dietmar Warning
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3def.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/cpdefs.h"


int
BSIM3soaCheck(CKTcircuit *ckt, GENmodel *inModel)
{
    BSIM3model *model = (BSIM3model *) inModel;
    BSIM3instance *here;
    double vgs, vgd, vgb, vds, vbs, vbd;    /* actual mos voltages */
    int maxwarns;
    static int warns_vgs = 0, warns_vgd = 0, warns_vgb = 0, warns_vds = 0, warns_vbs = 0, warns_vbd = 0;

    if (!ckt) {
        warns_vgs = 0;
        warns_vgd = 0;
        warns_vgb = 0;
        warns_vds = 0;
        warns_vbs = 0;
        warns_vbd = 0;
        return OK;
    }

    maxwarns = ckt->CKTsoaMaxWarns;

    for (; model; model = BSIM3nextModel(model)) {

        for (here = BSIM3instances(model); here; here = BSIM3nextInstance(here)) {

            vgs = ckt->CKTrhsOld [here->BSIM3gNode] -
                  ckt->CKTrhsOld [here->BSIM3sNodePrime];

            vgd = ckt->CKTrhsOld [here->BSIM3gNode] -
                  ckt->CKTrhsOld [here->BSIM3dNodePrime];

            vgb = ckt->CKTrhsOld [here->BSIM3gNode] -
                  ckt->CKTrhsOld [here->BSIM3bNode];

            vds = ckt->CKTrhsOld [here->BSIM3dNodePrime] -
                  ckt->CKTrhsOld [here->BSIM3sNodePrime];

            vbs = ckt->CKTrhsOld [here->BSIM3bNode] -
                  ckt->CKTrhsOld [here->BSIM3sNodePrime];

            vbd = ckt->CKTrhsOld [here->BSIM3bNode] -
                  ckt->CKTrhsOld [here->BSIM3dNodePrime];
 
            if (!model->BSIM3vgsrMaxGiven) {
                if (fabs(vgs) > model->BSIM3vgsMax)
                    if (warns_vgs < maxwarns) {
                        soa_printf(ckt, (GENinstance*) here,
                                   "Vgs=%g has exceeded Vgs_max=%g\n",
                                   vgs, model->BSIM3vgsMax);
                        warns_vgs++;
                    }
                if (!model->BSIM3vgbMaxGiven) {
                    if (fabs(vgb) > model->BSIM3vgsMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgs_max=%g\n",
                                       vgb, model->BSIM3vgsMax);
                            warns_vgb++;
                        }
                } else {
                    if (fabs(vgb) > model->BSIM3vgbMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgb_max=%g\n",
                                       vgb, model->BSIM3vgbMax);
                            warns_vgb++;
                        }
                }
            } else {
                if (model->BSIM3type > 0) {
                    if (vgs > model->BSIM3vgsMax)
                        if (warns_vgs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgs=%g has exceeded Vgs_max=%g\n",
                                       vgs, model->BSIM3vgsMax);
                            warns_vgs++;
                        }
                    if (-1*vgs > model->BSIM3vgsrMax)
                        if (warns_vgs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgs=%g has exceeded Vgsr_max=%g\n",
                                       vgs, model->BSIM3vgsrMax);
                            warns_vgs++;
                        }
                } else {
                    if (vgs > model->BSIM3vgsrMax)
                        if (warns_vgs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgs=%g has exceeded Vgsr_max=%g\n",
                                       vgs, model->BSIM3vgsrMax);
                            warns_vgs++;
                        }
                    if (-1*vgs > model->BSIM3vgsMax)
                        if (warns_vgs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgs=%g has exceeded Vgs_max=%g\n",
                                       vgs, model->BSIM3vgsMax);
                            warns_vgs++;
                        }
                }
            }

            if (!model->BSIM3vgdrMaxGiven) {
                if (fabs(vgd) > model->BSIM3vgdMax)
                    if (warns_vgd < maxwarns) {
                        soa_printf(ckt, (GENinstance*) here,
                                   "Vgd=%g has exceeded Vgd_max=%g\n",
                                   vgd, model->BSIM3vgdMax);
                        warns_vgd++;
                    }
            } else {
                if (model->BSIM3type > 0) {
                    if (vgd > model->BSIM3vgdMax)
                        if (warns_vgd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgd=%g has exceeded Vgd_max=%g\n",
                                       vgd, model->BSIM3vgdMax);
                            warns_vgd++;
                        }
                    if (-1*vgd > model->BSIM3vgdrMax)
                        if (warns_vgd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgd=%g has exceeded Vgdr_max=%g\n",
                                       vgd, model->BSIM3vgdrMax);
                            warns_vgd++;
                        }
                } else {
                    if (vgd > model->BSIM3vgdrMax)
                        if (warns_vgd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgd=%g has exceeded Vgdr_max=%g\n",
                                       vgd, model->BSIM3vgdrMax);
                            warns_vgd++;
                        }
                    if (-1*vgd > model->BSIM3vgdMax)
                        if (warns_vgd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgd=%g has exceeded Vgd_max=%g\n",
                                       vgd, model->BSIM3vgdMax);
                            warns_vgd++;
                        }
                }
            }

            if (fabs(vds) > model->BSIM3vdsMax)
                if (warns_vds < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "Vds=%g has exceeded Vds_max=%g\n",
                               vds, model->BSIM3vdsMax);
                    warns_vds++;
                }

            if (!model->BSIM3vgbrMaxGiven) {
                if (fabs(vgb) > model->BSIM3vgbMax)
                    if (warns_vgb < maxwarns) {
                        soa_printf(ckt, (GENinstance*) here,
                                   "Vgb=%g has exceeded Vgb_max=%g\n",
                                   vgb, model->BSIM3vgbMax);
                        warns_vgb++;
                    }
            } else {
                if (model->BSIM3type > 0) {
                    if (vgb > model->BSIM3vgbMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgb_max=%g\n",
                                       vgb, model->BSIM3vgbMax);
                            warns_vgb++;
                        }
                    if (-1*vgb > model->BSIM3vgbrMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgbr_max=%g\n",
                                       vgb, model->BSIM3vgbrMax);
                            warns_vgb++;
                        }
                } else {
                    if (vgb > model->BSIM3vgbrMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgbr_max=%g\n",
                                       vgb, model->BSIM3vgbrMax);
                            warns_vgb++;
                        }
                    if (-1*vgb > model->BSIM3vgbMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgb_max=%g\n",
                                       vgb, model->BSIM3vgbMax);
                            warns_vgb++;
                        }
                }
            }

            if (!model->BSIM3vbsrMaxGiven) {
                if (!model->BSIM3vbsMaxGiven) {
                    if (fabs(vbs) > model->BSIM3vbdMax)
                        if (warns_vbs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbs=%g has exceeded Vbd_max=%g\n",
                                       vbs, model->BSIM3vbdMax);
                            warns_vbs++;
                        }
                } else {
                    if (fabs(vbs) > model->BSIM3vbsMax)
                        if (warns_vbs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbs=%g has exceeded Vbs_max=%g\n",
                                       vbs, model->BSIM3vbsMax);
                            warns_vbs++;
                        }
                }
            } else {
                if (!model->BSIM3vbsMaxGiven) {
                    if (model->BSIM3type > 0) {
                        if (vbs > model->BSIM3vbdMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbd_max=%g\n",
                                           vbs, model->BSIM3vbdMax);
                                warns_vbs++;
                            }
                        if (-1*vbs > model->BSIM3vbsrMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbsr_max=%g\n",
                                           vbs, model->BSIM3vbsrMax);
                                warns_vbs++;
                            }
                    } else {
                        if (vbs > model->BSIM3vbsrMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbsr_max=%g\n",
                                           vbs, model->BSIM3vbsrMax);
                                warns_vbs++;
                            }
                        if (-1*vbs > model->BSIM3vbdMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbd_max=%g\n",
                                           vbs, model->BSIM3vbdMax);
                                warns_vbs++;
                            }
                    }
                } else {
                    if (model->BSIM3type > 0) {
                        if (vbs > model->BSIM3vbsMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbs_max=%g\n",
                                           vbs, model->BSIM3vbsMax);
                                warns_vbs++;
                            }
                        if (-1*vbs > model->BSIM3vbsrMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbsr_max=%g\n",
                                           vbs, model->BSIM3vbsrMax);
                                warns_vbs++;
                            }
                    } else {
                        if (vbs > model->BSIM3vbsrMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbsr_max=%g\n",
                                           vbs, model->BSIM3vbsrMax);
                                warns_vbs++;
                            }
                        if (-1*vbs > model->BSIM3vbsMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbs_max=%g\n",
                                           vbs, model->BSIM3vbsMax);
                                warns_vbs++;
                            }
                    }
                }
            }

            if (!model->BSIM3vbdrMaxGiven) {
                if (fabs(vbd) > model->BSIM3vbdMax)
                    if (warns_vbd < maxwarns) {
                        soa_printf(ckt, (GENinstance*) here,
                                   "Vbd=%g has exceeded Vbd_max=%g\n",
                                   vbd, model->BSIM3vbdMax);
                        warns_vbd++;
                    }
            } else {
                if (model->BSIM3type > 0) {
                    if (vbd > model->BSIM3vbdMax)
                        if (warns_vbd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbd=%g has exceeded Vbd_max=%g\n",
                                       vbd, model->BSIM3vbdMax);
                            warns_vbd++;
                        }
                    if (-1*vbd > model->BSIM3vbdrMax)
                        if (warns_vbd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbd=%g has exceeded Vbdr_max=%g\n",
                                       vbd, model->BSIM3vbdrMax);
                            warns_vbd++;
                        }
                } else {
                    if (vbd > model->BSIM3vbdrMax)
                        if (warns_vbd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbd=%g has exceeded Vbdr_max=%g\n",
                                       vbd, model->BSIM3vbdrMax);
                            warns_vbd++;
                        }
                    if (-1*vbd > model->BSIM3vbdMax)
                        if (warns_vbd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbd=%g has exceeded Vbd_max=%g\n",
                                       vbd, model->BSIM3vbdMax);
                            warns_vbd++;
                        }
                }
            }

        }
    }

    return OK;
}
