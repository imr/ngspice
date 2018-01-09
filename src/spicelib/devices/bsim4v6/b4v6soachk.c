/**********
Copyright 2013 Dietmar Warning. All rights reserved.
Author:   2013 Dietmar Warning
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v6def.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/cpdefs.h"


int
BSIM4v6soaCheck(CKTcircuit *ckt, GENmodel *inModel)
{
    BSIM4v6model *model = (BSIM4v6model *) inModel;
    BSIM4v6instance *here;
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

    for (; model; model = BSIM4v6nextModel(model)) {

        for (here = BSIM4v6instances(model); here; here = BSIM4v6nextInstance(here)) {

            vgs = ckt->CKTrhsOld [here->BSIM4v6gNodePrime] -
                  ckt->CKTrhsOld [here->BSIM4v6sNodePrime];

            vgd = ckt->CKTrhsOld [here->BSIM4v6gNodePrime] -
                  ckt->CKTrhsOld [here->BSIM4v6dNodePrime];

            vgb = ckt->CKTrhsOld [here->BSIM4v6gNodePrime] -
                  ckt->CKTrhsOld [here->BSIM4v6bNodePrime];

            vds = ckt->CKTrhsOld [here->BSIM4v6dNodePrime] -
                  ckt->CKTrhsOld [here->BSIM4v6sNodePrime];

            vbs = ckt->CKTrhsOld [here->BSIM4v6bNodePrime] -
                  ckt->CKTrhsOld [here->BSIM4v6sNodePrime];

            vbd = ckt->CKTrhsOld [here->BSIM4v6bNodePrime] -
                  ckt->CKTrhsOld [here->BSIM4v6dNodePrime];

            if (!model->BSIM4v6vgsrMaxGiven) {
                if (fabs(vgs) > model->BSIM4v6vgsMax)
                    if (warns_vgs < maxwarns) {
                        soa_printf(ckt, (GENinstance*) here,
                                   "Vgs=%g has exceeded Vgs_max=%g\n",
                                   vgs, model->BSIM4v6vgsMax);
                        warns_vgs++;
                    }
                if (!model->BSIM4v6vgbMaxGiven) {
                    if (fabs(vgb) > model->BSIM4v6vgsMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgs_max=%g\n",
                                       vgb, model->BSIM4v6vgsMax);
                            warns_vgb++;
                        }
                } else {
                    if (fabs(vgb) > model->BSIM4v6vgbMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgb_max=%g\n",
                                       vgb, model->BSIM4v6vgbMax);
                            warns_vgb++;
                        }
                }
            } else {
                if (model->BSIM4v6type > 0) {
                    if (vgs > model->BSIM4v6vgsMax)
                        if (warns_vgs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgs=%g has exceeded Vgs_max=%g\n",
                                       vgs, model->BSIM4v6vgsMax);
                            warns_vgs++;
                        }
                    if (-1*vgs > model->BSIM4v6vgsrMax)
                        if (warns_vgs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgs=%g has exceeded Vgsr_max=%g\n",
                                       vgs, model->BSIM4v6vgsrMax);
                            warns_vgs++;
                        }
                } else {
                    if (vgs > model->BSIM4v6vgsrMax)
                        if (warns_vgs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgs=%g has exceeded Vgsr_max=%g\n",
                                       vgs, model->BSIM4v6vgsrMax);
                            warns_vgs++;
                        }
                    if (-1*vgs > model->BSIM4v6vgsMax)
                        if (warns_vgs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgs=%g has exceeded Vgs_max=%g\n",
                                       vgs, model->BSIM4v6vgsMax);
                            warns_vgs++;
                        }
                }
            }

            if (!model->BSIM4v6vgdrMaxGiven) {
                if (fabs(vgd) > model->BSIM4v6vgdMax)
                    if (warns_vgd < maxwarns) {
                        soa_printf(ckt, (GENinstance*) here,
                                   "Vgd=%g has exceeded Vgd_max=%g\n",
                                   vgd, model->BSIM4v6vgdMax);
                        warns_vgd++;
                    }
            } else {
                if (model->BSIM4v6type > 0) {
                    if (vgd > model->BSIM4v6vgdMax)
                        if (warns_vgd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgd=%g has exceeded Vgd_max=%g\n",
                                       vgd, model->BSIM4v6vgdMax);
                            warns_vgd++;
                        }
                    if (-1*vgd > model->BSIM4v6vgdrMax)
                        if (warns_vgd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgd=%g has exceeded Vgdr_max=%g\n",
                                       vgd, model->BSIM4v6vgdrMax);
                            warns_vgd++;
                        }
                } else {
                    if (vgd > model->BSIM4v6vgdrMax)
                        if (warns_vgd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgd=%g has exceeded Vgdr_max=%g\n",
                                       vgd, model->BSIM4v6vgdrMax);
                            warns_vgd++;
                        }
                    if (-1*vgd > model->BSIM4v6vgdMax)
                        if (warns_vgd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgd=%g has exceeded Vgd_max=%g\n",
                                       vgd, model->BSIM4v6vgdMax);
                            warns_vgd++;
                        }
                }
            }

            if (fabs(vds) > model->BSIM4v6vdsMax)
                if (warns_vds < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "Vds=%g has exceeded Vds_max=%g\n",
                               vds, model->BSIM4v6vdsMax);
                    warns_vds++;
                }

            if (!model->BSIM4v6vgbrMaxGiven) {
                if (fabs(vgb) > model->BSIM4v6vgbMax)
                    if (warns_vgb < maxwarns) {
                        soa_printf(ckt, (GENinstance*) here,
                                   "Vgb=%g has exceeded Vgb_max=%g\n",
                                   vgb, model->BSIM4v6vgbMax);
                        warns_vgb++;
                    }
            } else {
                if (model->BSIM4v6type > 0) {
                    if (vgb > model->BSIM4v6vgbMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgb_max=%g\n",
                                       vgb, model->BSIM4v6vgbMax);
                            warns_vgb++;
                        }
                    if (-1*vgb > model->BSIM4v6vgbrMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgbr_max=%g\n",
                                       vgb, model->BSIM4v6vgbrMax);
                            warns_vgb++;
                        }
                } else {
                    if (vgb > model->BSIM4v6vgbrMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgbr_max=%g\n",
                                       vgb, model->BSIM4v6vgbrMax);
                            warns_vgb++;
                        }
                    if (-1*vgb > model->BSIM4v6vgbMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgb_max=%g\n",
                                       vgb, model->BSIM4v6vgbMax);
                            warns_vgb++;
                        }
                }
            }

            if (!model->BSIM4v6vbsrMaxGiven) {
                if (!model->BSIM4v6vbsMaxGiven) {
                    if (fabs(vbs) > model->BSIM4v6vbdMax)
                        if (warns_vbs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbs=%g has exceeded Vbd_max=%g\n",
                                       vbs, model->BSIM4v6vbdMax);
                            warns_vbs++;
                        }
                } else {
                    if (fabs(vbs) > model->BSIM4v6vbsMax)
                        if (warns_vbs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbs=%g has exceeded Vbs_max=%g\n",
                                       vbs, model->BSIM4v6vbsMax);
                            warns_vbs++;
                        }
                }
            } else {
                if (!model->BSIM4v6vbsMaxGiven) {
                    if (model->BSIM4v6type > 0) {
                        if (vbs > model->BSIM4v6vbdMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbd_max=%g\n",
                                           vbs, model->BSIM4v6vbdMax);
                                warns_vbs++;
                            }
                        if (-1*vbs > model->BSIM4v6vbsrMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbsr_max=%g\n",
                                           vbs, model->BSIM4v6vbsrMax);
                                warns_vbs++;
                            }
                    } else {
                        if (vbs > model->BSIM4v6vbsrMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbsr_max=%g\n",
                                           vbs, model->BSIM4v6vbsrMax);
                                warns_vbs++;
                            }
                        if (-1*vbs > model->BSIM4v6vbdMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbd_max=%g\n",
                                           vbs, model->BSIM4v6vbdMax);
                                warns_vbs++;
                            }
                    }
                } else {
                    if (model->BSIM4v6type > 0) {
                        if (vbs > model->BSIM4v6vbsMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbs_max=%g\n",
                                           vbs, model->BSIM4v6vbsMax);
                                warns_vbs++;
                            }
                        if (-1*vbs > model->BSIM4v6vbsrMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbsr_max=%g\n",
                                           vbs, model->BSIM4v6vbsrMax);
                                warns_vbs++;
                            }
                    } else {
                        if (vbs > model->BSIM4v6vbsrMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbsr_max=%g\n",
                                           vbs, model->BSIM4v6vbsrMax);
                                warns_vbs++;
                            }
                        if (-1*vbs > model->BSIM4v6vbsMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbs_max=%g\n",
                                           vbs, model->BSIM4v6vbsMax);
                                warns_vbs++;
                            }
                    }
                }
            }

            if (!model->BSIM4v6vbdrMaxGiven) {
                if (fabs(vbd) > model->BSIM4v6vbdMax)
                    if (warns_vbd < maxwarns) {
                        soa_printf(ckt, (GENinstance*) here,
                                   "Vbd=%g has exceeded Vbd_max=%g\n",
                                   vbd, model->BSIM4v6vbdMax);
                        warns_vbd++;
                    }
            } else {
                if (model->BSIM4v6type > 0) {
                    if (vbd > model->BSIM4v6vbdMax)
                        if (warns_vbd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbd=%g has exceeded Vbd_max=%g\n",
                                       vbd, model->BSIM4v6vbdMax);
                            warns_vbd++;
                        }
                    if (-1*vbd > model->BSIM4v6vbdrMax)
                        if (warns_vbd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbd=%g has exceeded Vbdr_max=%g\n",
                                       vbd, model->BSIM4v6vbdrMax);
                            warns_vbd++;
                        }
                } else {
                    if (vbd > model->BSIM4v6vbdrMax)
                        if (warns_vbd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbd=%g has exceeded Vbdr_max=%g\n",
                                       vbd, model->BSIM4v6vbdrMax);
                            warns_vbd++;
                        }
                    if (-1*vbd > model->BSIM4v6vbdMax)
                        if (warns_vbd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbd=%g has exceeded Vbd_max=%g\n",
                                       vbd, model->BSIM4v6vbdMax);
                            warns_vbd++;
                        }
                }
            }

        }
    }

    return OK;
}
