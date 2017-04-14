/**********
Copyright 2013 Dietmar Warning. All rights reserved.
Author:   2013 Dietmar Warning
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v4def.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/cpdefs.h"


int
BSIM4v4soaCheck(CKTcircuit *ckt, GENmodel *inModel)
{
    BSIM4v4model *model = (BSIM4v4model *) inModel;
    BSIM4v4instance *here;
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

    for (; model; model = model->BSIM4v4nextModel) {

        for (here = model->BSIM4v4instances; here; here = here->BSIM4v4nextInstance) {

            vgs = ckt->CKTrhsOld [here->BSIM4v4gNodePrime] -
                  ckt->CKTrhsOld [here->BSIM4v4sNodePrime];

            vgd = ckt->CKTrhsOld [here->BSIM4v4gNodePrime] -
                  ckt->CKTrhsOld [here->BSIM4v4dNodePrime];

            vgb = ckt->CKTrhsOld [here->BSIM4v4gNodePrime] -
                  ckt->CKTrhsOld [here->BSIM4v4bNodePrime];

            vds = ckt->CKTrhsOld [here->BSIM4v4dNodePrime] -
                  ckt->CKTrhsOld [here->BSIM4v4sNodePrime];

            vbs = ckt->CKTrhsOld [here->BSIM4v4bNodePrime] -
                  ckt->CKTrhsOld [here->BSIM4v4sNodePrime];

            vbd = ckt->CKTrhsOld [here->BSIM4v4bNodePrime] -
                  ckt->CKTrhsOld [here->BSIM4v4dNodePrime];

            if (!model->BSIM4v4vgsrMaxGiven) {
                if (fabs(vgs) > model->BSIM4v4vgsMax)
                    if (warns_vgs < maxwarns) {
                        soa_printf(ckt, (GENinstance*) here,
                                   "Vgs=%g has exceeded Vgs_max=%g\n",
                                   vgs, model->BSIM4v4vgsMax);
                        warns_vgs++;
                    }
                if (!model->BSIM4v4vgbMaxGiven) {
                    if (fabs(vgb) > model->BSIM4v4vgsMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgs_max=%g\n",
                                       vgb, model->BSIM4v4vgsMax);
                            warns_vgb++;
                        }
                } else {
                    if (fabs(vgb) > model->BSIM4v4vgbMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgb_max=%g\n",
                                       vgb, model->BSIM4v4vgbMax);
                            warns_vgb++;
                        }
                }
            } else {
                if (model->BSIM4v4type > 0) {
                    if (vgs > model->BSIM4v4vgsMax)
                        if (warns_vgs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgs=%g has exceeded Vgs_max=%g\n",
                                       vgs, model->BSIM4v4vgsMax);
                            warns_vgs++;
                        }
                    if (-1*vgs > model->BSIM4v4vgsrMax)
                        if (warns_vgs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgs=%g has exceeded Vgsr_max=%g\n",
                                       vgs, model->BSIM4v4vgsrMax);
                            warns_vgs++;
                        }
                } else {
                    if (vgs > model->BSIM4v4vgsrMax)
                        if (warns_vgs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgs=%g has exceeded Vgsr_max=%g\n",
                                       vgs, model->BSIM4v4vgsrMax);
                            warns_vgs++;
                        }
                    if (-1*vgs > model->BSIM4v4vgsMax)
                        if (warns_vgs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgs=%g has exceeded Vgs_max=%g\n",
                                       vgs, model->BSIM4v4vgsMax);
                            warns_vgs++;
                        }
                }
            }

            if (!model->BSIM4v4vgdrMaxGiven) {
                if (fabs(vgd) > model->BSIM4v4vgdMax)
                    if (warns_vgd < maxwarns) {
                        soa_printf(ckt, (GENinstance*) here,
                                   "Vgd=%g has exceeded Vgd_max=%g\n",
                                   vgd, model->BSIM4v4vgdMax);
                        warns_vgd++;
                    }
            } else {
                if (model->BSIM4v4type > 0) {
                    if (vgd > model->BSIM4v4vgdMax)
                        if (warns_vgd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgd=%g has exceeded Vgd_max=%g\n",
                                       vgd, model->BSIM4v4vgdMax);
                            warns_vgd++;
                        }
                    if (-1*vgd > model->BSIM4v4vgdrMax)
                        if (warns_vgd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgd=%g has exceeded Vgdr_max=%g\n",
                                       vgd, model->BSIM4v4vgdrMax);
                            warns_vgd++;
                        }
                } else {
                    if (vgd > model->BSIM4v4vgdrMax)
                        if (warns_vgd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgd=%g has exceeded Vgdr_max=%g\n",
                                       vgd, model->BSIM4v4vgdrMax);
                            warns_vgd++;
                        }
                    if (-1*vgd > model->BSIM4v4vgdMax)
                        if (warns_vgd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgd=%g has exceeded Vgd_max=%g\n",
                                       vgd, model->BSIM4v4vgdMax);
                            warns_vgd++;
                        }
                }
            }

            if (fabs(vds) > model->BSIM4v4vdsMax)
                if (warns_vds < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "Vds=%g has exceeded Vds_max=%g\n",
                               vds, model->BSIM4v4vdsMax);
                    warns_vds++;
                }

            if (!model->BSIM4v4vgbrMaxGiven) {
                if (fabs(vgb) > model->BSIM4v4vgbMax)
                    if (warns_vgb < maxwarns) {
                        soa_printf(ckt, (GENinstance*) here,
                                   "Vgb=%g has exceeded Vgb_max=%g\n",
                                   vgb, model->BSIM4v4vgbMax);
                        warns_vgb++;
                    }
            } else {
                if (model->BSIM4v4type > 0) {
                    if (vgb > model->BSIM4v4vgbMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgb_max=%g\n",
                                       vgb, model->BSIM4v4vgbMax);
                            warns_vgb++;
                        }
                    if (-1*vgb > model->BSIM4v4vgbrMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgbr_max=%g\n",
                                       vgb, model->BSIM4v4vgbrMax);
                            warns_vgb++;
                        }
                } else {
                    if (vgb > model->BSIM4v4vgbrMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgbr_max=%g\n",
                                       vgb, model->BSIM4v4vgbrMax);
                            warns_vgb++;
                        }
                    if (-1*vgb > model->BSIM4v4vgbMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgb_max=%g\n",
                                       vgb, model->BSIM4v4vgbMax);
                            warns_vgb++;
                        }
                }
            }

            if (!model->BSIM4v4vbsrMaxGiven) {
                if (!model->BSIM4v4vbsMaxGiven) {
                    if (fabs(vbs) > model->BSIM4v4vbdMax)
                        if (warns_vbs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbs=%g has exceeded Vbd_max=%g\n",
                                       vbs, model->BSIM4v4vbdMax);
                            warns_vbs++;
                        }
                } else {
                    if (fabs(vbs) > model->BSIM4v4vbsMax)
                        if (warns_vbs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbs=%g has exceeded Vbs_max=%g\n",
                                       vbs, model->BSIM4v4vbsMax);
                            warns_vbs++;
                        }
                }
            } else {
                if (!model->BSIM4v4vbsMaxGiven) {
                    if (model->BSIM4v4type > 0) {
                        if (vbs > model->BSIM4v4vbdMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbd_max=%g\n",
                                           vbs, model->BSIM4v4vbdMax);
                                warns_vbs++;
                            }
                        if (-1*vbs > model->BSIM4v4vbsrMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbsr_max=%g\n",
                                           vbs, model->BSIM4v4vbsrMax);
                                warns_vbs++;
                            }
                    } else {
                        if (vbs > model->BSIM4v4vbsrMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbsr_max=%g\n",
                                           vbs, model->BSIM4v4vbsrMax);
                                warns_vbs++;
                            }
                        if (-1*vbs > model->BSIM4v4vbdMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbd_max=%g\n",
                                           vbs, model->BSIM4v4vbdMax);
                                warns_vbs++;
                            }
                    }
                } else {
                    if (model->BSIM4v4type > 0) {
                        if (vbs > model->BSIM4v4vbsMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbs_max=%g\n",
                                           vbs, model->BSIM4v4vbsMax);
                                warns_vbs++;
                            }
                        if (-1*vbs > model->BSIM4v4vbsrMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbsr_max=%g\n",
                                           vbs, model->BSIM4v4vbsrMax);
                                warns_vbs++;
                            }
                    } else {
                        if (vbs > model->BSIM4v4vbsrMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbsr_max=%g\n",
                                           vbs, model->BSIM4v4vbsrMax);
                                warns_vbs++;
                            }
                        if (-1*vbs > model->BSIM4v4vbsMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbs_max=%g\n",
                                           vbs, model->BSIM4v4vbsMax);
                                warns_vbs++;
                            }
                    }
                }
            }

            if (!model->BSIM4v4vbdrMaxGiven) {
                if (fabs(vbd) > model->BSIM4v4vbdMax)
                    if (warns_vbd < maxwarns) {
                        soa_printf(ckt, (GENinstance*) here,
                                   "Vbd=%g has exceeded Vbd_max=%g\n",
                                   vbd, model->BSIM4v4vbdMax);
                        warns_vbd++;
                    }
            } else {
                if (model->BSIM4v4type > 0) {
                    if (vbd > model->BSIM4v4vbdMax)
                        if (warns_vbd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbd=%g has exceeded Vbd_max=%g\n",
                                       vbd, model->BSIM4v4vbdMax);
                            warns_vbd++;
                        }
                    if (-1*vbd > model->BSIM4v4vbdrMax)
                        if (warns_vbd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbd=%g has exceeded Vbdr_max=%g\n",
                                       vbd, model->BSIM4v4vbdrMax);
                            warns_vbd++;
                        }
                } else {
                    if (vbd > model->BSIM4v4vbdrMax)
                        if (warns_vbd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbd=%g has exceeded Vbdr_max=%g\n",
                                       vbd, model->BSIM4v4vbdrMax);
                            warns_vbd++;
                        }
                    if (-1*vbd > model->BSIM4v4vbdMax)
                        if (warns_vbd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbd=%g has exceeded Vbd_max=%g\n",
                                       vbd, model->BSIM4v4vbdMax);
                            warns_vbd++;
                        }
                }
            }

        }
    }

    return OK;
}
