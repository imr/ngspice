/**********
Copyright 2013 Dietmar Warning. All rights reserved.
Author:   2013 Dietmar Warning
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "hsmhv2def.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/cpdefs.h"


int
HSMHV2soaCheck(CKTcircuit *ckt, GENmodel *inModel)
{
    HSMHV2model *model = (HSMHV2model *) inModel;
    HSMHV2instance *here;
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

    for (; model; model = HSMHV2nextModel(model)) {

        for (here = HSMHV2instances(model); here; here = HSMHV2nextInstance(here)) {

            vgs = ckt->CKTrhsOld [here->HSMHV2gNode] -
                  ckt->CKTrhsOld [here->HSMHV2sNodePrime];

            vgd = ckt->CKTrhsOld [here->HSMHV2gNode] -
                  ckt->CKTrhsOld [here->HSMHV2dNodePrime];

            vgb = ckt->CKTrhsOld [here->HSMHV2gNode] -
                  ckt->CKTrhsOld [here->HSMHV2bNodePrime];

            vds = ckt->CKTrhsOld [here->HSMHV2dNode] -
                  ckt->CKTrhsOld [here->HSMHV2sNodePrime];

            vbs = ckt->CKTrhsOld [here->HSMHV2bNode] -
                  ckt->CKTrhsOld [here->HSMHV2sNodePrime];

            vbd = ckt->CKTrhsOld [here->HSMHV2bNode] -
                  ckt->CKTrhsOld [here->HSMHV2dNodePrime];

            if (!model->HSMHV2vgsrMaxGiven) {
                if (fabs(vgs) > model->HSMHV2vgsMax)
                    if (warns_vgs < maxwarns) {
                        soa_printf(ckt, (GENinstance*) here,
                                   "Vgs=%g has exceeded Vgs_max=%g\n",
                                   vgs, model->HSMHV2vgsMax);
                        warns_vgs++;
                    }
                if (!model->HSMHV2vgbMaxGiven) {
                    if (fabs(vgb) > model->HSMHV2vgsMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgs_max=%g\n",
                                       vgb, model->HSMHV2vgsMax);
                            warns_vgb++;
                        }
                } else {
                    if (fabs(vgb) > model->HSMHV2vgbMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgb_max=%g\n",
                                       vgb, model->HSMHV2vgbMax);
                            warns_vgb++;
                        }
                }
            } else {
                if (model->HSMHV2_type > 0) {
                    if (vgs > model->HSMHV2vgsMax)
                        if (warns_vgs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgs=%g has exceeded Vgs_max=%g\n",
                                       vgs, model->HSMHV2vgsMax);
                            warns_vgs++;
                        }
                    if (-1*vgs > model->HSMHV2vgsrMax)
                        if (warns_vgs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgs=%g has exceeded Vgsr_max=%g\n",
                                       vgs, model->HSMHV2vgsrMax);
                            warns_vgs++;
                        }
                } else {
                    if (vgs > model->HSMHV2vgsrMax)
                        if (warns_vgs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgs=%g has exceeded Vgsr_max=%g\n",
                                       vgs, model->HSMHV2vgsrMax);
                            warns_vgs++;
                        }
                    if (-1*vgs > model->HSMHV2vgsMax)
                        if (warns_vgs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgs=%g has exceeded Vgs_max=%g\n",
                                       vgs, model->HSMHV2vgsMax);
                            warns_vgs++;
                        }
                }
            }

            if (!model->HSMHV2vgdrMaxGiven) {
                if (fabs(vgd) > model->HSMHV2vgdMax)
                    if (warns_vgd < maxwarns) {
                        soa_printf(ckt, (GENinstance*) here,
                                   "Vgd=%g has exceeded Vgd_max=%g\n",
                                   vgd, model->HSMHV2vgdMax);
                        warns_vgd++;
                    }
            } else {
                if (model->HSMHV2_type > 0) {
                    if (vgd > model->HSMHV2vgdMax)
                        if (warns_vgd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgd=%g has exceeded Vgd_max=%g\n",
                                       vgd, model->HSMHV2vgdMax);
                            warns_vgd++;
                        }
                    if (-1*vgd > model->HSMHV2vgdrMax)
                        if (warns_vgd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgd=%g has exceeded Vgdr_max=%g\n",
                                       vgd, model->HSMHV2vgdrMax);
                            warns_vgd++;
                        }
                } else {
                    if (vgd > model->HSMHV2vgdrMax)
                        if (warns_vgd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgd=%g has exceeded Vgdr_max=%g\n",
                                       vgd, model->HSMHV2vgdrMax);
                            warns_vgd++;
                        }
                    if (-1*vgd > model->HSMHV2vgdMax)
                        if (warns_vgd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgd=%g has exceeded Vgd_max=%g\n",
                                       vgd, model->HSMHV2vgdMax);
                            warns_vgd++;
                        }
                }
            }

            if (fabs(vds) > model->HSMHV2vdsMax)
                if (warns_vds < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "Vds=%g has exceeded Vds_max=%g\n",
                               vds, model->HSMHV2vdsMax);
                    warns_vds++;
                }

            if (!model->HSMHV2vgbrMaxGiven) {
                if (fabs(vgb) > model->HSMHV2vgbMax)
                    if (warns_vgb < maxwarns) {
                        soa_printf(ckt, (GENinstance*) here,
                                   "Vgb=%g has exceeded Vgb_max=%g\n",
                                   vgb, model->HSMHV2vgbMax);
                        warns_vgb++;
                    }
            } else {
                if (model->HSMHV2_type > 0) {
                    if (vgb > model->HSMHV2vgbMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgb_max=%g\n",
                                       vgb, model->HSMHV2vgbMax);
                            warns_vgb++;
                        }
                    if (-1*vgb > model->HSMHV2vgbrMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgbr_max=%g\n",
                                       vgb, model->HSMHV2vgbrMax);
                            warns_vgb++;
                        }
                } else {
                    if (vgb > model->HSMHV2vgbrMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgbr_max=%g\n",
                                       vgb, model->HSMHV2vgbrMax);
                            warns_vgb++;
                        }
                    if (-1*vgb > model->HSMHV2vgbMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgb_max=%g\n",
                                       vgb, model->HSMHV2vgbMax);
                            warns_vgb++;
                        }
                }
            }

            if (!model->HSMHV2vbsrMaxGiven) {
                if (!model->HSMHV2vbsMaxGiven) {
                    if (fabs(vbs) > model->HSMHV2vbdMax)
                        if (warns_vbs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbs=%g has exceeded Vbd_max=%g\n",
                                       vbs, model->HSMHV2vbdMax);
                            warns_vbs++;
                        }
                } else {
                    if (fabs(vbs) > model->HSMHV2vbsMax)
                        if (warns_vbs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbs=%g has exceeded Vbs_max=%g\n",
                                       vbs, model->HSMHV2vbsMax);
                            warns_vbs++;
                        }
                }
            } else {
                if (!model->HSMHV2vbsMaxGiven) {
                    if (model->HSMHV2_type > 0) {
                        if (vbs > model->HSMHV2vbdMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbd_max=%g\n",
                                           vbs, model->HSMHV2vbdMax);
                                warns_vbs++;
                            }
                        if (-1*vbs > model->HSMHV2vbsrMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbsr_max=%g\n",
                                           vbs, model->HSMHV2vbsrMax);
                                warns_vbs++;
                            }
                    } else {
                        if (vbs > model->HSMHV2vbsrMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbsr_max=%g\n",
                                           vbs, model->HSMHV2vbsrMax);
                                warns_vbs++;
                            }
                        if (-1*vbs > model->HSMHV2vbdMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbd_max=%g\n",
                                           vbs, model->HSMHV2vbdMax);
                                warns_vbs++;
                            }
                    }
                } else {
                    if (model->HSMHV2_type > 0) {
                        if (vbs > model->HSMHV2vbsMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbs_max=%g\n",
                                           vbs, model->HSMHV2vbsMax);
                                warns_vbs++;
                            }
                        if (-1*vbs > model->HSMHV2vbsrMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbsr_max=%g\n",
                                           vbs, model->HSMHV2vbsrMax);
                                warns_vbs++;
                            }
                    } else {
                        if (vbs > model->HSMHV2vbsrMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbsr_max=%g\n",
                                           vbs, model->HSMHV2vbsrMax);
                                warns_vbs++;
                            }
                        if (-1*vbs > model->HSMHV2vbsMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbs_max=%g\n",
                                           vbs, model->HSMHV2vbsMax);
                                warns_vbs++;
                            }
                    }
                }
            }

            if (!model->HSMHV2vbdrMaxGiven) {
                if (fabs(vbd) > model->HSMHV2vbdMax)
                    if (warns_vbd < maxwarns) {
                        soa_printf(ckt, (GENinstance*) here,
                                   "Vbd=%g has exceeded Vbd_max=%g\n",
                                   vbd, model->HSMHV2vbdMax);
                        warns_vbd++;
                    }
            } else {
                if (model->HSMHV2_type > 0) {
                    if (vbd > model->HSMHV2vbdMax)
                        if (warns_vbd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbd=%g has exceeded Vbd_max=%g\n",
                                       vbd, model->HSMHV2vbdMax);
                            warns_vbd++;
                        }
                    if (-1*vbd > model->HSMHV2vbdrMax)
                        if (warns_vbd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbd=%g has exceeded Vbdr_max=%g\n",
                                       vbd, model->HSMHV2vbdrMax);
                            warns_vbd++;
                        }
                } else {
                    if (vbd > model->HSMHV2vbdrMax)
                        if (warns_vbd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbd=%g has exceeded Vbdr_max=%g\n",
                                       vbd, model->HSMHV2vbdrMax);
                            warns_vbd++;
                        }
                    if (-1*vbd > model->HSMHV2vbdMax)
                        if (warns_vbd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbd=%g has exceeded Vbd_max=%g\n",
                                       vbd, model->HSMHV2vbdMax);
                            warns_vbd++;
                        }
                }
            }

        }
    }

    return OK;
}
