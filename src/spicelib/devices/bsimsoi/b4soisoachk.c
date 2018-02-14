/**********
Copyright 2013 Dietmar Warning. All rights reserved.
Author:   2013 Dietmar Warning
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "b4soidef.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/cpdefs.h"


int
B4SOIsoaCheck(CKTcircuit *ckt, GENmodel *inModel)
{
    B4SOImodel *model = (B4SOImodel *) inModel;
    B4SOIinstance *here;
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

    for (; model; model = B4SOInextModel(model)) {

        for (here = B4SOIinstances(model); here; here = B4SOInextInstance(here)) {

            vgs = ckt->CKTrhsOld [here->B4SOIgNode] -
                  ckt->CKTrhsOld [here->B4SOIsNodePrime];

            vgd = ckt->CKTrhsOld [here->B4SOIgNode] -
                  ckt->CKTrhsOld [here->B4SOIdNodePrime];

            vgb = ckt->CKTrhsOld [here->B4SOIgNode] -
                  ckt->CKTrhsOld [here->B4SOIbNode];

            vds = ckt->CKTrhsOld [here->B4SOIdNodePrime] -
                  ckt->CKTrhsOld [here->B4SOIsNodePrime];

            vbs = ckt->CKTrhsOld [here->B4SOIbNode] -
                  ckt->CKTrhsOld [here->B4SOIsNodePrime];

            vbd = ckt->CKTrhsOld [here->B4SOIbNode] -
                  ckt->CKTrhsOld [here->B4SOIdNodePrime];

            if (!model->B4SOIvgsrMaxGiven) {
                if (fabs(vgs) > model->B4SOIvgsMax)
                    if (warns_vgs < maxwarns) {
                        soa_printf(ckt, (GENinstance*) here,
                                   "Vgs=%g has exceeded Vgs_max=%g\n",
                                   vgs, model->B4SOIvgsMax);
                        warns_vgs++;
                    }
                if (!model->B4SOIvgbMaxGiven) {
                    if (fabs(vgb) > model->B4SOIvgsMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgs_max=%g\n",
                                       vgb, model->B4SOIvgsMax);
                            warns_vgb++;
                        }
                } else {
                    if (fabs(vgb) > model->B4SOIvgbMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgb_max=%g\n",
                                       vgb, model->B4SOIvgbMax);
                            warns_vgb++;
                        }
                }
            } else {
                if (model->B4SOItype > 0) {
                    if (vgs > model->B4SOIvgsMax)
                        if (warns_vgs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgs=%g has exceeded Vgs_max=%g\n",
                                       vgs, model->B4SOIvgsMax);
                            warns_vgs++;
                        }
                    if (-1*vgs > model->B4SOIvgsrMax)
                        if (warns_vgs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgs=%g has exceeded Vgsr_max=%g\n",
                                       vgs, model->B4SOIvgsrMax);
                            warns_vgs++;
                        }
                } else {
                    if (vgs > model->B4SOIvgsrMax)
                        if (warns_vgs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgs=%g has exceeded Vgsr_max=%g\n",
                                       vgs, model->B4SOIvgsrMax);
                            warns_vgs++;
                        }
                    if (-1*vgs > model->B4SOIvgsMax)
                        if (warns_vgs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgs=%g has exceeded Vgs_max=%g\n",
                                       vgs, model->B4SOIvgsMax);
                            warns_vgs++;
                        }
                }
            }

            if (!model->B4SOIvgdrMaxGiven) {
                if (fabs(vgd) > model->B4SOIvgdMax)
                    if (warns_vgd < maxwarns) {
                        soa_printf(ckt, (GENinstance*) here,
                                   "Vgd=%g has exceeded Vgd_max=%g\n",
                                   vgd, model->B4SOIvgdMax);
                        warns_vgd++;
                    }
            } else {
                if (model->B4SOItype > 0) {
                    if (vgd > model->B4SOIvgdMax)
                        if (warns_vgd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgd=%g has exceeded Vgd_max=%g\n",
                                       vgd, model->B4SOIvgdMax);
                            warns_vgd++;
                        }
                    if (-1*vgd > model->B4SOIvgdrMax)
                        if (warns_vgd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgd=%g has exceeded Vgdr_max=%g\n",
                                       vgd, model->B4SOIvgdrMax);
                            warns_vgd++;
                        }
                } else {
                    if (vgd > model->B4SOIvgdrMax)
                        if (warns_vgd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgd=%g has exceeded Vgdr_max=%g\n",
                                       vgd, model->B4SOIvgdrMax);
                            warns_vgd++;
                        }
                    if (-1*vgd > model->B4SOIvgdMax)
                        if (warns_vgd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgd=%g has exceeded Vgd_max=%g\n",
                                       vgd, model->B4SOIvgdMax);
                            warns_vgd++;
                        }
                }
            }

            if (fabs(vds) > model->B4SOIvdsMax)
                if (warns_vds < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "Vds=%g has exceeded Vds_max=%g\n",
                               vds, model->B4SOIvdsMax);
                    warns_vds++;
                }

            if (!model->B4SOIvgbrMaxGiven) {
                if (fabs(vgb) > model->B4SOIvgbMax)
                    if (warns_vgb < maxwarns) {
                        soa_printf(ckt, (GENinstance*) here,
                                   "Vgb=%g has exceeded Vgb_max=%g\n",
                                   vgb, model->B4SOIvgbMax);
                        warns_vgb++;
                    }
            } else {
                if (model->B4SOItype > 0) {
                    if (vgb > model->B4SOIvgbMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgb_max=%g\n",
                                       vgb, model->B4SOIvgbMax);
                            warns_vgb++;
                        }
                    if (-1*vgb > model->B4SOIvgbrMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgbr_max=%g\n",
                                       vgb, model->B4SOIvgbrMax);
                            warns_vgb++;
                        }
                } else {
                    if (vgb > model->B4SOIvgbrMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgbr_max=%g\n",
                                       vgb, model->B4SOIvgbrMax);
                            warns_vgb++;
                        }
                    if (-1*vgb > model->B4SOIvgbMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgb_max=%g\n",
                                       vgb, model->B4SOIvgbMax);
                            warns_vgb++;
                        }
                }
            }

            if (!model->B4SOIvbsrMaxGiven) {
                if (!model->B4SOIvbsMaxGiven) {
                    if (fabs(vbs) > model->B4SOIvbdMax)
                        if (warns_vbs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbs=%g has exceeded Vbd_max=%g\n",
                                       vbs, model->B4SOIvbdMax);
                            warns_vbs++;
                        }
                } else {
                    if (fabs(vbs) > model->B4SOIvbsMax)
                        if (warns_vbs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbs=%g has exceeded Vbs_max=%g\n",
                                       vbs, model->B4SOIvbsMax);
                            warns_vbs++;
                        }
                }
            } else {
                if (!model->B4SOIvbsMaxGiven) {
                    if (model->B4SOItype > 0) {
                        if (vbs > model->B4SOIvbdMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbd_max=%g\n",
                                           vbs, model->B4SOIvbdMax);
                                warns_vbs++;
                            }
                        if (-1*vbs > model->B4SOIvbsrMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbsr_max=%g\n",
                                           vbs, model->B4SOIvbsrMax);
                                warns_vbs++;
                            }
                    } else {
                        if (vbs > model->B4SOIvbsrMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbsr_max=%g\n",
                                           vbs, model->B4SOIvbsrMax);
                                warns_vbs++;
                            }
                        if (-1*vbs > model->B4SOIvbdMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbd_max=%g\n",
                                           vbs, model->B4SOIvbdMax);
                                warns_vbs++;
                            }
                    }
                } else {
                    if (model->B4SOItype > 0) {
                        if (vbs > model->B4SOIvbsMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbs_max=%g\n",
                                           vbs, model->B4SOIvbsMax);
                                warns_vbs++;
                            }
                        if (-1*vbs > model->B4SOIvbsrMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbsr_max=%g\n",
                                           vbs, model->B4SOIvbsrMax);
                                warns_vbs++;
                            }
                    } else {
                        if (vbs > model->B4SOIvbsrMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbsr_max=%g\n",
                                           vbs, model->B4SOIvbsrMax);
                                warns_vbs++;
                            }
                        if (-1*vbs > model->B4SOIvbsMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbs_max=%g\n",
                                           vbs, model->B4SOIvbsMax);
                                warns_vbs++;
                            }
                    }
                }
            }

            if (!model->B4SOIvbdrMaxGiven) {
                if (fabs(vbd) > model->B4SOIvbdMax)
                    if (warns_vbd < maxwarns) {
                        soa_printf(ckt, (GENinstance*) here,
                                   "Vbd=%g has exceeded Vbd_max=%g\n",
                                   vbd, model->B4SOIvbdMax);
                        warns_vbd++;
                    }
            } else {
                if (model->B4SOItype > 0) {
                    if (vbd > model->B4SOIvbdMax)
                        if (warns_vbd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbd=%g has exceeded Vbd_max=%g\n",
                                       vbd, model->B4SOIvbdMax);
                            warns_vbd++;
                        }
                    if (-1*vbd > model->B4SOIvbdrMax)
                        if (warns_vbd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbd=%g has exceeded Vbdr_max=%g\n",
                                       vbd, model->B4SOIvbdrMax);
                            warns_vbd++;
                        }
                } else {
                    if (vbd > model->B4SOIvbdrMax)
                        if (warns_vbd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbd=%g has exceeded Vbdr_max=%g\n",
                                       vbd, model->B4SOIvbdrMax);
                            warns_vbd++;
                        }
                    if (-1*vbd > model->B4SOIvbdMax)
                        if (warns_vbd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbd=%g has exceeded Vbd_max=%g\n",
                                       vbd, model->B4SOIvbdMax);
                            warns_vbd++;
                        }
                }
            }

        }
    }

    return OK;
}
