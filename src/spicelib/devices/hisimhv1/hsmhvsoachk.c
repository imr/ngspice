/**********
Copyright 2013 Dietmar Warning. All rights reserved.
Author:   2013 Dietmar Warning
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "hsmhvdef.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/cpdefs.h"


int
HSMHVsoaCheck(CKTcircuit *ckt, GENmodel *inModel)
{
    HSMHVmodel *model = (HSMHVmodel *) inModel;
    HSMHVinstance *here;
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

    for (; model; model = HSMHVnextModel(model)) {

        for (here = HSMHVinstances(model); here; here = HSMHVnextInstance(here)) {

            vgs = ckt->CKTrhsOld [here->HSMHVgNode] -
                  ckt->CKTrhsOld [here->HSMHVsNodePrime];

            vgd = ckt->CKTrhsOld [here->HSMHVgNode] -
                  ckt->CKTrhsOld [here->HSMHVdNodePrime];

            vgb = ckt->CKTrhsOld [here->HSMHVgNode] -
                  ckt->CKTrhsOld [here->HSMHVbNodePrime];

            vds = ckt->CKTrhsOld [here->HSMHVdNode] -
                  ckt->CKTrhsOld [here->HSMHVsNodePrime];

            vbs = ckt->CKTrhsOld [here->HSMHVbNode] -
                  ckt->CKTrhsOld [here->HSMHVsNodePrime];

            vbd = ckt->CKTrhsOld [here->HSMHVbNode] -
                  ckt->CKTrhsOld [here->HSMHVdNodePrime];

            if (!model->HSMHVvgsrMaxGiven) {
                if (fabs(vgs) > model->HSMHVvgsMax)
                    if (warns_vgs < maxwarns) {
                        soa_printf(ckt, (GENinstance*) here,
                                   "Vgs=%g has exceeded Vgs_max=%g\n",
                                   vgs, model->HSMHVvgsMax);
                        warns_vgs++;
                    }
                if (!model->HSMHVvgbMaxGiven) {
                    if (fabs(vgb) > model->HSMHVvgsMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgs_max=%g\n",
                                       vgb, model->HSMHVvgsMax);
                            warns_vgb++;
                        }
                } else {
                    if (fabs(vgb) > model->HSMHVvgbMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgb_max=%g\n",
                                       vgb, model->HSMHVvgbMax);
                            warns_vgb++;
                        }
                }
            } else {
                if (model->HSMHV_type > 0) {
                    if (vgs > model->HSMHVvgsMax)
                        if (warns_vgs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgs=%g has exceeded Vgs_max=%g\n",
                                       vgs, model->HSMHVvgsMax);
                            warns_vgs++;
                        }
                    if (-1*vgs > model->HSMHVvgsrMax)
                        if (warns_vgs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgs=%g has exceeded Vgsr_max=%g\n",
                                       vgs, model->HSMHVvgsrMax);
                            warns_vgs++;
                        }
                } else {
                    if (vgs > model->HSMHVvgsrMax)
                        if (warns_vgs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgs=%g has exceeded Vgsr_max=%g\n",
                                       vgs, model->HSMHVvgsrMax);
                            warns_vgs++;
                        }
                    if (-1*vgs > model->HSMHVvgsMax)
                        if (warns_vgs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgs=%g has exceeded Vgs_max=%g\n",
                                       vgs, model->HSMHVvgsMax);
                            warns_vgs++;
                        }
                }
            }

            if (!model->HSMHVvgdrMaxGiven) {
                if (fabs(vgd) > model->HSMHVvgdMax)
                    if (warns_vgd < maxwarns) {
                        soa_printf(ckt, (GENinstance*) here,
                                   "Vgd=%g has exceeded Vgd_max=%g\n",
                                   vgd, model->HSMHVvgdMax);
                        warns_vgd++;
                    }
            } else {
                if (model->HSMHV_type > 0) {
                    if (vgd > model->HSMHVvgdMax)
                        if (warns_vgd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgd=%g has exceeded Vgd_max=%g\n",
                                       vgd, model->HSMHVvgdMax);
                            warns_vgd++;
                        }
                    if (-1*vgd > model->HSMHVvgdrMax)
                        if (warns_vgd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgd=%g has exceeded Vgdr_max=%g\n",
                                       vgd, model->HSMHVvgdrMax);
                            warns_vgd++;
                        }
                } else {
                    if (vgd > model->HSMHVvgdrMax)
                        if (warns_vgd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgd=%g has exceeded Vgdr_max=%g\n",
                                       vgd, model->HSMHVvgdrMax);
                            warns_vgd++;
                        }
                    if (-1*vgd > model->HSMHVvgdMax)
                        if (warns_vgd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgd=%g has exceeded Vgd_max=%g\n",
                                       vgd, model->HSMHVvgdMax);
                            warns_vgd++;
                        }
                }
            }

            if (fabs(vds) > model->HSMHVvdsMax)
                if (warns_vds < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "Vds=%g has exceeded Vds_max=%g\n",
                               vds, model->HSMHVvdsMax);
                    warns_vds++;
                }

            if (!model->HSMHVvgbrMaxGiven) {
                if (fabs(vgb) > model->HSMHVvgbMax)
                    if (warns_vgb < maxwarns) {
                        soa_printf(ckt, (GENinstance*) here,
                                   "Vgb=%g has exceeded Vgb_max=%g\n",
                                   vgb, model->HSMHVvgbMax);
                        warns_vgb++;
                    }
            } else {
                if (model->HSMHV_type > 0) {
                    if (vgb > model->HSMHVvgbMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgb_max=%g\n",
                                       vgb, model->HSMHVvgbMax);
                            warns_vgb++;
                        }
                    if (-1*vgb > model->HSMHVvgbrMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgbr_max=%g\n",
                                       vgb, model->HSMHVvgbrMax);
                            warns_vgb++;
                        }
                } else {
                    if (vgb > model->HSMHVvgbrMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgbr_max=%g\n",
                                       vgb, model->HSMHVvgbrMax);
                            warns_vgb++;
                        }
                    if (-1*vgb > model->HSMHVvgbMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgb_max=%g\n",
                                       vgb, model->HSMHVvgbMax);
                            warns_vgb++;
                        }
                }
            }

            if (!model->HSMHVvbsrMaxGiven) {
                if (!model->HSMHVvbsMaxGiven) {
                    if (fabs(vbs) > model->HSMHVvbdMax)
                        if (warns_vbs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbs=%g has exceeded Vbd_max=%g\n",
                                       vbs, model->HSMHVvbdMax);
                            warns_vbs++;
                        }
                } else {
                    if (fabs(vbs) > model->HSMHVvbsMax)
                        if (warns_vbs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbs=%g has exceeded Vbs_max=%g\n",
                                       vbs, model->HSMHVvbsMax);
                            warns_vbs++;
                        }
                }
            } else {
                if (!model->HSMHVvbsMaxGiven) {
                    if (model->HSMHV_type > 0) {
                        if (vbs > model->HSMHVvbdMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbd_max=%g\n",
                                           vbs, model->HSMHVvbdMax);
                                warns_vbs++;
                            }
                        if (-1*vbs > model->HSMHVvbsrMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbsr_max=%g\n",
                                           vbs, model->HSMHVvbsrMax);
                                warns_vbs++;
                            }
                    } else {
                        if (vbs > model->HSMHVvbsrMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbsr_max=%g\n",
                                           vbs, model->HSMHVvbsrMax);
                                warns_vbs++;
                            }
                        if (-1*vbs > model->HSMHVvbdMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbd_max=%g\n",
                                           vbs, model->HSMHVvbdMax);
                                warns_vbs++;
                            }
                    }
                } else {
                    if (model->HSMHV_type > 0) {
                        if (vbs > model->HSMHVvbsMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbs_max=%g\n",
                                           vbs, model->HSMHVvbsMax);
                                warns_vbs++;
                            }
                        if (-1*vbs > model->HSMHVvbsrMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbsr_max=%g\n",
                                           vbs, model->HSMHVvbsrMax);
                                warns_vbs++;
                            }
                    } else {
                        if (vbs > model->HSMHVvbsrMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbsr_max=%g\n",
                                           vbs, model->HSMHVvbsrMax);
                                warns_vbs++;
                            }
                        if (-1*vbs > model->HSMHVvbsMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbs_max=%g\n",
                                           vbs, model->HSMHVvbsMax);
                                warns_vbs++;
                            }
                    }
                }
            }

            if (!model->HSMHVvbdrMaxGiven) {
                if (fabs(vbd) > model->HSMHVvbdMax)
                    if (warns_vbd < maxwarns) {
                        soa_printf(ckt, (GENinstance*) here,
                                   "Vbd=%g has exceeded Vbd_max=%g\n",
                                   vbd, model->HSMHVvbdMax);
                        warns_vbd++;
                    }
            } else {
                if (model->HSMHV_type > 0) {
                    if (vbd > model->HSMHVvbdMax)
                        if (warns_vbd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbd=%g has exceeded Vbd_max=%g\n",
                                       vbd, model->HSMHVvbdMax);
                            warns_vbd++;
                        }
                    if (-1*vbd > model->HSMHVvbdrMax)
                        if (warns_vbd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbd=%g has exceeded Vbdr_max=%g\n",
                                       vbd, model->HSMHVvbdrMax);
                            warns_vbd++;
                        }
                } else {
                    if (vbd > model->HSMHVvbdrMax)
                        if (warns_vbd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbd=%g has exceeded Vbdr_max=%g\n",
                                       vbd, model->HSMHVvbdrMax);
                            warns_vbd++;
                        }
                    if (-1*vbd > model->HSMHVvbdMax)
                        if (warns_vbd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbd=%g has exceeded Vbd_max=%g\n",
                                       vbd, model->HSMHVvbdMax);
                            warns_vbd++;
                        }
                }
            }

        }
    }

    return OK;
}
