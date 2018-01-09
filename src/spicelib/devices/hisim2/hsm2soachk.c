/**********
Copyright 2013 Dietmar Warning. All rights reserved.
Author:   2013 Dietmar Warning
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "hsm2def.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/cpdefs.h"


int
HSM2soaCheck(CKTcircuit *ckt, GENmodel *inModel)
{
    HSM2model *model = (HSM2model *) inModel;
    HSM2instance *here;
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

    for (; model; model = HSM2nextModel(model)) {

        for (here = HSM2instances(model); here; here = HSM2nextInstance(here)) {

            vgs = ckt->CKTrhsOld [here->HSM2gNode] -
                  ckt->CKTrhsOld [here->HSM2sNodePrime];

            vgd = ckt->CKTrhsOld [here->HSM2gNode] -
                  ckt->CKTrhsOld [here->HSM2dNodePrime];

            vgb = ckt->CKTrhsOld [here->HSM2gNode] -
                  ckt->CKTrhsOld [here->HSM2bNodePrime];

            vds = ckt->CKTrhsOld [here->HSM2dNode] -
                  ckt->CKTrhsOld [here->HSM2sNodePrime];

            vbs = ckt->CKTrhsOld [here->HSM2bNode] -
                  ckt->CKTrhsOld [here->HSM2sNodePrime];

            vbd = ckt->CKTrhsOld [here->HSM2bNode] -
                  ckt->CKTrhsOld [here->HSM2dNodePrime];

            if (!model->HSM2vgsrMaxGiven) {
                if (fabs(vgs) > model->HSM2vgsMax)
                    if (warns_vgs < maxwarns) {
                        soa_printf(ckt, (GENinstance*) here,
                                   "Vgs=%g has exceeded Vgs_max=%g\n",
                                   vgs, model->HSM2vgsMax);
                        warns_vgs++;
                    }
                if (!model->HSM2vgbMaxGiven) {
                    if (fabs(vgb) > model->HSM2vgsMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgs_max=%g\n",
                                       vgb, model->HSM2vgsMax);
                            warns_vgb++;
                        }
                } else {
                    if (fabs(vgb) > model->HSM2vgbMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgb_max=%g\n",
                                       vgb, model->HSM2vgbMax);
                            warns_vgb++;
                        }
                }
            } else {
                if (model->HSM2_type > 0) {
                    if (vgs > model->HSM2vgsMax)
                        if (warns_vgs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgs=%g has exceeded Vgs_max=%g\n",
                                       vgs, model->HSM2vgsMax);
                            warns_vgs++;
                        }
                    if (-1*vgs > model->HSM2vgsrMax)
                        if (warns_vgs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgs=%g has exceeded Vgsr_max=%g\n",
                                       vgs, model->HSM2vgsrMax);
                            warns_vgs++;
                        }
                } else {
                    if (vgs > model->HSM2vgsrMax)
                        if (warns_vgs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgs=%g has exceeded Vgsr_max=%g\n",
                                       vgs, model->HSM2vgsrMax);
                            warns_vgs++;
                        }
                    if (-1*vgs > model->HSM2vgsMax)
                        if (warns_vgs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgs=%g has exceeded Vgs_max=%g\n",
                                       vgs, model->HSM2vgsMax);
                            warns_vgs++;
                        }
                }
            }

            if (!model->HSM2vgdrMaxGiven) {
                if (fabs(vgd) > model->HSM2vgdMax)
                    if (warns_vgd < maxwarns) {
                        soa_printf(ckt, (GENinstance*) here,
                                   "Vgd=%g has exceeded Vgd_max=%g\n",
                                   vgd, model->HSM2vgdMax);
                        warns_vgd++;
                    }
            } else {
                if (model->HSM2_type > 0) {
                    if (vgd > model->HSM2vgdMax)
                        if (warns_vgd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgd=%g has exceeded Vgd_max=%g\n",
                                       vgd, model->HSM2vgdMax);
                            warns_vgd++;
                        }
                    if (-1*vgd > model->HSM2vgdrMax)
                        if (warns_vgd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgd=%g has exceeded Vgdr_max=%g\n",
                                       vgd, model->HSM2vgdrMax);
                            warns_vgd++;
                        }
                } else {
                    if (vgd > model->HSM2vgdrMax)
                        if (warns_vgd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgd=%g has exceeded Vgdr_max=%g\n",
                                       vgd, model->HSM2vgdrMax);
                            warns_vgd++;
                        }
                    if (-1*vgd > model->HSM2vgdMax)
                        if (warns_vgd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgd=%g has exceeded Vgd_max=%g\n",
                                       vgd, model->HSM2vgdMax);
                            warns_vgd++;
                        }
                }
            }

            if (fabs(vds) > model->HSM2vdsMax)
                if (warns_vds < maxwarns) {
                    soa_printf(ckt, (GENinstance*) here,
                               "Vds=%g has exceeded Vds_max=%g\n",
                               vds, model->HSM2vdsMax);
                    warns_vds++;
                }

            if (!model->HSM2vgbrMaxGiven) {
                if (fabs(vgb) > model->HSM2vgbMax)
                    if (warns_vgb < maxwarns) {
                        soa_printf(ckt, (GENinstance*) here,
                                   "Vgb=%g has exceeded Vgb_max=%g\n",
                                   vgb, model->HSM2vgbMax);
                        warns_vgb++;
                    }
            } else {
                if (model->HSM2_type > 0) {
                    if (vgb > model->HSM2vgbMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgb_max=%g\n",
                                       vgb, model->HSM2vgbMax);
                            warns_vgb++;
                        }
                    if (-1*vgb > model->HSM2vgbrMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgbr_max=%g\n",
                                       vgb, model->HSM2vgbrMax);
                            warns_vgb++;
                        }
                } else {
                    if (vgb > model->HSM2vgbrMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgbr_max=%g\n",
                                       vgb, model->HSM2vgbrMax);
                            warns_vgb++;
                        }
                    if (-1*vgb > model->HSM2vgbMax)
                        if (warns_vgb < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vgb=%g has exceeded Vgb_max=%g\n",
                                       vgb, model->HSM2vgbMax);
                            warns_vgb++;
                        }
                }
            }

            if (!model->HSM2vbsrMaxGiven) {
                if (!model->HSM2vbsMaxGiven) {
                    if (fabs(vbs) > model->HSM2vbdMax)
                        if (warns_vbs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbs=%g has exceeded Vbd_max=%g\n",
                                       vbs, model->HSM2vbdMax);
                            warns_vbs++;
                        }
                } else {
                    if (fabs(vbs) > model->HSM2vbsMax)
                        if (warns_vbs < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbs=%g has exceeded Vbs_max=%g\n",
                                       vbs, model->HSM2vbsMax);
                            warns_vbs++;
                        }
                }
            } else {
                if (!model->HSM2vbsMaxGiven) {
                    if (model->HSM2_type > 0) {
                        if (vbs > model->HSM2vbdMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbd_max=%g\n",
                                           vbs, model->HSM2vbdMax);
                                warns_vbs++;
                            }
                        if (-1*vbs > model->HSM2vbsrMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbsr_max=%g\n",
                                           vbs, model->HSM2vbsrMax);
                                warns_vbs++;
                            }
                    } else {
                        if (vbs > model->HSM2vbsrMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbsr_max=%g\n",
                                           vbs, model->HSM2vbsrMax);
                                warns_vbs++;
                            }
                        if (-1*vbs > model->HSM2vbdMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbd_max=%g\n",
                                           vbs, model->HSM2vbdMax);
                                warns_vbs++;
                            }
                    }
                } else {
                    if (model->HSM2_type > 0) {
                        if (vbs > model->HSM2vbsMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbs_max=%g\n",
                                           vbs, model->HSM2vbsMax);
                                warns_vbs++;
                            }
                        if (-1*vbs > model->HSM2vbsrMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbsr_max=%g\n",
                                           vbs, model->HSM2vbsrMax);
                                warns_vbs++;
                            }
                    } else {
                        if (vbs > model->HSM2vbsrMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbsr_max=%g\n",
                                           vbs, model->HSM2vbsrMax);
                                warns_vbs++;
                            }
                        if (-1*vbs > model->HSM2vbsMax)
                            if (warns_vbs < maxwarns) {
                                soa_printf(ckt, (GENinstance*) here,
                                           "Vbs=%g has exceeded Vbs_max=%g\n",
                                           vbs, model->HSM2vbsMax);
                                warns_vbs++;
                            }
                    }
                }
            }

            if (!model->HSM2vbdrMaxGiven) {
                if (fabs(vbd) > model->HSM2vbdMax)
                    if (warns_vbd < maxwarns) {
                        soa_printf(ckt, (GENinstance*) here,
                                   "Vbd=%g has exceeded Vbd_max=%g\n",
                                   vbd, model->HSM2vbdMax);
                        warns_vbd++;
                    }
            } else {
                if (model->HSM2_type > 0) {
                    if (vbd > model->HSM2vbdMax)
                        if (warns_vbd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbd=%g has exceeded Vbd_max=%g\n",
                                       vbd, model->HSM2vbdMax);
                            warns_vbd++;
                        }
                    if (-1*vbd > model->HSM2vbdrMax)
                        if (warns_vbd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbd=%g has exceeded Vbdr_max=%g\n",
                                       vbd, model->HSM2vbdrMax);
                            warns_vbd++;
                        }
                } else {
                    if (vbd > model->HSM2vbdrMax)
                        if (warns_vbd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbd=%g has exceeded Vbdr_max=%g\n",
                                       vbd, model->HSM2vbdrMax);
                            warns_vbd++;
                        }
                    if (-1*vbd > model->HSM2vbdMax)
                        if (warns_vbd < maxwarns) {
                            soa_printf(ckt, (GENinstance*) here,
                                       "Vbd=%g has exceeded Vbd_max=%g\n",
                                       vbd, model->HSM2vbdMax);
                            warns_vbd++;
                        }
                }
            }

        }
    }

    return OK;
}
