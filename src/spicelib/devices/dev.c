/* NG-SPICE -- An electrical circuit simulator
 *
 * Copyright (c) 1990 University of California
 * Copyright (c) 2000 Arno W. Peters
 *
 * Permission to use, copy, modify, and distribute this software and
 * its documentation without fee, and without a written agreement is
 * hereby granted, provided that the above copyright notice, this
 * paragraph and the following three paragraphs appear in all copies.
 *
 * This software program and documentation are copyrighted by their
 * authors. The software program and documentation are supplied "as
 * is", without any accompanying services from the authors. The
 * authors do not warrant that the operation of the program will be
 * uninterrupted or error-free. The end-user understands that the
 * program was developed for research purposes and is advised not to
 * rely exclusively on the program for any reason.
 * 
 * IN NO EVENT SHALL THE AUTHORS BE LIABLE TO ANY PARTY FOR DIRECT,
 * INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
 * LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS
 * DOCUMENTATION, EVEN IF THE AUTHORS HAVE BEEN ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE. THE AUTHORS SPECIFICALLY DISCLAIMS ANY
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE
 * SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND THE AUTHORS
 * HAVE NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES,
 * ENHANCEMENTS, OR MODIFICATIONS. */

#include <config.h>
#include <assert.h>

#include <devdefs.h>
#include <ifsim.h>

#include "dev.h"

#define DEVICES_USED "asrc bjt bjt2 bsim1 bsim2 bsim3 bsim3v2 bsim3v1 bsim4 bsim3soipd bsim3soifd bsim3soidd cap cccs ccvs csw dio hfet hfet2 ind isrc jfet ltra mes mesa mos1 mos2 mos3 mos6 mos9 res soi3 sw tra urc vccs vcvs vsrc"

/*
 * Analyses
 */
#define AN_op
#define AN_dc
#define AN_tf
#define AN_ac
#define AN_tran
#define AN_pz
#define AN_disto
#define AN_noise
#define AN_sense

#define ANALYSES_USED "op dc tf ac tran pz disto noise sense"


#include "asrc/asrcitf.h"
#include "bjt/bjtitf.h"
/* #include "bjt2/bjt2itf.h" */
#include "bsim1/bsim1itf.h"
#include "bsim2/bsim2itf.h"
#include "bsim3/bsim3itf.h"
#include "bsim3v1/bsim3v1itf.h"
#include "bsim3v2/bsim3v2itf.h"
#include "bsim4/bsim4itf.h"
#include "bsim3soi_pd/b3soipditf.h"
#include "bsim3soi_fd/b3soifditf.h"
#include "bsim3soi_dd/b3soidditf.h"
#include "cap/capitf.h"
#include "cccs/cccsitf.h"
#include "ccvs/ccvsitf.h"
#include "csw/cswitf.h"
#include "dio/dioitf.h"
#include "hfet1/hfetitf.h"
#include "hfet2/hfet2itf.h"
#include "ind/inditf.h"
#include "isrc/isrcitf.h"
#include "jfet/jfetitf.h"
#include "jfet2/jfet2itf.h"
#include "ltra/ltraitf.h"
#include "mes/mesitf.h"
#include "mesa/mesaitf.h"
#include "mos1/mos1itf.h"
#include "mos2/mos2itf.h"
#include "mos3/mos3itf.h"
#include "mos6/mos6itf.h"
#include "mos9/mos9itf.h"
#include "res/resitf.h"
#include "soi3/soi3itf.h"
#include "sw/switf.h"
#include "tra/traitf.h"
#include "urc/urcitf.h"
#include "vccs/vccsitf.h"
#include "vcvs/vcvsitf.h"
#include "vsrc/vsrcitf.h"


#define DEVNUM 40

SPICEdev *DEVices[DEVNUM];


void
spice_init_devices(void)
{
    /* URC device MUST precede both resistors and capacitors */
    DEVices[ 0] = get_urc_info();
    DEVices[ 1] = get_asrc_info();
    DEVices[ 2] = get_bjt_info();
    DEVices[ 3] = get_bjt_info(); /* Quick hack until bjt2 works */
  /*  DEVices[ 3] = get bjt2_info(); */
    DEVices[ 4] = get_bsim1_info();
    DEVices[ 5] = get_bsim2_info();
    DEVices[ 6] = get_bsim3_info();
    DEVices[ 7] = get_bsim3v1_info();
    DEVices[ 8] = get_bsim3v2_info();
    DEVices[ 9] = get_bsim4_info();
    DEVices[10] = get_b3soipd_info();
    DEVices[11] = get_b3soifd_info();
    DEVices[12] = get_b3soidd_info();
    DEVices[13] = get_cap_info();
    DEVices[14] = get_cccs_info();
    DEVices[15] = get_ccvs_info();
    DEVices[16] = get_csw_info();
    DEVices[17] = get_dio_info();
    DEVices[18] = get_hfeta_info();
    DEVices[19] = get_hfet2_info();
    DEVices[20] = get_ind_info();
    DEVices[21] = get_mut_info();
    DEVices[22] = get_isrc_info();
    DEVices[23] = get_jfet_info();
    DEVices[24] = get_jfet2_info();
    DEVices[25] = get_ltra_info();
    DEVices[26] = get_mes_info();
    DEVices[27] = get_mesa_info();
    DEVices[28] = get_mos1_info();
    DEVices[29] = get_mos2_info();
    DEVices[30] = get_mos3_info();
    DEVices[31] = get_mos6_info();
    DEVices[32] = get_mos9_info();
    DEVices[33] = get_res_info();
    DEVices[34] = get_soi3_info();
    DEVices[35] = get_sw_info();
    DEVices[36] = get_tra_info();
    DEVices[37] = get_vccs_info();
    DEVices[38] = get_vcvs_info();
    DEVices[39] = get_vsrc_info();
    assert(40 == DEVNUM);
}


int
num_devices(void)
{
    return DEVNUM;
}


IFdevice **
devices_ptr(void)
{
    return (IFdevice **) DEVices;
}


SPICEdev **
devices(void)
{
    return DEVices;
}
