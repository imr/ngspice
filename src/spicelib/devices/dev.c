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

#include "dev.h"
#include "devdefs.h"

/* Enable the following devices */
#define DEV_asrc
#define DEV_bjt
#define DEV_bsim1
#define DEV_bsim2
#define DEV_bsim3
#define DEV_bsim4
#define DEV_bsim3v1
#define DEV_bsim3v2
#define DEV_cap
#define DEV_cccs
#define DEV_ccvs
#define DEV_csw
#define DEV_dio
#define DEV_ind
#define DEV_isrc
#define DEV_jfet
#define DEV_jfet2
#define DEV_ltra
#define DEV_mes
#define DEV_mos1
#define DEV_mos2
#define DEV_mos3
#define DEV_mos6
#define DEV_res
#define DEV_sw
#define DEV_tra
#define DEV_urc
#define DEV_vccs
#define DEV_vcvs
#define DEV_vsrc

#define DEVICES_USED "asrc bjt bsim1 bsim2 bsim3 bsim3v2 bsim3v1 cap cccs ccvs csw dio ind isrc jfet ltra mes mos1 mos2 mos3 mos6 res sw tra urc vccs vcvs vsrc"

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
#include "cap/capitf.h"
#include "cccs/cccsitf.h"
#include "ccvs/ccvsitf.h"
#include "csw/cswitf.h"
#include "dio/dioitf.h"
#include "ind/inditf.h"
#include "isrc/isrcitf.h"
#include "mos1/mos1itf.h"
#include "mos6/mos6itf.h"
#include "res/resitf.h"
#include "sw/switf.h"
#include "vccs/vccsitf.h"
#include "vcvs/vcvsitf.h"
#include "vsrc/vsrcitf.h"
#include "bsim1/bsim1itf.h"
#include "bsim2/bsim2itf.h"
#include "bsim3/bsim3itf.h"
#include "bsim4/bsim4itf.h"
#include "bsim3v1/bsim3v1itf.h"
#include "bsim3v2/bsim3v2itf.h"
#include "mos2/mos2itf.h"
#include "mos3/mos3itf.h"
#include "jfet/jfetitf.h"
#include "jfet2/jfet2itf.h"
#include "mes/mesitf.h"
#include "ltra/ltraitf.h"
#include "tra/traitf.h"
#include "urc/urcitf.h"


SPICEdev *DEVices[] = {
	/* URC must appear before the resistor, capacitor, and diode */
        &URCinfo,
        &ASRCinfo,
        &BJTinfo,
        &B1info,
        &B2info,
        &BSIM3info,
	&B4info,
	&BSIM3V2info,
	&BSIM3V1info,
        &CAPinfo,
        &CCCSinfo,
        &CCVSinfo,
        &CSWinfo,
        &DIOinfo,
        &INDinfo,
        &MUTinfo,
        &ISRCinfo,
        &JFETinfo,
        &JFET2info,
        &LTRAinfo,
        &MESinfo,
        &MOS1info,
        &MOS2info,
        &MOS3info,
        &MOS6info,
        &RESinfo,
        &SWinfo,
        &TRAinfo,
        &VCCSinfo,
        &VCVSinfo,
        &VSRCinfo,
};


int
num_devices(void)
{
    return sizeof(DEVices)/sizeof(SPICEdev *);
}

IFdevice **
devices_ptr(void)
{
    return (IFdevice **) DEVices;
}
