/* Configuration file for ng-spice */
#include <config.h>

#include "dev.h"
#include "devdefs.h"

/*
 * Devices
 */
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


/* my internal global constant for number of device types */
int DEVmaxnum = sizeof(DEVices)/sizeof(SPICEdev *);


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
