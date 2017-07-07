#!/bin/bash
hspice -hdlpath ../code -i idvgnmos.sp -o idvgnmos
hspice -hdlpath ../code -i idvgpmos.sp -o idvgpmos
hspice -hdlpath ../code -i idvdnmos.sp -o idvdnmos
hspice -hdlpath ../code -i idvdpmos.sp -o idvdpmos
hspice -hdlpath ../code -i gummel_n.sp -o gummel_n
hspice -hdlpath ../code -i gummel_p.sp -o gummel_p
hspice -hdlpath ../code -i invdc.sp -o invdc
hspice -hdlpath ../code -i inverter_transient.sp -o inverter_transient
hspice -hdlpath ../code -i ringosc_17stg.sp -o ringosc_17stg
hspice -hdlpath ../code -i ac.sp -o ac
hspice -hdlpath ../code -i noise.sp -o noise
hspice -hdlpath ../code -i rdsgeo.sp -o rdsgeo
hspice -hdlpath ../code -i cfrgeo.sp -o cfrgeo
