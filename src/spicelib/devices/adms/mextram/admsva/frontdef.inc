// Copyright (c) 2000-2007, NXP Semiconductor
// Copyright (c) 2007-2014, Delft University of Technology
// Copyright (c) 2015, Auburn University
// All rights reserved, see IP_NOTICE_DISCLAIMER_LICENSE for further information.

// Front definitions

`include "discipline.h"

// Numerical, physical and model constants
`define TEN_M40 1.0e-40
`define TEN_M07 1.0e-7
`define C2K     273.15
`define KB      1.3806226e-23
`define QQ      1.6021918e-19
`define KBdivQQ   8.61708691805812512584e-5
`define one_third 0.33333333333333333333
`define one_sixth 0.16666666666666666667
`define VDLOW   0.05
`define AJE     3.0
`define AJC     2.0
`define AJS     2.0
`define VEXLIM  400.0
`define PI      3.1415926

`define NGSPICE_ADMS

// Desriptions and units
//`ifdef __VAMS_COMPACT_MODELING__
`ifdef NGSPICE_ADMS
    `define OPP(nam,uni,des) (* desc="des", units="uni" *) real nam;
    `define PAR(des,uni) (* desc="des", units="uni" *) parameter real
    `define PAI(des,uni) (* desc="des", units="uni" *) parameter integer
`else
    `define OPP(nam,uni,des)
    `define PAR(des,uni) parameter real
    `define PAI(des,uni) parameter integer
`endif

`define PGIVEN $param_given

// ADMS specific definitions
`ifdef insideADMS
    `define P(p) (*p*)
    `define MODEL @(initial_model)
    `define INSTANCE @(initial_instance)
    `define NOISE @(noise)
`else
    `define P(p)
    `define MODEL
    `define INSTANCE
    `define NOISE
`endif

// Smooth limitting functions
`define max_hyp0(result, x, epsilon)\
    eps2 = epsilon * epsilon;\
    x2 = x * x;\
    if (x < 0.0)\
        result = 0.5 * eps2 / (sqrt(x2 + eps2) - x);\
    else\
        result = 0.5 * (sqrt(x2 + eps2) + x);\
    result=result

`define min_logexp(result, x, x0, a)\
    dxa = (x - x0) / (a);\
    if (x < x0)\
        result = x  - a * ln(1.0 + exp(dxa));\
    else\
        result = x0 - a * ln(1.0 + exp(-dxa));\
    result=result

`define max_logexp(result, x, x0, a)\
    dxa = (x - x0) / (a);\
    if (x < x0)\
        result = x0 + a * ln(1.0 + exp(dxa));\
    else\
        result = x  + a * ln(1.0 + exp(-dxa));\
    result=result

`define expLin(result, x)\
    if (x < `VEXLIM) begin\
        result = exp(x);\
    end else begin\
        expl = exp(`VEXLIM);\
        result = expl  * (1.0 + (x - `VEXLIM));\
    end

`define linLog(result, x, vlim)\
    if (x < vlim)\
        result = x;\
    else\
        result = vlim + ln(1.0 + (x - vlim));\
    result=result

//  Macros for the model/instance parameters
//
//  MPRxx    model    parameter real
//  MPIxx    model    parameter integer
//     ||
//     cc    closed lower bound, closed upper bound
//     oo    open   lower bound, open   upper bound
//     co    closed lower bound, open   upper bound
//     oc    open   lower bound, closed upper bound
//     cz    closed lower bound=0, open upper bound=inf
//     oz    open   lower bound=0, open upper bound=inf
//     nb    no bounds
//     ex    no bounds with exclude
//     sw    switch(integer only, values  0=false  and  1=true)
//     ty    switch(integer only, values -1=p-type and +1=n-type)
//
//
`define MPRnb(nam,def,uni,        des) (*units=uni,                   desc=des*) parameter real    nam=def ;
`define MPRex(nam,def,uni,exc,    des) (*units=uni,                   desc=des*) parameter real    nam=def exclude exc ;
`define MPRcc(nam,def,uni,lwr,upr,des) (*units=uni,                   desc=des*) parameter real    nam=def from[lwr:upr] ;
`define MPRoo(nam,def,uni,lwr,upr,des) (*units=uni,                   desc=des*) parameter real    nam=def from(lwr:upr) ;
`define MPRco(nam,def,uni,lwr,upr,des) (*units=uni,                   desc=des*) parameter real    nam=def from[lwr:upr) ;
`define MPRoc(nam,def,uni,lwr,upr,des) (*units=uni,                   desc=des*) parameter real    nam=def from(lwr:upr] ;
`define MPRcz(nam,def,uni,        des) (*units=uni,                   desc=des*) parameter real    nam=def from[  0:inf);
`define MPRoz(nam,def,uni,        des) (*units=uni,                   desc=des*) parameter real    nam=def from(  0:inf);

`define MPInb(nam,def,uni,        des) (*units=uni,                   desc=des*) parameter integer nam=def ;
`define MPIex(nam,def,uni,exc,    des) (*units=uni,                   desc=des*) parameter integer nam=def exclude exc ;
`define MPIcc(nam,def,uni,lwr,upr,des) (*units=uni,                   desc=des*) parameter integer nam=def from[lwr:upr] ;
`define MPIoo(nam,def,uni,lwr,upr,des) (*units=uni,                   desc=des*) parameter integer nam=def from(lwr:upr) ;
`define MPIco(nam,def,uni,lwr,upr,des) (*units=uni,                   desc=des*) parameter integer nam=def from[lwr:upr) ;
`define MPIoc(nam,def,uni,lwr,upr,des) (*units=uni,                   desc=des*) parameter integer nam=def from(lwr:upr] ;
`define MPIcz(nam,def,uni,        des) (*units=uni,                   desc=des*) parameter integer nam=def from[  0:inf);
`define MPIoz(nam,def,uni,        des) (*units=uni,                   desc=des*) parameter integer nam=def from(  0:inf);

`define MPIsw(nam,def,uni,        des) (*units=uni,                   desc=des*) parameter integer nam=def from[  0:  1] ;
`define MPIty(nam,def,uni,        des) (*units=uni,                   desc=des*) parameter integer nam=def from[ -1:  1] exclude 0 ;

