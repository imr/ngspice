
//
//  This file is the top-level declaration for the following CMC Verilog-A models:
//
//      r2_cmc            CMC 2-terminal resistor model
//      r2_et_cmc         CMC 2-terminal resistor model with self-heating
//

//
//  Physical constants and other generally useful numbers
//

`include "discipline.h"
`define TABS_NIST2004     2.73150000e+02    // (NIST2004) 0C in K
`define QQ_NIST2004       1.60217653e-19    // (NIST2004) mag. of electronic charge (C)
`define KB_NIST2004       1.38065050e-23    // (NIST2004) Boltzmann constant (J/K)
`define oneThird          3.3333333333333333e-01

//
//  Clipping macros, these smoothly limit to lower, upper, or both lower and upper
//  limits. Rather than using a sqrt or log-exp form, which affects values
//  everywhere, these use a conditional form that is continuous in function
//  and derivative. If a value is not clipped then no exp() evaluation occurs.
//  Smooth limiting is preferable to hard limiting (although latter can still
//  be useful for enforcing parameter limits) for bias dependent quantities
//  as derivatives do not become zero or have discontinuities.
//

`define CLIPL0p1(XCLIP,X,LOWER) \
    if (X<(LOWER+0.1)) \
        XCLIP    =  LOWER+0.1*exp(10.0*(X-LOWER)-1.0); \
    else \
        XCLIP    =  X;
`define CLIPU0p1(XCLIP,X,UPPER) \
    if (X>(UPPER-0.1)) \
        XCLIP    =  UPPER-0.1*exp(10.0*(UPPER-X)-1.0); \
    else \
        XCLIP    =  X;
`define CLIPB0p1(XCLIP,X,LOWER,UPPER) \
    if (X<(LOWER+0.1)) \
        XCLIP    =  LOWER+0.1*exp(10.0*(X-LOWER)-1.0); \
    else if (X>(UPPER-0.1)) \
        XCLIP    =  UPPER-0.1*exp(10.0*(UPPER-X)-1.0); \
    else \
        XCLIP    =  X;

`define CLIPL1p0(XCLIP,X,LOWER) \
    if (X<(LOWER+1.0)) \
        XCLIP    =  LOWER+exp(X-LOWER-1.0); \
    else \
        XCLIP    =  X;
`define CLIPU1p0(XCLIP,X,UPPER) \
    if (X>(UPPER-1.0)) \
        XCLIP    =  UPPER-exp(UPPER-X-1.0); \
    else \
        XCLIP    =  X;
`define CLIPB1p0(XCLIP,X,LOWER,UPPER) \
    if (X<(LOWER+1.0)) \
        XCLIP    =  LOWER+exp(X-LOWER-1.0); \
    else if (X>(UPPER-1.0)) \
        XCLIP    =  UPPER-exp(UPPER-X-1.0); \
    else \
        XCLIP    =  X;

`ifdef insideADMS
    `ifdef notInsideADMS
        `undef  notInsideADMS
    `endif
`else
    `define notInsideADMS
`endif
`ifdef __VAMS_COMPACT_MODELING__
    `ifdef not__VAMS_COMPACT_MODELING__
        `undef  not__VAMS_COMPACT_MODELING__
    `endif
`else
    `define not__VAMS_COMPACT_MODELING__
`endif

//
//  Conditional definitions of macros so that the code will work with
//  Verilog-A 2.1 and 2.2, and ADMS. The "des" description argument is intended to
//  be a short description, the "inf" information argument is intended to be
//  a detailed description (e.g. for display as part of on-line help).
//
//  MPR      model    parameter real
//  MPI      model    parameter integer
//  IPR      instance parameter real
//  IPI      instance parameter integer
//  IPM      instance parameter mFactor (multiplicity, implicit for LRM2.2)
//  OPP      operating point parameter, includes units and description for printing
//
//  There are some issues with passing range directives with some compilers,
//  so for each parameter declaration there are 5 versions:
//  cc       closed lower bound, closed upper bound
//  co       closed lower bound, open   upper bound
//  oc       open   lower bound, closed upper bound
//  oo       open   lower bound, open   upper bound
//  nb       no bounds
//

//`ifdef __VAMS_COMPACT_MODELING__
    `define ALIAS(alias,parameter) aliasparam alias = parameter;
    `define ERROR(str) \
        begin \
            $strobe(str); \
            $finish(1); \
        end
    `define WARNING(str) $strobe(str);
    `define OPP(nam,uni,des)               (*units=uni                   desc=des*) real nam;
    `define MPRcc(nam,def,uni,lwr,upr,des) (*units=uni,                  desc=des*) parameter real    nam=def from[lwr:upr];
    `define MPRco(nam,def,uni,lwr,upr,des) (*units=uni,                  desc=des*) parameter real    nam=def from[lwr:upr);
    `define MPRoc(nam,def,uni,lwr,upr,des) (*units=uni,                  desc=des*) parameter real    nam=def from(lwr:upr];
    `define MPRoo(nam,def,uni,lwr,upr,des) (*units=uni,                  desc=des*) parameter real    nam=def from(lwr:upr);
    `define MPRnb(nam,def,uni,        des) (*units=uni,                  desc=des*) parameter real    nam=def;
    `define MPIcc(nam,def,uni,lwr,upr,des) (*units=uni,                  desc=des*) parameter integer nam=def from[lwr:upr];
    `define MPIco(nam,def,uni,lwr,upr,des) (*units=uni,                  desc=des*) parameter integer nam=def from[lwr:upr);
    `define MPIoc(nam,def,uni,lwr,upr,des) (*units=uni,                  desc=des*) parameter integer nam=def from(lwr:upr];
    `define MPIoo(nam,def,uni,lwr,upr,des) (*units=uni,                  desc=des*) parameter integer nam=def from(lwr:upr);
    `define MPInb(nam,def,uni,        des) (*units=uni,                  desc=des*) parameter integer nam=def;
    `define IPRcc(nam,def,uni,lwr,upr,des) (*units=uni, type="instance", desc=des*) parameter real    nam=def from[lwr:upr];
    `define IPRco(nam,def,uni,lwr,upr,des) (*units=uni, type="instance", desc=des*) parameter real    nam=def from[lwr:upr);
    `define IPRoc(nam,def,uni,lwr,upr,des) (*units=uni, type="instance", desc=des*) parameter real    nam=def from(lwr:upr];
    `define IPRoo(nam,def,uni,lwr,upr,des) (*units=uni, type="instance", desc=des*) parameter real    nam=def from(lwr:upr);
    `define IPRnb(nam,def,uni,        des) (*units=uni, type="instance", desc=des*) parameter real    nam=def;
    `define IPIcc(nam,def,uni,lwr,upr,des) (*units=uni, type="instance", desc=des*) parameter integer nam=def from[lwr:upr];
    `define IPIco(nam,def,uni,lwr,upr,des) (*units=uni, type="instance", desc=des*) parameter integer nam=def from[lwr:upr);
    `define IPIoc(nam,def,uni,lwr,upr,des) (*units=uni, type="instance", desc=des*) parameter integer nam=def from(lwr:upr];
    `define IPIoo(nam,def,uni,lwr,upr,des) (*units=uni, type="instance", desc=des*) parameter integer nam=def from(lwr:upr);
    `define IPInb(nam,def,uni,        des) (*units=uni, type="instance", desc=des*) parameter integer nam=def;
    `define IPM                            parameter real      m=1   from(0:inf);
    `define TESTGIVEN(parameter) $param_given(parameter)
    `define GIVEN(parameter,variable,true,false) \
        begin \
            if ($param_given(parameter)) \
                variable = true; \
            else \
                variable = false; \
        end
//    `define SCALE \
//        begin \
//            if ($param_given(scale)) \
//                scaleFac =  scale; \
//            else \
//                scaleFac = $simparam("scale",1.0); \
//        end
//    `define SHRINKL \
//        begin \
//            if ($param_given(shrink)) \
//                shrinkL  =  1.0-0.01*shrink; \
//            else \
//                shrinkL  =  1.0-0.01*$simparam("shrink",0.0); \
//        end
//    `define RTHRESH \
//        begin \
//            if ($param_given(rthresh)) \
//                rthrR2   =  rthresh; \
//            else \
//                rthrR2   = $simparam("rthresh",1.0e-03); \
//        end
//`else // not__VAMS_COMPACT_MODELING__
//    `define ALIAS(alias,parameter)
//    `ifdef insideADMS
//        `define ERROR(str) \
//            begin \
//                $strobe(str); \
//                $finish(1); \
//            end
//        `define WARNING(str) $strobe(str);
//        `define OPP(nam,uni,des) real nam (*units=uni desc=des ask="yes"*);
//        `define MPRcc(nam,def,uni,lwr,upr,des) parameter real    nam=def from[lwr:upr] (*units=uni                 ask="yes" info=des*);
//        `define MPRco(nam,def,uni,lwr,upr,des) parameter real    nam=def from[lwr:upr) (*units=uni                 ask="yes" info=des*);
//        `define MPRoc(nam,def,uni,lwr,upr,des) parameter real    nam=def from(lwr:upr] (*units=uni                 ask="yes" info=des*);
//        `define MPRoo(nam,def,uni,lwr,upr,des) parameter real    nam=def from(lwr:upr) (*units=uni                 ask="yes" info=des*);
//        `define MPRnb(nam,def,uni,        des) parameter real    nam=def               (*units=uni                 ask="yes" info=des*);
//        `define MPIcc(nam,def,uni,lwr,upr,des) parameter integer nam=def from[lwr:upr] (*units=uni                 ask="yes" info=des*);
//        `define MPIco(nam,def,uni,lwr,upr,des) parameter integer nam=def from[lwr:upr) (*units=uni                 ask="yes" info=des*);
//        `define MPIoc(nam,def,uni,lwr,upr,des) parameter integer nam=def from(lwr:upr] (*units=uni                 ask="yes" info=des*);
//        `define MPIoo(nam,def,uni,lwr,upr,des) parameter integer nam=def from(lwr:upr) (*units=uni                 ask="yes" info=des*);
//        `define MPInb(nam,def,uni,        des) parameter integer nam=def               (*units=uni                 ask="yes" info=des*);
//        `define IPRcc(nam,def,uni,lwr,upr,des) parameter real    nam=def from[lwr:upr] (*units=uni type="instance" ask="yes" info=des*);
//        `define IPRco(nam,def,uni,lwr,upr,des) parameter real    nam=def from[lwr:upr) (*units=uni type="instance" ask="yes" info=des*);
//        `define IPRoc(nam,def,uni,lwr,upr,des) parameter real    nam=def from(lwr:upr] (*units=uni type="instance" ask="yes" info=des*);
//        `define IPRoo(nam,def,uni,lwr,upr,des) parameter real    nam=def from(lwr:upr) (*units=uni type="instance" ask="yes" info=des*);
//        `define IPRnb(nam,def,uni,        des) parameter real    nam=def               (*units=uni type="instance" ask="yes" info=des*);
//        `define IPIcc(nam,def,uni,lwr,upr,des) parameter integer nam=def from[lwr:upr] (*units=uni type="instance" ask="yes" info=des*);
//        `define IPIco(nam,def,uni,lwr,upr,des) parameter integer nam=def from[lwr:upr) (*units=uni type="instance" ask="yes" info=des*);
//        `define IPIoc(nam,def,uni,lwr,upr,des) parameter integer nam=def from(lwr:upr] (*units=uni type="instance" ask="yes" info=des*);
//        `define IPIoo(nam,def,uni,lwr,upr,des) parameter integer nam=def from(lwr:upr) (*units=uni type="instance" ask="yes" info=des*);
//        `define IPInb(nam,def,uni,        des) parameter integer nam=def               (*units=uni type="instance" ask="yes" info=des*);
//        `define IPM                            parameter real      m=1   from(0:inf)   (*units=""  type="instance" ask="yes" info="multiplicity factor"*);
//        `define TESTGIVEN(parameter) $given(parameter)
//        `define GIVEN(parameter,variable,true,false) \
//            begin \
//                if ($given(parameter)) \
//                    variable = true; \
//                else \
//                    variable = false; \
//            end
//        `define SCALE \
//            begin \
//                if ($given(scale)) \
//                    scaleFac =  scale; \
//                else \
//                    scaleFac = $scale; \
//            end
//        `define SHRINKL \
//            begin \
//                if ($given(shrink)) \
//                    shrinkL  =  1.0-0.01*shrink; \
//                else \
//                    shrinkL  =  $shrinkl("m"); \
//            end
//        `define RTHRESH rthrR2   =  rthresh;
//    `else // notInsideADMS
//        `define ERROR(str) \
//            begin \
//                $strobe(str); \
//                $finish(1); \
//            end
//        `define WARNING(str) $strobe(str);
//        `define OPP(nam,uni,des) real nam;
//        `define MPRcc(nam,def,uni,lwr,upr,des) parameter real    nam=def from[lwr:upr];
//        `define MPRco(nam,def,uni,lwr,upr,des) parameter real    nam=def from[lwr:upr);
//        `define MPRoc(nam,def,uni,lwr,upr,des) parameter real    nam=def from(lwr:upr];
//        `define MPRoo(nam,def,uni,lwr,upr,des) parameter real    nam=def from(lwr:upr);
//        `define MPRnb(nam,def,uni,        des) parameter real    nam=def;
//        `define MPIcc(nam,def,uni,lwr,upr,des) parameter integer nam=def from[lwr:upr];
//        `define MPIco(nam,def,uni,lwr,upr,des) parameter integer nam=def from[lwr:upr);
//        `define MPIoc(nam,def,uni,lwr,upr,des) parameter integer nam=def from(lwr:upr];
//        `define MPIoo(nam,def,uni,lwr,upr,des) parameter integer nam=def from(lwr:upr);
//        `define MPInb(nam,def,uni,        des) parameter integer nam=def;
//        `define IPRcc(nam,def,uni,lwr,upr,des) parameter real    nam=def from[lwr:upr];
//        `define IPRco(nam,def,uni,lwr,upr,des) parameter real    nam=def from[lwr:upr);
//        `define IPRoc(nam,def,uni,lwr,upr,des) parameter real    nam=def from(lwr:upr];
//        `define IPRoo(nam,def,uni,lwr,upr,des) parameter real    nam=def from(lwr:upr);
//        `define IPRnb(nam,def,uni,        des) parameter real    nam=def;
//        `define IPIcc(nam,def,uni,lwr,upr,des) parameter integer nam=def from[lwr:upr];
//        `define IPIco(nam,def,uni,lwr,upr,des) parameter integer nam=def from[lwr:upr);
//        `define IPIoc(nam,def,uni,lwr,upr,des) parameter integer nam=def from(lwr:upr];
//        `define IPIoo(nam,def,uni,lwr,upr,des) parameter integer nam=def from(lwr:upr);
//        `define IPInb(nam,def,uni,        des) parameter integer nam=def;
//        `define IPM                            parameter real      m=1   from(0:inf);
//        `define TESTGIVEN(parameter) 1
//        `define GIVEN(parameter,variable,true,false) variable = true;
        `define SCALE scaleFac = scale;
        `define SHRINKL shrinkL  = 1.0-0.01*shrink;
        `define RTHRESH rthrR2   = rthresh;
//    `endif
//`endif
