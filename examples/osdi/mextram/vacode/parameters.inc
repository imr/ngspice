// Copyright (c) 2000-2007, NXP Semiconductor
// Copyright (c) 2007-2014, Delft University of Technology
// Copyright (c) 2015-2020, Auburn University
// All rights reserved, see IP_NOTICE_DISCLAIMER_LICENSE for further information.

//Instance parameters
`IPRnb( dta            ,0.0            ,"degC"                                   ,"Difference between the local and global ambient temperatures" )
aliasparam trise = dta;
aliasparam dtemp = dta;
`IPRoo( mult           ,1.0            ,""            ,0.0         ,inf          ,"Multiplication factor" )

// Mextram parameters
`MPRco( version        ,505.20         ,""            ,505.20      ,505.21       ,"Model version" )
`MPIty( type           ,1              ,""                                       ,"Flag for NPN (1) or PNP (-1) transistor type" )
`MPRco( tref           ,25.0           ,"degC"        ,-273.0      ,inf          ,"Reference temperature" )
`MPIcc( exmod          ,1              ,""            ,0           ,3            ,"Flag for extended modeling of the reverse current gain" )
`MPIcc( exphi          ,1              ,""            ,0           ,1            ,"Flag for distributed high-frequency effects in transient" )
`MPIcc( exavl          ,0              ,""            ,0           ,1            ,"Flag for extended modeling of avalanche currents" )

`ifdef SUBSTRATE
    `MPIcc( exsub          ,1              ,""            ,0           ,1           ,"Flag for extended modeling of substrate currents" )
`endif

// main current parameters
`MPRoz( is             ,22.0a          ,"A"                                      ,"Saturation current of main current" )
`MPRco( nff            ,1.0            ,""            ,1.0         ,inf          ,"Non-ideality factor of forward main current" )
`MPRco( nfr            ,1.0            ,""            ,1.0         ,inf          ,"Non-ideality factor of reverse main current" )
`MPRco( ik             ,0.1            ,"A"           ,1.0p        ,inf          ,"CE high injection knee current" )
`MPRco( ver            ,2.5            ,"V"           ,0.01        ,inf          ,"Reverse Early voltage" )
`MPRco( vef            ,44.0           ,"V"           ,0.01        ,inf          ,"Forward Early voltage" )
`MPRoo( issr           ,1.0            ,""            ,0.0         ,inf          ,"Fraction of saturation current for reverse main current " )
// forward base current parameters
`MPRcz( ibi            ,0.1a           ,"A"                                      ,"Saturation current of ideal base current" )
`MPRco( nbi            ,1.0            ,""            ,0.1         ,inf          ,"Non-ideality factor of ideal base current" )
`MPRcz( ibis           ,0.0            ,"A"                                      ,"Saturation current of ideal side wall base current" )
`MPRco( nbis           ,1.0            ,""            ,0.1         ,inf          ,"Non-ideality factor of ideal side wall base current" )
`MPRcz( ibf            ,2.7f           ,"A"                                      ,"Saturation current of non-ideal forward base current" )
`MPRco( mlf            ,2.0            ,""            ,0.1         ,inf          ,"Non-ideality factor of non-ideal forward base current" )
`MPRcz( ibfs           ,0.0            ,"A"                                      ,"Saturation current of non-ideal side wall forward base current" )
`MPRco( mlfs           ,2.0            ,""            ,0.1         ,inf          ,"Non-ideality factor of non-ideal side wall forward base current" )
// reverse base current parameters
`MPRcz( ibx            ,3.14a          ,"A"                                      ,"Saturation current of extrinsic reverse base current" )
`MPRco( ikbx           ,14.29m         ,"A"           ,1.0p        ,inf          ,"Extrinsic CB high injection knee current" )
`MPRcz( ibr            ,1.0f           ,"A"                                      ,"Saturation current of non-ideal reverse base current" )
`MPRco( mlr            ,2.0            ,""            ,0.1         ,inf          ,"Non-ideality factor of non-ideal reverse base current" )
`MPRcc( xext           ,0.63           ,""            ,0.0         ,1.0          ,"Part of currents and charges that belong to extrinsic region" )
// EB tunneling model parameters
`MPRcz( izeb           ,0.0            ,"A"                                      ,"Pre-factor of EB Zener tunneling current" )
`MPRcz( nzeb           ,22.0           ,""                                       ,"Coefficient of EB Zener tunneling current" )
// CB tunneling model parameters
`MPRcz( izcb           ,0.0            ,"A"                                      ,"Pre-factor of CB Zener tunneling current" )
`MPRcz( nzcb           ,22.0           ,""                                       ,"Coefficient of CB Zener tunneling current" )
// zener tunneling model parameters
`MPRoo( vzmin          ,1.0u           ,"V"           ,0.0         ,inf          ,"Minimum junction reverse voltage to help avoid Zener current numerical issues close to zero bias" )
// 505 avalanche model parameters
`MPIcc( swavl          ,1              ,""            ,0           ,3            ,"Switch of avalanche factor Gem model" )
`MPRcz( aavl           ,400.0          ,""                                       ,"aavl of swavl=1,3 Gem models" )
`MPRoo( cavl           ,-370.0m        ,""            ,-inf        ,0.0          ,"cavl of swavl=1,3 Gem models" )
`MPRoz( itoavl         ,500.0m         ,"A"                                      ,"Current dependence parameter of swavl=1 Gem model" )
`MPRoz( bavl           ,25.0           ,""                                       ,"bavl of swavl=1,3 Gem models" )
`MPRnb( vdcavl         ,100.0m         ,"V"                                      ,"CB diffusion voltage dedicated for swavl=1 Gem model" )
// 504 avalanche model parameters
`MPRco( wavl           ,1.1u           ,"m"           ,1.0n        ,inf          ,"Epilayer thickness used in weak-avalanche model" )
`MPRco( vavl           ,3.0            ,"V"           ,0.01        ,inf          ,"Voltage determining curvature of avalanche current" )
`MPRcz( sfh            ,0.3            ,""                                       ,"Current spreading factor of avalanche model when exavl=1" )
// 505.2.0 avalanche model parameters
`MPRco( ihcavl         ,4.0m           ,"A"           ,1.0p        ,inf          ,"Critical current for velocity saturation of swavl=3 Gem model" )
`MPRoo( davl           ,-370.0m        ,""            ,-inf        ,0.0          ,"Coefficient for controlling decrease of Gem with current in swavl=3 Gem model" )
`MPRoo( eavl           ,-370.0m        ,""            ,-inf        ,0.0          ,"Coefficient for controlling increase of Gem with current in swavl=3 extended Gem model" )
`MPRoo( aexavl         ,0.3            ,""            ,0.0         ,inf          ,"Smoothness parameter for onset of swavl=3 extended Gem model" )
`MPRco( ionexavl       ,4.0m           ,"A"           ,1.0p        ,inf          ,"Onset current of swavl=3 extended Gem model" )
`MPIcc( swgemlim       ,1              ,""            ,0           ,1            ,"Switch of limiting of avalanche factor Gem model" )
// resistance parameters
`MPRco( re             ,5.0            ,"Ohm"         ,1.0m        ,inf          ,"Emitter resistance" )
`MPRco( rbc            ,23.0           ,"Ohm"         ,1.0m        ,inf          ,"Constant part of base resistance" )
`MPRco( rbv            ,18.0           ,"Ohm"         ,1.0m        ,inf          ,"Zero-bias value of variable part of the base resistance" )
`MPRco( rcc            ,12.0           ,"Ohm"         ,1.0m        ,inf          ,"Constant part of collector resistance" )
`MPRcz( rcblx          ,0.0            ,"Ohm"                                    ,"Resistance Collector Buried Layer extrinsic" )
`MPRcz( rcbli          ,0.0            ,"Ohm"                                    ,"Resistance Collector Buried Layer Intrinsic" )
// epilayer dc parameters
`MPRco( rcv            ,150.0          ,"Ohm"         ,1.0m        ,inf          ,"Resistance of un-modulated epilayer" )
`MPRco( scrcv          ,1250.0         ,"Ohm"         ,1.0m        ,inf          ,"Space charge resistance of epilayer" )
`MPRco( ihc            ,4.0m           ,"A"           ,1.0p        ,inf          ,"Critical current for velocity saturation in epilayer" )
`MPRco( axi            ,0.3            ,""            ,0.02        ,inf          ,"Smoothness parameter for onset of quasi-saturation" )
`MPRco( vdc            ,0.68           ,"V"           ,0.05        ,inf          ,"CB diffusion voltage" )
// EB junction capacitance parameters
`MPRcz( cje            ,73.0f          ,"F"                                      ,"Zero-bias EB depletion capacitance" )
`MPRco( vde            ,0.95           ,"V"           ,0.05        ,inf          ,"EB diffusion voltage" )
`MPRco( pe             ,0.4            ,""            ,0.01        ,0.99         ,"EB grading coefficient" )
`MPRcc( xcje           ,0.4            ,""            ,0.0         ,1.0          ,"Sidewall fraction of EB depletion capacitance" )
`MPRcz( cbeo           ,0.0            ,"F"                                      ,"EB overlap capacitance" )
// CB junction capacitance parameters
`MPRcz( cjc            ,78.0f          ,"F"                                      ,"Zero-bias CB depletion capacitance" )
`MPRco( vdcctc         ,0.68           ,"V"           ,0.05        ,inf          ,"CB diffusion voltage of depletion capacitance" )
`MPRco( pc             ,0.5            ,""            ,0.01        ,0.99         ,"CB grading coefficient" )
`MPIcc( swvchc         ,0              ,""            ,0           ,1            ,"Switch of Vch for CB depletion capacitance" )
`MPIcc( swvjunc        ,0              ,""            ,0           ,2            ,"Switch of Vjunc for collector junction capacitance" )
`MPRco( xp             ,0.35           ,""            ,0.0         ,0.99         ,"Constant part of Cjc" )
`MPRco( mc             ,0.5            ,""            ,0.0         ,1.0          ,"Coefficient for current modulation of CB depletion capacitance" )
`MPRcc( xcjc           ,32.0m          ,""            ,0.0         ,1.0          ,"Fraction of CB depletion capacitance under the emitter" )
`MPRcz( cbco           ,0.0            ,"F"                                      ,"CB overlap capacitance" )
`MPIcc( swqex          ,0              ,""            ,0           ,1            ,"Switch for CB diffusion capacitance" )
`MPRco( vdcex          ,0.68           ,"V"           ,0.05        ,inf          ,"CB diffusion voltage of diffusion capacitance" )
// Breakdown for CB junction leakage parameters (not avalanche of IN)
`MPRoc( vbrcb          ,100.0          ,"V"           ,0.0         ,2000.0       ,"Breakdown voltage for CB junction leakage" )
`MPRoc( pbrcb          ,4.0            ,"V"           ,0.0         ,500.0        ,"Breakdown onset tuning parameter for CB junction leakage" )
`MPRoc( frevcb         ,1000.0         ,""            ,1e1         ,1e10         ,"Coefficient for limiting CB junction breakdown leakage current" )
`MPIsw( swjbrcb        ,0              ,""                                       ,"Switch for breakdown in CB junction leakage" )
// transit time parameters
`MPRco( mtau           ,1.0            ,""            ,0.1         ,inf          ,"Non-ideality factor of emitter stored charge" )
`MPRcz( taue           ,2.0p           ,"s"                                      ,"Minimum transit time of stored emitter charge" )
`MPRoz( taub           ,4.2p           ,"s"                                      ,"Transit time of stored base charge" )
`MPRcz( tepi           ,41.0p          ,"s"                                      ,"Transit time of stored epilayer charge" )
`MPRcz( taur           ,520.0p         ,"s"                                      ,"Transit time of reverse extrinsic stored base charge" )
`MPRcz( tauex          ,10.0p          ,"s"                                      ,"Transit time of reverse extrinsic stored epilayer charge of swqex=1" )
// heterojunction parameters
`MPRnb( deg            ,0.0            ,"eV"                                     ,"Bandgap difference over the base" )
// neutral base recombination parameter
`MPRcz( xrec           ,0.0            ,""                                       ,"Pre-factor of the recombination part of Ib1" )
// charge partition parameters
`MPRcc( xqb            ,`one_third     ,""            ,0.0         ,1.0          ,"Emitter-fraction of base diffusion charge" )
`MPRcc( ke             ,0.0            ,""            ,0.0         ,1.0          ,"Fraction of QE in excess phase shift" )
// general temperature scaling parameters
`MPRnb( aqbo           ,0.3            ,""                                       ,"Temperature coefficient of zero-bias base charge" )
`MPRnb( ae             ,0.0            ,""                                       ,"Temperature coefficient of resistivity of the emitter" )
`MPRnb( ab             ,1.0            ,""                                       ,"Temperature coefficient of resistivity of the base" )
`MPRnb( aepi           ,2.5            ,""                                       ,"Temperature coefficient of resistivity of the epilayer" )
`MPRnb( aepiex         ,2.5            ,""                                       ,"Temperature coefficient of reverse transit time of the extrinsic epilayer" )
`MPRnb( aex            ,0.62           ,""                                       ,"Temperature coefficient of resistivity of the extrinsic base" )
`MPRnb( ac             ,2.0            ,""                                       ,"Temperature coefficient of resistivity of the collector contact" )
`MPRnb( acx            ,1.3            ,""                                       ,"Temperature coefficient of extrinsic reverse base current" )
`MPRcz( acbl           ,2.0            ,""                                       ,"Temperature coefficient of resistivity of the collector buried layer" )
`MPRco( vgb            ,1.17           ,"V"           ,0.1         ,inf          ,"Band-gap voltage of base" )
`MPRco( vgc            ,1.18           ,"V"           ,0.1         ,inf          ,"Band-gap voltage of collector" )
`MPRco( vge            ,1.12           ,"V"           ,0.1         ,inf          ,"Band-gap voltage of emitter" )
`MPRco( vgcx           ,1.125          ,"V"           ,0.1         ,inf          ,"Band-gap voltage of extrinsic collector" )
`MPRco( vgj            ,1.15           ,"V"           ,0.1         ,inf          ,"Band-gap voltage recombination EB junction" )
`MPRco( vgzeb          ,1.15           ,"V"           ,0.1         ,inf          ,"Band-gap voltage at Tref for EB tunneling" )
`MPRnb( avgeb          ,4.73e-4        ,"V/K"                                    ,"Temperature coefficient of band-gap voltage for EB tunneling" )
`MPRcz( tvgeb          ,636.0          ,"K"                                      ,"Temperature coefficient of band-gap voltage for EB tunneling" )
`MPRco( vgzcb          ,1.15           ,"V"           ,0.1         ,inf          ,"Band-gap voltage at Tref for CB tunneling" )
`MPRnb( avgcb          ,4.73e-4        ,"V/K"                                    ,"Temperature coefficient of band-gap voltage for CB tunneling" )
`MPRcz( tvgcb          ,636.0          ,"K"                                      ,"Temperature coefficient of band-gap voltage for CB tunneling" )
`MPRnb( dvgte          ,0.05           ,"V"                                      ,"Band-gap voltage difference of emitter stored charge" )
`MPRnb( dais           ,0.0            ,""                                       ,"Fine tuning of temperature dependence of CE saturation current" )
`MPRnb( tnff           ,0.0            ,"/K"                                     ,"Temperature coefficient of nff" )
`MPRnb( tnfr           ,0.0            ,"/K"                                     ,"Temperature coefficient of nfr" )
`MPRnb( tbavl          ,500.0u         ,""                                       ,"Temperature scaling parameter of bavl when swavl=1" )
// 1/f noise parameters
`MPRco( af             ,2.0            ,""            ,0.01        ,inf          ,"Exponent of Flicker-noise of ideal base current" )
`MPRco( afn            ,2.0            ,""            ,0.01        ,inf          ,"Exponent of Flicker-noise of non-ideal base current" )
`MPRcz( kf             ,20.0p          ,""                                       ,"Flicker-noise coefficient of ideal base current" )
`MPRcz( kfn            ,20.0p          ,""                                       ,"Flicker-noise coefficient of non-ideal base current" )
// avalanche noise switch
`MPIcc( kavl           ,0              ,""            ,0           ,1            ,"Switch for white noise contribution due to avalanche" )
// correlated noise parameters
`MPIcc( kc             ,0              ,""            ,0           ,2            ,"Switch for RF correlation noise model selection" )
`MPRcc( ftaun          ,0.0            ,""            ,0.0         ,1.0          ,"Fraction of noise transit time to total transit time" )

`ifdef SUBSTRATE
    `MPRcz( iss            ,48.0a          ,"A"                                      ,"Saturation current of parasitic BCS transistor main current" )
    `MPRcz( icss           ,0.0            ,"A"                                      ,"CS ideal saturation current" )
    `MPRco( iks            ,545.5u         ,"A"           ,1.0p        ,inf          ,"Knee current for BCS transistor main current" )
    `MPRco( ikcs           ,50.0u          ,"A"           ,1.0p        ,inf          ,"Knee current for CS junction diode current" )
    `MPRcz( cjs            ,315.0f         ,"F"                                      ,"Zero-bias CS depletion capacitance" )
    `MPRoo( vds            ,0.62           ,"V"           ,0.05        ,inf          ,"CS diffusion voltage" )
    `MPRoo( ps             ,0.34           ,""            ,0.01        ,0.99         ,"CS grading coefficient" )
    `MPRco( vgs            ,1.20           ,"V"           ,0.1         ,inf          ,"Band-gap voltage of the substrate" )
    `MPRnb( as             ,1.58           ,""                                       ,"Substrate temperature coefficient" )
    `MPRnb( asub           ,2.0            ,""                                       ,"Temperature coefficient for mobility of minorities in the substrate" )
    `MPRcc( xisubi         ,0.0            ,""            ,0.0         ,1.0          ,"Part of substrate current that belongs to intrinsic region" )
    `MPIsw( swvsch         ,0              ,""                                       ,"Switch for Vsc induced high injection in main currents of BCS transistors and CS diode current" )
`endif

// self heating parameters
`ifdef SELFHEATING
    `MPRoz( rth            ,300.0          ,"K/W"                                    ,"Thermal resistance" )
    `MPRcz( cth            ,3.0n           ,"J/K"                                    ,"Thermal capacitance" )
    `MPRnb( ath            ,0.0            ,""                                       ,"Temperature coefficient of thermal resistance" )
`endif

// reliability modeling parameters
`MPRcz( isibrel        ,0.0            ,"A"                                     ,"Saturation current of base current for reliability simulation" )
`MPRco( nfibrel        ,2.0            ,""            ,0.1         ,inf         ,"Non-ideality factor of base current for reliability simulation" )

`MPRcc( vexlim         ,400.0          ,""            ,40.0        ,400.0       ,"Upper limit of exp() function argument for convergence" )
