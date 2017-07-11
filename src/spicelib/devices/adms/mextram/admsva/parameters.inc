// Mextram parameters  
  
parameter integer LEVEL = 504      from [504:505)      
                 `ATTR(desc="Model level");  
parameter real    TREF  = 25.0     from [-273.0:inf)   
                 `ATTR(desc="Reference temperature");   
parameter real    DTA   = 0.0                          
                 `ATTR(desc="Difference between the local and global ambient temperatures");
parameter integer EXMOD = 1        from [0:1]          
                 `ATTR(desc="Flag for extended modeling of the reverse current gain");   
parameter integer EXPHI = 1        from [0:1]          
                 `ATTR(desc="Flag for the distributed high-frequency effects in transient");   
parameter integer EXAVL = 0        from [0:1]          
                 `ATTR(desc="Flag for extended modeling of avalanche currents");   
   
parameter real    IS    = 22.0a    from (0.0:inf)      
                 `ATTR(desc="Collector-emitter saturation current");    
parameter real    IK    = 0.1      from [1.0p:inf)     
                 `ATTR(desc="Collector-emitter high injection knee current");   
parameter real    VER   = 2.5      from [0.01:inf)     
                 `ATTR(desc="Reverse Early voltage");   
parameter real    VEF   = 44.0     from [0.01:inf)     
                 `ATTR(desc="Forward Early voltage");   
parameter real    BF    = 215.0    from [0.1m:inf)     
                 `ATTR(desc="Ideal forward current gain");   
parameter real    IBF   = 2.7f     from [0.0:inf)      
                 `ATTR(desc="Saturation current of the non-ideal forward base current");   
parameter real    MLF   = 2.0      from [0.1:inf)      
                 `ATTR(desc="Non-ideality factor of the non-ideal forward base current");   
parameter real    XIBI  = 0.0      from [0.0:1.0]      
                 `ATTR(desc="Part of ideal base current that belongs to the sidewall");   
// begin: RvdT, November 2008, BE tunneling current parameters:
parameter real    IZEB   = 0.0     from [0.0:inf)      
                 `ATTR(desc="Pre-factor of emitter-base Zener tunneling current");
parameter real    NZEB   = 22.0     from [0.0:inf)      
                 `ATTR(desc="Coefficient of emitter-base Zener tunneling current");
// end: RvdT, November 2008, EB tunneling current parameters:
parameter real    BRI   = 7.0      from [1.0e-4:inf)   
                 `ATTR(desc="Ideal reverse current gain");   
parameter real    IBR   = 1.0f     from [0.0:inf)      
                 `ATTR(desc="Saturation current of the non-ideal reverse base current");   
parameter real    VLR   = 0.2                          
                 `ATTR(desc="Cross-over voltage of the non-ideal reverse base current");   
parameter real    XEXT  = 0.63     from [0.0:1.0]      
                 `ATTR(desc="Part of currents and charges that belong to extrinsic region");     
   
parameter real    WAVL  = 1.1u     from [1.0n:inf)     
                 `ATTR(desc="Epilayer thickness used in weak-avalanche model");   
parameter real    VAVL  = 3.0      from [0.01:inf)     
                 `ATTR(desc="Voltage determining curvature of avalanche current");   
parameter real    SFH   = 0.3      from [0.0:inf)      
                 `ATTR(desc="Current spreading factor of avalanche model when EXAVL=1");   
// RvdT, 22-02-2008: for MXT 504.7
// increased lower clipping values RE, RBC, RBV, RCC, RCV, SCRCV 
// from 1u to 1m: 
parameter real    RE    = 5.0      from [1.0m:inf)     
                 `ATTR(desc="Emitter resistance");   
parameter real    RBC   = 23.0     from [1.0m:inf)     
                 `ATTR(desc="Constant part of the base resistance");   
parameter real    RBV   = 18.0     from [1.0m:inf)     
                 `ATTR(desc="Zero-bias value of the variable part of the base resistance");   
parameter real    RCC   = 12.0     from [1.0m:inf)     
                 `ATTR(desc="Constant part of the collector resistance");   
parameter real    RCV   = 150.0    from [1.0m:inf)     
                 `ATTR(desc="Resistance of the un-modulated epilayer");   
parameter real    SCRCV = 1250.0   from [1.0m:inf)     
                 `ATTR(desc="Space charge resistance of the epilayer");   
parameter real    IHC   = 4.0m     from [1.0p:inf)     
                 `ATTR(desc="Critical current for velocity saturation in the epilayer");   
parameter real    AXI   = 0.3      from [0.02:inf)     
                 `ATTR(desc="Smoothness parameter for the onset of quasi-saturation");   
  
parameter real    CJE    = 73.0f   from [0.0:inf)      
                 `ATTR(desc="Zero-bias emitter-base depletion capacitance");   
parameter real    VDE    = 0.95    from [0.05:inf)     
                 `ATTR(desc="Emitter-base diffusion voltage");   
parameter real    PE     = 0.4     from [0.01:0.99)    
                 `ATTR(desc="Emitter-base grading coefficient");   
parameter real    XCJE   = 0.4     from [0.0:1.0]      
                 `ATTR(desc="Sidewall fraction of the emitter-base depletion capacitance");   
parameter real    CBEO   = 0.0     from [0.0:inf)      
                 `ATTR(desc="Emitter-base overlap capacitance");     
  
parameter real    CJC    = 78.0f   from [0.0:inf)      
                 `ATTR(desc="Zero-bias collector-base depletion capacitance");   
parameter real    VDC    = 0.68    from [0.05:inf)     
                 `ATTR(desc="Collector-base diffusion voltage");   
parameter real    PC     = 0.5     from [0.01:0.99)    
                 `ATTR(desc="Collector-base grading coefficient");   
parameter real    XP     = 0.35    from [0.0:0.99)     
                 `ATTR(desc="Constant part of Cjc");   
parameter real    MC     = 0.5     from [0.0:1.0)      
                 `ATTR(desc="Coefficient for current modulation of CB depletion capacitance");     
parameter real    XCJC   = 32.0m   from [0.0:1.0]      
                 `ATTR(desc="Fraction of CB depletion capacitance under the emitter");   
// RvdT, 30-11-2007: introduced RCBLX and RCBLI:
parameter real    RCBLX   = 0.0    from [0.0:inf)      
                 `ATTR(desc="Resistance Collector Buried Layer eXtrinsic");   
parameter real    RCBLI   = 0.0    from [0.0:inf)      
                 `ATTR(desc="Resistance Collector Buried Layer Intrinsic");   
parameter real    CBCO   = 0.0     from [0.0:inf)      
                 `ATTR(desc="Collector-base overlap capacitance");    
  
parameter real    MTAU    = 1.0    from [0.1:inf)      
                 `ATTR(desc="Non-ideality factor of the emitter stored charge");   
parameter real    TAUE    = 2.0p   from [0.0:inf)      
                 `ATTR(desc="Minimum transit time of stored emitter charge");   
parameter real    TAUB    = 4.2p   from (0.0:inf)      
                 `ATTR(desc="Transit time of stored base sharge");   
parameter real    TEPI    = 41.0p  from [0.0:inf)      
                 `ATTR(desc="Transit time of stored epilayer charge");   
parameter real    TAUR    = 520.0p from [0.0:inf)      
                 `ATTR(desc="Transit time of reverse extrinsic stored base charge");    
   
parameter real    DEG     = 0.0                        
                 `ATTR(desc="Bandgap difference over the base");   
parameter real    XREC    = 0.0    from [0.0:inf)      
                  `ATTR(desc="Pre-factor of the recombination part of Ib1");     
   
parameter real    AQBO    = 0.3                        
                 `ATTR(desc="Temperature coefficient of the zero-bias base charge");   
parameter real    AE      = 0.0                        
                 `ATTR(desc="Temperature coefficient of the resistivity of the emitter");   
parameter real    AB      = 1.0                        
                 `ATTR(desc="Temperature coefficient of the resistivity of the base");   
parameter real    AEPI    = 2.5                        
                 `ATTR(desc="Temperature coefficient of the resistivity of the epilayer");   
parameter real    AEX     = 0.62                       
                 `ATTR(desc="Temperature coefficient of the resistivity of the extrinsic base");     
parameter real    AC      = 2.0                        
                 `ATTR(desc="Temperature coefficient of the resistivity of the collector contact");   
// RvdT, 30-01-2007: introduced ACBL
parameter real    ACBL    = 2.0    from [0.0:inf)                            
                 `ATTR(desc="Temperature coefficient of the resistivity of the collector buried layer");   
parameter real    DVGBF   = 50.0m                      
                 `ATTR(desc="Band-gap voltage difference of the forward current gain");    
parameter real    DVGBR   = 45.0m                      
                 `ATTR(desc="Band-gap voltage difference of the reverse current gain");   
parameter real    VGB     = 1.17   from [0.1:inf)      
                 `ATTR(desc="Band-gap voltage of the base");   
parameter real    VGC     = 1.18   from [0.1:inf)      
                 `ATTR(desc="Band-gap voltage of the collector");   
parameter real    VGJ     = 1.15   from [0.1:inf)      
                 `ATTR(desc="Band-gap voltage recombination emitter-base junction");   
parameter real    VGZEB     = 1.15   from [0.1:inf)      
                 `ATTR(desc="Band-gap voltage at Tref of Zener effect emitter-base junction");
parameter real    AVGEB     = 4.73e-4   from (-inf:inf)      
                 `ATTR(desc="Temperature coefficient band-gap voltage for Zener effect emitter-base junction");
parameter real    TVGEB     = 636.0   from [0.0:inf)      
                 `ATTR(desc="Temperature coefficient band-gap voltage for Zener effect emitter-base junction");
parameter real    DVGTE   = 0.05                       
                 `ATTR(desc="Band-gap voltage difference of emitter stored charge");   
parameter real    DAIS    = 0.0                       
                 `ATTR(desc="Fine tuning of temperature dependence of C-E saturation current");  
   
parameter real    AF      = 2.0    from [0.01:inf)     
                 `ATTR(desc="Exponent of the Flicker-noise");   
parameter real    KF      = 20.0p  from [0.0:inf)      
                 `ATTR(desc="Flicker-noise coefficient of the ideal base current");   
parameter real    KFN     = 20.0p  from [0.0:inf)      
                 `ATTR(desc="Flicker-noise coefficient of the non-ideal base current");   
parameter integer KAVL    = 0      from [0:1]          
                 `ATTR(desc="Switch for white noise contribution due to avalanche"); 

`ifdef SUBSTRATE   
parameter real    ISS     = 48.0a  from [0.0:inf)      
                 `ATTR(desc="Base-substrate saturation current");
parameter real    ICSS    = -1.0   from (-inf:inf)      
                 `ATTR(desc="Collector-substrate ideal saturation current");   
parameter real    IKS     = 250.0u from [1.0p:inf)     
                 `ATTR(desc="Base-substrate high injection knee current");   
parameter real    CJS     = 315.0f from [0:inf)      
                 `ATTR(desc="Zero-bias collector-substrate depletion capacitance");   
parameter real    VDS     = 0.62   from (0.05:inf)     
                 `ATTR(desc="Collector-substrate diffusion voltage");   
parameter real    PS      = 0.34   from (0.01:0.99)    
                 `ATTR(desc="Collector-substrate grading coefficient");   
parameter real    VGS     = 1.20   from [0.1:inf)      
                 `ATTR(desc="band-gap voltage of the substrate");   
parameter real    AS      = 1.58                       
                 `ATTR(desc="Substrate temperature coefficient");
parameter real    ASUB    = 2.0                       
                 `ATTR(desc="Temperature coefficient for mobility of minorities in the substrate");   
`endif
 
`ifdef SELFHEATING
parameter real    RTH     = 300.0  from (0.0:inf)      
                 `ATTR(desc="Thermal resistance");   
parameter real    CTH     = 3.0n   from [0.0:inf)      
                 `ATTR(desc="Thermal capacitance"); 
parameter real    ATH     = 0.0                        
                 `ATTR(desc="Temperature coefficient of the thermal resistance");      
`endif

parameter real    MULT    = 1.0      from (0.0:inf)      
                 `ATTR(desc="Multiplication factor");  
 
// Non-standard (additional) model parameters   
// (introduced for the users' convenience)  
  
`ifdef insideADMS 
parameter integer TYPE    = 1      from [-1:1]  
                 `ATTR(desc="Flag for NPN (1) or PNP (-1) transistor type");   
`else 
parameter integer TYPE    = 1      from [-1:1] exclude 0;   
`endif 
parameter real    GMIN    = 1.0e-13  from (0:1e-10]  
                 `ATTR(desc="Minimum conductance");  
 
