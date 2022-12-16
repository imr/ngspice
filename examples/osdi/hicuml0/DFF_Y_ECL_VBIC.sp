ECL DFF VBIC test case
*
V6 D GND PULSE(-.25 0 0 1P 1P .25N .50N)   
V5 D_BAR GND PULSE(0 -.25 0 1P 1P .25N .50N)   
V4 CLK GND PULSE(-0.9 -1.2 0 1P 1P .125N .25N)   
V3 CLK_BAR GND PULSE(-1.2 -0.9 0 1P 1P .125N .25N)   
VVCS NET2 GND DC -0.8 
R6 GND NET6  800 
R5 GND NET11  800 
R4 NET12 VEE  350 
VVEE VEE GND DC -2.0 
R3 GND NET10  800 
R2 NET13 VEE  350 
R1 GND Q  800 
QVLGNPN16 NET7 CLK_BAR NET4 VEE NPN_VBIC_VLG
QVLGNPN15 NET5 CLK NET3 VEE NPN_VBIC_VLG
QVLGNPN14 NET10 Q NET7 VEE NPN_VBIC_VLG
QVLGNPN13 Q NET6 NET9 VEE NPN_VBIC_VLG
QVLGNPN12 NET6 NET11 NET5 VEE NPN_VBIC_VLG
QVLGNPN11 NET11 D_BAR NET8 VEE NPN_VBIC_VLG
QVLGNPN10 NET4 NET2 NET13 VEE NPN_VBIC_VLG
QVLGNPN9 NET3 NET2 NET12 VEE NPN_VBIC_VLG
QVLGNPN8 NET8 CLK_BAR NET3 VEE NPN_VBIC_VLG
QVLGNPN7 NET9 CLK NET4 VEE NPN_VBIC_VLG
QVLGNPN6 Q NET10 NET7 VEE NPN_VBIC_VLG
QVLGNPN5 NET10 NET11 NET9 VEE NPN_VBIC_VLG
QVLGNPN4 NET11 NET6 NET5 VEE NPN_VBIC_VLG
QVLGNPN3 NET6 D NET8 VEE NPN_VBIC_VLG
*
.MODEL NPN_VBIC_VLG NPN LEVEL=4
+TNOM    = 27             RCI     = 1E3            RCX     = 50                 
+VO      = 1.5            GAMM    = 3.402097E-11   HRCF    = 1                  
+RBX     = 243            RBI     = 20             RE      = 30                 
+RS      = 0              RBP     = 0              IS      = 8.084033E-18       
+NF      = 1              NR      = 1.005          FC      = 0.5                
+CJE     = 2.083234E-15   PE      = 0.8793669      ME      = 0.3108762          
+CJC     = 1.803275E-15   PC      = 0.5512188      MC      = 0.4454263          
+CJCP    = 8E-15          PS      = 0.66956        MS      = 0.2243             
+IBEI    = 4.542609E-20   WBE     = 1              NEI     = 1                  
+IBEN    = 3.275162E-16   NEN     = 1.5543186      IBCI    = 3.594252E-19       
+NCI     = 0.996          IBCN    = 1.717776E-17   NCN     = 1.202521           
+AVC1    = 3E-4           AVC2    = 1E-5           ISP     = 1.332E-19          
+WSP     = 1              NFP     = 1              IBEIP   = 0                  
+IBENP   = 0              IBCIP   = 0              NCIP    = 1                  
+IBCNP   = 0              NCNP    = 2              VEF     = 109.6523           
+VER     = 2.2052435      IKF     = 6.03524E-3     IKR     = 1.807895E-4        
+IKP     = 2.908576E-5    TF      = 1.1E-12        XTF     = 21.5423            
+VTF     = 12.4758677     ITF     = 0.0175231      TR      = 2.23542E-9       

.SAVE V(D) V(CLK) V(Q)
.control
TRAN 0.25p 5n
rusage
set color0=white
set xbrushwidth=2
plot V(D) V(CLK) V(Q)
*quit
.endc
.END
