HICUM0 Output Test Ic=f(Vc,Ib)

IB 0 B 200n
VC C 0 2.0
VS S 0 0.0
Q1 C B 0 S NPN_VBIC_VLG

.control
dc vc 0.0 5.0 0.05 ib 10u 100u 10u
run
plot abs(i(vc))
plot v(dt)
.endc

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

.end
