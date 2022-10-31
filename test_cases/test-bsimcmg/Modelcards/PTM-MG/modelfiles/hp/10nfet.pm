** PTM-MG 10nm HSPICE Model Card for HP NFET 
** Nominal VDD=0.75V

.model nfet nmos level = 72 
+ bulkmod = 1 
+lmin    = 1e-008          lmax    = 3e-008        
************************************************************
*                         general                          *
************************************************************
*========================= flags ==========================*
+version = 105.03          bulkmod = 1               igcmod  = 1               igbmod  = 1             
+gidlmod = 0               iimod   = 0               geomod  = 1               rdsmod  = 1             
+rgatemod= 0               rgeomod = 1               shmod   = 0               nqsmod  = 0             
+coremod = 0               cgeomod = 2               capmod  = 0               tnom    = 300.15        
*======================== process =========================*
+eot     = 6.8e-010        eotbox  = 1.4e-007        tfin    = 8e-009          toxp    = 1.2e-009      
+nbody   = 2.5e+022        phig    = 4.4212           epsrox  = 3.9             epsrsub = 11.9          
+easub   = 4.05            ni0sub  = 1.1e+016        bg0sub  = 1.12            nc0sub  = 2.86e+025     
+nsd     = 3e+026          ngate   = 0               nfin    = 1               d       = 4e-008        
+nseg    = 5               l       = 1.4e-008      
*========================== w/l ===========================*
+xl      = 0               lint    = 6e-010          ll      = 0               lln     = 1             
+llc     = 0               dlc     = 0               dlbin   = 0               hfin    = 2.1e-008      
+fech    = 1               deltaw  = 0               deltawcv= 0               fechcv  = 1             
+hepi    = 8.5e-009      
*======================== geometry ========================*
+tsili   = 7e-009          rhoc    = 6e-13        cratio  = 0.5             deltaprsd= 0             
+sdterm  = 0               ldg     = 2e-009          epsrsp  = 3.9             tgate   = 8e-009        
+tmask   = 0               asiliend= 0               arsdend = 0               prsdend = 0             
+nsde    = 6e+025          rgeoa   = 1               rgeob   = 0               rgeoc   = 0             
+rgeod   = 0               rgeoe   = 0               cgeoa   = 1               cgeob   = 0             
+cgeoc   = 0               cgeod   = 0               cgeoe   = 1             
*===================== model_selector =====================*
************************************************************
*                            dc                            *
************************************************************
+cit     = 1.3e-005      
*========================== vth ===========================*
+cdsc    = 0.032          cdscd   = 0.007           dvt0    = 0.01            dvt1    = 0.69          
+phin    = 0.05            eta0    = 0.6778            dsub    = 0.9             k1rsce  = 0             
+lpe0    = 5e-009          dvtshift= 0               qmfactor= 0               qmtceniv= 0             
+qmtcencv= 0               etaqm   = 0.54            qm0     = 0.001143        pqm     = 0.66          
+qm0acc  = 0.001           pqmacc  = 0.66            delvfbacc= 0               u0      = 0.0568        
*======================== mobility ========================*
+etamob  = 2               up      = 0               lpa     = 1               ua      = 0.3           
+aua     = 0               bua     = 1e-007          eu      = 1.8             ud      = 0             
+aud     = 0               bud     = 5e-008          ucs     = 0               rdswmin = 0             
*======================= resistance =======================*
+rdsw    = 100             ardsw   = 0               brdsw   = 1e-007          prwg    = 0             
+wr      = 1               rswmin  = 0               rsw     = 0              arsw    = 0             
+brsw    = 1e-007          rdwmin  = 0               rdw     = 0              ardw    = 0             
+brdw    = 1e-007          rgfin   = 0.001           rgext   = 0               rshs    = 0             
+vsat    = 132000        
*======================= saturation =======================*
+deltavsat= 1               ksativ  = 0.8             avsat   = 0               avsat1  = 0             
+bvsat   = 1e-007          bvsat1  = 1e-007          mexp    = 3               amexp   = 0             
+bmexp   = 1               ptwg    = 0               aptwg   = 0               bptwg   = 1e-007        
*========================== rout ==========================*
+pclm    = 0.2             pclmcv  = 0.013           apclm   = 0               bpclm   = 1e-007        
+pclmg   = 0               pclmgcv = 0               vasat   = 0.6             vasatcv = 0.2           
+pdibl1  = 1.3             pdibl2  = 0.0002          drout   = 1.06            pvag    = 1             
+fpitch  = 2.8e-008      
*====================== self-heating ======================*
+rth0    = 0.01            cth0    = 1e-005          wth0    = 0             
************************************************************
*                         leakage                          *
************************************************************
+aigbinv = 0.0111        
*========================== igb ===========================*
+bigbinv = 0.03            cigbinv = 0.006           eigbinv = 1.1             nigbinv = 1.2             
+aigbacc = 0.0149          bigbacc = 0.000949        cigbacc = 0.075           nigbacc = 3             
+aigc    = 0.0136        
*========================== igc ===========================*
+bigc    = 0.00171         cigc    = 0.075           nigc    = 1             
*======================== igs/igd =========================*
+dlcigs  = 0               dlcigd  = 0               aigs    = 0.0136          aigd    = 0.0136        
+bigs    = 0.00171         bigd    = 0.00171         cigs    = 0.075           cigd    = 0.075         
+poxedge = 1               agidl   = 5.729e-012    
*======================= gidl/gisl ========================*
+agisl   = 5.729e-012      bgidl   = 3e+008          bgisl   = 3e+008          egidl   = 0.1           
+egisl   = 0.1             alpha0  = 0             
*========================== isub ==========================*
+alpha1  = 0               alphaii = 0               betaii0 = 0               betaii1 = 0             
+betaii2 = 0.1             esatii  = 10000000        lii     = 5e-010          sii0    = 0.5           
+sii1    = 0.1             sii2    = 0               siid    = 0               beta0   = 30            
+lintigen= 0             
*================ generation/recombination ================*
+ntgen   = 1               aigen   = 0               bigen   = 0             
************************************************************
*                            rf                            *
************************************************************
+xrcrg1  = 12            
*==================== nonquasi-static =====================*
+xrcrg2  = 1             
************************************************************
*                         junction                         *
************************************************************
*======================== current =========================*
+jss     = 0.0001          jsd     = 0.0001          jsws    = 0               jswgs   = 0             
+njs     = 1               njd     = 1               ijthsfwd= 0.1             ijthdfwd= 0.1           
+ijthsrev= 0.1             ijthdrev= 0.1             bvs     = 10              bvd     = 10            
+xjbvs   = 1               xjbvd   = 1               cjs     = 0.0005        
*====================== capacitance =======================*
+cjd     = 0.0005          cjsws   = 5e-010          cjswgs  = 0               pbs     = 1             
+pbd     = 1               pbsws   = 1               mjs     = 0.5             mjd     = 0.5           
+mjsws2  = 0.33            mjswd2  = 0.33            mjsws   = 0.33            sjs     = 0             
+sjsws   = 0               sjswgs  = 0               mjs2    = 0.125           mjd2    = 0.125         
************************************************************
*                       capacitance                        *
************************************************************
*====================== capacitance =======================*
+cgsp    = 0               cgdp    = 0               cdsp    = 0               cfs     = 2.56e-011     
+cfd     = 2.65e-011       covs    = 2.5e-011        covd    = 2.5e-011        cgsl    = 0             
+ckappas = 0.6             cgbo    = 0               cgbl    = 0             
************************************************************
*                       temperature                        *
************************************************************
+tbgasub = 0.000702      
*======================== process =========================*
+tbgbsub = 1108            kt1     = 0             
*========================== vth ===========================*
+kt1l    = 0               ute     = 0             
*======================== mobility ========================*
+utl     = -0.0015         ua1     = 0.001032        ud1     = 0               ucste   = -0.00478      
+at      = -0.00156      
*========================== vsat ==========================*
+ptwgt   = 0.004           tmexp   = 0               prt     = 0.001         
*======================== resistor ========================*
+iit     = -0.5          
*========================== isub ==========================*
+tii     = 0               tgidl   = -0.0003       
*========================== gidl ==========================*
+igt     = 2.5           
*====================== gatecurrent =======================*
*======================== junction ========================*
+tcj     = 0               tcjsw   = 0               tcjswg  = 0               tpb     = 0             
+tpbsw   = 0               tpbswg  = 0               xtis    = 3               xtid    = 3             
+xtss    = 0.02            xtssws  = 0.02            xtsswgs = 0.02            tnjts   = 0             
+tnjtssw = 0               tnjtsswg= 0             
************************************************************
*                          noise                           *
************************************************************
+ef      = 1             
*========================== 1/f ===========================*
+lintnoi = 0               em      = 41000000        noia    = 6.25e+039       noib    = 3.125e+024    
+noic    = 87500000        ntnoi   = 1             
