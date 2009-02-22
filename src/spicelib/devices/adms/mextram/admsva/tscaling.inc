// Temperature scaling of parameters   
      
   // The excess transistor temperature due to the self-heating 
`ifdef SELFHEATING
   Tki = V(dt);                      
   // *** Convergence related smoothing ***
   if (Tki < 0.0) begin  
       Tki = - ln(1.0 - Tki);  
   end 
   `linLog(Vdt, Tki, 200.0);
//   `min_logexp(Vdt, Tki, 200.0, 10.0);  
`else
   Vdt = 0.0;
`endif
                                       
   // Temperature variables
   Trk = TREF + `C2K;   
    
`ifdef insideADMS 
   Tk   = Trk + DTA + Vdt;     
   Tamb = Trk + DTA;  
`else 
   Tk   = $temperature + DTA + Vdt;     
   Tamb = $temperature + DTA;  
`endif 
    
   tN  = Tk / Trk;    
   Vt  = `KBdivQQ * Tk;    
   Vtr = `KBdivQQ * Trk;   
   VtINV = 1.0 / Vt;   
   VtrINV = 1.0 / Vtr;     
   VdtINV = VtINV - VtrINV;     
   
   // Depletion capacitances   
   
   UdeT = -3.0 * Vt * ln(tN) + VDE * tN + (1.0 - tN) * VGB;   
   `max_logexp(VDE_T, `VDLOW, UdeT, Vt);  
   
   UdcT = -3.0 * Vt * ln(tN) + VDC * tN + (1.0 - tN) * VGC;   
   `max_logexp(VDC_T, `VDLOW, UdcT, Vt);  
 
`ifdef SUBSTRATE
   UdsT = -3.0 * Vt * ln(tN) + VDS * tN + (1.0 - tN) * VGS;   
   `max_logexp(VDS_T, `VDLOW, UdsT, Vt);  
`endif
        
   CJE_T = CJE * pow(VDE / VDE_T, PE);   

`ifdef SUBSTRATE
   CJS_T = CJS * pow(VDS / VDS_T, PS);   
`endif
                                         
   CJCscale = ((1.0 - XP) * pow(VDC / VDC_T, PC) + XP);   
   CJCscaleINV = 1.0 / CJCscale;    
   
   CJC_T = CJC * CJCscale;   
   XP_T  = XP  * CJCscaleINV;   
      
   // Resistances   
      
   RE_T  = RE  * pow(tN, AE);   
   RBV_T = RBV * pow(tN, AB - AQBO);   
   RBC_T = RBC * pow(tN, AEX);   

// RvdT, 30-11-2007: new collector resistances RCCxx_T, RCCex_T, RCCin_T
   RCCxx_T = RCC * pow(tN, AC);   
   RCCex_T = RCBLX * pow(tN, ACBL);   
   RCCin_T = RCBLI * pow(tN, ACBL);   

   RCV_T = RCV * pow(tN, AEPI);    
   
   // Current gains   
      
   BF_T  = BF  * pow(tN, AE - AB - AQBO) * exp(-DVGBF * VdtINV);    
   BRI_T = BRI * exp(-DVGBR * VdtINV);    
   
   // Currents and voltages   
   
   IS_T  = IS  * pow(tN, 4.0 - AB - AQBO + DAIS) * exp(-VGB * VdtINV);    
   IK_T  = IK  * pow(tN, 1.0 - AB);   
   IBF_T = IBF * pow(tN, 6.0 - 2.0 * MLF) * exp(-VGJ * VdtINV / MLF);   
   IBR_T = IBR * tN * tN * exp(-VGC * VdtINV / 2.0);   
   VEF_T = VEF * pow(tN, AQBO) * CJCscaleINV;   
   VER_T = VER * pow(tN, AQBO) * pow(VDE / VDE_T, -PE);   

`ifdef SUBSTRATE
   ISS_T = ISS * pow(tN, 4.0 - AS) * exp(-VGS * VdtINV);    
   if ((ISS_T > 0.0))    
      IKS_T = IKS * pow(tN, 1.0 - AS) * (IS_T / IS) * (ISS / ISS_T);   
   else    
      IKS_T = IKS * pow(tN, 1.0 - AS);   
`endif
       
   // Transit times   
   
   TAUE_T = TAUE * pow(tN, AB - 2.0) * exp(-DVGTE * VdtINV);   
   TAUB_T = TAUB * pow(tN, AQBO + AB - 1.0);   
   TEPI_T = TEPI * pow(tN, AEPI - 1.0);    
   TAUR_T = TAUR * (TAUB_T + TEPI_T) / (TAUB + TEPI);    
      
   // Avalanche constant      
                             
   Tk300 = Tk - 300.0;
// RvdT, 15-02-2008: prevent division by zero and overflow at high temperatures:
   if (Tk < 525.0)
    begin
     BnT = Bn * (1.0 + 7.2e-4 * Tk300 - 1.6e-6 * Tk300 * Tk300) ;   
    end
   else
    begin
     BnT = Bn * 1.081 ;
    end
  
   // Heterojunction features   
   
   DEG_T = DEG * pow(tN, AQBO);      
   
`ifdef SELFHEATING
  // Tempearature scaling of the thermal resistance   
 
   RTH_Tamb = RTH * pow(Tamb / Trk, ATH); 
`endif
   
// MULT - scaling    
     
   IS_TM  = IS_T  * MULT;       
   IK_TM  = IK_T  * MULT;   
   IBF_TM = IBF_T * MULT;    
   IBR_TM = IBR_T * MULT;    
   IHC_M  = IHC   * MULT;    
`ifdef SUBSTRATE
   ISS_TM = ISS_T * MULT;    
   IKS_TM = IKS_T * MULT;    
`endif
   CJE_TM = CJE_T * MULT;    
   CJC_TM = CJC_T * MULT;    

`ifdef SUBSTRATE
   CJS_TM = CJS_T * MULT;    
`endif
                
   RE_TM      = RE_T     * invMULT;       
   RBC_TM     = RBC_T    * invMULT;   
   RBV_TM     = RBV_T    * invMULT;    
// RvdT, 30-01-2007: new collector resistances:
   RCCxx_TM   = RCCxx_T  * invMULT;    
   RCCex_TM   = RCCex_T  * invMULT;    
   RCCin_TM   = RCCin_T  * invMULT;    
   RCV_TM     = RCV_T    * invMULT;    
 

// RvdT, 03-12-2007: new collector conductances
   if (RCC > 0.0) 
    begin 
     GCCxx_TM = 1.0 / RCCxx_TM ;   
    end
   else
    begin
     GCCxx_TM = 0 ;   
    end 

   if (RCBLX > 0.0) 
    begin 
     GCCex_TM = 1.0 / RCCex_TM ;   
    end
   else
    begin
     GCCex_TM = 0 ;   
    end 

   if (RCBLI > 0.0) 
    begin 
     GCCin_TM = 1.0 / RCCin_TM ;   
    end
   else
    begin
     GCCin_TM = 0 ;   
    end 

`ifdef SELFHEATING
   RTH_Tamb_M = RTH_Tamb * invMULT;    
`endif
