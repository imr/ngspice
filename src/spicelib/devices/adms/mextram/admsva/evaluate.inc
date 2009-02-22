// Evaluate model equations 
  
begin  // Currents and sharges
// Nodal biases   
  
   Vb2c1 = TYPE * V(b2, c1);  
   Vb2c2 = TYPE * V(b2, c2);  
   Vb2e1 = TYPE * V(b2, e1);  
   Vb1e1 = TYPE * V(b1, e1);  
   Vb1b2 = TYPE * V(b1, b2);  
`ifdef SUBSTRATE
   Vsc1  = TYPE * V(s,  c1);  
`endif
   Vc1c2 = TYPE * V(c1, c2);  
   Vee1  = TYPE * V(e,  e1);  
   Vbb1  = TYPE * V(b,  b1);  
   Vbe   = TYPE * V(b,  e);  
   Vbc   = TYPE * V(b,  c);   

/* RvdT, 03-12-2007, voltage differences 
 associated with distributed parasitic collector.
 Evaluated taking values of resistances into account:
 in case of vanishing resistance corresponding node 
 is not addressed: */

if (RCBLX > 0.0)
 begin
  if (RCBLI > 0.0)
    begin
     Vc4c1 = TYPE * V(c4, c1);  
     Vc3c4 = TYPE * V(c3, c4);  
    end
   else
    begin
     Vc4c1 = 0 ;  
     Vc3c4 = TYPE * V(c3, c1);  
    end
 end
else
 begin
  if (RCBLI > 0.0)
    begin
     Vc4c1 = TYPE * V(c4, c1);  
     Vc3c4 = 0 ;  
    end
   else
    begin
     Vc4c1 = 0 ;  
     Vc3c4 = 0 ;  
    end
 end

   Vb1c4 = Vb1b2 + Vb2c2 - Vc1c2 - Vc4c1 ;  
   Vcc3  = - Vbc + Vbb1 + Vb1c4 - Vc3c4 ;  
   Vbc3  = Vbc + Vcc3 ;

`ifdef  SUBSTRATE
Vsc4 = Vsc1 - Vc4c1 ;
Vsc3 = Vsc4 - Vc3c4 ;
`endif


// Exponential bias terms  
  
   `expLin(eVb2c2,Vb2c2 * VtINV)  
   `expLin(eVb2e1,Vb2e1 * VtINV)    
   `expLin(eVb1e1,Vb1e1 * VtINV)   
   `expLin(eVb1c4,Vb1c4 * VtINV)  
   `expLin(eVb1b2,Vb1b2 * VtINV)  
   `expLin(eVbc3,Vbc3  * VtINV)  
`ifdef SUBSTRATE
   `expLin(eVsc1,Vsc1  * VtINV)  
`endif
     
   `expLin(eVbc3VDC,(Vbc3  - VDC_T) * VtINV)                      
   `expLin(eVb1c4VDC,(Vb1c4 - VDC_T) * VtINV)  
   `expLin(eVb2c2VDC,(Vb2c2 - VDC_T) * VtINV)  
   `expLin(eVb2c1VDC,(Vb2c1 - VDC_T) * VtINV)  
  
// Governing equations  
  
   // Epilayer model   
   
   K0 = sqrt(1.0 + 4.0 * eVb2c2VDC);   
   Kw = sqrt(1.0 + 4.0 * eVb2c1VDC);    
   pW = 2.0 *  eVb2c1VDC / (1.0 + Kw);   
   if (pW < `TEN_M40) pW = 0; 
   Ec = Vt * (K0 - Kw - ln((K0 + 1.0) / (Kw + 1.0)) );    
   Ic1c2 =  (Ec + Vc1c2) / RCV_TM;     
  
   if (Ic1c2 > 0.0) begin   
      
     `linLog(tmpV,Vb2c1,100.0);
     Vqs_th = VDC_T + 2.0 * Vt *    
              ln(0.5 * Ic1c2 * RCV_TM * VtINV + 1.0) - tmpV;     
     eps_VDC = 0.2 * VDC_T;  
     `max_hyp0(Vqs, Vqs_th, eps_VDC);  
     Iqs = Vqs * (Vqs + IHC_M * SCRCV_M) / (SCRCV_M * (Vqs + IHC_M * RCV_TM));  
  
     Ic1c2_Iqs = Ic1c2 / Iqs;  
     `max_logexp(alpha1, Ic1c2_Iqs, 1.0, AXI);    
     alpha = alpha1 / (1.0 + AXI * ln(1.0 + exp(-1.0 / AXI)));   
     vyi = Vqs / (IHC_M * SCRCV_M);    
     yi = (1.0 + sqrt(1.0 + 4.0 * alpha * vyi * (1.0 + vyi))) /   
          (2.0 * alpha * (1.0 + vyi));    
  
     xi_w = 1.0 - yi / (1.0 + pW * yi);   
     gp0 = 0.5 * Ic1c2 * RCV_TM * xi_w * VtINV;    
     
     gp0_help = 2.0 * gp0 + pW * (pW + gp0 + 1.0);  
     gp02 = 0.5 * (gp0 - 1.0);   
     sqr_arg = gp02 * gp02 + gp0_help;  
     if (gp0 >= 1.0)  
        p0star =  gp02 + sqrt(sqr_arg);    
     else  
        p0star = gp0_help / (sqrt(sqr_arg) - gp02);  
     if (p0star < `TEN_M40) p0star = 0.0; 
 
 
     eVb2c2star = p0star * (p0star + 1.0) * exp(VDC_T * VtINV);    
     B1 = 0.5 * SCRCV_M * (Ic1c2 - IHC_M);   
     B2 = SCRCV_M * RCV_TM * IHC_M * Ic1c2;    
     Vxi0 = B1 + sqrt(B1 * B1 + B2);    
     Vch = VDC_T * (0.1 + 2.0 * Ic1c2 / (Ic1c2 + Iqs));     
     Icap = IHC_M * Ic1c2 / (IHC_M + Ic1c2);   
     Icap_IHC = IHC_M / (IHC_M + Ic1c2);  
      
   end else begin    
         
     p0star = 2.0 * eVb2c2VDC / (1.0 + K0);    
     eVb2c2star = eVb2c2;   
     if ((abs(Vc1c2) < 1.0e-5 * Vt) ||   
         (abs(Ec) < `TEN_M40 * Vt * (K0 + Kw))) 
        begin   
          pav = 0.5 * (p0star + pW);   
          xi_w = pav / (pav + 1.0);   
        end

     else 
        begin   
          xi_w = Ec / (Ec + Vb2c2 - Vb2c1);    
        end   
   
     Vxi0 = Vc1c2;    
     Vch = 0.1 * VDC_T;                    
     Icap = Ic1c2;   
     Icap_IHC = 1.0 - Icap / IHC_M;  
   
   end    
   
   // Effective emitter junction capacitance bias   
    
   Vfe = VDE_T * (1.0 - pow(`AJE , -1.0 / PE));   
   a_VDE = 0.1 * VDE_T;  
   `min_logexp(Vje, Vb2e1, Vfe, a_VDE);  
   Vte = VDE_T / (1.0 - PE) * (1.0 - pow(1.0 - Vje / VDE_T, 1.0 - PE)) +   
        `AJE * (Vb2e1 - Vje);   
      
   // Effective collector junction capacitance bias   
    
   Vjunc = Vb2c1 + Vxi0;   
   bjc = (`AJC - XP_T) / (1.0 - XP_T);   
   Vfc = VDC_T * (1.0 - pow(bjc, -1.0 / PC));    
   `min_logexp(Vjc, Vjunc, Vfc, Vch);  
   fI = pow(Icap_IHC, MC);    
   Vcv = VDC_T / (1.0 - PC) * (1.0 - fI * pow(1.0 - Vjc / VDC_T, 1.0 - PC)) +   
         fI * bjc * (Vjunc - Vjc);    
   Vtc = (1.0 - XP_T) * Vcv + XP_T * Vb2c1;    
  
   // Transfer current   
    
   If0 = 4.0 * IS_TM / IK_TM;  
   f1 =  If0 * eVb2e1;    
   n0 =  f1 / (1.0 + sqrt(1.0 + f1));    
   f2 =  If0 * eVb2c2star;      
   nB =  f2 / (1.0 + sqrt(1.0 + f2));    
  
   if (DEG == 0.0) 
        q0I = 1.0 + Vte / VER_T + Vtc / VEF_T;    
   else 
      begin  
        termE = (Vte / VER_T + 1.0) * DEG_T * VtINV;  
        termC = -Vtc / VEF_T * DEG_T * VtINV;  
        q0I = (exp(termE) - exp(termC)) /   
              (exp(DEG_T * VtINV) - 1.0);  
      end  
  
   `max_hyp0(q1I, q0I, 0.1);   
   qBI = q1I * (1.0 + 0.5 * (n0 + nB));    
   
   Ir = IS_TM *  eVb2c2star;    
   If = IS_TM * eVb2e1;   
   In = (If - Ir) / qBI;    
    
   // Base and substrate current(s)   
                
   Ibf0 = IS_TM / BF_T;     
   if (XREC == 0.0)  
      Ib1 = (1.0 - XIBI) * Ibf0 * (eVb2e1 - 1.0);   
   else  
      Ib1 = (1.0 - XIBI) * Ibf0 * ((1.0 - XREC) * (eVb2e1 - 1.0) +  
            XREC * (eVb2e1 + eVb2c2star - 2.0) * (1.0 + Vtc / VEF_T));    
  
   Ib1_s = XIBI * Ibf0 * (eVb1e1 - 1.0);   
   `expLin(tmpExp,Vb2e1 * VtINV / MLF)
   Ib2 = IBF_TM * (tmpExp - 1.0) + GMIN * Vb2e1;       
   `expLin(tmpExp,0.5 * Vb1c4 * VtINV)
   Ib3 = IBR_TM * (eVb1c4 - 1.0) /   
         (tmpExp + exp(0.5 * VLR * VtINV)) +    
         GMIN  * Vb1c4;    
      
   // Iex, Isub (XIex, XIsub)   
  
   g1 = If0 * eVb1c4;                   
   g2 = 4.0 * eVb1c4VDC;     
   nBex = g1 / (1.0 + sqrt(1.0 + g1));    
   pWex = g2 / (1.0 + sqrt(1.0 + g2));    
   Iex = (1.0 / BRI_T) * (0.5 * IK_TM * nBex - IS_TM);    

`ifdef SUBSTRATE
   Isub = 2.0 * ISS_TM * (eVb1c4 - 1.0) /   
          (1.0 + sqrt(1.0 + 4.0 * (IS_TM / IKS_TM) * eVb1c4));     
   Isf =  ISS_TM * (eVsc1 - 1.0);   
`endif
  
   XIex =0.0; 

`ifdef SUBSTRATE
   XIsub = 0.0;    
`endif

   if (EXMOD == 1) 
    begin   
  
      Iex =   Iex *  Xext1;     

`ifdef SUBSTRATE
      Isub  = Isub * Xext1;    
`endif
  
      Xg1 = If0 * eVbc3;    
      XnBex = Xg1 / (1.0 + sqrt(1.0 + Xg1));    
      XIMex = XEXT * (0.5 * IK_TM * XnBex - IS_TM) / BRI_T;   

`ifdef SUBSTRATE
      XIMsub = XEXT * 2.0 * ISS_TM * (eVbc3 - 1.0) /   
               (1.0 + sqrt(1.0 + 4.0 * IS_T / IKS_T * eVbc3));    
      Vex_bias = XEXT * (IS_TM / BRI_T + ISS_TM) * RCCxx_TM;  
`else
      XIMsub = 0.0;
      Vex_bias = XEXT * (IS_TM / BRI_T) * RCCxx_TM;  
`endif
      
      Vex = Vt * (2.0 - ln( Vex_bias * VtINV));    
      vdif = Vbc3 - Vex;  
      `max_hyp0(VBex, vdif, 0.11);  
  
      Fex = VBex /(Vex_bias + (XIMex + XIMsub) * RCCxx_TM + VBex);     
      XIex = Fex * XIMex;   

`ifdef SUBSTRATE
      XIsub = Fex * XIMsub;     
`endif
    end     
   else
    begin
     Fex = 0;
     XnBex = 0 ;
    end

   // Variable base resistance   
        
   q0Q = 1.0 + Vte / VER_T + Vtc / VEF_T;    
   `max_hyp0(q1Q, q0Q, 0.1);   
   qBQ = q1Q * (1.0 + 0.5 * (n0 + nB));    
  
   Rb2 = 3.0 * RBV_TM / qBQ;   
   Ib1b2 =  (2.0 * Vt * (eVb1b2 - 1.0) + Vb1b2) / Rb2;   
   
   // Weak-avalanche current   
      
   Iavl = 0.0;  
   Gem  = 0.0;  
   if ((Ic1c2 > 0.0) && (Vb2c1 < VDC_T)) begin   
  
      dEdx0 = 2.0 * VAVL / (WAVL * WAVL);    
      sqr_arg = (VDC_T - Vb2c1) / Icap_IHC;  
      xd = sqrt(2.0 * sqr_arg / dEdx0);      
      if (EXAVL == 0.0) 
         Weff = WAVL;   
      else 
         begin            
           xi_w1 = 1.0 - 0.5 * xi_w;
           Weff = WAVL * xi_w1 * xi_w1;    
         end
      Wd = xd * Weff / sqrt(xd * xd + Weff * Weff);    
      Eav = (VDC_T - Vb2c1) / Wd;   
      E0 = Eav + 0.5 * Wd * dEdx0 * Icap_IHC;   
      
      if (EXAVL == 0)  
         Em = E0;   
      else 
         begin               
           SHw = 1.0 + 2.0 * SFH * (1.0 + 2.0 * xi_w);    
           Efi = (1.0 + SFH) / (1.0 + 2.0 * SFH);   
           Ew = Eav - 0.5 * Wd * dEdx0 * (Efi - Ic1c2 / (IHC_M * SHw));   
           sqr_arg = (Ew - E0) * (Ew - E0) + 0.1 * Eav * Eav * Icap / IHC_M;   
           Em = 0.5 * (Ew + E0 + sqrt(sqr_arg));  
         end   
   
      EmEav_Em = (Em - Eav) / Em;   
      if (abs(EmEav_Em) > `TEN_M07) 
         begin   
           lambda = 0.5 * Wd / EmEav_Em;    
           Gem = An / BnT * Em * lambda *   
                (exp(-BnT / Em) - exp(-BnT / Em * (1.0 + Weff / lambda)) );   
         end
      else    
         Gem = An * Weff * exp(-BnT / Em);   
   
      Gmax = Vt / (Ic1c2 * (RBC_TM + Rb2)) +  qBI / BF_T +   
             RE_TM / (RBC_TM + Rb2);   
      Iavl = Ic1c2 * Gem  / (Gem +Gem / Gmax + 1.0);   
   end    

   if (eVb2c2star > 0.0)   
      Vb2c2star = Vt * ln(eVb2c2star);   
   else  
      Vb2c2star = Vb2c2;  
    
`ifdef SELFHEATING
   // Power dissipation   
 
// RvdT 03-12-2007, modified power equation due to distribution collector resistance

   power =  In * (Vb2e1 - Vb2c2star) +    
            Ic1c2 * (Vb2c2star - Vb2c1) -   
            Iavl  * Vb2c2star +   
            Vee1  * Vee1  / RE_TM +   
            Vcc3  * Vcc3  * GCCxx_TM +   
            Vc3c4 * Vc3c4 * GCCex_TM +   
            Vc4c1 * Vc4c1 * GCCin_TM +   
            Vbb1  * Vbb1  / RBC_TM +   
            Ib1b2 * Vb1b2 +   
            (Ib1 + Ib2) * Vb2e1 +   
            Ib1_s * Vb1e1 +   
`ifdef SUBSTRATE
            (Iex + Ib3) * Vb1c4 + XIex  * Vbc3 + 
              Isub * (Vb1c4 - Vsc4) +   
             XIsub * (Vbc3 - Vsc3) +   
              Isf * Vsc1;   
`else
            (Iex + Ib3) * Vb1c4 + XIex * Vbc3;   
`endif
    
`endif 


   // Charges   
   
   Qte = (1.0 - XCJE) * CJE_TM * Vte;    
   `min_logexp(Vje_s, Vb1e1, Vfe, a_VDE);  
   Qte_s = XCJE * CJE_TM * (VDE_T / (1.0 - PE) *    
           (1.0 - pow(1.0 - Vje_s / VDE_T, 1.0 - PE)) +    
            `AJE * (Vb1e1 - Vje_s));    
       
   Qtc = XCJC * CJC_TM * Vtc;    
   Qb0 = TAUB_T * IK_TM;   
   Qbe_qs = 0.5 * Qb0 * n0 * q1Q;   
   Qbc_qs = 0.5 * Qb0 * nB * q1Q;    
    
   a_VDC = 0.1 * VDC_T;  
   `min_logexp(Vjcex, Vb1c4, Vfc, a_VDC);  
   Vtexv = VDC_T / (1.0 - PC) * (1.0 - pow(1.0 - Vjcex / VDC_T, 1.0 - PC)) +   
           bjc * (Vb1c4 - Vjcex);   
   Qtex = CJC_TM * ((1.0 - XP_T) * Vtexv + XP_T * Vb1c4) *    
          (1.0 - XCJC) * (1.0 - XEXT);      
  
   `min_logexp(XVjcex, Vbc3, Vfc, a_VDC);  
   XVtexv = VDC_T / (1.0 - PC) * (1.0 - pow(1.0 - XVjcex / VDC_T, 1.0 - PC)) +   
            bjc * (Vbc3 - XVjcex);   
   XQtex = CJC_TM * ((1.0 - XP_T) * XVtexv + XP_T * Vbc3) *    
           (1.0 - XCJC) * XEXT;    
    
`ifdef SUBSTRATE
   a_VDS = 0.1 * VDS_T;  
   Vfs = VDS_T * (1.0 - pow(`AJS , -1.0 / PS));   
   `min_logexp(Vjs, Vsc1, Vfs, a_VDS);  
   Qts = CJS_TM * (VDS_T / (1.0 - PS) *    
         (1.0 - pow(1.0 - Vjs / VDS_T, 1.0 - PS)) +  `AJS * (Vsc1 - Vjs));   
`endif
  
   Qe0 = TAUE_T * IK_TM * pow(IS_TM / IK_TM, 1.0 / MTAU);   
   `expLin(tmpExp,Vb2e1 / (MTAU * Vt))
   Qe = Qe0 * (tmpExp - 1.0);   
                                                                 
   Qepi0 = 4.0 * TEPI_T * Vt / RCV_TM;   
   Qepi = 0.5 * Qepi0 * xi_w * (p0star + pW + 2.0);   
                                                             
   Qex = TAUR_T * 0.5 * (Qb0 * nBex + Qepi0 * pWex) / (TAUB_T + TEPI_T);        
   XQex = 0.0;   
  
   if (EXMOD == 1) begin       
  
      Qex = Qex * (1.0 - XEXT);    
      Xg2 = 4.0 * eVbc3VDC;    
      XpWex = Xg2 / (1.0 + sqrt(1.0 + Xg2));    
      XQex = 0.5 * Fex * XEXT * TAUR_T *   
             (Qb0 * XnBex + Qepi0 * XpWex) / (TAUB_T + TEPI_T);   
  
   end    
  
   Qb1b2 = 0.0;  
   if (EXPHI == 1) 
      begin  
        dVteVje = pow(1.0 - Vje / VDE_T, -PE) - `AJE;  
        Vb2e1Vfe = (Vb2e1 - Vfe) / a_VDE;  
        if (Vb2e1Vfe < 0.0)  
           dVjeVb2e1 = 1.0 / (1.0 + exp(Vb2e1Vfe));  
        else  
           dVjeVb2e1 = exp(- Vb2e1Vfe) / (1.0 + exp(- Vb2e1Vfe));  
       
        dVteVb2e1 = dVteVje * dVjeVb2e1 + `AJE;   
        dQteVb2e1 = (1.0 - XCJE) * CJE_TM * dVteVb2e1;  
  
        dn0Vb2e1 = If0 * eVb2e1 * VtINV * (0.5 / sqrt(1.0 + f1));  
        dQbeVb2e1 = 0.5 * Qb0 * q1Q * dn0Vb2e1;   
  
        dQeVb2e1 = (Qe + Qe0) / (MTAU * Vt);   
   
        Qb1b2 = 0.2 * Vb1b2 * (dQteVb2e1 + dQbeVb2e1 + dQeVb2e1);   
  
        Qbc = Qbe_qs / 3.0 + Qbc_qs;  
        Qbe = 2.0 * Qbe_qs / 3.0;  
      end  
    else
      begin
        Qbe = Qbe_qs;
        Qbc = Qbc_qs;
      end
      

// Add branch current contributions  
  
   // Static currents 
   I(c1, c2) <+ TYPE * Ic1c2;   
   I(c2, e1) <+ TYPE * In;   
   I(b1, e1) <+ TYPE * Ib1_s;    
   I(b2, e1) <+ TYPE * (Ib1 + Ib2);   
`ifdef SUBSTRATE
   I(b1, s)  <+ TYPE * Isub;   
   I(b,  s)  <+ TYPE * XIsub;    
   I(s,  c1) <+ TYPE * Isf;   
`endif
   I(b1, b2) <+ TYPE * Ib1b2;    
   I(b2, c2) <+ TYPE * (-1.0 * Iavl);  
   I(e,  e1) <+ TYPE * Vee1 / RE_TM;   
   I(b,  b1) <+ TYPE * Vbb1 / RBC_TM;

`ifdef SELFHEATING
   // Electrical equivalent for the thermal network 
   I(dt) <+ V(dt) / RTH_Tamb_M;     
   I(dt) <+ ddt(CTH_M * V(dt));   
   I(dt) <+ -1.0 *  power;  
`endif
    
   // Electrical equivalent for the correlated noise 
   I(noi, e1)    <+ V(noi, e1); 
   cor_exp_1 = sqrt(1.0 + 2.0 * Gem) * V(noi,e1); 
   I(b2, e1) <+ cor_exp_1; 
   cor_exp_2 = (2.0 + 2.0 * Gem) / sqrt(1.0 + 2.0 * Gem) * V(noi, e1); 
   I(e1, c2) <+ cor_exp_2; 
 
   // Dynamic currents 
   I(b2, e1) <+ ddt(TYPE * (Qte + Qbe + Qe));   
   I(b1, e1) <+ ddt(TYPE * (Qte_s));   
   I(b2, c2) <+ ddt(TYPE * (Qtc + Qbc + Qepi));   
`ifdef SUBSTRATE
   I(s,  c1) <+ ddt(TYPE * Qts);   
`endif
   I(b1, b2) <+ ddt(TYPE * Qb1b2);  
   I(b,   e) <+ ddt(TYPE * CBEO_M * Vbe);   
   I(b,   c) <+ ddt(TYPE * CBCO_M * Vbc);  
 
   end  // Currents and charges 


/* RvdT, Delft Univ. Tech. 03-12-2007.
Distribution of parasitic collector resistance.
This construct supports the case 
RCBLI = 0.0 and or RCBLX = 0.0 .
It is up to the compiler to adjust the circuit topology
and perform a node-collapse in such cases. */
if (RCBLX > 0.0)
 begin
  I(b,  c3) <+ TYPE * XIex;   
  I(c,  c3) <+ TYPE * Vcc3  * GCCxx_TM ;
  I(b,  c3) <+ ddt(TYPE * (XQtex + XQex));    
  if (RCBLI > 0.0)
   begin
     I(c4, c1) <+ TYPE * Vc4c1 * GCCin_TM;    
     I(b1, c4) <+ TYPE * (Ib3 + Iex);    
     I(c3, c4) <+ TYPE * Vc3c4 * GCCex_TM ;    
     I(b1, c4) <+ ddt(TYPE * (Qtex + Qex));        
   end
  else
   begin
     V(c4, c1) <+ 0.0 ;    
     I(b1, c1) <+ TYPE * (Ib3 + Iex);
     I(b1, c1) <+ ddt(TYPE * (Qtex + Qex));            
     I(c3, c1) <+ TYPE * Vc3c4 * GCCex_TM ;    
   end
 end
else
 begin
  V(c3, c4) <+ 0 ;    
       if (RCBLI > 0.0)
          begin
            I(b,  c4) <+ TYPE * XIex;   
            I(c,  c4) <+ TYPE * Vcc3  * GCCxx_TM ;
            I(c4, c1) <+ TYPE * Vc4c1 * GCCin_TM;    
            I(b1, c4) <+ TYPE * (Ib3 + Iex);    
            I(b1, c4) <+ ddt(TYPE * (Qtex + Qex)); 
            I(b,  c4) <+ ddt(TYPE * (XQtex + XQex));    
          end  
       else
          begin 
            I(b,  c1) <+ TYPE * XIex;   
            I(c,  c1) <+ TYPE * Vcc3  * GCCxx_TM ;
            V(c4, c1) <+ 0.0 ;    
            I(b1, c1) <+ TYPE * (Ib3 + Iex);
            I(b1, c1) <+ ddt(TYPE * (Qtex + Qex));
            I(b,  c1) <+ ddt(TYPE * (XQtex + XQex));                
            I(c3, c1) <+ TYPE * Vc3c4 * GCCex_TM ;
          end
 end
 
// Noise sources 
  
`NOISE begin  
 
   // Thermal noise 
   common = 4.0 * `KB * Tk;  
   powerREC = common / RE_TM;  // Emitter resistance 
   powerRBC = common / RBC_TM; // Base resistance 
 // RvdT, 03-12-2007: distributed collector resistance
   powerRCCxx = common * GCCxx_TM; // Collector resistance 
   powerRCCex = common * GCCex_TM; // Collector resistance 
   powerRCCin = common * GCCin_TM; // Collector resistance 
   powerRBV = common / Rb2 * (4.0 * eVb1b2 + 5.0) / 3.0; // Variable base resistance  
 
   // Collector current shot noise 
   powerCCS = 2.0 * `QQ * (If + Ir) / qBI; 
 
   // Forward base current shot noise and 1/f noise 
   powerFBCS = 2.0 * `QQ * (abs(Ib1) + abs(Ib2)); 
   powerFBC1fB1 = (1.0 - XIBI) * pow((abs(Ib1) / (1 - XIBI)), AF) * KF_M; 
   exponentFBC1fB2 = (2.0 * (MLF - 1.0)) + (AF * (2.0 - MLF)); 
   powerFBC1fB2 = KFN_M * pow(abs(Ib2), exponentFBC1fB2); 
 
  // Emitter-base sidewall current shot and 1/f noise 
   powerEBSCS = 2.0 * `QQ * abs(Ib1_s); 
   if (XIBI == 0)  
       powerEBSC1f = 0.0; 
   else  
       powerEBSC1f = KF_M * XIBI * pow((abs(Ib1_s / XIBI)), AF); 
 
  // Reverse base current shot noise and 1/f noise 
   powerRBCS = 2.0 * `QQ * abs(Ib3); 
   powerRBC1f = KF_M * pow(abs(Ib3), AF); 
  
   // Extrinsic current shot noise and 1/f noise 
   powerExCS = 2.0 * `QQ * abs(Iex); 
   powerExC1f = KF_M * (1 - (EXMOD * XEXT)) *  
                pow((abs(Iex) / (1 - (EXMOD * XEXT))), AF); 
   powerExCSMOD = 2.0 * `QQ * abs(XIex) * EXMOD; 
   if (XEXT == 0.0)  
       powerExC1fMOD = 0.0; 
   else  
       powerExC1fMOD = KF_M * EXMOD * XEXT * pow((abs(XIex) / XEXT), AF); 
 
`ifdef SUBSTRATE
   // Substrate current shot noise (between nodes B1 and S, resp. B and S) 
   powerSubsCS_B1S = 2.0 * `QQ * abs(Isub); 
   powerSubsCS_BS  = 2.0 * `QQ * abs(XIsub); 
`endif
 

   // Noise due to the avalanche 
 //   twoqIavl = KAVL * 2.0 * `QQ * Iavl; 
    twoqIavl  = KAVL*Gem*powerCCS;
   powerCCS_A  = powerCCS + twoqIavl * (3.0 + 2.0 * Gem 
                 - (2.0 + 2.0 * Gem)*(2.0 + 2.0 * Gem)/(1.0 + 2.0 * Gem) ); 
 
   // Add noise sources 
   I(e, e1)   <+ white_noise(powerREC);           // "emitter resistance" 
   I(b, b1)   <+ white_noise(powerRBC);           // "base resistance" 

   I(b1, b2)  <+ white_noise(powerRBV);           // "variable baseresistance"  
    
   I(noi, e1) <+ white_noise(twoqIavl);           // "avalanche" 
   I(c2, e1)  <+ white_noise(powerCCS_A);         // "col_emi_shot" 
   I(b2, e1)  <+ white_noise(powerFBCS);          // "bas_emi_forw" 
    
   I(b2, e1)  <+ flicker_noise(powerFBC1fB1, 1);  // "bas_emi_forw" 
   I(b2, e1)  <+ flicker_noise(powerFBC1fB2, 1);  // "bas_emi_forw" 
   I(e1, b1)  <+ white_noise(powerEBSCS);         // "emi_bas_side" 
   I(e1, b1)  <+ flicker_noise(powerEBSC1f, 1);   // "emi_bas_side" 
   I(b1, c1)  <+ white_noise(powerRBCS);          // "bas_col_reve" 
   I(b1, c1)  <+ flicker_noise(powerRBC1f, 1);    // "bas_col_reve" 
   I(b1, c1)  <+ white_noise(powerExCS);          // "Ext_bas_col" 
   I(b1, c1)  <+ flicker_noise(powerExC1f, 1);    // "Ext_bas_col" 
   I(b, c1)   <+ white_noise(powerExCSMOD);       // "Ext_bas_col" 
   I(b, c1)   <+ flicker_noise(powerExC1fMOD, 1); // "Ext_bas_col" 
`ifdef SUBSTRATE
   I(b1, s)   <+ white_noise(powerSubsCS_B1S);    // "bas_sub_current" 
   I(b, s)    <+ white_noise(powerSubsCS_BS);     // "bas_sub_current" 
`endif

/* RvdT, Delft University of Technology 03-12-2007, 
Noise voltage associated with distributed parasitic collector.
In case of vanishing resistance corresponding node 
is not addressed: */

 // RvdT, 31-01-2007: distributed collector resistance

if (RCBLX > 0.0)
 begin
  if (RCBLI > 0.0)
    begin /* all branches exist */
      I(c,  c3)  <+ white_noise(powerRCCxx); // "collector plug resistance" 
      I(c3, c4)  <+ white_noise(powerRCCex); // "extrinsic collector BL resistance" 
      I(c4, c1)  <+ white_noise(powerRCCin); // "intrinsic collector BL resistance" 
    end
   else
    begin  /* only Rcblx exists */
      I(c,  c3)  <+ white_noise(powerRCCxx); // "collector plug resistance" 
      I(c3, c1)  <+ white_noise(powerRCCex); // "extrinsic collector BL resistance" 
    end
 end
else
 begin
  if (RCBLI > 0.0)
    begin   /* only Rcbli exists */
      I(c,  c4)  <+ white_noise(powerRCCxx); // "collector plug resistance" 
      I(c4, c1)  <+ white_noise(powerRCCin); // "intrinsic collector BL resistance" 
    end
   else
    begin /* neither Rcblx nor Rcbli exists */
      I(c,  c1)  <+ white_noise(powerRCCxx); // "collector plug resistance" 
    end
 end

  
end // Noise sources 

