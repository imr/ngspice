// Copyright (c) 2000-2007, NXP Semiconductor
// Copyright (c) 2007-2014, Delft University of Technology
// Copyright (c) 2015, Auburn University
// All rights reserved, see IP_NOTICE_DISCLAIMER_LICENSE for further information.

// Noise sources 
  
`NOISE begin  
 
   // Thermal noise 
   common = 4.0 * `KB * Tk;  
   powerREC   = common / RE_TM;  // Emitter resistance 
   powerRBC   = common / RBC_TM; // Base resistance 
   powerRCCxx = common * GCCxx_TM; // Collector resistance 
   powerRCCex = common * GCCex_TM; // Collector resistance 
   powerRCCin = common * GCCin_TM; // Collector resistance 
   powerRBV   = common / Rb2 * (4.0 * eVb1b2 + 5.0)  * `one_third ; // Variable base resistance  

   // Main current shot noise  
   In_N   = (If + Ir) / qBI;
   powerCCS = 2.0 * `QQ * abs(In_N);

   // Weak-avalanche current shot noise
   if (KAVL > 0) begin
       Gem_N = abs(Iavl / In_N);
   end else begin
       Gem_N = 0.0;
   end

   powerIIS = 2.0 * `QQ * Iavl * (Gem_N + 1);

   // Transit time for noise
   if (In_N > 0.0) begin
       Taub_N = (Qbe + Qbc) / In_N;
   end else begin        
       Taub_N = TAUB_T * q1Q * qBI;
   end

   // RF correlation noise model switch
    if (KC == 1) begin
        // use charge partition for noise transit time
        taun = XQB * Taub_N;
    end else if (KC == 2) begin
        // use fraction of transit time for noise transit time
        taun = FTAUN * Taub_N;
    end else begin // KC == 0
        // no correlation noise
        taun = 0;
    end

   // Forward base current shot noise and 1/f noise 
   powerFBCS = 2.0 * `QQ * (abs(Ib1) + abs(Ib2) + abs(Izteb)); 
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

   // Reference un-correlated current shot noise sources
   I(noi) <+ white_noise(powerCCS);
   I(noi) <+ V(noi);
   
   // Implementing correlated noise sources
   I(b2, e1) <+ taun * ddt(V(noi));
   I(c2, b2) <+ Gem_N * V(noi);
   I(c2, e1) <+ V(noi);
   
   // Implementing un-correlated noise sources
   I(c2, b2) <+ white_noise(powerIIS);
   I(b2, e1) <+ white_noise(powerFBCS);
   
   // Add noise sources  
   I(e, e1)    <+ white_noise(powerREC);            // "emitter resistance" 
   I(b, b1)    <+ white_noise(powerRBC);            // "base resistance" 
   I(b1, b2)   <+ white_noise(powerRBV);            // "variable base resistance"  
   I(b2, e1)   <+ flicker_noise(powerFBC1fB1, 1);   // "bas_emi_forw" 
   I(b2, e1)   <+ flicker_noise(powerFBC1fB2, 1);   // "bas_emi_forw" 
   I(e1, b1)   <+ white_noise(powerEBSCS);          // "emi_bas_side" 
   I(e1, b1)   <+ flicker_noise(powerEBSC1f, 1);    // "emi_bas_side"
   I(b1, c4)   <+ white_noise(powerRBCS);           // "bas_col_reve"
   I(b1, c4)   <+ flicker_noise(powerRBC1f, 1);     // "bas_col_reve"
   I(b1, c4)   <+ white_noise(powerExCS);           // "Ext_bas_col"
   I(b1, c4)   <+ flicker_noise(powerExC1f, 1);     // "Ext_bas_col"
   I(b, c3)    <+ white_noise(powerExCSMOD);        // "Ext_bas_col"
   I(b, c3)    <+ flicker_noise(powerExC1fMOD, 1);  // "Ext_bas_col"
   
`ifdef SUBSTRATE
   I(b1, s)   <+ white_noise(powerSubsCS_B1S);    // "bas_sub_current" 
   I(b, s)    <+ white_noise(powerSubsCS_BS);     // "bas_sub_current" 
`endif

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
    
end // Noise 

