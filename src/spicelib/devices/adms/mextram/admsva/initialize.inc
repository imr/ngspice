// Initialze model constants  

     // Impact ionization constants (NPN - PNP) 
   
if (TYPE == 1) begin  
  
   An = 7.03e7;   
   Bn = 1.23e8;   
  
end else begin  
                          
   An = 1.58e8;   
   Bn = 2.04e8;   
  
end  
 
Xext1 = 1.0 - XEXT; 

    // Temperature independent  MULT scaling 

`ifdef SELFHEATING
   CTH_M  = CTH   * MULT;    
`endif

   CBEO_M = CBEO  * MULT;  
   CBCO_M = CBCO  * MULT;  

   invMULT    = 1.0 / MULT;   
   SCRCV_M    = SCRCV    * invMULT;    

   KF_M     = KF  * pow(MULT, 1.0 - AF);   
   KFN_M    = KFN * pow(MULT, 1.0 - (2.0 * (MLF - 1.0) + AF * (2.0 - MLF)));   

