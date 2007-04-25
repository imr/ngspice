======================================================================================
======================================================================================

  ---------------------------
  Verilog-A definition of PSP
  ---------------------------


  (c) Copyright 2006, All Rights Reserved, NXP Semiconductors


  Version: PSP 102.1 (including JUNCAP2 200.2), October 2006 (Simkit 2.4)

======================================================================================
======================================================================================

 Authors: G.D.J. Smit, A.J. Scholten, and D.B.M. Klaassen (NXP Semiconductors Research)
          R. van Langevelde (Philips Research)
          G. Gildenblat, X. Li, and W. Wu (The Arizona State University)



The most recent version of the model code, the documentation, and contact information
can be found on:

         http://PSPmodel.asu.edu/
or         
         http://www.nxp.com/Philips_Models/

======================================================================================
======================================================================================

This package consists of several files:

     - readme.txt                     This file

     - psp102.va                      Main file for global ("geometrical") model
     - psp102b.va                     Main file for global binning model
     - psp102e.va                     Main file for local  ("electrical") model
     - psp102_nqs.va                  Main file for global ("geometrical") model with NQS-effects
     - psp102b_nqs.va                 Main file for global binning model with NQS-effects
     - psp102e_nqs.va                 Main file for local  ("electrical") model with NQS-effects
     - juncap200.va                   Main file for JUNCAP2 stand-alone model

     - SIMKIT_macrodefs.include       Common macro definitions
     - PSP102_macrodefs.include       Macro definitions for PSP
     - PSP102_module.include          Actual model code for intrinsic MOS model
     - PSP102_binning.include         Geometry scaling equation for binning 
     - PSP102_binpars.include         Parameterlist for global PSP binning model
     - PSP102_nqs_macrodefs.include   Macro definitions for PSP-NQS
     - PSP102_InitNQS.include         PSP-NQS initialization code
     - PSP102_ChargesNQS.include      Calculation of NQS-charge contributions
     - JUNCAP200_macrodefs.include    Macro definitions for JUNCAP2 model
     - JUNCAP200_parlist.include      JUNCAP2 parameter list
     - JUNCAP200_varlist.include      JUNCAP2 variable declarations
     - JUNCAP200_InitModel.include    JUNCAP2 model initialization code

======================================================================================
======================================================================================

Usage
-----

Depending which model one wants to use, one should compile one of the seven .va-files
(psp102.va, psp102b.va, psp102e.va, psp102_nqs.va, psp102b_nqs.va, psp102e_nqs.va, and
juncap200.va). The module names are "PSP102VA" and "PSPNQS102VA" for the global PSP-model
(QS and NQS, respectively), and similarly "PSP102BVA" and "PSPNQS102BVA" for the binning
PSP-model, "PSP102EVA" and "PSPNQS102EVA" for the local PSP-model, and "JUNCAP200" for
the JUNCAP2-model.


======================================================================================
======================================================================================

Release notes va-code of PSP 102.1, including JUNCAP2 200.2 (October 2006)
--------------------------------------------------------------------------

PSP 102.1 is backwards compatible with the previous version, PSP 102.0, and
resolves some minor implementation issues and bugs. Next to the existing
verilog-A implementation, a test version of the NQS model is now available
in the SiMKit.

- Added clipping boundaries for SWNQS.
- Removed several "empty statements".
- Resolved SpectreRF hidden state problem
- Solved minor bugs in stress model
- Solved minor bug in juncap model
- Changed the NQS-module names in the verilog-A code
- Made some implementation changes for optimization and maintenance purposes
   * Introduced verilog-macro for nodes G/GP, B/BP, B/BS, and B/BD
   * Make drain junction voltage equal to V(D, B) instead of V(D, S) + V(S, B)
   * Extra intermediate variables for parasitic resistor noise densities
   * Modified implementation of NQS-model

======================================================================================
======================================================================================

The functionality of the Verilog-A code in this package is the same as that of the
C-code, which is contained in SIMKIT version 2.4. Note that Operating Point information
is available only in the C-code, not in Verilog-A code.


The PSP-NQS model is provided as Verilog-A code. In SiMKit 2.4, for the first time a
test version of the PSP-NQS model is included. This implementation circumvents the
of the SpectreVerilog-A-generated C-code being too large to compile. Moreover, it is
computationally more efficient as it uses less rows in the simulator matrix. On the
other hand, this implementation has some known limitations. More information is
available from the authors. Further improvements are expected in future releases.


This Verilog-A code of PSP is primarily intended as a source for C-code generation
using ADMS. Most of the testing has been done on the C-code which was generated from it.


The authors want to thank Laurent Lemaitre and Colin McAndrew (Freescale)
for their help with ADMS and the implementation of the model code. Geoffrey
Coram (Analog Devices) is acknowledged for his useful comments on the Verilog-A
code.
