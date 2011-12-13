======================================================================================
======================================================================================

  ---------------------------
  Verilog-A definition of PSP
  ---------------------------


  (c) Copyright 2007, All Rights Reserved, NXP Semiconductors


  Version: PSP 102.1 (including JUNCAP2 200.2), April 2007 (Simkit 2.5)

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

Release notes va-code of PSP 102.1, including JUNCAP2 200.2 (April 2007)
------------------------------------------------------------------------

Focus in this release has been on improving the simulation speed of PSP and JUNCAP2.
The model equations in this release of PSP 102.1 are identical to those in the
October 2006 release. This version features some minor impelementation changes
w.r.t. the previous release.

The main changes have been in the SiMKit version generated from this verilog-A
implementation: improvements in the automatic C-code generation process
and compilation of the C-code. The result is reflected in the SiMKit 2.5 version of
PSP, which shows a very significant simulation speed improvement w.r.t SiMKit 2.4.

The minor implementation changes in the verilog-A code will have some positive effect
on the simulation speed of the verilog-A version as well. Note, however, that the
simulation speed of the verilog-A version of PSP and the improvement w.r.t. the
previous version strongly depend on the verilog-A compiler used. 

PSP 102.1 is backwards compatible with the previous version, PSP 102.0.


======================================================================================
======================================================================================

The functionality of the Verilog-A code in this package is the same as that of the
C-code, which is contained in SIMKIT version 2.5. Note that Operating Point information
is available only in the C-code, not in Verilog-A code.


The PSP-NQS model is provided as Verilog-A code. In SiMKit 2.5, a test version of
the PSP-NQS model is included (identical to that in SiMKit 2.4). This implementation
circumvents the problem of the SpectreVerilog-A-generated C-code being too large to
compile. Moreover, it is computationally more efficient as it uses less rows in the
simulator matrix. On the other hand, this implementation has some known limitations.
More information is available from the authors. Further improvements are expected in
future releases.


This Verilog-A code of PSP is primarily intended as a source for C-code generation
using ADMS. Most of the testing has been done on the C-code which was generated from it.


The authors want to thank Laurent Lemaitre and Colin McAndrew (Freescale)
for their help with ADMS and the implementation of the model code. Geoffrey
Coram (Analog Devices) is acknowledged for useful comments on the Verilog-A
code.
