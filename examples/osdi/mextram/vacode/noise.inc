// Copyright (c) 2000-2007, NXP Semiconductor
// Copyright (c) 2007-2014, Delft University of Technology
// Copyright (c) 2015-2020, Auburn University
// All rights reserved, see IP_NOTICE_DISCLAIMER_LICENSE for further information.

// Noise sources
// Thermal noise
common = 4.0 * `KB * Tk;
power_rec = common / re_t;  // Emitter resistance
power_rbc = common / rbc_t; // Constant Base resistance
power_rcc_xx = common * gcc_xx_t; // Collector resistance
power_rcc_ex = common * gcc_ex_t; // Collector resistance
power_rcc_in = common * gcc_in_t; // Collector resistance
power_rbv = common / Rb2 * (4.0 * eVb1b2 + 5.0) * `one_third; // Variable base resistance

// Main current shot noise
In_N = (If + Ir) / qBI;
powerCCS = 2.0 * `QQ * abs(In_N);

// Weak-avalanche current shot noise
if (kavl > 0) begin
    Gem_N = abs(Iavl / In_N);
end else begin
    Gem_N = 0.0;
end

powerIIS = 2.0 * `QQ * Iavl * (Gem_N + 1.0);

// Transit time for noise
if (In_N > 0.0) begin
    Taub_N = (Qbe + Qbc) / In_N;
end else begin
    Taub_N = taub_t * q1Q * qBI;
end

// RF correlation noise model switch
if (kc == 1) begin
    // use charge partition for noise transit time
    taun = xqb * Taub_N;
end else if (kc == 2) begin
    // use fraction of transit time for noise transit time
    taun = ftaun * Taub_N;
end else begin
    // kc == 0, no correlation noise
    taun = 0.0;
end

// Forward base current shot noise
powerFBCS = 2.0 * `QQ * abs(Ib1 + Ib2 - Izteb);

// Ideal forward base current 1/f noise
Ib_fwd_ideal_tot = Ib1 + Ib1_s;
powerFBC1f = kf * pow(abs(Ib_fwd_ideal_tot), af);
if (Ib_fwd_ideal_tot < 0) begin
    powerFBC1f = -powerFBC1f;
end

// Non-ideal forward base current 1/f noise
Ib_fwd_non_ideal_tot = Ib2 + Ib2_s + Ibrel;
powerNFBC1f = kfn * pow(abs(Ib_fwd_non_ideal_tot), afn);
if (Ib_fwd_non_ideal_tot < 0) begin
    powerNFBC1f = -powerNFBC1f;
end

// Emitter-base sidewall current shot
powerEBSCS = 2.0 * `QQ * abs(Ib1_s + Ib2_s + Ibrel);

// Reverse base current shot noise and 1/f noise
powerRBCS = 2.0 * `QQ * abs(Ib3);
powerRBC1f = kf * pow(abs(Ib3), af);
if (Ib3 < 0) begin
    powerRBC1f = -powerRBC1f;
end
powerZTCB = 2.0 * `QQ * abs(Iztcb);

// Extrinsic current shot noise and 1/f noise
powerExCS = 2.0 * `QQ * abs(Iex);
powerExC1f = kf * (1.0 - (exmod * xext)) *
                    pow((abs(Iex) / (1.0 - (exmod * xext))), af);
if (Iex < 0) begin
    powerExC1f = -powerExC1f;
end
powerExCSMOD = 2.0 * `QQ * abs(XIex) * exmod;
if (xext == 0.0) begin
    powerExC1fMOD = 0.0;
end else begin
    powerExC1fMOD = kf * exmod * xext * pow((abs(XIex) / xext), af);
end
if (XIex < 0) begin
    powerExC1fMOD = -powerExC1fMOD;
end

`ifdef SUBSTRATE
    // Substrate current shot noise (between nodes B1 and S, resp. B and S)
    powerSubsCS_B2S = 2.0 * `QQ * abs(Isub_int);
    powerSubsCS_B1S = 2.0 * `QQ * abs(Isub);
    powerSubsCS_BS  = 2.0 * `QQ * abs(XIsub);
`endif

// Reference un-correlated current shot noise sources
I(noi) <+ white_noise(powerCCS * mult, "in");
I(noi) <+ V(noi);

// Implementing correlated noise sources
I(b2, e1) <+ taun * ddt(V(noi));
I(c2, b2) <+ Gem_N * V(noi);
I(c2, e1) <+ V(noi);

// Implementing un-correlated noise sources
I(c2, b2) <+ white_noise(powerIIS * mult, "iavl");
I(b2, e1) <+ white_noise(powerFBCS * mult, "ib2e1");

// Add noise sources
I(e, e1)    <+ white_noise(power_rec * mult, "re");
I(b, b1)    <+ white_noise(power_rbc * mult, "rbc");
I(b1, b2)   <+ white_noise(power_rbv * mult, "rbv");
I(b2, e1)   <+ flicker_noise(powerFBC1f * mult, 1, "ib2e1_f");
I(b1, e1)   <+ flicker_noise(powerNFBC1f * mult, 1, "ib1e1_f");
I(b1, e1)   <+ white_noise(powerEBSCS * mult, "ib1e1");
I(b1, c4)   <+ white_noise(powerRBCS * mult, "ib3");
I(b1, c4)   <+ flicker_noise(powerRBC1f * mult, 1, "ib3_f");
I(c2, b2)   <+ white_noise(powerZTCB * mult, "iztcb");
I(b1, c4)   <+ white_noise(powerExCS * mult, "iex");
I(b1, c4)   <+ flicker_noise(powerExC1f * mult, 1, "iex_f");
I(b, c3)    <+ white_noise(powerExCSMOD * mult, "xiex");
I(b, c3)    <+ flicker_noise(powerExC1fMOD * mult, 1, "xiex_f");

`ifdef SUBSTRATE
    I(b2, s)   <+ white_noise(powerSubsCS_B2S * mult, "isub_int");
    I(b1, s)   <+ white_noise(powerSubsCS_B1S * mult, "isub");
    I(b, s)    <+ white_noise(powerSubsCS_BS * mult, "xisub");
`endif

if (rcblx > 0.0) begin
    if (rcbli > 0.0) begin /* all branches exist */
        I(c,  c3)  <+ white_noise(power_rcc_xx * mult, "rcc");
        I(c3, c4)  <+ white_noise(power_rcc_ex * mult, "rcblx");
        I(c4, c1)  <+ white_noise(power_rcc_in * mult, "rcbli");
    end else begin  /* only Rcblx exists */
        I(c,  c3)  <+ white_noise(power_rcc_xx * mult, "rcc");
        I(c3, c1)  <+ white_noise(power_rcc_ex * mult, "rcblx");
    end
end else begin
    if (rcbli > 0.0) begin   /* only Rcbli exists */
        I(c,  c4)  <+ white_noise(power_rcc_xx * mult, "rcc");
        I(c4, c1)  <+ white_noise(power_rcc_in * mult, "rcbli");
    end else begin /* neither Rcblx nor Rcbli exists */
        I(c,  c1)  <+ white_noise(power_rcc_xx * mult, "rcc");
    end
end

