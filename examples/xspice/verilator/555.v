// Very simple logic for a 555 timer simulation

`timescale 1us/100ns

module VL555(Trigger, Threshold, Reset, Q, Qbar);
   input wire                Trigger, Threshold, Reset; // Reset is active low.
   output reg		     Q;
   output wire		     Qbar;

   wire			     ireset, go;

   assign Qbar = !Q;

   // The datasheet implies that Trigger overrides Threshold.

   assign go = Trigger & Reset;
   assign ireset = (Threshold & !Trigger) | !Reset;

   initial begin
      Q = 0;
   end

   always @(posedge(go), posedge(ireset)) begin
      Q = go;
   end
endmodule
