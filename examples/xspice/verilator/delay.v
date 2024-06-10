`timescale 1us/100ns

module delay(out);
   output reg [4:0] out;
   reg		    t;

   initial out = 0;
   always begin
      #1;
      t = out[4];
      out <<= 1;
      out[0] = ~t;
   end
endmodule; // delay
