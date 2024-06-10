`timescale 1us/100ns

//`include "constants.vams"
`define M_TWO_PI 6.28318530717958647652

module pwm(out);
   output reg out;
   parameter  Cycles = 1000, Samples = 1000;
   integer    i, j, width;
   real	      sine;

   initial begin
      i = 0;
      j = 0;
      width = Cycles / 2;
      out = 0;
   end

   always begin
      #1;
      ++i;
      if (i == width)
	out = 0;
      if (i == Cycles) begin
	 i = 0;
	 ++j;
	 if (j == Samples)
	   j = 0;
	 sine = $sin(j * `M_TWO_PI / Samples);
	 width = $rtoi(Samples * (1.0 + sine) / 2.0);
	 out = (width == 0) ? 0 : 1;
      end
   end
endmodule // pwm
