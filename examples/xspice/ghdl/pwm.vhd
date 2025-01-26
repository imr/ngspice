-- Very simple logic for PWM waveform generation.

library ieee;
use ieee.std_logic_1164.all;
use ieee.math_real.all;

entity pwm is
  port ( output : out std_logic := '1');
end pwm;

architecture pwm_impl of pwm is
  constant Cycles : Integer := 1000;
  constant Samples : Integer := 1000;
  constant uSec : Time := 1 us;
begin
  process
    variable j : Integer := 0;
    variable width : Integer := Cycles / 2;
    variable sine : Real;
  begin
    wait for width * uSec;
    output <= '0';
    wait for (Cycles - width) * uSec;
    j := j + 1;
    if j = Samples then
      j := 0;
    end if;
    sine := sin(real(j) * MATH_2_PI / real(Samples));
    width := integer(real(Samples) * (1.0 + sine) / 2.0);
    if width = 0 then
      output <= '0';
    else
      output <= '1';
    end if;
  end process;
end pwm_impl;

