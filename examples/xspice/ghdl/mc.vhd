-- Test crude algorithm for a simple motor controller.

library ieee;
use ieee.std_logic_1164.all;

entity mc is
  -- Control values that may be overrriden by the "sim_args" option
  -- in a netlist's .model line.  Intgers are used as GHDL does not
  -- support overriding Real values.
  generic (
    Target    : Integer := 4000; -- notional RPM
    Dead_Band : Integer := 200;
    Full_Band : Integer := 600
  );
  port (
    Zero    : in  std_logic;
    Tach    : in  std_logic;
    Trigger : out std_logic
  );
end mc;

architecture mc_arch of mc is
  shared variable Speed : Real := 0.0;
begin
  Tachometer : process (Tach) is
    variable Time  : Real := 0.0;
    variable Last_pulse : Real := 0.0;
  begin
    if rising_edge(Tach) or falling_edge(Tach) then
      Time := real(now / 1 ns) * 1.0e-9;
      Speed := 30.0 / (Time - Last_pulse);
      Last_pulse := Time;
    end if;
  end process Tachometer;

  Controller : process (Zero) is
    variable Skip     : Integer := 0;
    variable Count    : Integer := 0;
    variable Even     : Boolean := True;
    variable Was_even : Boolean := True;
    variable Error    : Integer;
  begin
    -- Trigger triac on zero crossings, preventing DC current.

    if rising_edge(Zero) then
      Even := not Even;
      if (Count >= Skip) and (Even /= Was_even) then
        Trigger <= '1';
        Was_even := Even;
        if Count > Skip then
          Count := 0;
        else
          Count := -1;
        end if;
      else
        Trigger <= '0';
      end if;
      Count := Count + 1;

      -- A very crude feedback mechanism.

      Error := integer(Speed) - Target;
      if Error > Full_Band then
        -- No drive
        Skip := 1;
        Count := 0;
      elsif Error < -Full_Band then
        -- Full Drive
        Skip := 0;
      elsif Error > Dead_Band then
        Skip := Skip + 1;
      elsif Error < -Dead_Band then
        if Skip > 0 then
          Skip := Skip - 1;
        end if;
      end if;

    end if;
  end process Controller;
end mc_arch;
