-- Very simple logic for a 555 timer simulation

library ieee;
use ieee.std_logic_1164.all;

entity timer_core is
  port (
    Trigger, Threshold, Reset : in std_logic;
    Q, Qbar : out std_logic
  );
end timer_core;

architecture rtl of timer_core is
  signal result : std_logic := '0';
  signal ireset, go : std_logic;
begin
  go <= Trigger and Reset;
  ireset <= (Threshold and not go) or not Reset;
  Q <= result;
  Qbar <= not result;

  process (go, ireset)
  begin
    if rising_edge(go) or rising_edge(ireset) then
      result <= go;
    end if;
  end process;
end rtl;

