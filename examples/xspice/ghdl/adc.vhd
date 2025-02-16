library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity adc is
  generic ( Bits : integer := 6 );
  port (
    Clk : in std_logic;
    Comp : in std_logic;
    Start : in std_logic;
    Sample : out std_logic;
    Done : out std_logic;
    Result : out unsigned(0 to Bits - 1)
  );
end entity; 

architecture ghdl_adc of adc is
  signal SR : unsigned(0 to Bits - 1);
  signal Result_Reg : unsigned(0 to Bits - 1);
  signal Sample_Reg : std_logic := '0';
  signal Running : std_logic := '0';
begin
  Result <= Result_Reg;
  Sample <= Sample_Reg;
  
  process (Clk)
    constant Zeros : unsigned(0 to Bits - 1) := (others => '0');
    variable NextSR : unsigned(0 to Bits - 1);
  begin
    if rising_edge(Clk) then
      if Running = '1' then
        if Sample_Reg = '1' then
          Sample_Reg <= '0';
          SR(Bits - 1) <= '1';
          Result_Reg(Bits - 1) <= '1';
        else
          if SR /= 0 then
            NextSR := shift_left(SR, 1);
            if Comp = '1' then
              Result_Reg <= (Result_Reg and not SR) or NextSR;
            else
              Result_Reg <= Result_Reg or NextSR;
            end if;
            SR <= NextSR;
          else
            Running <= '0';
            Done <= '1';
          end if;
        end if;
      else
        if Start = '1' then
          Running <= '1';
          Sample_Reg <= '1';
          Done <= '0';
          SR <= Zeros;
          Result_Reg <= Zeros;
        end if;
      end if;
    end if;
  end process;
end architecture;

