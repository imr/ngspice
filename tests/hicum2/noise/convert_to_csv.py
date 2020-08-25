""" using a Python script to convert the ADS data to readable CSV and create a small plot of the results -> using DMT (our extraction toolkit)
"""
from DMT.core import DutType
from DMT.ADS import DutAds

dut = DutAds(None, DutType.npn, 'not used', reference_node='E')

dut.import_output_data('tests/hicum2/noise/spectra.raw', key='results')

df = dut.data['results']

df.to_csv('result.csv')