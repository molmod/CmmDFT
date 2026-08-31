from cmmdft.units_constants import *

SAFT_pars_dict = {'CH4' : (16.031*amu, 3.70051*angstrom, 150*boltzmann, 1.0),
                  'C2H6': (30.047*amu, 3.51681*angstrom, 191.45389*boltzmann, 1.60686),
                  'C2H4': (28.031*amu, 3.40769*angstrom, 177.32571*boltzmann, 1.59196),
                  'C3H6': (42.047*amu, 3.53847*angstrom, 207.73187*boltzmann, 1.95306),
                  'C3H8': (44.063*amu, 3.6244*angstrom, 209.08586*boltzmann, 1.98602),
                  'CO2' : (43.99*amu,  2.57855*angstrom, 153.31864*boltzmann, 2.53096)}

chk_dict = {'CH4_AA': 'InputFiles/Guest/CH4/struct_AA_TraPPE.chk',
            'CO2_AA': 'InputFiles/Guest/CO2/struct_AA.chk',
            'C2H6_CA': 'InputFiles/Guest/C2H6/struct_CA.chk',
            'C2H6_AA': 'InputFiles/Guest/C2H6/struct_AA_TraPPE.chk',
            'C2H4_CA': 'InputFiles/Guest/C2H4/struct_CA.chk',
            'C2H4_AA': 'InputFiles/Guest/C2H4/struct_AA.chk',
            'CH4_UA': 'InputFiles/Guest/CH4/struct_UA.chk',
            'C3H6_CA' : 'InputFiles/Guest/C3H6/struct_CA.chk',
            'C3H8_CA' : 'InputFiles/Guest/C3H8/struct_CA.chk',
            }

pars_dict = {'CH4_AA': 'InputFiles/Guest/CH4/pars_vdw_AA_trappe.txt',
            'CO2_AA': 'InputFiles/Guest/CO2/pars_eivdw_AA_trappe.txt',
            'C2H6_CA': 'InputFiles/Guest/C2H6/pars_vdw_CA_trappe.txt',
            'C2H6_AA': 'InputFiles/Guest/C2H6/pars_vdw_AA_trappe.txt',
            'C2H4_CA': 'InputFiles/Guest/C2H4/pars_vdw_CA_trappe.txt',
            'C2H4_AA': 'InputFiles/Guest/C2H4/pars_vdw_AA_trappe.txt',
            'CH4_UA': 'InputFiles/Guest/CH4/pars_vdw_UA_trappe.txt',
            'C3H6_CA' : 'InputFiles/Guest/C3H6/pars_vdw_CA_trappe.txt',
            'C3H8_CA' : 'InputFiles/Guest/C3H8/pars_vdw_CA_trappe.txt',
            }

LJ_pars_dict = {'CH4': (16*amu, 3.75*angstrom, 0.29411*kcalmol),
            'CO2': (12.011*amu+2*15.999*amu, 3.75*angstrom, 0.4691794*kcalmol),
            'C2H6': (12.011*amu*2+6*1*amu, 4.41*angstrom, 0.46209*kcalmol),
            'C2H4':(2.011*amu*2+4*1*amu, 4.24*angstrom, 0.42730*kcalmol)
            }