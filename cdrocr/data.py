#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
#-------------------------
# imports
#-------------------------
class card:
    height   =  614
    width    =  1024
    class nid:
        class front:
            face        =   [26, 219, 252, 451]
            sign        =   [26, 461, 252, 574]    
            box_dict = {
                    "Bangla Name"    :  [420, 230, 1000, 260],
                    "English Name"   :  [420, 290, 1000, 320],
                    "Fathers Name"   :  [420, 350, 1000, 380],
                    "Mothers Name"   :  [420, 410, 1000, 440],
                    "Date of Birth"  :  [550, 470, 1000, 500],
                    "ID No."         :  [500, 550, 1000, 580]
            }
    class smart:
        class front:
            face        =   [57, 182, 319, 481]
            sign        =   [57, 494, 319, 591]        
            box_dict={
                    "Bangla Name"    :   [325, 200, 750, 220],
                    "English Name"   :   [325, 280, 750, 300],
                    "Fathers Name"   :   [325, 350, 750, 370],
                    "Mothers Name"   :   [325, 440, 750, 460],
                    "Date of Birth"  :   [465, 510, 750, 530],
                    "ID No."         :   [465, 560, 750, 580]
                    }
