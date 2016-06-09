# -*- coding: utf-8 -*-
u"""Interface class for creating and running simulations using OLIVE
:copyright: Copyright (c) 2016 RadiaSoft LLC.  All Rights Reserved.
:license: http://www.apache.org/licenses/LICENSE-2.0.html
"""

from olive.fields import field_data

class interface:


    def add_modes(self, mode_array):
        """Input an array of four-component arrays, one for each mode:
        [frequency, k-vector, horizontal power, vertical power]"""

        for array in mode_array:
            if type(array[0]) != float:
                raise TypeError('frequency must be a floating point number')
            if type(array[1]) != float:
                raise TypeError('mode k-vector must be a floating point '
                                'number')
            if type(array[2]) != int:
                raise TypeError('horizontal exponents must be integers')
            if type(array[3]) != int:
                raise TypeError('vertical exponents must be integers')

        self.sim_fields = field_data.field_data()
        self.sim_fields.create_modes(mode_array)
