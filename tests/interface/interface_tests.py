from olive.interface import interface

def test_add_modes():

    passed = True

    my_interface = interface.interface()
    # Check that the type checking is correct
    modes_error_0 = [[1,2.,3,4]]    # should raise exception for first element
    modes_error_1 = [[1.,2,3,4]]    # should raise exception for second element
    modes_error_2 = [[1.,2.,3.,4]]  # should raise exception for third element
    modes_error_3 = [[1.,2.,3,4.]]  # should raise exception for fourth element
    modes_error_4 = [[1.,2.,3,4]]   # should work

    msg = 'Testing add_modes'

    try:
        my_interface.add_modes(modes_error_0)
    except TypeError:
        passed = True
        msg += '\n Test 0 passed'
    else:
        passed = False
        msg += '\n Test 0 failed'

    try:
        my_interface.add_modes(modes_error_1)
    except TypeError:
        passed = True
        msg += '\n Test 1 passed'
    else:
        passed = False
        msg += '\n Test 1 failed'

    try:
        my_interface.add_modes(modes_error_2)
    except TypeError:
        passed = True
        msg += '\n Test 2 passed'
    else:
        passed = False
        msg += '\n Test 2 failed'

    try:
        my_interface.add_modes(modes_error_3)
    except TypeError:
        passed = True
        msg += '\n Test 3 passed'
    else:
        passed = False
        msg += '\n Test 3 failed'

    try:
        my_interface.add_modes(modes_error_4)
    except TypeError:
        passed = False
        msg += '\n Test 4 failed'
    else:
        passed = True
        msg += '\n Test 4 passed'

    print msg

test_add_modes()