import numpy as np
from struct import unpack, calcsize


class readSDDS:
    """
    Class for reading SDDS data files.
    Usage:
        After initialization parameter data can be read out using the 'read_params' method
        and column data through the 'read_columns' method.

    Caveats:
        - System is little-endian
        - Data stored in binary
        - No array data (only parameters and columns)
        - No string data (unless stored as fixed value)
    """

    def __init__(self, input_file, verbose=False):
        """
        Initialize the read in.

        Parameters
        ----------
        input_file: str
            Name of binary SDDS file to read.
        verbose: Boolean
            Print additional data about detailing intermediate read in process.
        """

        self.openf = open(input_file, 'r')
        self.verbose = verbose
        self.header = []
        self.params = []

        self.param_key = 'i'  # Include row count with parameters
        self.column_key = ''
        self.pointer = 0

        self.param_size = 0
        self.column_size = 0

        self.param_names = ['rowCount']

    def _read_header(self):
        """
        Read in ASCII data of the header to string and organize.
        """

        while True:
            new_line = self.openf.readline()
            if new_line.find('&data') == 0:
                self.header.append(new_line)
                break
            else:
                self.header.append(new_line)

        self.pointer = self.openf.tell()

        return self.header

    def _parse_header(self):
        """
        Parse header data to instruct unpacking procedure.
        """

        self.parsef = True

        columns = []

        # Find Parameters and Column
        for line in self.header:
            if line.find('&parameter') == 0:
                self.params.append(line)
            if line.find('&column') == 0:
                columns.append(line)

        # TODO: Need to check data types on other systems (weird that need use i for longs here)
        # Construct format string for parameters and columns
        for param in self.params:
            if param.find('type=double') > -1:
                self.param_key += 'd'
            elif param.find('type=long') > -1:
                self.param_key += 'i'
            elif param.find('type=short') > -1:
                self.param_key += 's'
            else:
                pass

        for column in columns:
            if column.find('type=double') > -1:
                self.column_key += 'd'
            elif column.find('type=long') > -1:
                self.column_key += 'i'
            elif column.find('type=short') > -1:
                self.column_key += 's'
            else:
                pass

        for param in self.params:
            i0 = param.find('name') + 5
            ie = param[i0:].find(',')
            self.param_names.append(param[i0:i0+ie])

        self.param_size = calcsize(self.param_key)
        self.column_size = calcsize(self.column_key)

        if self.verbose:
            print "Parameter unpack size: %s bytes \nColumn unpack size: %s bytes" % (self.param_size, self.column_size)

        if self.verbose:
            print "Parameter key string: %s \nColumn key string: %s" % (self.param_key, self.column_key)

        return self.param_key, self.column_key

    def read_params(self):
        """
        Read parameter data from the SDDS file.

        Returns
        -------
        parameters: dictionary
            Dictionary object with parameters names and values.
        """

        try:
            self.parsef
        except AttributeError:
            self._read_header()
            self._parse_header()
            if self.verbose:
                print "Header data read and parsed."

        parameters = {}

        # Reset pointer back to beginning of binary data
        self.openf.seek(self.pointer)

        param_data = unpack(self.param_key, self.openf.read(self.param_size))

        for param, value in zip(self.param_names, param_data):
            parameters[param] = value

        self.row_count = parameters['rowCount']

        return parameters

    def read_columns(self):
        """
        Read column data from the SDDS file.

        Returns
        -------
        columns: ndarray
            NumPy array with column data.
        """

        try:
            self.row_count
        except AttributeError:
            self.read_params()

        columns = []

        for i in range(self.row_count):
            columns.append(unpack(self.column_key, self.openf.read(self.column_size)))

        return np.asarray(columns)


if __name__ == '__main__':
    test = readSDDS('../../tests/interface/test.bun')
    tpar = test.read_params()
    tcol = test.read_columns()
    print tcol[0:2, :]
    print tpar
    print tcol.shape