import numpy as np
from struct import pack, unpack, calcsize
from sys import byteorder


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


headSDDS = "SDDS1\n"
columnSDDS = """&column name=col%d, type=double, &end\n"""
parameterSDDS = """&parameter name=col%d, type=double, &end\n"""
columnAttributeStr = ['name=', 'type=', 'units=', 'symbol=', 'format_string=', 'description=']
parameterAttributeStr = ['name=', 'type=', 'units=', 'symbol=', 'format_string=', 'description=']


class writeSDDS:
    """
    Implements an SDDS class in Python.
    Can be used to write out data stored as NumPy arrays or single values stored as a variable
    Included methods:
        SDDS.create_column
        SDDS.create_param
        SDDS.save_sdds
    Does not support creating multi-page SDDS files at this time.
    Acceptable values for colType/parType:
        short
        long
        double
        character (not recommended)
        string    (not recommended)

    @author: Chris
    """

    # TODO: Test binary writeout more carefully. (Multiple types)

    sddsIdentifier = headSDDS

    def __init__(self, page=1, readInFormat='numpyArray'):
        """
        Initialize SDDS object for storing parameter/column data and writing.
        """
        self.format = readInFormat
        self.endianness = byteorder
        self.page = page
        self.columns = []
        self.columnData = []
        self.columnAttributes = []
        self.parameters = []
        self.parameterData = []
        self.parameterAttributes = []

    def create_column(self, colName, colData, colType, colUnits='', colSymbol='', colFormatStr='', colDescription=''):
        """
        Creates a column data object that can be written to file.

        Parameters
        ----------
        colName: str
            Name of the column.
        colData: ndarray (Data type must match 'colType')
        colType: str
            Data type for the column. Must match data type contained in 'colData'. See Description of the class
            for available data types to write.
        colUnits: str (optional)
            String with the units of the column. To be written out to file.
        colSymbol: str (optional)
            Optional symbol string that can be written out to the file. See SDDS manual for syntax.
        colFormatStr: str (optional)
            May specify the form of the printf string for use by SDDS.
        colDescription: str (optional)
            Optional description of the column to write to file.
        """

        self.columns.append(colName)
        self.columnAttributes.append([])
        self.columnAttributes[-1].append(colName)
        self.columnAttributes[-1].append(colType)
        self.columnData.append(colData)
        for attribute in (colUnits, colSymbol, colFormatStr, colDescription):
            if attribute:
                self.columnAttributes[-1].append(attribute)
            else:
                self.columnAttributes[-1].append('')

    def create_param(self, parName, parData, parType, parUnits='', parSymbol='', parFormatStr='', parDescription=''):
        """
        Creates a parameter data object that can be written to file.

        Parameters
        ----------
        parName: str
            Name of the parameter.
        parData: short, long, or double
            Data being written to the SDDS file.
        parType: str
            Data type for the parameter. Must match data type for the variable being written.
            See Description of the class for available data types to write.
        parUnits: str (optional)
            String with the units of the parameter. To be written to file.
        parSymbol: str (optional)
            Optional symbol string that can be written out to the file. See SDDS manual for syntax.
        parFormatStr: str (optional)
            May specify form of the printf string for use by SDDS.
        parDescription: str (optional)
            Optional description of the parameter to be written to the file.
        """

        self.parameters.append(parName)
        self.parameterAttributes.append([])
        self.parameterAttributes[-1].append(parName)
        self.parameterAttributes[-1].append(parType)
        self.parameterData.append(parData)
        for attribute in (parUnits, parSymbol, parFormatStr, parDescription):
            if attribute:
                self.parameterAttributes[-1].append(attribute)
            else:
                self.parameterAttributes[-1].append('')

    def save_sdds(self, fileName, dataMode='ascii'):
        """
        Saves the parameters and columns to file. Parameters and columns are written to the file in the order
        that they were created in the writeSDDS object.

        Parameters
        ----------
        fileName: str
            Name of the file to be written.
        dataMode: Either 'ascii' or 'binary'
            Write mode for the file. Data will either be written to the file in ascii or binary mode.
        """

        self.dataMode = dataMode
        columnString = ''
        parameterString = ''
        outputFile = open(fileName, 'w')

        if len(self.columnData) > 1:
            try:
                columnDataPrint = np.column_stack(self.columnData)
            except ValueError:
                print 'ERROR: All columns on a page must have same length'
        elif len(self.columnData) == 1:
            columnDataPrint = self.columnData[0]
        else:
            columnDataPrint = np.empty([0])

        # Begin header writeout
        outputFile.write(self.sddsIdentifier)
        outputFile.write('!# %s-endian\n' % self.endianness)

        for parameter in self.parameterAttributes:
            for attribute in zip(parameter, parameterAttributeStr):
                if attribute[0]:
                    parameterString = ''.join((parameterString, '%s%s, ' % (attribute[1], attribute[0])))
            outputFile.write('&parameter %s&end\n' % parameterString)
            parameterString = ''

        for column in self.columnAttributes:
            for attribute in zip(column, columnAttributeStr):
                if attribute[0]:
                    columnString = ''.join((columnString,'%s%s, ' % (attribute[1], attribute[0])))
            outputFile.write('&column %s &end\n' % columnString)
            columnString = ''

        outputFile.write('&data mode=%s, &end\n' % self.dataMode)

        # Begin data writeout
        if self.dataMode == 'ascii':
                outputFile.write('%s\n' % columnDataPrint.shape[0])
        if self.dataMode == 'binary':
                outputFile.write('%s' % pack('I', columnDataPrint.shape[0]))

        for parameter in self.parameterData:
            if self.dataMode == 'ascii':
                outputFile.write('%s\n' % parameter)
            if self.dataMode == 'binary':
                outputFile.write('%s' % (pack('d', parameter)))

        if self.dataMode == 'ascii':
            np.savetxt(outputFile, columnDataPrint)
        elif self.dataMode == 'binary':
            columnDataPrint.astype('float64').tofile(outputFile)
        else:
            print "NOT A DEFINED DATA TYPE"

        outputFile.close()
