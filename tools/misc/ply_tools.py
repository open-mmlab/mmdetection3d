import sys

import numpy as np

# Define PLY types
ply_dtypes = dict([(b'int8', 'i1'), (b'char', 'i1'), (b'uint8', 'u1'),
                   (b'uchar', 'u1'), (b'int16', 'i2'), (b'short', 'i2'),
                   (b'uint16', 'u2'), (b'ushort', 'u2'), (b'int32', 'i4'),
                   (b'int', 'i4'), (b'uint32', 'u4'), (b'uint', 'u4'),
                   (b'float32', 'f4'), (b'float', 'f4'), (b'float64', 'f8'),
                   (b'double', 'f8')])

# Numpy reader format
valid_formats = {
    'ascii': '',
    'binary_big_endian': '>',
    'binary_little_endian': '<'
}


def read_ply(filename, triangular_mesh=False):
    """Read ".ply" files.

    Args:
        filename (str): the name of the file to read.

    Examples:
        Store data in file
        >>> points = np.random.rand(5, 3)
        >>> values = np.random.randint(2, size=10)
        >>> write_ply('example.ply',[points, values],['x', 'y', 'z', 'values'])
        Read the file
        >>> data = read_ply('example.ply')
        >>> values = data['values']
        array([0, 0, 1, 1, 0])
        >>> points = np.vstack((data['x'], data['y'], data['z'])).T
        array([[ 0.466  0.595  0.324]
               [ 0.538  0.407  0.654]
               [ 0.850  0.018  0.988]
               [ 0.395  0.394  0.363]
               [ 0.873  0.996  0.092]])
    """

    with open(filename, 'rb') as plyfile:

        # Check if the file start with ply
        if b'ply' not in plyfile.readline():
            raise ValueError('The file does not start with the word ply')

        # get binary_little/big or ascii
        fmt = plyfile.readline().split()[1].decode()
        if fmt == 'ascii':
            raise ValueError('The file is not binary')

        # get extension for building the numpy dtypes
        ext = valid_formats[fmt]

        # PointCloud reader vs mesh reader
        if triangular_mesh:

            # Parse header
            num_points, num_faces, properties = parse_mesh_header(plyfile, ext)

            # Get point data
            vertex_data = np.fromfile(
                plyfile, dtype=properties, count=num_points)

            # Get face data
            face_properties = [('k', ext + 'u1'), ('v1', ext + 'i4'),
                               ('v2', ext + 'i4'), ('v3', ext + 'i4')]
            faces_data = np.fromfile(
                plyfile, dtype=face_properties, count=num_faces)

            # Return vertex data and concatenated faces
            faces = np.vstack(
                (faces_data['v1'], faces_data['v2'], faces_data['v3'])).T
            data = [vertex_data, faces]

        else:

            # Parse header
            num_points, properties = parse_header(plyfile, ext)

            # Get data
            data = np.fromfile(plyfile, dtype=properties, count=num_points)

    return data


def write_ply(filename, field_list, field_names, triangular_faces=None):
    """Write ".ply" files.

    Args:
        filename (str):the name of the file to which the data is saved.
            A '.ply' extension will be appended to the
            file name if it does no already have one.
        field_list (list|tuple|array):the fields to be saved in the ply file.
            Either numpy array, list of numpy arrays or tuple of numpy arrays.
            Each 1D numpy array and each column of 2D numpy arrays
            are considered as one field.
        field_names (list):
            the name of each fields as a list of strings.
            Has to be the same length as the number of fields.

    Examples:
        >>> points = np.random.rand(10, 3)
        >>> write_ply('example1.ply', points, ['x', 'y', 'z'])

        >>> values = np.random.randint(2, size=10)
        >>> write_ply('example2.ply',[points,values],['x', 'y', 'z', 'values'])

        >>> colors = np.random.randint(255, size=(10,3), dtype=np.uint8)
        >>> field_names = ['x', 'y', 'z', 'red', 'green', 'blue', 'values']
        >>> write_ply('example3.ply', [points, colors, values], field_names)
    """

    # Format list input to the right form
    field_list = list(field_list) if (type(field_list) == list
                                      or type(field_list) == tuple) else list(
                                          (field_list, ))
    for i, field in enumerate(field_list):
        if field.ndim < 2:
            field_list[i] = field.reshape(-1, 1)
        if field.ndim > 2:
            print('fields have more than 2 dimensions')
            return False

            # check all fields have the same number of data
    n_points = [field.shape[0] for field in field_list]
    if not np.all(np.equal(n_points, n_points[0])):
        print('wrong field dimensions')
        return False

        # Check if field_names and field_list have same nb of column
    n_fields = np.sum([field.shape[1] for field in field_list])
    if (n_fields != len(field_names)):
        print('wrong number of field names')
        return False

    # Add extension if not there
    if not filename.endswith('.ply'):
        filename += '.ply'

    # open in text mode to write the header
    with open(filename, 'w') as plyfile:

        # First magical word
        header = ['ply']

        # Encoding format
        header.append('format binary_' + sys.byteorder + '_endian 1.0')

        # Points properties description
        header.extend(header_properties(field_list, field_names))

        # Add faces if needded
        if triangular_faces is not None:
            header.append('element face {:d}'.format(
                triangular_faces.shape[0]))
            header.append('property list uchar int vertex_indices')

        # End of header
        header.append('end_header')

        # Write all lines
        for line in header:
            plyfile.write('%s\n' % line)

    # open in binary/append to use tofile
    with open(filename, 'ab') as plyfile:

        # Create a structured array
        i = 0
        type_list = []
        for fields in field_list:
            for field in fields.T:
                type_list += [(field_names[i], field.dtype.str)]
                i += 1
        data = np.empty(field_list[0].shape[0], dtype=type_list)
        i = 0
        for fields in field_list:
            for field in fields.T:
                data[field_names[i]] = field
                i += 1

        data.tofile(plyfile)

        if triangular_faces is not None:
            triangular_faces = triangular_faces.astype(np.int32)
            type_list = [('k', 'uint8')] + [(str(ind), 'int32')
                                            for ind in range(3)]
            data = np.empty(triangular_faces.shape[0], dtype=type_list)
            data['k'] = np.full((triangular_faces.shape[0], ),
                                3,
                                dtype=np.uint8)
            data['0'] = triangular_faces[:, 0]
            data['1'] = triangular_faces[:, 1]
            data['2'] = triangular_faces[:, 2]
            data.tofile(plyfile)
    return True


def header_properties(field_list, field_names):
    """Points properties description
    Args:
        field_list (list|tuple|np.array):
            the fields to be saved in the ply file.
            Either numpy array, list of numpy arrays or tuple of numpy arrays.
            Each 1D numpy array and each column of 2D numpy arrays
            are considered as one field.
        field_names (list):
            the name of each fields as a list of strings.
            Has to be the same length as the number of fields.
    """
    # List of lines to write
    lines = []

    # First line describing element vertex
    lines.append('element vertex %d' % field_list[0].shape[0])

    # Properties lines
    i = 0
    for fields in field_list:
        for field in fields.T:
            lines.append('property %s %s' % (field.dtype.name, field_names[i]))
            i += 1

    return lines


def parse_mesh_header(plyfile, ext):
    """Parse header.

    Args:
        plyfile (file): the plyfile object
        ext (str): extension for building the numpy dtypes
    """
    # Variables
    line = []
    vertex_properties = []
    num_points = None
    num_faces = None
    current_element = None

    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        # Find point element
        if b'element vertex' in line:
            current_element = 'vertex'
            line = line.split()
            num_points = int(line[2])

        elif b'element face' in line:
            current_element = 'face'
            line = line.split()
            num_faces = int(line[2])

        elif b'property' in line:
            if current_element == 'vertex':
                line = line.split()
                vertex_properties.append(
                    (line[2].decode(), ext + ply_dtypes[line[1]]))
            elif current_element == 'vertex':
                if not line.startswith('property list uchar int'):
                    raise ValueError('Unsupported faces property : ' + line)

    return num_points, num_faces, vertex_properties


def parse_header(plyfile, ext):
    """Parse header.

    Args:
        plyfile (file): the plyfile object
        ext (str): extension for building the numpy dtypes
    """
    # Variables
    line = []
    properties = []
    num_points = None

    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        if b'element' in line:
            line = line.split()
            num_points = int(line[2])

        elif b'property' in line:
            line = line.split()
            properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))

    return num_points, properties
