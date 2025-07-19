from tkinter import Tk, Canvas, Event
from math import sin, cos, radians, pi
from typing import cast, Any

lastCubeTag: str = ""
lastFaceTag: str = ""
cursorPositions: list[list[float]] = []
dirVector: "Vector3D"
rubicksCube: list[list[list["Cube"]]]
phantomCube: "Cube"
root: Tk
size: int



def sinD(angle: float) -> float:
    """Sine funciton in degrees

    Args:
        angle (float): Angle in degrees

    Returns:
        float: Value if sine of angle
    """
    return sin(radians(angle))


def cosD(angle: float) -> float:
    """Cosine function in degrees

    Args:
        angle (float): Angle in degrees

    Returns:
        float: Value of cosine of angle
    """
    return cos(angle * pi / 180)


def rotateMatrix(matrix: list[list[Any]], clockwise: bool) -> None:
    """Rotates square matrix by reference.

    Args:
        matrix (list[list[Any]]): The reference to the matrix that will be rotated
        clockwise (bool): Direction the matrix will be rotated
    """
    matrixSize = len(matrix)
    if (clockwise):
        # Flipping matrix along diagonal
        for i in range(matrixSize):
            for j in range(matrixSize):
                if (j < i): continue
                temp: Any = matrix[i][j]
                matrix[i][j] = matrix[j][i]
                matrix[j][i] = temp
        # Swapping left and right columns
        for k in range(matrixSize):
            temp = matrix[k][0]
            matrix[k][0] = matrix[k][-1]
            matrix[k][-1] = temp
    else:
        # Rotating matrix 3 times to rotate counter clockwise
        for i in range(3):
            rotateMatrix(matrix, True)


def matrixMultiply(matrix: list[list[float]], vector: tuple[float, ...]) -> tuple[float, ...]:
    """Multiplies matrix by vector

    Args:
        matrix (list[list[float]]): input 3x3 matrix
        vector (tuple[float, ...]): input 3x1 vector

    Returns:
        tuple[float, ...]: resultant vector
    """
    output: list[float] = [0, 0, 0]

    for i in range(3):
        result: float = 0
        for j in range(3):
            result += matrix[i][j] * vector[j]
        output[i] = result

    return tuple(output)


def transform(matrix: list[list[float]], v: "Vector3D") -> "Vector3D":
    """Transforms vector by given matrix

    Args:
        matrix (list[list[float]]): Transformation matrix
        v (Vector3D): Input vector

    Returns:
        Vector3D: Transformed vector
    """
    newCoords = matrixMultiply(matrix, tuple([v.x, v.y, v.z]))

    return Vector3D(newCoords[0], newCoords[1], newCoords[2])


def resetCursorPositions() -> None:
    """Resets cursorPositions variable
    """
    global cursorPositions
    cursorPositions = []


def reset(w: Canvas, d: float, factor: float) -> None:
    """Resets variables and bind canvas again to enable dragging a side.

    Args:
        w (Canvas): Canvas
        d (float): Distance to the camera
        factor (float): Scale factor for image
    """
    global lastCubeTag, lastFaceTag
    lastCubeTag = ""
    lastFaceTag = ""
    root.bind("<B1-Motion>", lambda e: handleDrag(e, w, d, factor))


def getCurrentUnits() -> tuple["Vector3D", ...]:
    """Gives unit vectors in current rotation of the cube.

    Returns:
        tuple["Vector3D"]: Unit vectors in current rotation
    """
    global phantomCube

    newX = phantomCube.points[4] - phantomCube.points[0] # 4, 0
    newY = phantomCube.points[3] - phantomCube.points[1] # 3, 1
    newZ = phantomCube.points[0] - phantomCube.points[1] # 1, 0

    newX *= 0.5
    newY *= 0.5
    newZ *= 0.5
    return tuple([newX, newY, newZ])


def handleRotationDrag(e: Event, w: Canvas, d: float, factor: float) -> None:
    """Rotates the cube when user drags with right click.

    Args:
        e (Event): Event returned by bind method of canvas
        w (Canvas): Canvas
        d (float): Distance to the camera
        factor (float): Scale factor for the image
    """
    global cursorPositions
    if (len(cursorPositions) < 10):
        cursorPositions.append([e.x, e.y])
        return

    start: list[float] = cursorPositions[0]
    end: list[float] = cursorPositions[-1]

    Dx: float = end[1] - start[1]
    Dy: float = end[0] - start[0]

    width: int = w.winfo_width()
    height: int = w.winfo_height()

    fractionX: float = Dx / width
    fractionY: float = Dy / height

    angleX: float = .5 * 360 * fractionX
    angleY: float = .5 * 360 * fractionY


    rotate(-angleX, -angleY, 0)
    draw(w, d, factor)
    cursorPositions = []


def getCubesIndexes(cubeTag1: str, cubeTag2: str) -> tuple[int, ...]:
    """Calculates the layer, row and column of cubes in Rubik's cube from given tags.

    Args:
        cubeTag1 (str): Canvas tag of first cube
        cubeTag2 (str): Canvas tag of second cube

    Returns:
        tuple[int]: Returns the tuple of layer, row and column for each cube.
        Or (-1,) if cube tags are invalid.
    """
    try:
        cube1Number: int = int(cubeTag1[4:])    
        cube2Number: int = int(cubeTag2[4:])
    except:
        return (-1,)
    square = size * size
    layer1: int = cube1Number // square
    row1: int = (cube1Number % square) // size
    column1: int = (cube1Number % square) % size
    layer2: int = cube2Number // square
    row2: int = (cube2Number % square) // size
    column2: int = (cube2Number % square) % size

    return tuple(list([layer1, row1, column1, layer2, row2, column2]))


def handleDrag(e: Event, w: Canvas, d: float, factor: float) -> None:
    """This function determines what side of cube should be rotated,
    in which direction and around which axis,
    and rotates the side, when user drags with left mouse button.

    Args:
        e (Event): Event returned by canvas bind method
        w (Canvas): Canvas
        d (float): Distance to the camera
        factor (float): Scale factor for the image
    """
    global lastCubeTag, lastFaceTag
    overlaps: tuple[int, ...] = w.find_overlapping(e.x, e.y, e.x, e.y)
    if (len(overlaps) <= 0): return
    tags: list[str] = cast(str, w.itemcget(overlaps[0], "tags")).split(' ') # type: ignore
    if (len(tags) <= 0): return
    cubeTag: str = tags[0]
    if (lastCubeTag == cubeTag): return

    if (lastCubeTag == ""):
        lastCubeTag = cubeTag
        lastFaceTag = tags[3]
        return
    root.unbind("<B1-Motion>")

    try:
        layer1, row1, column1, layer2, row2, column2 = getCubesIndexes(lastCubeTag, cubeTag)
    except:
        return
    cube1: Cube = rubicksCube[layer1][row1][column1]
    cube2: Cube = rubicksCube[layer2][row2][column2]

    faceIndex: int = int(lastFaceTag[4:]) # Getting index of the face that was hit
    p1: Vector3D = cube1.points[cube1.faces[faceIndex][0]]
    p2: Vector3D = cube1.points[cube1.faces[faceIndex][2]]
    start: Vector3D = (p1 + p2) * .5 # It is in the center of face of the cube
    end = cube2.center

    directionVector: Vector3D = end - start # The direction in which cube was dragged
    unitVectors = getCurrentUnits()   

    axis: str = ''

        
    smallest = float("inf")
    axis = ''
    if (abs(dot(directionVector, unitVectors[0])) < smallest):
        smallest = abs(dot(directionVector, unitVectors[0]))
        axis = 'X'
    if (abs(dot(directionVector, unitVectors[1])) < smallest):
        smallest = abs(dot(directionVector, unitVectors[1]))
        axis = 'Y'
    if (abs(dot(directionVector, unitVectors[2])) < smallest):
        smallest = abs(dot(directionVector, unitVectors[2]))
        axis = 'Z'
    
    if (axis == ''):
        print("ERROR: no suitable axis")
        lastCubeTag = ""
        return
    

    # Roation around a point:
    # Substract point of rotation from the point we want to rotate
    # Rotate the point
    # Add back the point of rotation to the rotated point

    i = unitVectors[0]
    j = unitVectors[1]
    k = unitVectors[2]

    matrix = [[i.x, j.x, k.x],
              [i.y, j.y, k.y],
              [i.z, j.z, k.z]]
    inverseMatrix = [
        [matrix[0][0], matrix[1][0], matrix[2][0]],
        [matrix[0][1], matrix[1][1], matrix[2][1]],
        [matrix[0][2], matrix[1][2], matrix[2][2]],
    ]

    side: list[list[Cube]] # The side that will be rotated
    angle: float # The angle by which the side will be rotated
    clockwise: bool # The direction by which the matrix of side will be rotated

    directionVector = transform(inverseMatrix, directionVector)

    match axis:
        case 'X':
            # Getting the direction vector to face in only 1 way in the original state
            # from there it is easy to determine where is the direction of rotation
            while (directionVector.y > 0 or abs(directionVector.y) > abs(directionVector.z)):
                directionVector.rotate(90, 0, 0)
            directionVector = transform(matrix, directionVector)
            if (dot(directionVector, unitVectors[2]) > 0):
                angle = 90
                clockwise = True
            else:
                angle = -90
                clockwise = False
            
            # column is the same
            side = []
            for i in range(size):
                row: list[Cube] = []
                for j in range(size):
                    row.append(rubicksCube[i][j][column1])
                side.append(row)

            # Translating every point into it's original state
            # and then rotating it there
            for i in range(size):
                for j in range(size):
                    for point in range(8):
                        side[i][j].points[point] = transform(inverseMatrix, side[i][j].points[point])
                    side[i][j].rotate(angle, 0, 0)
            # Rotating the side in direction of rotation on screen
            rotateMatrix(side, clockwise)
            # Changing tags
            for i in range(size):
                for j in range(size):
                    side[i][j].id = f"cube{i * size * size + j * size + column1}"
                    rubicksCube[i][j][column1] = side[i][j]

        case 'Y':
            while (directionVector.z < 0 or abs(directionVector.z) > abs(directionVector.x)):
                directionVector.rotate(0, 90, 0)
            directionVector = transform(matrix, directionVector)
            if (dot(directionVector, unitVectors[0]) < 0):
                # swipe to the left
                angle = 90
                clockwise = False
            else:
                angle = -90
                clockwise = True
            
            # layer is the same
            side = []
            for j in range(size):
                row: list[Cube] = []
                for k in range(size):
                    row.append(rubicksCube[layer1][j][k])
                side.append(row)
            
            # Translating every point into it's original state
            # and then rotating it there
            for i in range(size):
                for j in range(size):
                    for point in range(8):
                        side[i][j].points[point] = transform(inverseMatrix, side[i][j].points[point])

                    side[i][j].rotate(0, angle, 0)

            rotateMatrix(side, clockwise)
            # Changing cube tags
            for j in range(size):
                for k in range(size):
                    side[j][k].id = f"cube{layer1 * size * size + j * size + k}"
                    rubicksCube[layer1][j][k] = side[j][k]

        case 'Z':
            # For sampled direction vector:
            # axes are x & y
            # y has to be negative
            # y has to be smaller than x
            # determine the side of rotation based on x axis
            # if x is positive, then 90, -90 otherwise
            while (directionVector.y > 0 or abs(directionVector.y) > abs(directionVector.x)):
                directionVector.rotate(0, 0, 90)
            directionVector = transform(matrix, directionVector)

            if (dot(directionVector, unitVectors[0]) < 0):
                angle = 90
                clockwise = False
            else:
                angle = -90
                clockwise = True
            
            
            # The row is the same
            side = []
            for i in range(size):
                row: list[Cube] = []
                for k in range(size):
                    row.append(rubicksCube[i][row1][k])
                side.append(row)

            # Translating every point into it's original state
            # and then rotating it there
            for i in range(size):
                for j in range(size):
                    for point in range(8):
                        side[i][j].points[point] = transform(inverseMatrix, side[i][j].points[point])

                    side[i][j].rotate(0, 0, angle)
            
            rotateMatrix(side, clockwise)
            # Changing cube tags
            for i in range(size):
                for k in range(size):
                    side[i][k].id = f"cube{i * size * size + row1 * size + k}"
                    rubicksCube[i][row1][k] = side[i][k]

        case _:
            return
    
    # Translating all the points into th rotated state
    # and changing centers
    for i in range(size):
        for j in range(size):
            for point in range(8):
                side[i][j].points[point] = transform(matrix, side[i][j].points[point])
            newCenter = (side[i][j].points[1] + side[i][j].points[6]) * 0.5
            side[i][j].center = newCenter

    draw(w, d, factor)

    lastCubeTag = ""
    root.unbind("<B1-Motion>")


def handleRotate(e: Event, w: Canvas, d: float, factor: float) -> None:
    """Rotates the cube around an axis depending onn key press.

    Args:
        e (Event): Event returned by canvas bind method
        w (Canvas): Canvas
        d (float): Distance to the camera
        factor (float): Scale factor for the image
    """
    match(e.char):
        case 'w':
            rotate(10, 0, 0)
        case 's':
            rotate(-10, 0, 0)
        case 'd':
            rotate(0, -10, 0)
        case 'a':
            rotate(0, 10, 0)
        case 'q':
            rotate(0, 0, 10)
        case 'e':
            rotate(0, 0, -10)
        case _:
            return
    draw(w, d, factor)


class Vector2D:
    def __init__(self, x: float, y: float, scaleFactor: float) -> None:
        self.x = x * scaleFactor
        self.y = y * scaleFactor
    
    def __str__(self) -> str:
        return f"X: {self.x}, Y: {self.y}"


class Vector3D:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x: float = x
        self.y: float = y
        self.z: float = z
    
    def get2D(self, d: float, scaleFactor: float) -> Vector2D:
        coof = (d / (d + self.z))
        return Vector2D(self.x * coof, self.y * coof, scaleFactor)
    
    def rotate(self, x: float, y: float, z: float) -> "Vector3D":
        xCord = self.x
        yCord = self.y
        zCord = self.z
        self.y = yCord * cosD(x) - zCord * sinD(x)
        self.z = yCord * sinD(x) + zCord * cosD(x)

        xCord = self.x
        yCord = self.y
        zCord = self.z

        self.x = xCord * cosD(y) + zCord * sinD(y)
        self.z = -xCord * sinD(y) + zCord * cosD(y)

        xCord = self.x
        yCord = self.y
        zCord = self.z

        self.x = xCord * cosD(z) - yCord * sinD(z)
        self.y = xCord * sinD(z) + yCord * cosD(z)
        
        return self

    def __sub__(self, other: "Vector3D") -> "Vector3D":
        x: float = self.x - other.x
        y: float = self.y - other.y
        z: float = self.z - other.z
        return Vector3D(x, y, z)
    
    def __add__(self, other: "Vector3D") -> "Vector3D":
        x: float = self.x + other.x
        y: float = self.y + other.y
        z: float = self.z + other.z
        return Vector3D(x, y, z)

    def __str__(self) -> str:
        return f"X: {self.x}, Y: {self.y}, Z: {self.z}"

    def __setitem__(self, other: "Vector3D") -> "Vector3D":
        return self.__add__(other)

    def __mul__(self, num: float) -> "Vector3D":
        x = self.x * num
        y = self.y * num
        z = self.z * num
        return Vector3D(x, y, z)


class Cube:
    def __init__(self, id: str, center: Vector3D) -> None:
        self.points: list[Vector3D] = [Vector3D(-1, -1, 1), Vector3D(-1, -1, -1),
             Vector3D(-1,  1, 1), Vector3D(-1,  1, -1), 
             Vector3D( 1, -1, 1), Vector3D( 1, -1, -1), 
             Vector3D( 1,  1, 1), Vector3D( 1,  1, -1)]
        self.center: Vector3D = center
        for i in range(len(self.points)):
            self.points[i] += self.center
        self.id = id

        self.edges: list[tuple[int, int]] = []
        self.faces: list[tuple[int, int, int]] = []
        
        self.colors: list[str | None] = []
        self.allowedError = .5
    
    def addEdge(self, p1: int, p2: int) -> "Cube":   
        self.edges.append((p1, p2))
        return self

    def addEdges(self, edges: list[tuple[int, int]]) -> "Cube":
        self.edges = edges
        return self
    
    def addFace(self, face: str, color: str | None) -> "Cube":
        match face:
            case "left":
                self._addFace((1, 0, 2))
                self._addFace((2, 3, 1))
            case "top":
                self._addFace((3, 2, 6))
                self._addFace((6, 7, 3))
            case "back":
                self._addFace((6, 2, 0))
                self._addFace((0, 4, 6))
            case "right":
                self._addFace((7, 6, 4))
                self._addFace((4, 5, 7))
            case "bottom":
                self._addFace((4, 0, 1))
                self._addFace((1, 5, 4))
            case "front":
                self._addFace((1, 3, 7))
                self._addFace((7, 5, 1))
            case _:
                return self
        self._addFaceColor(color)

        return self 

    def _addFace(self, facePoints: tuple[int, int, int]) -> "Cube":
        self.faces.append(facePoints)
        return self

    def addFaces(self, faces: list[tuple[int, int, int]]) -> "Cube":
        self.faces = faces
        return self
    
    def _addFaceColor(self, color: str | None) -> "Cube":
        self.colors.extend([color, color])
        return self
    
    def addColor(self, color: str) -> "Cube":
        self.colors.append(color)
        return self
    
    def changeCenter(self, newCenter: Vector3D) -> "Cube":
        self.points: list[Vector3D] = [Vector3D(-1, -1, 1), Vector3D(-1, -1, -1),
             Vector3D(-1,  1, 1), Vector3D(-1,  1, -1), 
             Vector3D( 1, -1, 1), Vector3D( 1, -1, -1), 
             Vector3D( 1,  1, 1), Vector3D( 1,  1, -1)]
        self.center: Vector3D = newCenter
        for i in range(len(self.points)):
            self.points[i] += self.center
        return self

    def getDefaultFaces(self) -> "Cube":
        self.faces = [(1, 0, 2), (2, 3, 1), # left face
            (3, 2, 6), (6, 7, 3), # top face
            (6, 2, 0), (0, 4, 6), # back face
            (7, 6, 4), (4, 5, 7), # right face
            (4, 0, 1), (1, 5, 4), # bottom face
            (1, 3, 7), (7, 5, 1)  # front face
        ]
        return self

    def rotate(self, x: float, y: float, z: float) -> "Cube":
        for i in range(len(self.points)):
            self.points[i].rotate(x, y, z)
        self.center.rotate(x, y, z)
        return self

    #TODO possibly improve by using threads
    def draw(self, w: Canvas, distance: float, factor: float) -> None:
        w.delete(self.id)
        if (self.colors == []):
            self.colors = ["yellow", "green", "purple", "red", "orange", "white",
                  "blue", "brown", "pink", "black", "cyan", "violet"]
        for face in range(len(self.faces)):
            # This is the only fix that works, if I leave out
            # one side of the cube, it will not render properly.
            # E.g if I render in order all sides except bottom
            # then the one after it (front) will behave as bottom
            # one but it will still be drawn on the front side.
            if (self.colors[face] is None):
                continue
            
            color: str | None = self.colors[face]
            assert color is not None # Making sure color is not None

            v1: Vector3D = self.points[self.faces[face][0]] - self.points[self.faces[face][1]]
            v2: Vector3D = self.points[self.faces[face][2]] - self.points[self.faces[face][1]]
            p: Vector3D = cross(v1, v2) # This is perpendicular to triangle vector
            direction = dot(dirVector, p)
            if (abs(direction) < self.allowedError):
                continue
            if (direction < 0):
                continue
            faceTags: list[str] = [self.id, "cube", color, f"face{face}"]

            drawTriangle(w, (self.points[self.faces[face][0]].get2D(distance, factor),
                             self.points[self.faces[face][1]].get2D(distance, factor),
                             self.points[self.faces[face][2]].get2D(distance, factor)),
                         color, faceTags)
            drawLine(w, self.points[self.faces[face][0]].get2D(distance, factor), self.points[self.faces[face][1]].get2D(distance, factor), [self.id])
            drawLine(w, self.points[self.faces[face][1]].get2D(distance, factor), self.points[self.faces[face][2]].get2D(distance, factor), [self.id])
        
        for start, end in self.edges:
            drawLine(w, self.points[start].get2D(distance, factor),
                     self.points[end].get2D(distance, factor), [self.id])
        w.update()
    
    def __str__(self) -> str:
        return f"id: {self.id}, center: {self.center}\n points: {self.points}"


def dot(v: Vector3D, w: Vector3D) -> float:
    """Dot product of 2 vectors

    Args:
        v (Vector3D): First vector
        w (Vector3D): Second vector

    Returns:
        float: Result of dot product
    """    
    return v.x * w.x + v.y * w.y + v.z * w.z


def cross(v: Vector3D, w: Vector3D) -> Vector3D:
    """Cross product of 2 vectors.

    Args:
        v (Vector3D): First vector
        w (Vector3D): Second vector

    Returns:
        Vector3D: Returns new vector
    """    
    x = v.y * w.z - w.y * v.z
    y = v.z * w.x - w.z * v.x
    z = v.x * w.y - w.x * v.y
    return Vector3D(x, y, z)


def drawLine(w: Canvas, point1: Vector2D, point2: Vector2D, tags: list[str]) -> None:
    """Draws line in cartesian coordiantes centered in center of canvas.

    Args:
        w (Canvas): Canvas
        point1 (Vector2D): First point of line
        point2 (Vector2D): Second point of line
        tags (list[str]): Tags that will be given to the line
    """
    w.create_line(point1.x + 400, 400 - point1.y, point2.x + 400, 400 - point2.y, fill= "black", tags= tags) # type: ignore


def drawTriangle(w: Canvas, points: tuple[Vector2D, Vector2D, Vector2D], color: str, tags: list[str]) -> None:
    """Draws triangle in cartesian coordinates centered on center of canvas.

    Args:
        w (Canvas): Canvas
        points (tuple[Vector2D, Vector2D, Vector2D]): Points of triangle
        color (str): Triangles color
        tags (list[str]): Tags that will be given to the triangle
    """
    w.create_polygon(points[0].x + 400, 400 - points[0].y, # Adjusted coordinates for first point
                     points[1].x + 400, 400 - points[1].y, # Adjusting coordinates for second point
                     points[2].x + 400, 400 - points[2].y, # Adjusting coordinates for thrird point
                     fill= color,
                     tag= tags) # type: ignore


def draw(w: Canvas, distance: float, factor: float):
    """Draws the Rubik's cube.

    Args:
        w (Canvas): Canvas
        distance (float): Distance to the camera
        factor (float): Scale factor for the image
    """
    for i in range(size):
        for j in range(size):
            for k in range(size):
                rubicksCube[i][j][k].draw(w, distance, factor)


def rotate(x: float, y: float, z: float):
    """Rotates the Rubik's cube.

    Args:
        x (float): Angle of rotation around x-axis
        y (float): Angle of rotation around y-axis
        z (float): Angle of rotation around z-axis
    """
    for i in range(size):
        for j in range(size):
            for k in range(size):
                rubicksCube[i][j][k].rotate(x, y, z)
    phantomCube.rotate(x, y, z)


def main() -> None:
    """Main function, here all global varianbles are asigned.
    """
    global root, dirVector, rubicksCube, phantomCube, size
    root = Tk()

    d = 20.0
    dirVector = Vector3D(0, 0, 1)
    factor = 50
    size = 3

    w = Canvas(root, width= 801, height= 801, bg= "grey")
    w.pack()

    rubicksCube = []

    phantomCube = Cube("phantom", Vector3D(0, 0, 0))
    phantomCube.addFace("left", None)
    phantomCube.addFace("top", None)
    phantomCube.addFace("back", None)
    phantomCube.addFace("right", None)
    phantomCube.addFace("bottom", None)
    phantomCube.addFace("front", None)

    counter = 0
    for i in range(size):
        layer: list[list[Cube]] = []
        for j in range(size):
            row: list[Cube] = []
            for k in range(size):
                c = Cube(f"cube{counter}", Vector3D(2*k - (size - 1), -2*i + (size - 1), 2*j - (size - 1)))
                c.addFace("left", "orange") if (k == 0) else c.addFace("left", None) # left face
                
                c.addFace("top", "yellow") if (i == 0) else c.addFace("top", None) # top face
                
                c.addFace("back", "green") if (j == size - 1) else c.addFace("back", None)

                c.addFace("right", "red") if (k == size - 1) else c.addFace("right", None)
                
                c.addFace("bottom", "white") if (i == size - 1) else c.addFace("bottom", None)
                
                c.addFace("front", "blue") if (j == 0) else c.addFace("front", None)
                
                row.append(c)
                c.draw(w, d, factor)
                counter += 1
            layer.append(row)
        rubicksCube.append(layer)


    root.bind("<Key>", lambda e: handleRotate(e, w, d, factor))
    root.bind("<Button>", lambda e: reset(w, d, factor))
    root.bind("<B1-Motion>", lambda e: handleDrag(e, w, d, factor))
    root.bind("<B3-Button>", lambda e: resetCursorPositions())
    root.bind("<B3-Motion>", lambda e: handleRotationDrag(e, w, d, factor))


    root.mainloop()


if (__name__ == "__main__"):
    main()