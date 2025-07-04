from tkinter import Tk, Canvas, Event
from math import sin, cos, radians, pi
from typing import cast, Any
from math import sqrt
from copy import deepcopy

lastCubeTag: str = ""
lastFaceTag: str = ""
cursorPositions: list[list[float]] = []
dirVector: "Vector3D"
rubicksCube: list[list[list["Cube"]]]
phantomCube: "Cube"
root: Tk
# w: Canvas
size: int



def sinD(angle: float):
    return sin(radians(angle))


def cosD(angle: float):
    return cos(angle * pi / 180)


def rotateMatrix(matrix: list[list[Any]], clockwise: bool) -> None:
    matrixSize = len(matrix)
    if (clockwise):
        for i in range(matrixSize):
            for j in range(matrixSize):
                if (j < i): continue
                temp: Any = matrix[i][j]
                matrix[i][j] = matrix[j][i]
                matrix[j][i] = temp
        for k in range(matrixSize):
            temp = matrix[k][0]
            matrix[k][0] = matrix[k][-1]
            matrix[k][-1] = temp
    else:
        for i in range(3):
            rotateMatrix(matrix, True)


def matrixMultiply(matrix: list[list[float]], vector: tuple[float, ...]) -> tuple[float, ...]:
    output: list[float] = [0, 0, 0]

    for i in range(3):
        result: float = 0
        for j in range(3):
            result += matrix[i][j] * vector[j]
        output[i] = result

    return tuple(output)


def transform(matrix: list[list[float]], v: "Vector3D") -> "Vector3D":
    newCoords = matrixMultiply(matrix, tuple([v.x, v.y, v.z]))

    return Vector3D(newCoords[0], newCoords[1], newCoords[2])


def resetCursorPositions() -> None:
    global cursorPositions
    cursorPositions = []


def reset(w: Canvas, d: float, factor: float) -> None:
    global lastCubeTag, lastFaceTag
    lastCubeTag = ""
    lastFaceTag = ""
    root.bind("<B1-Motion>", lambda e: handleDrag(e, w, d, factor))


def getCurrentUnits():
    global phantomCube

    newX = phantomCube.points[4] - phantomCube.points[0] # 4, 0
    newY = phantomCube.points[3] - phantomCube.points[1] # 3, 1
    newZ = phantomCube.points[0] - phantomCube.points[1] # 1, 0

    newX *= 0.5
    newY *= 0.5
    newZ *= 0.5
    return tuple([newX, newY, newZ])


def handleRotationDrag(e: Event, w: Canvas, d: float, factor: float) -> None:
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

    # Full width crossed should be 0.5 full rotations
    # Full height crossed should be 0.5 full rotations

    fractionX: float = Dx / width
    fractionY: float = Dy / height

    angleX: float = .5 * 360 * fractionX
    angleY: float = .5 * 360 * fractionY


    rotate(-angleX, -angleY, 0)
    draw(w, d, factor)
    cursorPositions = []


def handleDrag(e: Event, w: Canvas, d: float, factor: float) -> None:
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
  
    #TODO make these compatible with size
    cube1Number: int = int(lastCubeTag[4:])
    layer1: int = cube1Number // 9
    row1: int = (cube1Number % 9) // 3
    column1: int = (cube1Number % 9) % 3
    try:
        cube2Number: int = int(cubeTag[4:])
    except:
        return
    layer2: int = cube2Number // 9
    row2: int = (cube2Number % 9) // 3
    column2: int = (cube2Number % 9) % 3

    cube1: Cube = rubicksCube[layer1][row1][column1]
    cube2: Cube = rubicksCube[layer2][row2][column2]

    faceIndex: int = int(lastFaceTag[4:]) # Getting index of the face that was hit
    p1: Vector3D = cube1.points[cube1.faces[faceIndex][0]]
    p2: Vector3D = cube1.points[cube1.faces[faceIndex][2]]
    start: Vector3D = (p1 + p2) * .5 # It is in the center of the cube
    end = deepcopy(cube2.center)

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


    directionVector = transform(inverseMatrix, directionVector)

    match axis:
        case 'X':
            # Getting the direction vector to face in only 1 way in the original state
            # from there it is easy to determine where is the direction of rotation
            while (directionVector.y > 0 or abs(directionVector.y) > abs(directionVector.z)):
                directionVector.rotate(90, 0, 0)
            directionVector = transform(matrix, directionVector)
            if (dot(directionVector, unitVectors[2]) > 0):
                angle: float = 90
                clockwise: bool = True
            else:
                angle: float = -90
                clockwise: bool = False
            
            side: list[list[Cube]] = []
            for i in range(size):
                row: list[Cube] = []
                for j in range(size):
                    row.append(rubicksCube[i][j][column1])
                side.append(row)

            for i in range(size):
                for j in range(size):
                    for point in range(8):
                        side[i][j].points[point] = transform(inverseMatrix, side[i][j].points[point])
                    side[i][j].rotate(angle, 0, 0)
            
            rotateMatrix(side, clockwise)
            # Changing tags
            for i in range(size):
                for j in range(size):
                    side[i][j].id = f"cube{i * 9 + j * 3 + column1}"
                    rubicksCube[i][j][column1] = side[i][j]

        case 'Y':
            while (directionVector.z < 0 or abs(directionVector.z) > abs(directionVector.x)):
                directionVector.rotate(0, 90, 0)
            directionVector = transform(matrix, directionVector)
            # layer is the same
            if (dot(directionVector, unitVectors[0]) < 0):
                # swipe to the left
                angle: float = 90
                clockwise: bool = False
            else:
                angle: float = -90
                clockwise: bool = True
            
            side: list[list[Cube]] = []
            for j in range(size):
                row: list[Cube] = []
                for k in range(size):
                    row.append(rubicksCube[layer1][j][k])
                side.append(row)
            

            for i in range(size):
                for j in range(size):
                    for point in range(8):
                        side[i][j].points[point] = transform(inverseMatrix, side[i][j].points[point])

                    side[i][j].rotate(0, angle, 0)

            rotateMatrix(side, clockwise)
            for j in range(size):
                for k in range(size):
                    side[j][k].id = f"cube{layer1 * 9 + j * 3 + k}"
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
                angle: float = 90
                clockwise: bool = False
            else:
                angle: float = -90
                clockwise: bool = True
            
            # The row is the same
            
            side: list[list[Cube]] = []
            for i in range(size):
                row: list[Cube] = []
                for k in range(size):
                    row.append(rubicksCube[i][row1][k])
                side.append(row)


            for i in range(size):
                for j in range(size):
                    for point in range(8):
                        side[i][j].points[point] = transform(inverseMatrix, side[i][j].points[point])

                    side[i][j].rotate(0, 0, angle)
            
            rotateMatrix(side, clockwise)
            for i in range(size):
                for k in range(size):
                    side[i][k].id = f"cube{i * 9 + row1 * 3 + k}"
                    rubicksCube[i][row1][k] = side[i][k]

        case _:
            return
    
    # Changing centers
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
    match(e.char):
        case 'w':
            rotate(10, 0, 0)
            draw(w, d, factor)
        case 's':
            rotate(-10, 0, 0)
            draw(w, d, factor)
        case 'd':
            rotate(0, -10, 0)
            draw(w, d, factor)
        case 'a':
            rotate(0, 10, 0)
            draw(w, d, factor)
        case 'q':
            rotate(0, 0, 10)
            draw(w, d, factor)
        case 'e':
            rotate(0, 0, -10)
            draw(w, d, factor)
        case _:
            return
    getCurrentUnits()



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
    return v.x * w.x + v.y * w.y + v.z * w.z


def cross(v: Vector3D, w: Vector3D) -> Vector3D:
    x = v.y * w.z - w.y * v.z
    y = v.z * w.x - w.z * v.x
    z = v.x * w.y - w.x * v.y
    return Vector3D(x, y, z)


faceDirections: dict[str, Vector3D] = {
    "orange": Vector3D( 1,  0,  0),
    "yellow": Vector3D( 0, -1,  0),
    "blue":   Vector3D( 0,  0,  1),
    "red":    Vector3D(-1,  0,  0),
    "green":  Vector3D( 0,  0, -1),
    "white":  Vector3D( 0,  1,  0)
}


def drawLine(w: Canvas, point1: Vector2D, point2: Vector2D, tags: list[str]) -> None:
    w.create_line(point1.x + 400, 400 - point1.y, point2.x + 400, 400 - point2.y, fill= "black", tags= tags) # type: ignore


def drawTriangle(w: Canvas, points: tuple[Vector2D, Vector2D, Vector2D], color: str, tags: list[str]) -> None:
    w.create_polygon(points[0].x + 400, 400 - points[0].y, # Adjusted coordinates for first point
                     points[1].x + 400, 400 - points[1].y, # Adjusting coordinates for second point
                     points[2].x + 400, 400 - points[2].y, # Adjusting coordinates for thrird point
                     fill= color,
                     tag= tags) # type: ignore


def draw(w: Canvas, distance: float, factor: float):
    for i in range(size):
        for j in range(size):
            for k in range(size):
                rubicksCube[i][j][k].draw(w, distance, factor)


def rotate(x: float, y: float, z: float):
    for i in range(size):
        for j in range(size):
            for k in range(size):
                # if (1 == i == j == k):
                #     continue
                rubicksCube[i][j][k].rotate(x, y, z)
    phantomCube.rotate(x, y, z)


def getLength(v: Vector3D) -> float:
    return sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)


def rotateAroundAxis(v: Vector3D, k: Vector3D, theta: float) -> Vector3D:
    return (v * cosD(theta)) + (cross(k, v) * sinD(theta)) + (k * dot(k, v) * (1 - cosD(theta)))

def main() -> None:
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
                c = Cube(f"cube{counter}", Vector3D(2*k - 2, -2*i + 2, 2*j - 2))
                c.addFace("left", "orange") if (k == 0) else c.addFace("left", None) # left face
                
                c.addFace("top", "yellow") if (i == 0) else c.addFace("top", None) # top face
                
                # if (j == size - 1): # back face
                c.addFace("back", "green") if (j == size - 1) else c.addFace("back", None)
                # else:
                #     c.addFace("back", None)
                
                # if (k == size - 1): # right face
                c.addFace("right", "red") if (k == size - 1) else c.addFace("right", None)
                # else:
                #     c.addFace("right", None)
                
                # if (i == size - 1): # bottom face
                c.addFace("bottom", "white") if (i == size - 1) else c.addFace("bottom", None)
                # else:
                #     c.addFace("bottom", None)
                
                # if (j == 0): # front face
                c.addFace("front", "blue") if (j == 0) else c.addFace("front", None)
                # else:
                #     c.addFace("front", None)
                
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