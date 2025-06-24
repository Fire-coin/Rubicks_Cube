from tkinter import Tk, Canvas, Event
from math import sin, cos, radians, pi
from typing import cast, Any

def sinD(angle: float):
    return sin(radians(angle))

def cosD(angle: float):
    return cos(angle * pi / 180)

# def transform(vectors: list["Vector3D"]):
#     c1 = rubicksCube[0][0][0]
#     # c2 = rubicksCube[0][0][1]

#     newX = c1.points[0] # 4, 0
#     newY = ... # 3, 1
#     newZ = ... # 1, 0



lastCubeTag: str = ""

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


def handleDrag(e: Event, w: Canvas, d: float, factor: float) -> None:
    global lastCubeTag
    overlaps: tuple[int, ...] = w.find_overlapping(e.x, e.y, e.x, e.y)
    if (len(overlaps) <= 0): return
    tags: list[str] = cast(str, w.itemcget(overlaps[0], "tags")).split(' ') # type: ignore
    # print(tags)
    if (len(tags) <= 0): return
    cubeTag: str = tags[0]

    if (lastCubeTag == cubeTag): return

    if (lastCubeTag == ""):
        lastCubeTag = cubeTag
        return
    print(lastCubeTag, cubeTag)
    try:
        color: str = tags[2]
    except IndexError:
        return

    cube1Number: int = int(lastCubeTag[4:])
    layer1: int = cube1Number // 9
    row1: int = (cube1Number % 9) // 3
    column1: int = (cube1Number % 9) % 3

    cube2Number: int = int(cubeTag[4:])
    layer2: int = cube2Number // 9
    row2: int = (cube2Number % 9) // 3
    column2: int = (cube2Number % 9) % 3

    cube1: Cube = rubicksCube[layer1][row1][column1]
    cube2: Cube = rubicksCube[layer2][row2][column2]

    directionVector: Vector3D = cube2.center - cube1.center
    
    unitVector: Vector3D = faceDirections[color]

    directionVector += unitVector

    axis: str = ''

    if (dot(directionVector, Vector3D(1, 0, 0)) == 0): # the x-axis
        axis = 'X'
    elif (dot(directionVector, Vector3D(0, 1, 0))): # the y-axis
        axis = 'Y'
    elif (dot(directionVector, Vector3D(0, 0, 1))): # the z-axis
        axis = 'Z'
    
    if (axis == ''):
        print("ERROR: no suitable axis")
        lastCubeTag = ""
        return
    
    directionVector -= unitVector

    match axis:
        case 'X':
            if (directionVector.y + directionVector.z > 0):
                side: list[list[Cube]] = []
                for i in range(3):
                    row: list[Cube] = []
                    for j in range(3):
                        row.append(rubicksCube[i][j][column1])
                    side.append(row)
                rotateMatrix(side, True)
                for i in range(3):
                    for j in range(3):
                        # newCenter = Vector3D(2*k - 2, -2*i + 2, 2*j - 2)
                        side[i][j].rotate(90, 0, 0)
                        # side[i][j].changeCenter(newCenter)
        case _:
            pass
    draw(w, d, factor)
    lastCubeTag = ""


# 1000 for loop iterations do not affect rendering time
def handleRotate(e: Event, w: Canvas, d: float, factor: float) -> None:
    # matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # rotateMatrix(matrix, True)
    # print(matrix)
    match(e.char):
        case 'w':
            rotate(10, 0, 0)
            # cube2.rotate(10, 0, 0)
            draw(w, d, factor)
            # cube2.draw(w, d, factor)
        case 's':
            rotate(-10, 0, 0)
            # cube2.rotate(-10, 0, 0)
            draw(w, d, factor)
            # cube2.draw(w, d, factor)
        case 'd':
            rotate(0, -10, 0)
            # cube2.rotate(0, -10, 0)
            draw(w, d, factor)
            # cube2.draw(w, d, factor)
        case 'a':
            rotate(0, 10, 0)
            # cube2.rotate(0, 10, 0)
            draw(w, d, factor)
            # cube2.draw(w, d, factor)
        case 'q':
            rotate(0, 0, 10)
            # cube2.rotate(0, 0, 10)
            draw(w, d, factor)
            # cube2.draw(w, d, factor)
        case 'e':
            rotate(0, 0, -10)
            # cube2.rotate(0, 0, -10)
            draw(w, d, factor)
            # cube2.draw(w, d, factor)
        case _:
            return




class Vector2D:
    def __init__(self, x: float, y: float, scaleFactor: float) -> None:
        self.x = x * scaleFactor
        self.y = y * scaleFactor



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


class Cube:

    def __init__(self, id: str, center: Vector3D) -> None:
        self.points: list[Vector3D] = [Vector3D(-1, -1, 1), Vector3D(-1, -1, -1),
             Vector3D(-1,  1, 1), Vector3D(-1,  1, -1), 
             Vector3D( 1, -1, 1), Vector3D( 1, -1, -1), 
             Vector3D( 1,  1, 1), Vector3D( 1,  1, -1)]
        self.id = id
        self.center: Vector3D = center
        for i in range(len(self.points)):
            self.points[i] += self.center

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
        for i in range(len(self.points)):
            self.points[i] -= self.center
            self.points[i] += newCenter
        self.center = newCenter
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
        return self

    #TODO possibly improve by using threads
    def draw(self, w: Canvas, distance: float, factor: float) -> None:
        w.delete(self.id)
        if (self.colors == []):
            self.colors = ["yellow", "green", "purple", "red", "orange", "white",
                  "blue", "brown", "pink", "black", "cyan", "violet"]
        # print(self.id)
        # print(self.faces)
        # print()
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

            v1: Vector3D = self.points[faces[face][0]] - self.points[faces[face][1]]
            v2: Vector3D = self.points[faces[face][2]] - self.points[faces[face][1]]
            p: Vector3D = cross(v1, v2) # This is perpendicular to triangle vector
            # print(face, p.x, p.y, p.z)
            direction = dot(directionVector, p)
            if (abs(direction) < self.allowedError):
                continue
            # print(self.id, self.faces)
            # print(direction)
            # drawLine(w, self.points[faces[face][0]].get2D(distance, factor), p.get2D(distance, factor), self.id)
            # drawLine(w, self.points[faces[face][1]].get2D(distance, factor), p.get2D(distance, factor), self.id)
            # drawLine(w, self.points[faces[face][2]].get2D(distance, factor), p.get2D(distance, factor), self.id)
            # if (self.colors[face] == "white" or self.colors[face] == "red"):
            #     print(self.id)
            #     print(self.colors[face])
            #     print(self.faces[face])
            #     print(self.points[self.faces[face][0]])
            #     print(self.points[self.faces[face][1]])
            #     print(self.points[self.faces[face][2]])
            #     # print(v1)
            #     # print(v2)
            #     # print(p)
            #     print(direction)
            #     print()
            #     drawLine(w, self.points[faces[face][1]].get2D(distance, factor), p.get2D(distance, factor), self.id)
            if (direction >= 0):
                continue
            # print(direction)
            # lightMagnitude = dot(lightDirection, p)
            # norm = abs(lightMagnitude) / abs(dot(p, p))
            # shade = int(150 * (1 + norm))
            # print(norm, shade, lightMagnitude)
            # color: str = self.colors[face]
            faceTags: list[str] = [self.id, "cube", color]

            drawTriangle(w, (self.points[self.faces[face][0]].get2D(distance, factor),
                             self.points[self.faces[face][1]].get2D(distance, factor),
                             self.points[self.faces[face][2]].get2D(distance, factor)),
                         color, faceTags)
            drawLine(w, self.points[self.faces[face][0]].get2D(distance, factor), self.points[self.faces[face][1]].get2D(distance, factor), [self.id])
            drawLine(w, self.points[self.faces[face][1]].get2D(distance, factor), self.points[self.faces[face][2]].get2D(distance, factor), [self.id])
        
        # for start, end in self.edges:
        #     drawLine(w, self.points[start].get2D(distance, factor),
        #              self.points[end].get2D(distance, factor), self.id)
        w.update()

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
                # if (1 == i == j == k):
                #     continue
                rubicksCube[i][j][k].draw(w, distance, factor)

def rotate(x: float, y: float, z: float):
    for i in range(size):
        for j in range(size):
            for k in range(size):
                # if (1 == i == j == k):
                #     continue
                rubicksCube[i][j][k].rotate(x, y, z)

root = Tk()

d = 20.0
directionVector = Vector3D(0, 0, -1)
lightDirection = Vector3D(-1, -1, 0)
factor = 50
size = 3

w = Canvas(root, width= 801, height= 801, bg= "grey")
w.pack()

# p1 = Vector3D(0, 0, 1.0)
# p2 = Vector3D(1.0, 1.0, 1.0)

# cube = [p0, p1, p2, p3, p4, p5, p6, p7]

connections = [(0, 1), (0, 2), (0, 4),
(1, 3), (1, 5),
(2, 3), (2, 6),
(3, 7),
(4, 5), (4, 6),
(5, 7),
(6, 7)]

faces = [(1, 0, 2), (2, 3, 1), # left face
         (3, 2, 6), (6, 7, 3), # top face
         (6, 2, 0), (0, 4, 6), # back face
         (7, 6, 4), (4, 5, 7), # right face
         (4, 0, 1), (1, 5, 4), # bottom face
         (1, 3, 7), (7, 5, 1)] # front face

rubicksCube: list[list[list[Cube]]] = []

counter = 0
for i in range(size):
    layer: list[list[Cube]] = []
    for j in range(size):
        row: list[Cube] = []
        for k in range(size):
            # if (1 == i == j == k):
            #     row.append(Cube(f"cube{counter}"))
            c = Cube(f"cube{counter}", Vector3D(2*k - 2, -2*i + 2, 2*j - 2))
            # c.getDefaultFaces()
            if (k == 0): # left face
                c.addFace("left", "orange")
            else:
                c.addFace("left", None)
            
            if (i == 0): # top face
                c.addFace("top", "yellow")
            else:
                c.addFace("top", None)
            
            if (j == size - 1): # back face9
                c.addFace("back", "green")
            else:
                c.addFace("back", None)
            
            if (k == size - 1): # right face
                c.addFace("right", "red")
            else:
                c.addFace("right", None)
            
            if (i == size - 1): # bottom face
                c.addFace("bottom", "white")
            else:
                c.addFace("bottom", None)
            
            if (j == 0): # front face
                c.addFace("front", "blue")
            else:
                c.addFace("front", None)
            
            row.append(c)
            c.draw(w, d, factor)
            counter += 1
        layer.append(row)
    rubicksCube.append(layer)



cube = Cube("cube1", Vector3D(0, 0, 0))
cube2 = Cube("cube2", Vector3D(2, 0, 0))

#TODO problem happens when faces are not added in the order I created them in
# cube.addFace("left", "orange")
# cube.addFace("top", "yellow")
# cube.getDefaultFaces()
cube.addFace("left", "orange")
cube.addFace("top", "yellow")
cube.addFace("back", "green")
cube.addFace("right", None)
cube.addFace("bottom", "white")
cube.addFace("front", "blue")

cube2.addFace("left", None)
cube2.addFace("top", "yellow")
cube2.addFace("back", "green")
cube2.addFace("right", "red")
cube2.addFace("bottom", "white")
cube2.addFace("front", "blue")
# cube2.getDefaultFaces()
# cube2._addFaceColor("orange")
# cube2._addFaceColor("yellow")
# cube2._addFaceColor("green")
# cube2._addFaceColor("red")
# cube2._addFaceColor("white")
# cube2._addFaceColor("blue")


# cube.addEdges(connections)
# cube.addFaces(faces)
# cube.getDefaultFaces()
# cube.draw(w, d, factor)
# cube2.draw(w, d, factor)
# root.bind("<B3-Motion>", lambda e: handleRotate(e, w, d, factor))
root.bind("<Key>", lambda e: handleRotate(e, w, d, factor))
root.bind("<B1-Motion>", lambda e: handleDrag(e, w, d, factor))

# for point in cube.points:
#     print(point.x, point.y, point.z)
# for i in range(36):
#     w.after(400)
#     cube.rotate(0, 10, 0)
#     cube.draw(w, d, factor)
#     for point in cube.points:
#         print(point.x, point.y, point.z)
#     print('\n')
# cube.rotate(20, 20, 0)
# cube.draw(w, d, factor)

# for c in connections:
#     drawLine(w, c[0].get2D(d, factor), c[1].get2D(d, factor))


# drawLine(w, p1.get2D(d, factor), p2.get2D(d, factor))



root.mainloop()