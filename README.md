# Rubik's_Cube

3x3 Rubik's cube made in python with tkinter.

All the math skills required I got from this 3b1b series about linear algebra: https://www.youtube.com/watch?v=k7RM-ot2NWY&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&pp=0gcJCRIBOCosWNin

The key points in making this cube were:

1. Projection of 3D vectors on 2D space
2. Rotating cube around x, y and z axis using rotation metrices
3. Adding faces using triangles
4. Rendering only the triangles that can be seen using cross and dot products of vectors
5. Stacking cubes to make 1 big cube
6. Rotating the side:
   i) Keeping track of transformed unit vectors i-hat, j-hat and k-hat
   ii) Translating whole side into original position (centered at origin) using inverse matrix made of transformed unit vectors, then rotating it
   iii) Tranlating whole side back into rotated state using matrix of transformed unit vector
   iv) rotating the matrix individual cubes are stored in
