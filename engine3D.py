import pygame
import math


class Obj:
    def __init__(self, mesh):
        self.mesh = mesh

    @staticmethod
    def loadObj(fp: str):
        with open(fp) as file:
            points = []
            mesh = Mesh()
            for ln in file.readlines():
                if ln[0] == "v":
                    points.append(Vector(float(ln.split()[1]) + 1, float(ln.split()[2]) + 1, float(ln.split()[3]) + 1))

                if ln[0] == "f":
                    mesh.tris.append(Triangle([points[int(ln.split()[1]) - 1], points[int(ln.split()[2]) - 1], points[int(ln.split()[3]) - 1]]))
        o = Obj(mesh)
        return o


class Vector:
    def __init__(self, x=0., y=0., z=0., w=1.):
        self.x, self.y, self.z, self.w = x, y, z, w

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other: float):
        return Vector(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other: float):
        return Vector(self.x / other, self.y / other, self.z / other)

    def dotProduct(self, other) -> int:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def __len__(self) -> float:
        return math.sqrt(self.dotProduct(self))

    def normalize(self):
        length = self.__len__()
        if length:
            return Vector(self.x / length, self.y / length, self.z / length)
        return self

    def crossProduct(self, other):
        v = Vector()
        v.x = self.y * other.z - self.z * other.y
        v.y = self.z * other.x - self.x * other.z
        v.z = self.x * other.y - self.y * other.x
        return v

    @staticmethod
    def intersectPlane(plane_p, plane_n, lineStart, lineEnd):
        plane_n = plane_n.normalize()
        plane_d = -plane_n.dotProduct(plane_p)
        ad = lineStart.dotProduct(plane_n)
        bd = lineEnd.dotProduct(plane_n)
        t = (-plane_d - ad) / (bd - ad)
        lineStartToEnd = lineEnd - lineStart
        lineToIntersect = lineStartToEnd * t
        return lineStart + lineToIntersect


class Triangle:
    def __init__(self, p: [Vector, Vector, Vector], color=(0, 0, 0), luminance=0.1):
        self.p = p
        self.c = color
        self.l = luminance

    @staticmethod
    def clipAgainstPlane(in_tri, plane_p: Vector, plane_n: Vector, clippingColors=False):
        plane_n = plane_n.normalize()

        def dist(p: Vector):
            return plane_n.x * p.x + plane_n.y * p.y + plane_n.z * p.z - plane_n.dotProduct(plane_p)

        inside_points = []

        outside_points = []

        d0 = dist(in_tri.p[0])
        d1 = dist(in_tri.p[1])
        d2 = dist(in_tri.p[2])

        if d0 >= 0:
            inside_points.append(in_tri.p[0])
        else:
            outside_points.append(in_tri.p[0])
        if d1 >= 0:
            inside_points.append(in_tri.p[1])
        else:
            outside_points.append(in_tri.p[1])
        if d2 >= 0:
            inside_points.append(in_tri.p[2])
        else:
            outside_points.append(in_tri.p[2])

        if len(inside_points) == 0:
            #  no valid triangles
            return 0, None, None

        elif len(inside_points) == 3:
            #  all of triangle is inside plane
            return 1, in_tri, None

        elif len(inside_points) == 1 and len(outside_points) == 2:
            #  triangle gets clipped into smaller triangle
            out_tri = Triangle([
                inside_points[0],
                Vector.intersectPlane(plane_p, plane_n, inside_points[0], outside_points[0]),
                Vector.intersectPlane(plane_p, plane_n, inside_points[0], outside_points[1])
            ], in_tri.c, in_tri.l)

            if clippingColors:
                out_tri.c = (255 * in_tri.l, 0, 0)

            return 1, out_tri, None

        elif len(inside_points) == 2 and len(outside_points) == 1:
            out_tri_1 = Triangle([
                inside_points[0],
                inside_points[1],
                Vector.intersectPlane(plane_p, plane_n, inside_points[0], outside_points[0])
            ], in_tri.c, in_tri.l)
            out_tri_2 = Triangle([
                inside_points[1],
                out_tri_1.p[2],
                Vector.intersectPlane(plane_p, plane_n, inside_points[1], outside_points[0])
            ], in_tri.c, in_tri.l)

            if clippingColors:
                out_tri_1.c = (0, 255 * in_tri.l, 0)
                out_tri_2.c = (0, 0, 255 * in_tri.l)

            return 2, out_tri_1, out_tri_2


class Mesh:
    def __init__(self):
        self.tris = []


class Matrix4x4:
    def __init__(self):
        self.m = [[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]]

    class static:
        @staticmethod
        def RotationZ(Theta):
            matrix = Matrix4x4()
            matrix.m[0][0] = math.cos(Theta)
            matrix.m[0][1] = math.sin(Theta)
            matrix.m[1][0] = -math.sin(Theta)
            matrix.m[1][1] = math.cos(Theta)
            matrix.m[2][2] = 1.
            matrix.m[3][3] = 1.
            return matrix

        @staticmethod
        def RotationY(Theta):
            matrix = Matrix4x4()
            matrix.m[0][0] = math.cos(Theta)
            matrix.m[0][2] = math.sin(Theta)
            matrix.m[2][0] = -math.sin(Theta)
            matrix.m[1][1] = 1.
            matrix.m[2][2] = math.cos(Theta)
            matrix.m[3][3] = 1.
            return matrix

        @staticmethod
        def RotationX(Theta):
            matrix = Matrix4x4()
            matrix.m[0][0] = 1.
            matrix.m[1][1] = math.cos(Theta)
            matrix.m[1][2] = math.sin(Theta)
            matrix.m[2][1] = -math.sin(Theta)
            matrix.m[2][2] = math.cos(Theta)
            matrix.m[3][3] = 1.
            return matrix

        @staticmethod
        def Translation(x: float, y: float, z: float):
            matrix = Matrix4x4()
            matrix.m[0][0] = 1.
            matrix.m[1][1] = 1.
            matrix.m[2][2] = 1.
            matrix.m[3][3] = 1.
            matrix.m[3][0] = x
            matrix.m[3][1] = y
            matrix.m[3][2] = z
            return matrix

        @staticmethod
        def Projection(AspectRatio, FovRad, Far, Near):
            matrix = Matrix4x4()
            matrix.m[0][0] = AspectRatio * FovRad
            matrix.m[1][1] = FovRad
            matrix.m[2][2] = Far / (Far - Near)
            matrix.m[3][2] = (-Far * Near) / (Far - Near)
            matrix.m[2][3] = 1.
            matrix.m[3][3] = 0.
            return matrix

        @staticmethod
        def Identity():
            matrix = Matrix4x4()
            matrix.m[0][0] = 1.
            matrix.m[1][1] = 1.
            matrix.m[2][2] = 1.
            matrix.m[3][3] = 1.
            return matrix

        @staticmethod
        def PointAt(pos: Vector, target: Vector, up: Vector):
            newForward = (target - pos).normalize()

            a = newForward * up.dotProduct(newForward)
            newUp = (up - a).normalize()

            newRight = newUp.crossProduct(newForward)

            matrix = Matrix4x4()
            matrix.m[0][0] = newRight.x
            matrix.m[0][1] = newRight.y
            matrix.m[0][2] = newRight.z
            matrix.m[0][3] = 0.0
            matrix.m[1][0] = newUp.x
            matrix.m[1][1] = newUp.y
            matrix.m[1][2] = newUp.z
            matrix.m[1][3] = 0.0
            matrix.m[2][0] = newForward.x
            matrix.m[2][1] = newForward.y
            matrix.m[2][2] = newForward.z
            matrix.m[2][3] = 0.0
            matrix.m[3][0] = pos.x
            matrix.m[3][1] = pos.y
            matrix.m[3][2] = pos.z
            matrix.m[3][3] = 1.0
            return matrix

    def MultiplyVector(self, i: Vector):
        v = Vector()
        v.x = i.x * self.m[0][0] + i.y * self.m[1][0] + i.z * self.m[2][0] + i.w * self.m[3][0]
        v.y = i.x * self.m[0][1] + i.y * self.m[1][1] + i.z * self.m[2][1] + i.w * self.m[3][1]
        v.z = i.x * self.m[0][2] + i.y * self.m[1][2] + i.z * self.m[2][2] + i.w * self.m[3][2]
        v.w = i.x * self.m[0][3] + i.y * self.m[1][3] + i.z * self.m[2][3] + i.w * self.m[3][3]
        return v

    def __mul__(self, other):  # MultiplyMatrix
        matrix = Matrix4x4()
        for c in range(4):
            for r in range(4):
                matrix.m[r][c] = self.m[r][0] * other.m[0][c] + self.m[r][1] * other.m[1][c] + self.m[r][2] * other.m[2][c] + self.m[r][3] * other.m[3][c]
        return matrix

    def QuickInverse(self):  # only works with rotation/translation matrices
        matrix = Matrix4x4()
        matrix.m[0][0] = self.m[0][0]
        matrix.m[0][1] = self.m[1][0]
        matrix.m[0][2] = self.m[2][0]
        matrix.m[0][3] = 0.0
        matrix.m[1][0] = self.m[0][1]
        matrix.m[1][1] = self.m[1][1]
        matrix.m[1][2] = self.m[2][1]
        matrix.m[1][3] = 0.0
        matrix.m[2][0] = self.m[0][2]
        matrix.m[2][1] = self.m[1][2]
        matrix.m[2][2] = self.m[2][2]
        matrix.m[2][3] = 0.0
        matrix.m[3][0] = -(self.m[3][0] * matrix.m[0][0] + self.m[3][1] * matrix.m[1][0] + self.m[3][2] * matrix.m[2][0])
        matrix.m[3][1] = -(self.m[3][0] * matrix.m[0][1] + self.m[3][1] * matrix.m[1][1] + self.m[3][2] * matrix.m[2][1])
        matrix.m[3][2] = -(self.m[3][0] * matrix.m[0][2] + self.m[3][1] * matrix.m[1][2] + self.m[3][2] * matrix.m[2][2])
        matrix.m[3][3] = 1.0
        return matrix


class Camera:
    def __init__(self):
        self.pos = Vector()
        self.yaw = 0
        self.dir = Vector()


class Render:
    def __init__(self, display: pygame.display, Near=.01, Far=1000., Fov=90.):
        self.Theta = 0
        self.trianglesToRaster = []
        self.display = display

        AspectRatio = display.get_height() / display.get_width()
        FovRad = 1.0 / math.tan(Fov * 0.5 / 180.0 * 3.14159)
        self.matProj = Matrix4x4.static.Projection(AspectRatio, FovRad, Far, Near)

    def render(self, obj: Obj, camera: Camera, offset=16, wireframe=False, clippingColors=False):
        trianglesToRaster = []
        # self.Theta += 0.01  #  rotation

        matRotX = Matrix4x4.static.RotationX(self.Theta)
        matRotZ = Matrix4x4.static.RotationZ(self.Theta * 0.5)

        matTrans = Matrix4x4.static.Translation(0., 0., offset)

        matWorld = (matRotZ * matRotX) * matTrans

        up = Vector(0, 1, 0)
        target = Vector(0, 0, 1)
        matCameraRot = Matrix4x4.static.RotationY(camera.yaw)

        camera.dir = matCameraRot.MultiplyVector(target)

        target = camera.pos + camera.dir

        matCamera = Matrix4x4.static.PointAt(camera.pos, target, up)

        matView = matCamera.QuickInverse()

        for tri in obj.mesh.tris:
            triTransformed = Triangle([
                matWorld.MultiplyVector(tri.p[0]),
                matWorld.MultiplyVector(tri.p[1]),
                matWorld.MultiplyVector(tri.p[2])
            ])

            # get lines either side of triangle
            line1: Vector = triTransformed.p[1] - triTransformed.p[0]
            line2: Vector = triTransformed.p[2] - triTransformed.p[0]

            # create normal
            normal: Vector = line1.crossProduct(line2).normalize()

            # get ray from triangle to camera
            cameraRay: Vector = triTransformed.p[0] - camera.pos

            # if the ray's angle is aligned with the normal
            if normal.dotProduct(cameraRay) < 0.:
                # calculate shading
                light = Vector(z=-1.)
                light.normalize()
                luminance = max(0.1, light.dotProduct(normal))
                triTransformed.c = (int(255 * luminance), int(255 * luminance), int(255 * luminance))
                triTransformed.l = luminance

                # world space -> view space
                triViewed = Triangle([
                    matView.MultiplyVector(triTransformed.p[0]),
                    matView.MultiplyVector(triTransformed.p[1]),
                    matView.MultiplyVector(triTransformed.p[2])
                ], triTransformed.c, triTransformed.l)

                ClippedTriangles, tri1, tri2 = Triangle.clipAgainstPlane(triViewed, Vector(0, 0, 1), Vector(0, 0, 1), clippingColors=clippingColors)

                for triangleClipped in [tri1, tri2]:
                    if triangleClipped:
                        # 3D -> 2D PROJECTION
                        triProjected = Triangle([
                            self.matProj.MultiplyVector(triangleClipped.p[0]),
                            self.matProj.MultiplyVector(triangleClipped.p[1]),
                            self.matProj.MultiplyVector(triangleClipped.p[2])
                        ], triangleClipped.c, triangleClipped.l)

                        # normalise
                        triProjected.p[0] = triProjected.p[0] / triProjected.p[0].w
                        triProjected.p[1] = triProjected.p[1] / triProjected.p[1].w
                        triProjected.p[2] = triProjected.p[2] / triProjected.p[2].w

                        # offset onto screen
                        screen_offset = Vector(1, 1, 0)
                        triProjected.p[0] = triProjected.p[0] + screen_offset
                        triProjected.p[1] = triProjected.p[1] + screen_offset
                        triProjected.p[2] = triProjected.p[2] + screen_offset

                        triProjected.p[0].x *= 0.5 * self.display.get_width()
                        triProjected.p[0].y *= 0.5 * self.display.get_height()
                        triProjected.p[1].x *= 0.5 * self.display.get_width()
                        triProjected.p[1].y *= 0.5 * self.display.get_height()
                        triProjected.p[2].x *= 0.5 * self.display.get_width()
                        triProjected.p[2].y *= 0.5 * self.display.get_height()

                        if luminance > 0:
                            trianglesToRaster.append(triProjected)

        trianglesToRaster.sort(key=lambda x: (x.p[0].z + x.p[1].z + x.p[2].z) / 3, reverse=True)

        for tri in trianglesToRaster:
            tris = [tri]
            clipped = [None, None]
            newTriangles = 1

            for p in range(4):
                trisToAdd = 0
                while newTriangles > 0:
                    test = tris.pop(0)
                    newTriangles -= 1

                    if p == 0:
                        trisToAdd, clipped[0], clipped[1] = Triangle.clipAgainstPlane(test, Vector(), Vector(0, 1), clippingColors=clippingColors)
                    elif p == 1:
                        trisToAdd, clipped[0], clipped[1] = Triangle.clipAgainstPlane(test, Vector(0, self.display.get_height()-1), Vector(0, -1), clippingColors=clippingColors)
                    elif p == 2:
                        trisToAdd, clipped[0], clipped[1] = Triangle.clipAgainstPlane(test, Vector(), Vector(1), clippingColors=clippingColors)
                    elif p == 3:
                        trisToAdd, clipped[0], clipped[1] = Triangle.clipAgainstPlane(test, Vector(self.display.get_width()-1), Vector(-1), clippingColors=clippingColors)

                    if clipped[0]:
                        tris.append(clipped[0])
                    if clipped[1]:
                        tris.append(clipped[1])

                newTriangles = len(tris)

        for tri in trianglesToRaster:
            points = [(tri.p[0].x, self.display.get_height() - tri.p[0].y),
                      (tri.p[1].x, self.display.get_height() - tri.p[1].y),
                      (tri.p[2].x, self.display.get_height() - tri.p[2].y)]
            pygame.draw.polygon(self.display, tri.c, points)
            if wireframe:
                pygame.draw.polygon(self.display, (0, 0, 0), points, 1)
