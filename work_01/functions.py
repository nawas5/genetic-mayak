import numpy as np
import random
import matplotlib.path as pltPath
from sklearn.preprocessing import StandardScaler


def createRoom(xcoord, ycoord):
    '''
    функция создания комнаты по заданным координатам
    :param ncorners:
    :param xcoord:
    :param ycoord:
    :return:
    '''
    ncorners = len(xcoord)
    room = np.zeros((ncorners, 2))
    for corner in range(ncorners):
        room[corner, 0] = xcoord[corner]
        room[corner, 1] = ycoord[corner]
    return room


def findMinMaxRoom(xcoord, ycoord):
    '''
    размеры комнаты
    :param xcoord:
    :param ycoord:
    :return:
    '''
    return min(xcoord), max(xcoord), min(ycoord), max(ycoord)


def createPoints(stepx, stepy, xcoord, ycoord):
    '''
    формирование внутренних и внешних точек
    :param stepx:
    :param stepy:
    :param ncorners:
    :param xcoord:
    :param ycoord:
    :return:
    '''
    ncorners = len(xcoord)
    room = createRoom(xcoord, ycoord)
    xrmin, xrmax, yrmin, yrmax = findMinMaxRoom(xcoord, ycoord)
    x = np.arange(xrmin, xrmax + stepx, stepx)
    y = np.arange(yrmin, yrmax + stepy, stepy)
    xx, yy = np.meshgrid(x, y)

    path = pltPath.Path(room)

    points = []

    for i in range(len(xx)):
        for j in range(len(xx[i])):
            points.append([xx[i][j], yy[i][j]])

    inside2 = path.contains_points(points)

    pointsIn = []
    pointsOut = []

    for i in range(len(inside2)):
        flag = 0
        if inside2[i]:
            for j in range(len(room)):
                xc1, yc1 = room[j]
                if j != (ncorners - 1):
                    xc2, yc2 = room[j + 1]
                else:
                    xc2, yc2 = room[0]

                x, y = points[i]

                k1 = abs(xc1 - xc2)
                k2 = abs(yc1 - yc2)
                s1 = np.sqrt(k1 * k2 + k2 * k2)

                k1 = abs(xc1 - x)
                k2 = abs(yc1 - y)
                s2 = np.sqrt(k1 * k2 + k2 * k2)

                k1 = abs(xc2 - x)
                k2 = abs(yc2 - y)
                s3 = np.sqrt(k1 * k2 + k2 * k2)

                s1 = np.round(s1 * 1e4) / 1e-4
                s2 = s2 + s3
                s2 = np.round(s2 * 1e4) / 1e-4

                if s2 == s1:
                    flag = 1

            if flag == 1:
                pointsOut.append(points[i])
            else:
                pointsIn.append(points[i])

        else:
            pointsOut.append(points[i])

    pointsIn = np.array(pointsIn)
    pointsOut = np.array(pointsOut)

    return pointsIn, pointsOut


def standardScalerPoints(xcoord, ycoord, pointsIn, pointsOut):
    '''
    стандартизация данных, чтобы они были распределены нормально
    :param xcoord:
    :param ycoord:
    :param pointsIn:
    :param pointsOut:
    :return:
    '''
    scaler = StandardScaler()
    room = createRoom(xcoord, ycoord)
    roomScaler = scaler.fit_transform(room.reshape(-1, 1))
    roomScaler = roomScaler.reshape(len(room), 2)

    pointsInScaler = scaler.transform(pointsIn.reshape(-1, 1))
    pointsInScaler = pointsInScaler.reshape(len(pointsIn), 2)
    pointsOutScaler = scaler.transform(pointsOut.reshape(-1, 1))
    pointsOutScaler = pointsOutScaler.reshape(len(pointsOut), 2)

    return roomScaler, pointsInScaler, pointsOutScaler


def createCoordAnchors(nanchors, xcoord, ycoord):
    xrmin, xrmax, yrmin, yrmax = findMinMaxRoom(xcoord, ycoord)
    coordAnchors = np.zeros((nanchors, 2))
    for anchor in range(nanchors):
        coordAnchors[anchor, 0] = random.uniform(xrmin, xrmax)
        coordAnchors[anchor, 1] = random.uniform(yrmin, yrmax)
    return coordAnchors


def createPointsIndoors(coordAnchors, pointsIn, nanchors):
    index = []
    pointsInMetka = pointsIn
    for point in range(len(pointsIn)):
        for anchor in range(nanchors):
            if pointsIn[point][0] == coordAnchors[anchor][0] and pointsIn[point][1] == coordAnchors[anchor][1]:
                index.append(point)
    if index:
        pointsInMetka = np.delete(pointsIn, index, axis=0)
    return pointsInMetka
