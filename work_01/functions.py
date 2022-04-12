import numpy as np
import random
import matplotlib.path as pltPath
from sklearn.preprocessing import StandardScaler


def createRoom(xcoord, ycoord):
    '''
    определение комнаты по заданным координатам
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
    опеределение минимальных и максимальных размеров комнаты
    :param xcoord:
    :param ycoord:
    :return:
    '''
    return min(xcoord), max(xcoord), min(ycoord), max(ycoord)


def createPoints(stepx, stepy, xcoord, ycoord):
    '''
    формирование массивов внутренних и внешних точек
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
    стандартизация данных
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

    return roomScaler, pointsInScaler, pointsOutScaler, scaler


def createCoordAnchors(nanchors, xcoord, ycoord):
    '''
    генерировние координат маяков
    :param nanchors:
    :param xcoord:
    :param ycoord:
    :return:
    '''
    xrmin, xrmax, yrmin, yrmax = findMinMaxRoom(xcoord, ycoord)
    coordAnchors = np.zeros((nanchors, 2))
    for anchor in range(nanchors):
        coordAnchors[anchor, 0] = random.uniform(xrmin, xrmax)
        coordAnchors[anchor, 1] = random.uniform(yrmin, yrmax)
    return coordAnchors


def createPointsIndoors(coordAnchors, pointsIn, nanchors):
    '''
    удаление точки, если она совпала с координатой маяка
    :param coordAnchors:
    :param pointsIn:
    :param nanchors:
    :return:
    '''
    index = []
    pointsInMetka = pointsIn
    for point in range(len(pointsIn)):
        for anchor in range(nanchors):
            if pointsIn[point][0] == coordAnchors[anchor][0] and pointsIn[point][1] == coordAnchors[anchor][1]:
                index.append(point)
    if index:
        pointsInMetka = np.delete(pointsIn, index, axis=0)
    return pointsInMetka


def segmentCrossing(room, pointInMetka, coordAnchor):
    '''
    ставится flag метке, сколько маяков видит метку
    0 - пересечений нет, маяк видит метку
    1 - есть одно пересечение, маяк не видит метку
    :param room:
    :param pointInMetka:
    :param coordAnchor:
    :return:
    '''
    flag = 0
    ncorners = len(room)

    x1_1, y1_1 = coordAnchor
    x1_2, y1_2 = pointInMetka

    A1 = y1_1 - y1_2
    B1 = x1_2 - x1_1
    C1 = x1_1 * y1_2 - x1_2 * y1_1

    def point(x, y):
        if min(x1_1, x1_2) <= x <= max(x1_1, x1_2) and min(y1_1, y1_2) <= y <= max(y1_1, y1_2):
            flag = 1
        else:
            flag = 0
        return flag

    for j in range(ncorners):

        x2_1, y2_1 = room[j]
        if j != (ncorners - 1):
            x2_2, y2_2 = room[j + 1]
        else:
            x2_2, y2_2 = room[0]

        A2 = y2_1 - y2_2
        B2 = x2_2 - x2_1
        C2 = x2_1 * y2_2 - x2_2 * y2_1

        if B1 * A2 - B2 * A1 and A1:
            y = (C2 * A1 - C1 * A2) / (B1 * A2 - B2 * A1)
            x = (-C1 - B1 * y) / A1
            flag += point(x, y)
        elif B1 * A2 - B2 * A1 and A2:
            y = (C2 * A1 - C1 * A2) / (B1 * A2 - B2 * A1)
            x = (-C2 - B2 * y) / A2
            flag += point(x, y)
        else:
            flag += 0
    return flag


def estimateDOP(room, pointsIn, coordAnchors, nanchors, DOPmax=20):
    '''
    расчет геометрического фактора
    :param room:
    :param pointsIn:
    :param coordAnchors:
    :param nanchors:
    :param DOPmax:
    :return:
    '''
    DOP = np.zeros(len(pointsIn))
    DOP += DOPmax
    pointsVisible = np.zeros(len(pointsIn))

    for point, pointIn in enumerate(pointsIn):
        rastMatrix = np.zeros(nanchors)
        gradMatrix = np.zeros((nanchors, 2))
        flagAnchor = 0

        for anchor, coordAnchor in enumerate(coordAnchors):
            flag = segmentCrossing(room, coordAnchor, pointIn)

            if flag == 0:
                flagAnchor += 1
                rastMatrix[anchor] = np.sqrt((pointIn[0] - coordAnchor[0]) ** 2 + (pointIn[1] - coordAnchor[1]) ** 2)
                # if rastMatrix[anchor]:
                gradMatrix[anchor] = (pointIn - coordAnchor) / rastMatrix[anchor]

            if flagAnchor >= 3:
                pointsVisible[point] = 1
                DOP[point] = np.sqrt(np.trace(np.linalg.inv(gradMatrix.T.dot(gradMatrix))))

    return DOP, pointsVisible


def getSurvPopul(popul, populMetka, val, nsurv, reverse):
    '''
    функция получения популяции
    :param popul:
    :param populMetka:
    :param val:
    :param nsurv:
    :param reverse:
    :return:
    '''
    newPopul = []
    newPopulMetka = []
    sval = sorted(val, reverse=reverse)
    for i in range(nsurv):
        index = val.index(sval[i])
        newPopul.append(popul[index])
        newPopulMetka.append(populMetka[index])
    return newPopul, newPopulMetka, sval


def getParents(currPopul, currPopulMetka, nsurv):
    '''
    функция получения родителей
    :param currPopul:
    :param currPopulMetka:
    :param nsurv:
    :return:
    '''
    indexp1 = random.randint(0, nsurv - 1)
    indexp2 = random.randint(0, nsurv - 1)
    botp1 = currPopul[indexp1]
    botp2 = currPopul[indexp2]
    botp1Metka = currPopulMetka[indexp1]
    botp2Metka = currPopulMetka[indexp2]
    return botp1, botp1Metka, botp2, botp2Metka


def crossPointFrom2Parents(botp1, botp2, idx):
    '''
    функция смешивания (кроссинговера) двух родителей
    :param botp1:
    :param botp2:
    :param idx:
    :return:
    '''
    pindex = random.random()
    if pindex < 0.5:
        x = botp1[idx]
    else:
        x = botp2[idx]
    return x
