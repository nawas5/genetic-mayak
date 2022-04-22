import numpy as np
import random
import matplotlib.path as pltPath
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go


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

def polyDistPoint(xanchor, yanchor, xcoord, ycoord):
    '''
    сдвигать маяки на границы комнаты, если они вылетели за её пределы
    :param xanchor:
    :param yanchor:
    :param xcoord:
    :param ycoord:
    :return:
    '''

    if (xcoord[0] != xcoord[-1]) or (ycoord[0] != ycoord[-1]):
        xcoord = np.append(xcoord, xcoord[0])
        ycoord = np.append(ycoord, ycoord[0])

    A = -np.diff(ycoord)
    B = np.diff(xcoord)
    C = ycoord[1:] * xcoord[:-1] - xcoord[1:] * ycoord[:-1]

    AB = 1 / (A ** 2 + B ** 2)
    vv = (A * xanchor + B * yanchor + C) - 1e-3
    xpoint = xanchor - (A * AB) * vv
    ypoint = yanchor - (B * AB) * vv

    idx_x = (((xpoint >= xcoord[:-1]) & (xpoint <= xcoord[1:])) | ((xpoint >= xcoord[1:]) & (xpoint <= xcoord[:-1])))
    idx_y = (((ypoint >= ycoord[:-1]) & (ypoint <= ycoord[1:])) | ((ypoint >= ycoord[1:]) & (ypoint <= ycoord[:-1])))
    idx = idx_x & idx_y

    dcoord = np.sqrt((xcoord[:-1] - xanchor) ** 2 + (ycoord[:-1] - yanchor) ** 2)

    if not any(idx):
        i = np.argmin(dcoord)
        xpoly = xcoord[i]
        ypoly = ycoord[i]
    else:
        dpoint = np.sqrt((xpoint[idx] - xanchor) ** 2 + (ypoint[idx] - yanchor) ** 2)
        i_coord = np.argmin(dcoord)
        i_point = np.argmin(dpoint)
        i = np.argmin([i_coord, i_point])
        if i == 0:
            xpoly = xpoint[i_coord]
            ypoly = ypoint[i_coord]
        elif i == 1:
            idxs = np.where(idx)[0]
            xpoly = xpoint[idxs[i_point]]
            ypoly = ypoint[idxs[i_point]]

    return xpoly, ypoly

def polyCoordAnchors(xcoord, ycoord, room, coordAnchors):

    polyCoordAnchor = np.zeros(coordAnchors.shape)
    path = pltPath.Path(room)
    for i, coordAnchor in enumerate(coordAnchors):
        inside = path.contains_points(coordAnchor.reshape((1,2)))
        if not inside:
            xpoly, ypoly = polyDistPoint(coordAnchor[0], coordAnchor[1], xcoord, ycoord)
            polyCoordAnchor[i] = [xpoly, ypoly]
        else:
            polyCoordAnchor[i] = coordAnchor
    return polyCoordAnchor


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

    for point, pointIn in enumerate(pointsIn):
        rastMatrix = np.zeros(nanchors)
        gradMatrix = np.zeros((nanchors, 2))
        flagAnchor = 0

        for anchor, coordAnchor in enumerate(coordAnchors):
            flag = segmentCrossing(room, coordAnchor, pointIn)

            if flag == 0:
                flagAnchor += 1
                rastMatrix[anchor] = np.sqrt((pointIn[0] - coordAnchor[0]) ** 2 + (pointIn[1] - coordAnchor[1]) ** 2)
                gradMatrix[anchor] = (pointIn - coordAnchor) / rastMatrix[anchor]

            if flagAnchor >= 3:
                DOP[point] = np.sqrt(np.trace(np.linalg.inv(gradMatrix.T.dot(gradMatrix))))

    return DOP


def getSurvPopul(popul, val, nsurv, reverse):
    '''
    функция получения популяции
    :param popul:
    :param val:
    :param nsurv:
    :param reverse:
    :return:
    '''
    newPopul = []
    sval = sorted(val, reverse=reverse)
    for i in range(nsurv):
        index = val.index(sval[i])
        newPopul.append(popul[index])
    return newPopul, sval


def getParents(currPopul, nsurv):
    '''
    функция получения родителей
    :param currPopul:
    :param nsurv:
    :return:
    '''
    indexp1 = random.randint(0, nsurv - 1)
    indexp2 = random.randint(0, nsurv - 1)
    botp1 = currPopul[indexp1]
    botp2 = currPopul[indexp2]
    return botp1, botp2


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


def plotRoom(room, pointsIn, pointsOut):
    roomPlot = list(room).copy()
    roomPlot.append(roomPlot[0])
    xr, yr = zip(*roomPlot)

    xin, yin = pointsIn[:, 0], pointsIn[:, 1]
    xout, yout = pointsOut[:, 0], pointsOut[:, 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xr, y=yr,
                             mode='lines',
                             name='path'))
    fig.add_trace(go.Scatter(x=xin, y=yin,
                             mode='markers',
                             name='points in path'))

    fig.add_trace(go.Scatter(x=xout, y=yout,
                             mode='markers', name='points out path'))

    fig.show()


def plotAnchors(coordAnchors, room):
    roomPlot = list(room).copy()
    roomPlot.append(roomPlot[0])
    xr, yr = zip(*roomPlot)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xr, y=yr,
                             mode='lines',
                             name='path'))
    fig.add_trace(go.Scatter(x=coordAnchors[:, 0], y=coordAnchors[:, 1],
                             mode='markers',
                             name='coord mayak'))

    fig.show()


def plotDOPfactor(rooms, coordMayak, DOP, pointsInMetka):
    rooms_plot = list(rooms).copy()
    rooms_plot.append(rooms_plot[0])
    xr, yr = zip(*rooms_plot)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=xr, y=yr, z=np.zeros(len(xr)),
                               mode='lines',
                               name="rooms"))

    fig.add_trace(go.Mesh3d(x=(pointsInMetka[:, 0]),
                            y=(pointsInMetka[:, 1]),
                            z=DOP,
                            opacity=0.7,
                            color='blue',
                            name='DOP'
                            ))

    fig.add_trace(go.Scatter3d(x=coordMayak[:, 0],
                               y=coordMayak[:, 1],
                               z=np.zeros(len(coordMayak)),
                               mode='markers',
                               name='coord mayak'
                               ))

    fig.show()
