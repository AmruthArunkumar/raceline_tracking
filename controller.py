import numpy as np
from numpy.typing import ArrayLike
import math

from simulator import RaceTrack

# GLOBAL
prevErrorDelta = 0.0
prevErrorV = 0.0

KpV = 3
KdV = 0.2
KiV = 1

KpDELTA = 19
KdDELTA = 0.1
KiDELTA = 1.5

def getLookaheadPoints(closest: int, path: np.ndarray, lookahead_distances: list[int] = [30]) -> list[np.ndarray]:
    """
    Get raceline points `lookahead_distances` ahead of current position
    """
    N = path.shape[0]
    i = closest
    dist = 0.0
    results = []
    lookahead_index = 0

    while dist < lookahead_distances[-1]:
        j = (i + 1) % N # cyclic
        space = np.linalg.norm(path[i] - path[j])
        dist += space
        i = j
        while lookahead_index < len(lookahead_distances) and dist >= lookahead_distances[lookahead_index]:
            results.append(path[i])
            lookahead_index += 1
        if space == 0: break

    return results

def lower_controller(
    state : ArrayLike, # car state: [sx, sy, delta, v, phi]
    desired : ArrayLike, # desired: [delta_r, v_r]
    parameters : ArrayLike # car parameters
) -> ArrayLike:
    assert(desired.shape == (2,))

    global prevErrorDelta, prevErrorV

    # Parse inputs
    pos = [state[0], state[1]] # current position
    delta = state[2] # current heading
    v = state[3] # current speed
    phi = state[4] # current steering angle

    deltaR = desired[0] # reference steering
    vR = desired[1] # reference velocity

    minSteerRate = parameters[7]
    maxSteerRate = parameters[9]
    minAccel = parameters[8]
    maxAccel = parameters[10]

    # Error Function
    vError = vR - v
    deltaError = deltaR - delta

    derivativeV = (vError - prevErrorV) / 0.1
    integralV = vError * 0.1
    prevErrorV = vError

    derivativeDelta = (deltaError - prevErrorDelta) / 0.1
    integralDelta = deltaError * 0.1
    prevErrorDelta = deltaError

    a = KpV*vError + KdV*derivativeV + KiV*integralV
    deltaRate = KpDELTA*deltaError + KdDELTA*derivativeDelta + KiDELTA*integralDelta

    a = np.clip(a, minAccel, maxAccel)
    deltaRate = np.clip(deltaRate, minSteerRate, maxSteerRate)

    # print("C {:.2f}, {:.2f}".format(deltaRate, a))

    return np.array([deltaRate, a]).T

def get_alpha(lookAheadPoint: np.ndarray, pos: list[float, float], phi: np.float64):
    dx = lookAheadPoint[0] - pos[0]
    dy = lookAheadPoint[1] - pos[1]

    desiredPhi = np.arctan2(dy, dx)
    headingError = desiredPhi - phi

    return np.arctan2(np.sin(headingError), np.cos(headingError))

def get_angles(points: list[tuple[np.ndarray]]):
    """
    Computes angles between a list of points. Each list element is a triplet of vectors.
    """
    results = []

    for triplet in points:
        # angle1 = np.arctan((triplet[1][1] - triplet[0][1]) / (triplet[1][0] - triplet[0][0]))
        # angle2 = np.arctan((triplet[2][1] - triplet[1][1]) / (triplet[2][0] - triplet[1][0]))
        # results.append(angle2 - angle1)

        # a dot b = |a||b|cos(theta)
        vec1 = triplet[1] - triplet[0]
        vec2 = triplet[2] - triplet[1]
        dot = sum(vec1 * vec2)
        angle = np.acos(dot / (math.sqrt(vec1[0] ** 2 + vec1[1] ** 2) * math.sqrt(vec2[0] ** 2 + vec2[1] ** 2)))

        results.append(angle)

    return results

def controller(
    state : ArrayLike, # car state: [sx, sy, phi, v, delta]
    parameters : ArrayLike, # car parameters: [l_wb, a_max, v_max, delta_max, etc]
    racetrack : RaceTrack # track and reference path
) -> ArrayLike:
    # Parse inputs
    pos = [state[0], state[1]]
    delta = state[2]
    v = state[3]
    phi = state[4]

    wb = parameters[0]
    deltaMin = parameters[1]
    deltaMax = parameters[4]
    vMin = parameters[2]
    vMax = parameters[5]

    path = racetrack.centerline

    closest = np.argmin(np.linalg.norm(path - pos, axis=1))
    # lookahead_distances = [30, 50, 70, 100, 130, 170, 200, 230, 260]
    lookahead_distances = [10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250]
    lookahead_indexes = []
    padded_lookahead_distances = [] # Compute distances with += 10 around each point as well
    for dist in lookahead_distances:
        if len(padded_lookahead_distances) == 0 or padded_lookahead_distances[-1] != dist - 10:
            lookahead_indexes.append(len(padded_lookahead_distances) + 1)
            padded_lookahead_distances.extend((dist - 10, dist, dist + 10))
        else:
            lookahead_indexes.append(len(padded_lookahead_distances))
            padded_lookahead_distances.extend((dist, dist + 10))

    lookahead_points = getLookaheadPoints(closest, path, padded_lookahead_distances)
    turn_angles = get_angles([(lookahead_points[i - 1], lookahead_points[i], lookahead_points[i + 1]) for i in lookahead_indexes])
    # print(", ".join(["({:.2f}, {:.2f})".format(x, y) for [x, y] in lookahead_points]))
    # print(", ".join(["{:.2f}".format(a) for a in turn_angles]))

    # TODO: this is actually a pretty bad way to calculate our required steering angle and we should change to turn_angles
    alpha_point = lookahead_points[3]
    alpha = get_alpha(alpha_point, pos, phi)

    # print("DIFF {:.5f}, {:.5f}".format(alpha, turn_angles[0]))

    pure_pursuit_magnification = 1.1
    deltaR = np.arctan(2 * pure_pursuit_magnification * wb * np.sin(alpha) / lookahead_distances[1]) # Pure pursuit
    deltaR = np.clip(deltaR, deltaMin, deltaMax)

    # print(", ".join("{:.2f}".format(a) for a in turn_angles))
    steering_compensation = max([np.abs(angle) for index, angle in enumerate(turn_angles)])
    steering_compensation *= 0 if steering_compensation <= 0.1 else 1.5

    # print("{:.2f} {:.2f}".format(steering_factor, steering_compensation))

    vR = vMax / (1 + steering_compensation)
    # print("{:.2f}, {:.2f}, {:.2f}".format(steering_compensation, deltaR, vR))
    vR = np.clip(vR, vMin, vMax)

    # print("{:.2f}, {:.2f}, [{:.2f}, {:.2f}], [{:.2f}, {:.2f}]".format(turn_factor, vR, pos[0], pos[1], lookaheadPoint[0], lookaheadPoint[1]))

    return np.array([deltaR, vR]).T
