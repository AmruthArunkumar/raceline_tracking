import numpy as np
from numpy.typing import ArrayLike

from simulator import RaceTrack

# GLOBAL
prevErrorDelta = 0.0
prevErrorV = 0.0

LOOKAHEAD_DISTANCE_TURN = 30

KpV = 1.0
KdV = 0.2
KiV = 0.05

KpDELTA = 4.0
KdDELTA = 0.5
KiDELTA = 0.1

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

    return np.array([deltaRate, a]).T

def getAlpha(lookAheadPoint: np.ndarray, pos: list[float, float], phi: np.float64):
    dx = lookAheadPoint[0] - pos[0]
    dy = lookAheadPoint[1] - pos[1]

    desiredPhi = np.arctan2(dy, dx)
    headingError = desiredPhi - phi

    return np.arctan2(np.sin(headingError), np.cos(headingError))

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
    lookaheadPoints = getLookaheadPoints(closest, path, [LOOKAHEAD_DISTANCE_TURN])

    alphas = [getAlpha(point, pos, phi) for point in lookaheadPoints]

    deltaR = np.arctan(2 * wb * np.sin(alphas[0]) / LOOKAHEAD_DISTANCE_TURN)
    deltaR = np.clip(deltaR, deltaMin, deltaMax)

    vR = vMax / (1 + 10 * max([np.abs(alpha) for alpha in alphas]))  
    vR = np.clip(vR, vMin, vMax)

    # print("{:.2f}, {:.2f}, [{:.2f}, {:.2f}], [{:.2f}, {:.2f}]".format(turn_factor, vR, pos[0], pos[1], lookaheadPoint[0], lookaheadPoint[1]))

    return np.array([deltaR, vR]).T
