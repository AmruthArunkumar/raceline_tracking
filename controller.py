import numpy as np
from numpy.typing import ArrayLike

from simulator import RaceTrack

# GLOBAL
prevErrorDelta = 0.0
prevErrorV = 0.0

# PARAMETERS
LOOKAHEAD_DISTANCE = 30

KpV = 1.0
KdV = 0.2
KiV = 0.05

KpDELTA = 4.0
KdDELTA = 0.5
KiDELTA = 0.1

def getLookaheadPoint(closest: int, path: np.ndarray):
    N = path.shape[0]
    i = closest
    dist = 0.0

    while dist < LOOKAHEAD_DISTANCE:
        j = (i + 1) % N # cyclic
        space = np.linalg.norm(path[i] - path[j])
        dist += space
        i = j
        if space == 0: break

    return path[i]

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
    lookaheadPoint = getLookaheadPoint(closest, path)

    dx = lookaheadPoint[0] - pos[0]
    dy = lookaheadPoint[1] - pos[1]

    desiredPhi = np.arctan2(dy, dx)
    headingError = desiredPhi - phi
    alpha = np.arctan2(np.sin(headingError), np.cos(headingError))

    deltaR = np.arctan(2 * wb * np.sin(alpha) / LOOKAHEAD_DISTANCE)
    deltaR = np.clip(deltaR, deltaMin, deltaMax)

    turn_factor = np.abs(alpha)
    vR = vMax / (1 + 5 * turn_factor)  
    vR = np.clip(vR, vMin, vMax)

    return np.array([deltaR, vR]).T
