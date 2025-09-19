import airsim
import cv2
import numpy as np
import time
import math
from drone_controller import DroneController
from perception import Perception

# -- Parameters ---
# --- GPS navigation Parameters --
TARGET_POSITION_X = 55.76
TARGET_POSITION_Y = 27.06
GPS_ARRIVAL_THRESHOLD = 30.5  # how close (in meters) to be considered "arrived", i like keeping it a bit far to expand usage of OpenCv.

# --- World Coordinate "Look and Move" Approach Parameters ---
APPROACH_AREA_THRESHOLD = 75000
CENTERING_THRESHOLD = 15      # how close to center (pixels) to be considered "centered" which is state of landing.
APPROACH_STEP_DISTANCE = 2.0  # old approach not rlly needed for now
APPROACH_DESCENT_PER_STEP = 0.0  # How much to descend per move step (meters, positive for down).
APPROACH_SPEED = 2.5          # increased speed for faster approach while not overshooting.
YAW_GAIN = 0.05               # Increased for more aggressive centering if value is too low then u sometimes land fatr.
APPROACH_ALTITUDE = -0.2      # very low altitude for the search phase so open cv has no issues spotting pad.
SEARCH_YAW_RATE = 30.0        # Increased yaw rate during search for faster scanning, to slow wastes time (improved speed by 86%).
LOWER_FRAME_THRESHOLD_FACTOR = 0.9  # increased to trigger only when very low in frame.
MIN_AREA_FOR_LOWER_CHECK = 10000  # Min area to trigger the "too low" check (avoid false positives from far away).

# - Bird's eye Landing Parameters -
BIRDSEYE_ALTITUDE = -3.0      # Alt for a wider, more stable view so u can see whole pad to center better.
FINAL_DESCENT_ALTITUDE = -0.1 # drone gets to 10cmm before final landing api call.
LANDING_CORRECTION_GAIN = 0.03  # increased gain for faster corrections during final descent.
OVERHEAD_CENTERING_GAIN = 0.02  # Gain for lateral corrections during overhead centering (still testing this value)
OVERHEAD_CENTERING_THRESHOLD = 20  # Pixel threshold for considering centered overhead (bottom camera also sometimes causes issue when approaching from nearby).
MAX_LATERAL_SPEED = 0.5  # increased max lateral speed for better correction (so no infinite hovering around).
DESCENT_SPEED = 0.5  # Rduced descent speed to allow more time for corrections(sweet spot).
FRAME_WAIT = 0.03             # reduced wait for faster loop(more hz but much faster).

# ---  Continuous Approach Parameters ---
CONTINUOUS_FORWARD_SPEED_MAX = APPROACH_SPEED
CONTINUOUS_FORWARD_SPEED_MIN = 0.5  # min forward speed when off center (changing this value can cause overshooting)
CENTERING_SPEED_FACTOR = 0.5       # how much centering affects forward speed (0-1

def get_forward_vector(quaternion):
    """Calculates the 2D forward vector in the world frame from an AirSim quaternion."""
    x = 1 - 2 * (quaternion.y_val**2 + quaternion.z_val**2)
    y = 2 * (quaternion.x_val * quaternion.y_val + quaternion.w_val * quaternion.z_val) #will prolly explain why i ended up using this equation in the readme
    norm = math.sqrt(x**2 + y**2)
    if norm == 0: return np.array([1.0, 0.0])  # Default to x-axis if norm is zero
    return np.array([x / norm, y / norm])

if __name__ == "__main__":
    drone = DroneController()
    perception = Perception(drone.client)

    # -- Take-off --
    print("Arming and taking off...")
    drone.takeoff()
    time.sleep(1)

    state = "NAVIGATING_TO_TARGET"
    
    try:
        while True:
            camera = "bottom_center" if state in ["FINAL_DESCENT", "CENTER_OVERHEAD", "REACQUIRE_OVERHEAD", "REACQUIRE_ASCEND", "BIRDSEYE_ASCEND"] else "front_center"
            frame = perception.get_frame(camera)
            h, w, _ = frame.shape
            cx, cy = w / 2, h / 2
            target_info, processed_frame = perception.find_landing_pad(frame)

            # --- State Machine ---
            if state == "NAVIGATING_TO_TARGET":
                print("STATE: NAVIGATING_TO_TARGET")
                pos = drone.client.getMultirotorState().kinematics_estimated.position # this helped a lot with refinding target
                distance = math.sqrt((TARGET_POSITION_X - pos.x_val)**2 + (TARGET_POSITION_Y - pos.y_val)**2)
                print(f"  -> Distance to target area: {distance:.2f}m") # basically derive how far we are
                
                if distance < GPS_ARRIVAL_THRESHOLD: # change this value when needed in paramaters
                    print("  -> Arrived. Switching to DESCEND_TO_APPROACH_ALT.")
                    drone.hover()
                    state = "DESCEND_TO_APPROACH_ALT"
                else:
                    drone.client.moveToPositionAsync(TARGET_POSITION_X, TARGET_POSITION_Y, -5.0, velocity=5)

            elif state == "DESCEND_TO_APPROACH_ALT":
                print(f"STATE: DESCEND_TO_APPROACH_ALT to {APPROACH_ALTITUDE}")
                drone.client.moveToZAsync(APPROACH_ALTITUDE, 1.0).join()
                drone.hover()
                state = "SEARCH"

            elif state == "SEARCH":
                print("STATE: SEARCH (Front Camera)")
                if target_info:
                    print("  -> Target found. Switching to APPROACH.")
                    state = "APPROACH"
                else:
                    # Rotate faster to find target quicker
                    drone.client.moveByVelocityAsync(0, 0, 0, 0.1, yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=SEARCH_YAW_RATE))

            elif state == "APPROACH":
                print("STATE: APPROACH (Continuous)")
                if not target_info:
                    print("  -> Lost target during approach. Switching to REACQUIRE_ASCEND.")
                    drone.hover()
                    state = "REACQUIRE_ASCEND"
                    continue

                (px, py), area = target_info
                if area >= APPROACH_AREA_THRESHOLD:
                    print("  -> Target is close. Switching to BIRDSEYE_ASCEND.")
                    drone.hover()
                    state = "BIRDSEYE_ASCEND"
                    continue

                # Check if target is too low in the frame (likely below the drone, causing circling), but only if close enough
                if area > MIN_AREA_FOR_LOWER_CHECK and py > h * LOWER_FRAME_THRESHOLD_FACTOR:
                    print("  -> Target detected too low in frame (and large enough). Switching to BIRDSEYE_ASCEND to avoid circling.")
                    drone.hover()
                    state = "BIRDSEYE_ASCEND"
                    continue

                # Continuous movement: yaw to center while moving forward at variable speed (prolly dont want to change these values)
                err_x = px - cx
                yaw_rate = err_x * YAW_GAIN
                
                # Forward speed scales with how centered the target is (full speed when centered)
                centering_error_ratio = abs(err_x) / cx
                forward_speed = CONTINUOUS_FORWARD_SPEED_MAX * (1 - CENTERING_SPEED_FACTOR * min(centering_error_ratio, 1.0))
                forward_speed = max(forward_speed, CONTINUOUS_FORWARD_SPEED_MIN)
                
                # optional descent rate proportional to forward speed
                descent_rate = (APPROACH_DESCENT_PER_STEP / APPROACH_STEP_DISTANCE) * forward_speed
                
                # Use body frame velocity for forward movement while yawing (airsim corddinates flip when drone flips)
                drone.client.moveByVelocityBodyFrameAsync(
                    forward_speed, 0, descent_rate, 0.1,
                    yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
                )

            elif state == "REACQUIRE_ASCEND":
                print(f"STATE: REACQUIRE_ASCEND -> to {BIRDSEYE_ALTITUDE}m.")
                current_alt = drone.client.getMultirotorState().kinematics_estimated.position.z_val
                if abs(current_alt - BIRDSEYE_ALTITUDE) < 0.2:
                    drone.hover()
                    state = "REACQUIRE_OVERHEAD"
                else:
                    vz = -1.0 if current_alt > BIRDSEYE_ALTITUDE else 1.0
                    drone.client.moveByVelocityAsync(0, 0, vz, 0.1)  # Short durations for smoother control

            elif state == "REACQUIRE_OVERHEAD":
                print("STATE: REACQUIRE_OVERHEAD (Bottom Camera)")
                if target_info:
                    print("  -> Target reacquired. Proceeding to CENTER_OVERHEAD.")
                    state = "CENTER_OVERHEAD"
                else:
                    print("  -> Failed to reacquire. Returning to SEARCH.")
                    state = "SEARCH"
                time.sleep(0.5)

            elif state == "BIRDSEYE_ASCEND":
                print(f"STATE: BIRDSEYE_ASCEND -> target {BIRDSEYE_ALTITUDE}m")
                drone.client.moveToZAsync(BIRDSEYE_ALTITUDE, 1.5).join()
                drone.hover()
                print("  -> Altitude adjusted. Proceeding to CENTER_OVERHEAD.")
                state = "CENTER_OVERHEAD"

            elif state == "CENTER_OVERHEAD":
                print("STATE: CENTER_OVERHEAD (Bottom Camera)")
                if not target_info:
                    print("  -> Lost target overhead. Switching to REACQUIRE_OVERHEAD.")
                    state = "REACQUIRE_OVERHEAD"
                    continue

                (px, py), _ = target_info
                err_x = px - cx
                err_y = py - cy

                if abs(err_x) < OVERHEAD_CENTERING_THRESHOLD and abs(err_y) < OVERHEAD_CENTERING_THRESHOLD:
                    print("  -> Centered overhead. Proceeding to FINAL_DESCENT.")
                    state = "FINAL_DESCENT"
                    continue

                vx = -err_y * OVERHEAD_CENTERING_GAIN
                vy = err_x * OVERHEAD_CENTERING_GAIN
                vx, vy = np.clip([vx, vy], -MAX_LATERAL_SPEED, MAX_LATERAL_SPEED)
                vz = 0.0  # No vertical movement, just lateral centering

                print(f"  -> Correcting position: Vel(X:{vx:.2f}, Y:{vy:.2f})")
                drone.client.moveByVelocityAsync(vx, vy, vz, 0.1)

            elif state == "FINAL_DESCENT":
                current_alt = drone.client.getMultirotorState().kinematics_estimated.position.z_val
                if current_alt >= FINAL_DESCENT_ALTITUDE:
                    print("  -> Reached final landing altitude.")
                    break

                vx, vy = 0.0, 0.0
                if target_info:
                    (px, py), _ = target_info
                    err_x, err_y = px - cx, py - cy
                    # use a proportional gain for 'gentle' final corrections
                    vx = -err_y * LANDING_CORRECTION_GAIN
                    vy = err_x * LANDING_CORRECTION_GAIN
                    vx, vy = np.clip([vx, vy], -MAX_LATERAL_SPEED, MAX_LATERAL_SPEED)

                vz = DESCENT_SPEED
                print(f"STATE: FINAL_DESCENT -> Alt: {current_alt:.2f}m, Correcting Vel(X:{vx:.2f}, Y:{vy:.2f})")
                drone.client.moveByVelocityAsync(vx, vy, vz, 0.1)

            cv2.imshow("Drone Perception", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(FRAME_WAIT)

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Landing drone...")
    finally:
        print("Initiating final landing sequence.")
        drone.land()
        cv2.destroyAllWindows()