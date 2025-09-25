import airsim
import cv2
import numpy as np
import time
import math
from drone_controller import DroneController
from perception import Perception

# --- Parameters ---
# --- GPS Navigation Parameters ---
TARGET_POSITION_X = 181.46
TARGET_POSITION_Y = -2.11
DYNAMIC_GPS_THRESHOLDS = [30.5, 25.0, 20.0, 15.0, 10.0, 5.0] 

# --- Lidar Obstacle Avoidance Parameters ---
LIDAR_ENABLED = True
LIDAR_SAFETY_DISTANCE = 10.0   # Distance to react.
LIDAR_SECTORS = 5             # Number of sectors.
NAVIGATION_SPEED = 3.0        # Speed when clear.
EVASION_TURN_RATE = 90.0      # Turn rate.
EVASION_SLIDE_DURATION = 3.0  # Slide duration.
EVASION_REVERSE_DURATION = 2.0  # Reverse duration.
INTERIM_ARRIVAL_THRESHOLD = 1.0 # Interim threshold.
TRAPPED_THRESHOLD = 20        # Frames blocked before reverse.
CLEAR_PATH_THRESHOLD = 20     # Frames clear to reset detour.

# --- Continuous Approach Parameters ---
APPROACH_AREA_THRESHOLD = 75000
APPROACH_SPEED = 2.5
YAW_GAIN = 0.05
APPROACH_ALTITUDE = -0.2
SEARCH_YAW_RATE = 30.0
LOWER_FRAME_THRESHOLD_FACTOR = 0.9
MIN_AREA_FOR_LOWER_CHECK = 10000
CONTINUOUS_FORWARD_SPEED_MIN = 0.5
CENTERING_SPEED_FACTOR = 0.5

# --- Bird's-Eye Landing Parameters ---
BIRDSEYE_ALTITUDE = -3.0
FINAL_DESCENT_ALTITUDE = -0.1
LANDING_CORRECTION_GAIN = 0.03
OVERHEAD_CENTERING_GAIN = 0.02
OVERHEAD_CENTERING_THRESHOLD = 20
MAX_LATERAL_SPEED = 0.5
DESCENT_SPEED = 0.5
FRAME_WAIT = 0.05 

def get_forward_vector(quaternion):
    """Calculates the 2D forward vector in the world frame from an AirSim quaternion."""
    x = 1 - 2 * (quaternion.y_val**2 + quaternion.z_val**2)
    y = 2 * (quaternion.x_val * quaternion.y_val + quaternion.w_val * quaternion.z_val)
    norm = math.sqrt(x**2 + y**2)
    if norm == 0: return np.array([1.0, 0.0])
    return np.array([x / norm, y / norm])

def transform_points_to_body_frame(points, drone_position, drone_orientation):
    """Transform points from world coordinates to drone body coordinates."""
    # Convert quaternion to rotation matrix
    w, x, y, z = drone_orientation.w_val, drone_orientation.x_val, drone_orientation.y_val, drone_orientation.z_val
    rotation_matrix = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    
    # Traslate points to dronecentered coordinates
    translated_points = points - np.array([drone_position.x_val, drone_position.y_val, drone_position.z_val])
    
    # Rotoate points to align with drone orientation
    body_points = np.dot(translated_points, rotation_matrix.T)
    
    return body_points

def process_lidar_data(lidar_data, sectors, drone_position, drone_orientation):
    """Processes Lidar point cloud into distance-per-sector, transforming to body frame."""
    if not hasattr(lidar_data, 'point_cloud') or len(lidar_data.point_cloud) < 3:
        return [100.0] * sectors

    point_cloud = lidar_data.point_cloud
    num_elements = len(point_cloud)
    num_points = num_elements // 3
    # (fix for potential incomplete data)
    points = np.array(point_cloud[:num_points * 3], dtype=np.dtype('f4'))
    points = np.reshape(points, (num_points, 3))
    
    body_points = transform_points_to_body_frame(points, drone_position, drone_orientation)
    
    front_points = body_points[body_points[:, 0] > 0.1]
    if front_points.shape[0] == 0:
        return [100.0] * sectors

    x, y = front_points[:, 0], front_points[:, 1]
    horizontal_distances = np.sqrt(x**2 + y**2)
    angles = np.degrees(np.arctan2(y, x))
    
    sector_width = 180 / sectors
    sector_distances = [100.0] * sectors
    
    for i in range(len(angles)):
        angle = angles[i] + 90
        if 0 <= angle < 180:
            sector_index = int(angle // sector_width)
            if horizontal_distances[i] < sector_distances[sector_index]:
                sector_distances[sector_index] = float(horizontal_distances[i])

    return sector_distances

if __name__ == "__main__":
    drone = DroneController()
    perception = Perception(drone.client)

    print("Arming and taking off...")
    drone.takeoff()
    time.sleep(1)

    state = "NAVIGATING_TO_TARGET"
    
    current_gps_threshold_index = 0
    gps_arrival_threshold = DYNAMIC_GPS_THRESHOLDS[current_gps_threshold_index]
    search_rotation_tracker = 0.0
    need_compute_interim = True
    interim_target_x, interim_target_y = 0.0, 0.0
    evasion_state = "SEEKING_GOAL"
    evasion_slide_timer = 0
    evasion_reverse_timer = 0
    all_blocked_count = 0
    detour_direction = 0
    clear_path_count = 0

    try:
        while True:
            try:  # Inner try to catch loop only errors
                drone_state = drone.client.getMultirotorState()
                pos = drone_state.kinematics_estimated.position
                orientation = drone_state.kinematics_estimated.orientation
                
                # Calculate 3D distance to landing pad
                distance_to_pad = math.sqrt(
                    (TARGET_POSITION_X - pos.x_val)**2 + 
                    (TARGET_POSITION_Y - pos.y_val)**2 + 
                    (0 - pos.z_val)**2
                )
                print(f"Distance to landing pad: {distance_to_pad:.2f} m")
                
                lidar_data = None
                if LIDAR_ENABLED and state == "NAVIGATING_TO_TARGET":
                    lidar_data = drone.client.getLidarData(lidar_name="Lidar")

                camera = "bottom_center" if state in ["FINAL_DESCENT", "CENTER_OVERHEAD", "REACQUIRE_OVERHEAD", "REACQUIRE_ASCEND", "BIRDSEYE_ASCEND"] else "front_center"
                frame = perception.get_frame(camera)
                h, w, _ = frame.shape
                cx, cy = w / 2, h / 2
                target_info, processed_frame = perception.find_landing_pad(frame)

                if state == "NAVIGATING_TO_TARGET":
                    distance_to_original = math.sqrt((TARGET_POSITION_X - pos.x_val)**2 + (TARGET_POSITION_Y - pos.y_val)**2)
                    print(f"STATE: NAVIGATING_TO_TARGET (Threshold: {gps_arrival_threshold:.1f}m) (Mode: {evasion_state})")
                    
                    if need_compute_interim:
                        print("  -> Computing interim waypoint.")
                        if distance_to_original <= gps_arrival_threshold:
                            print("  -> Already within threshold. Proceeding to search.")
                            state = "DESCEND_TO_APPROACH_ALT"
                            need_compute_interim = True
                            continue
                        
                        vx = TARGET_POSITION_X - pos.x_val
                        vy = TARGET_POSITION_Y - pos.y_val
                        norm = math.sqrt(vx**2 + vy**2)
                        vx /= norm
                        vy /= norm
                        
                        interim_target_x = TARGET_POSITION_X - vx * gps_arrival_threshold
                        interim_target_y = TARGET_POSITION_Y - vy * gps_arrival_threshold
                        need_compute_interim = False
                        print(f"  -> Interim target: ({interim_target_x:.2f}, {interim_target_y:.2f})")

                    distance_to_interim = math.sqrt((interim_target_x - pos.x_val)**2 + (interim_target_y - pos.y_val)**2)
                    
                    if distance_to_interim < INTERIM_ARRIVAL_THRESHOLD:
                        print("  -> Reached interim waypoint. Proceeding to search.")
                        drone.hover()
                        state = "DESCEND_TO_APPROACH_ALT"
                        continue

                    if LIDAR_ENABLED and lidar_data:
                        sector_distances_list = process_lidar_data(lidar_data, LIDAR_SECTORS, pos, orientation)
                        sector_distances = np.array(sector_distances_list)
                        center_sector = LIDAR_SECTORS // 2
                        sector_width = 180.0 / LIDAR_SECTORS
                        print(f"  -> Lidar (m): {[f'{d:.1f}' for d in sector_distances]}")

                        # Define wide path sectoors (center and adjacent)
                        start_sector = max(0, center_sector - 1)
                        end_sector = min(LIDAR_SECTORS, center_sector + 2)
                        clear_sectors = sector_distances[start_sector:end_sector]

                        forward_velocity, strafe_velocity, yaw_rate = 0.0, 0.0, 0.0

                        goal_vector_world = np.array([interim_target_x - pos.x_val, interim_target_y - pos.y_val])
                        _, _, drone_yaw = airsim.to_eularian_angles(orientation)
                        cos_yaw, sin_yaw = math.cos(drone_yaw), math.sin(drone_yaw)
                        goal_x_body = goal_vector_world[0] * cos_yaw + goal_vector_world[1] * sin_yaw
                        goal_y_body = -goal_vector_world[0] * sin_yaw + goal_vector_world[1] * cos_yaw
                        desired_angle = math.degrees(math.atan2(goal_y_body, goal_x_body))

                        if evasion_state == "SEEKING_GOAL":
                            if sector_distances[center_sector] < LIDAR_SAFETY_DISTANCE:
                                if detour_direction == 0:
                                    left_distance = max(sector_distances[0:center_sector]) if center_sector > 0 else 0
                                    right_distance = max(sector_distances[center_sector+1:]) if center_sector + 1 < LIDAR_SECTORS else 0
                                    if left_distance > right_distance:
                                        detour_direction = -1
                                    else:
                                        detour_direction = 1
                                    print(f"  -> Starting detour {'left' if detour_direction == -1 else 'right'}")
                                print("  -> Obstacle detected! Switching to EVASION_TURN.")
                                evasion_state = "EVASION_TURN"
                                all_blocked_count = 0
                                clear_path_count = 0
                            else:
                                forward_velocity = NAVIGATION_SPEED
                                strafe_velocity = np.clip(goal_y_body, -0.5, 0.5)
                                yaw_rate = desired_angle
                                yaw_rate = np.clip(yaw_rate, -EVASION_TURN_RATE, EVASION_TURN_RATE)
                                clear_path_count += 1
                                if clear_path_count > CLEAR_PATH_THRESHOLD:
                                    detour_direction = 0
                                    clear_path_count = 0
                                    print("  -> Detour complete, reset direction.")
                        
                        elif evasion_state == "EVASION_TURN":
                            yaw_rate = EVASION_TURN_RATE * detour_direction
                            forward_velocity = 0.0
                            strafe_velocity = 0.0
                            print(f"  -> Turning {'left' if detour_direction == -1 else 'right'}.")
                            if np.max(sector_distances) < LIDAR_SAFETY_DISTANCE:
                                all_blocked_count += 1
                                print("  -> All sectors blocked!")
                            else:
                                all_blocked_count = 0
                                print("  -> Clear sector available.")
                            if all_blocked_count > TRAPPED_THRESHOLD:
                                print("  -> Trapped for too long! Switching to EVASION_REVERSE.")
                                evasion_state = "EVASION_REVERSE"
                                evasion_reverse_timer = time.time() + EVASION_REVERSE_DURATION
                                all_blocked_count = 0
                            if min(clear_sectors) > LIDAR_SAFETY_DISTANCE * 1.2:
                                print("  -> Wide path is clear. Switching to EVASION_SLIDE.")
                                evasion_state = "EVASION_SLIDE"
                                evasion_slide_timer = time.time() + EVASION_SLIDE_DURATION

                        elif evasion_state == "EVASION_SLIDE":
                            if min(clear_sectors) < LIDAR_SAFETY_DISTANCE:
                                print("  -> Obstacle in wide path during slide! Back to EVASION_TURN.")
                                evasion_state = "EVASION_TURN"
                                all_blocked_count = 0
                            elif time.time() > evasion_slide_timer:
                                print("  -> Slide complete. Switching back to SEEKING_GOAL.")
                                evasion_state = "SEEKING_GOAL"
                            else:
                                print("  -> Sliding forward to clear obstacle.")
                                forward_velocity = NAVIGATION_SPEED * 0.5
                                strafe_velocity = 0.0
                                yaw_rate = 0.0
                        
                        elif evasion_state == "EVASION_REVERSE":
                            if time.time() > evasion_reverse_timer:
                                print("  -> Reverse complete. Switching back to EVASION_TURN.")
                                evasion_state = "EVASION_TURN"
                                all_blocked_count = 0
                            else:
                                print("  -> Reversing to gain distance.")
                                forward_velocity = -NAVIGATION_SPEED * 0.5
                                strafe_velocity = 0.0
                                yaw_rate = 0.0
                        
                        drone.client.moveByVelocityBodyFrameAsync(
                            float(forward_velocity), float(strafe_velocity), 0, FRAME_WAIT, 
                            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=float(yaw_rate))
                        )
                    else: 
                        drone.client.moveToPositionAsync(interim_target_x, interim_target_y, -5.0, velocity=5)

                elif state == "DESCEND_TO_APPROACH_ALT":
                    print(f"STATE: DESCEND_TO_APPROACH_ALT to {APPROACH_ALTITUDE}")
                    drone.client.moveToZAsync(APPROACH_ALTITUDE, 1.0).join()
                    drone.hover()
                    search_rotation_tracker = 0.0
                    state = "SEARCH"
                    need_compute_interim = True
                    evasion_state = "SEEKING_GOAL"

                elif state == "SEARCH":
                    print("STATE: SEARCH (Front Camera)")
                    if target_info:
                        print("  -> Target found. Switching to APPROACH.")
                        state = "APPROACH"
                    else:
                        drone.client.moveByVelocityAsync(0, 0, 0, FRAME_WAIT, yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=float(SEARCH_YAW_RATE)))
                        search_rotation_tracker += SEARCH_YAW_RATE * FRAME_WAIT
                        if search_rotation_tracker >= 360.0:
                            print("  -> Full 360 search complete, target not found.")
                            current_gps_threshold_index += 1
                            if current_gps_threshold_index < len(DYNAMIC_GPS_THRESHOLDS):
                                gps_arrival_threshold = DYNAMIC_GPS_THRESHOLDS[current_gps_threshold_index]
                                print(f"  -> Tightening GPS threshold to: {gps_arrival_threshold}m")
                                need_compute_interim = True 
                                state = "NAVIGATING_TO_TARGET"
                            else:
                                print("  -> All GPS thresholds tried. Landing.")
                                break

                elif state == "APPROACH":
                    print("STATE: APPROACH (Continuous)")
                    if not target_info:
                        print("  -> Lost target during approach. Switching to REACQUIRE_ASCEND.")
                        drone.hover()
                        state = "REACQUIRE_ASCEND"
                        continue
                    (px, py), area = target_info
                    if area >= APPROACH_AREA_THRESHOLD or (area > MIN_AREA_FOR_LOWER_CHECK and py > h * LOWER_FRAME_THRESHOLD_FACTOR):
                        print("  -> Target close or low in frame. Switching to BIRDSEYE_ASCEND.")
                        drone.hover()
                        state = "BIRDSEYE_ASCEND"
                        continue
                    err_x = px - cx
                    yaw_rate = err_x * YAW_GAIN
                    centering_error_ratio = abs(err_x) / cx
                    forward_speed = APPROACH_SPEED * (1 - CENTERING_SPEED_FACTOR * min(centering_error_ratio, 1.0))
                    forward_speed = max(forward_speed, CONTINUOUS_FORWARD_SPEED_MIN)
                    drone.client.moveByVelocityBodyFrameAsync(
                        float(forward_speed), 0, 0, FRAME_WAIT,
                        yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=float(yaw_rate))
                    )

                elif state == "REACQUIRE_ASCEND":
                    print(f"STATE: REACQUIRE_ASCEND -> to {BIRDSEYE_ALTITUDE}m.")
                    if abs(pos.z_val - BIRDSEYE_ALTITUDE) < 0.2:
                        drone.hover()
                        state = "REACQUIRE_OVERHEAD"
                    else:
                        vz = -1.0 if pos.z_val > BIRDSEYE_ALTITUDE else 1.0
                        drone.client.moveByVelocityAsync(0, 0, float(vz), FRAME_WAIT)

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
                    err_x, err_y = px - cx, py - cy
                    if abs(err_x) < OVERHEAD_CENTERING_THRESHOLD and abs(err_y) < OVERHEAD_CENTERING_THRESHOLD:
                        print("  -> Centered overhead. Proceeding to FINAL_DESCENT.")
                        state = "FINAL_DESCENT"
                        continue
                    vx = -err_y * OVERHEAD_CENTERING_GAIN
                    vy = err_x * OVERHEAD_CENTERING_GAIN
                    vx, vy = np.clip([vx, vy], -MAX_LATERAL_SPEED, MAX_LATERAL_SPEED)
                    print(f"  -> Correcting position: Vel(X:{vx:.2f}, Y:{vy:.2f})")
                    drone.client.moveByVelocityAsync(float(vx), float(vy), 0, FRAME_WAIT)

                elif state == "FINAL_DESCENT":
                    if pos.z_val >= FINAL_DESCENT_ALTITUDE:
                        print("  -> Reached final landing altitude.")
                        break
                    vx, vy = 0.0, 0.0
                    if target_info:
                        (px, py), _ = target_info
                        err_x, err_y = px - cx, py - cy
                        vx = -err_y * LANDING_CORRECTION_GAIN
                        vy = err_x * LANDING_CORRECTION_GAIN
                        vx, vy = np.clip([vx, vy], -MAX_LATERAL_SPEED, MAX_LATERAL_SPEED)
                    vz = DESCENT_SPEED
                    print(f"STATE: FINAL_DESCENT -> Alt: {pos.z_val:.2f}m, Correcting Vel(X:{vx:.2f}, Y:{vy:.2f})")
                    drone.client.moveByVelocityAsync(float(vx), float(vy), float(vz), FRAME_WAIT)

                cv2.imshow("Drone Perception", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                time.sleep(FRAME_WAIT)
            except Exception as e:
                print(f"Error in main loop: {str(e)}. Continuing...")
                time.sleep(0.1)  # Brief pause to avoid spam

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Landing drone...")
    finally:
        print("Initiating final landing sequence.")
        drone.land()
        cv2.destroyAllWindows()