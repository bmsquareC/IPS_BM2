# calibrate_intrinsics.py
import cv2
import numpy as np
import time
import os
import config

def calibrate_camera_intrinsics(camera_index, board_size, square_size, num_images=25):
    """
    Performs intrinsic camera calibration using a chessboard pattern.

    Args:
        camera_index (int): The index of the camera to calibrate.
        board_size (tuple): Dimensions of the chessboard (inner corners, width x height).
        square_size (float): Side length of a chessboard square in meters.
        num_images (int): Number of calibration images to capture.

    Returns:
        bool: True if calibration was successful, False otherwise.
    """
    print(f"--- Starting Intrinsic Calibration for Camera {camera_index} ---")
    print(f"Board size: {board_size}, Square size: {square_size}m")
    print(f"Need {num_images} valid chessboard views.")
    print("Show the chessboard pattern to the camera from various angles and distances.")
    print("Press 'c' to capture an image when corners are detected.")
    print("Press 'q' to quit.")

    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(board_w-1, board_h-1,0)
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp = objp * square_size  # Scale to real-world size in meters

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_index}")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    images_captured = 0
    last_capture_time = time.time()
    min_capture_interval = 1.0 # Prevent capturing too fast

    while images_captured < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret_corners, corners = cv2.findChessboardCorners(gray, board_size, None)

        display_frame = frame.copy()
        # If found, add object points, image points (after refining them)
        if ret_corners:
            # Refine corner locations
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Draw and display the corners
            cv2.drawChessboardCorners(display_frame, board_size, corners2, ret_corners)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('c') and (time.time() - last_capture_time > min_capture_interval):
                print(f"Image {images_captured + 1}/{num_images} captured.")
                objpoints.append(objp)
                imgpoints.append(corners2)
                images_captured += 1
                last_capture_time = time.time()
                # Visual feedback
                cv2.putText(display_frame, f"Captured: {images_captured}/{num_images}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(f'Calibration - Camera {camera_index}', display_frame)
                cv2.waitKey(500) # Pause briefly after capture

            elif key == ord('q'):
                print("Calibration aborted by user.")
                cap.release()
                cv2.destroyAllWindows()
                return False
            else:
                 cv2.putText(display_frame, f"Found! Press 'c'. ({images_captured}/{num_images})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        else:
            cv2.putText(display_frame, f"Searching... ({images_captured}/{num_images})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Calibration aborted by user.")
                cap.release()
                cv2.destroyAllWindows()
                return False

        cv2.imshow(f'Calibration - Camera {camera_index}', display_frame)

    cap.release()
    cv2.destroyAllWindows()

    if len(objpoints) < 5: # Need sufficient views for good calibration
         print("Error: Not enough valid images captured for calibration.")
         return False

    print("\nPerforming calibration...")
    # Perform calibration
    # Use cv2.CALIB_FIX_PRINCIPAL_POINT if center is known, cv2.CALIB_ZERO_TANGENT_DIST etc if needed
    ret_cal, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if ret_cal:
        print("Calibration Successful!")
        print("Camera Matrix (mtx):\n", mtx)
        print("Distortion Coefficients (dist):\n", dist)

        # Calculate reprojection error
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        print(f"Total Reprojection Error: {mean_error / len(objpoints)}")

        # --- Save Calibration Data ---
        output_dir = config.CALIBRATION_DATA_DIR
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = config.INTRINSIC_FILE_TEMPLATE.format(camera_index)
        np.savez(filename, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
        print(f"Calibration data saved to: {filename}")
        return True
    else:
        print("Calibration Failed!")
        return False

if __name__ == "__main__":
    # Example usage: Calibrate camera 0
    cam_id_to_calibrate = 0 # Change this to calibrate other cameras
    if cam_id_to_calibrate in config.CAMERA_INDICES:
         calibrate_camera_intrinsics(
             camera_index=cam_id_to_calibrate,
             board_size=config.CHESSBOARD_SIZE,
             square_size=config.CHESSBOARD_SQUARE_SIZE,
             num_images=20 # Adjust number of images needed
         )
    else:
        print(f"Error: Camera index {cam_id_to_calibrate} not found in config.CAMERA_INDICES")