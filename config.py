# config.py
import numpy as np
import cv2

# --- Camera Configuration ---
NUM_CAMERAS = 4
# Assign correct camera indices for your system (e.g., 0, 1, 2, 3)
# Use V4L2 utilities (like `v4l2-ctl --list-devices`) on Linux if unsure
CAMERA_INDICES = [0, 1, 2, 3] #Should Change it 
FRAME_WIDTH = 640  # Desired frame width
FRAME_HEIGHT = 480 # Desired frame height

# --- Calibration Files ---
INTRINSIC_FILE_TEMPLATE = "calibration_data/cam{}_intrinsics.npz"
EXTRINSIC_FILE_TEMPLATE = "calibration_data/cam{}_extrinsics_world.npz"
CALIBRATION_DATA_DIR = "calibration_data" # Ensure this directory exists

# --- Intrinsic Calibration Parameters ---
CHESSBOARD_SIZE = (9, 6)  # Number of inner corners (width, height)
CHESSBOARD_SQUARE_SIZE = 0.025  # Size of a chessboard square in meters

# --- World Object Definition (Triangle + ArUco) ---
# Define the 3D coordinates of the ball centers in the world frame.
# Assume ArUco is at (0, 0, 0). Triangle is equilateral.
# Adjust TRIANGLE_SIDE_LENGTH based on your physical setup.
TRIANGLE_SIDE_LENGTH = 0.07 # meters (7 cm)
# Calculate vertex positions for an equilateral triangle centered at origin (on XY plane)
# Height of equilateral triangle: sqrt(3)/2 * side
h = np.sqrt(3) / 2 * TRIANGLE_SIDE_LENGTH
r = TRIANGLE_SIDE_LENGTH / np.sqrt(3) # Distance from center to vertex
# Ball 1 (Top vertex)
BALL_WORLD_POS_1 = np.array([0.0, r, 0.0], dtype=np.float32)
# Ball 2 (Bottom Left)
BALL_WORLD_POS_2 = np.array([-TRIANGLE_SIDE_LENGTH / 2.0, -r/2.0, 0.0], dtype=np.float32)
# Ball 3 (Bottom Right)
BALL_WORLD_POS_3 = np.array([ TRIANGLE_SIDE_LENGTH / 2.0, -r/2.0, 0.0], dtype=np.float32)

# List of the 3 ball world positions (ensure order matches detection logic)
# We will sort detected balls by Y-coordinate (top, middle, middle) then X (for the two middle)
# So order should be: Top, Bottom-Left, Bottom-Right
BALLS_WORLD_POS = np.array([BALL_WORLD_POS_1, BALL_WORLD_POS_2, BALL_WORLD_POS_3], dtype=np.float32)

# --- ArUco Marker Parameters ---
ARUCO_DICT_NAME = cv2.aruco.DICT_4X4_50 # Choose an appropriate dictionary
ARUCO_DICTIONARY = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_NAME)
ARUCO_PARAMETERS = cv2.aruco.DetectorParameters()
ARUCO_MARKER_ID = 7 # ID of the marker placed at the center
ARUCO_MARKER_SIZE = 0.10 # Size of the ArUco marker side in meters (10 cm)
# World coordinates of ArUco corners (counter-clockwise order starting from top-left)
# Relative to the center (0,0,0)
half_m = ARUCO_MARKER_SIZE / 2.0
ARUCO_WORLD_CORNERS = np.array([
    [-half_m, half_m, 0.0], # Top-left
    [ half_m, half_m, 0.0], # Top-right
    [ half_m,-half_m, 0.0], # Bottom-right
    [-half_m,-half_m, 0.0]  # Bottom-left
], dtype=np.float32)

# --- Red Ball Detection Parameters (HSV Color Range) ---
# **IMPORTANT**: Tune these values for your specific lighting and red balls!
# Use a tool like HSV color picker (many available online or as apps)
RED_LOWER = np.array([0, 120, 70])   # Lower bound for Red in HSV
RED_UPPER = np.array([10, 255, 255]) # Upper bound for Red in HSV
RED_LOWER2 = np.array([170, 120, 70]) # Lower bound for Red (wraps around hue)
RED_UPPER2 = np.array([180, 255, 255])# Upper bound for Red (wraps around hue)

# Ball detection parameters
MIN_BALL_AREA = 50     # Minimum pixel area to consider a contour a ball
MAX_BALL_AREA = 5000   # Maximum pixel area
MIN_BALL_CIRCULARITY = 0.6 # Minimum circularity (1.0 is perfect circle)

# --- Positioning System Parameters ---
POSE_ESTIMATION_METHOD = 'BALLS' # 'BALLS' or 'ARUCO' or 'BOTH'
MIN_CONFIDENCE_BALLS = 3 # Minimum number of balls required for pose estimation
MIN_CONFIDENCE_ARUCO = 1 # Minimum number of ArUco markers required

# --- Positioning System Parameters ---
POSE_ESTIMATION_METHOD = 'BALLS' # 'BALLS' or 'ARUCO' or 'BOTH'
MIN_CONFIDENCE_BALLS = 3 # Minimum number of balls required for pose estimation
MIN_CONFIDENCE_ARUCO = 1 # Minimum number of ArUco markers required

# --- PnP RANSAC Parameters (for Ball-based Pose Estimation) ---
# Use SOLVEPNP_EPNP (Efficient PnP), SOLVEPNP_ITERATIVE, SOLVEPNP_P3P, SOLVEPNP_SQPNP etc.
PNP_METHOD = cv2.SOLVEPNP_EPNP
# RANSAC parameters
RANSAC_ITERATIONS = 100     # Number of RANSAC iterations
RANSAC_REPROJECTION_ERROR = 8.0 # Maximum allowed reprojection error (pixels) for inliers
RANSAC_CONFIDENCE = 0.99    # Desired confidence level (probability)
MIN_PNP_INLIERS = 3         # Minimum number of inlier points required for a valid RANSAC pose

# --- Display ---
DISPLAY_SCALE = 0.8 # Scale factor for displayed images