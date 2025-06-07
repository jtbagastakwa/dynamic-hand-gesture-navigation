import cv2
import mediapipe as mp
import pyautogui
import time
import math
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="Dynamic Hand Gesture Navigation",
    page_icon="👋",
    layout="wide",
)

# ===================================================================
# --- Sidebar ---
# ===================================================================
with st.sidebar:
    st.title("ℹ️ Gesture Guide")

    # Functionally Grouped Descriptions
    st.subheader("Mouse Pointer Control")
    st.markdown("""
    - **Move Cursor:** Point with your Index Finger.
    - **Left Click / Drag:** Pinch Thumb and Index finger.
    - **Right Click:** Pinch Thumb and Middle finger.
    - **Double Click:** Bring two Index Fingers together.
    """)

    st.subheader("System & Mode Control")
    st.markdown("""
    - **Cycle Modes:** Pinch Thumb and Pinky finger to switch between Scroll, Volume, Zoom, etc.
    - **Copy (Normal Mode):** Make a "Love" sign 🤘.
    - **Paste (Normal Mode):** Make a Closed Fist ✊.
    """)

    st.subheader("Mode-Specific Actions")
    st.markdown("""
    - **In Scroll Mode:**
        - **Scroll:** Move a Closed Fist ✊ up or down.
    - **In Volume Mode:**
        - **Change Volume:** Move a Closed Fist ✊ up or down.
    - **In Zoom Mode:**
        - **Zoom:** Move two Index Fingers 👆👆 apart or closer.
    - **In Enter/Backspace Mode:**
        - **Press Enter:** Make a "Love" sign 🤘.
        - **Press Backspace:** Make a Closed Fist ✊.
    """)


# ===================================================================
# --- Main UI Layout ---
# ===================================================================
# Centered Title using Markdown
st.markdown("<h1 style='text-align: center;'>👋 Dynamic Hand Gesture Navigation 👋</h1>", unsafe_allow_html=True)

st.markdown("---")

st.markdown("")
st.markdown("")

# Create columns to constrain the video feed size
# The middle column will be twice the size of the side columns, making the video feed smaller.
col_left_padding, col_main_content, col_right_padding = st.columns([1, 2, 1])


with col_main_content:
    # --- Video Feed and Buttons ---
    video_placeholder = st.empty()

    # --- Side-by-side Start/Stop Buttons ---
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Start Navigation", type="primary", use_container_width=True):
            st.session_state.is_running = True

    with col_btn2:
        if st.button("Stop Navigation", use_container_width=True):
            st.session_state.is_running = False

# ===================================================================
# --- Backend Logic & Initialization ---
# ===================================================================

# --- Initialize MediaPipe and PyAutoGUI ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

screen_width, screen_height = pyautogui.size()
pyautogui.FAILSAFE = False

# --- Cursor Control Parameters ---
CAM_ACTIVE_X_MIN = 0.25
CAM_ACTIVE_X_MAX = 0.75
CAM_ACTIVE_Y_MIN = 0.25
CAM_ACTIVE_Y_MAX = 0.75

# --- Smoothing and Cooldown (Hardcoded original values) ---
prev_x, prev_y = 0, 0
smoothing_factor = 0.2
action_cooldown_time = 0.65
last_action_time = 0
gesture_label = ""

# --- Gesture Specific Variables & Thresholds (Hardcoded original values) ---
is_pinching_for_action = False
pinch_start_time = 0
pinch_action_threshold = 0.045
PINCH_DRAG_ACTIVATION_DURATION = 0.3
is_dragging = False

MIDDLE_THUMB_PINCH_THRESHOLD = 0.04

# Mode states
current_active_mode = "NONE"
is_actively_zooming = False
initial_two_hand_zoom_dist = None
last_zoom_scroll_time = 0
ZOOM_SCROLL_COOLDOWN = 0.08
ZOOM_SENSITIVITY_THRESHOLD = 0.02
ZOOM_SCROLL_MULTIPLIER = 20

PINKY_THUMB_PINCH_THRESHOLD = 0.05
DOUBLE_CLICK_TWO_INDEX_THRESHOLD = 0.04

LOVE_SIGN_FINGER_STRAIGHT_THRESHOLD = 0.05
LOVE_SIGN_FINGER_CURLED_Y_FACTOR = 0.95
CLOSED_FIST_THUMB_ACROSS_DIST = 0.07

prev_fist_y_for_scroll = None
last_scroll_action_time = 0
FIST_Y_SENSITIVITY_FOR_SCROLL = 0.02
SCROLL_STEP_AMOUNT = 50
SCROLL_COOLDOWN = 0.05

prev_fist_y_for_volume = None
last_volume_change_time = 0
FIST_Y_SENSITIVITY_FOR_VOLUME = 0.025
VOLUME_CHANGE_STEP_COUNT = 5
VOLUME_CHANGE_COOLDOWN = 0.1


# --- Core Detection Functions from your script ---
def calculate_landmark_distance(lm1, lm2):
    if lm1 is None or lm2 is None: return float('inf')
    return math.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2 + (lm1.z - lm2.z)**2)

def get_landmark(landmarks_list, landmark_id):
    if landmarks_list and 0 <= landmark_id < len(landmarks_list):
        return landmarks_list[landmark_id]
    return None

def is_finger_extended_and_pointing_up(landmarks_list_param, tip_id, pip_id, mcp_id, straight_dist_thresh):
    tip = get_landmark(landmarks_list_param, tip_id)
    pip = get_landmark(landmarks_list_param, pip_id)
    mcp = get_landmark(landmarks_list_param, mcp_id)
    if tip and pip and mcp:
        is_pointing_up = tip.y < pip.y and pip.y < mcp.y
        is_straight = calculate_landmark_distance(tip, mcp) > straight_dist_thresh
        return is_straight and is_pointing_up
    return False

def is_finger_curled(landmarks_list_param, tip_id, pip_id):
    tip = get_landmark(landmarks_list_param, tip_id)
    pip = get_landmark(landmarks_list_param, pip_id)
    if tip and pip:
        return tip.y > pip.y
    return False

def check_love_sign_gesture(hand_landmarks_list_param):
    index_ok = is_finger_extended_and_pointing_up(hand_landmarks_list_param, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.INDEX_FINGER_MCP, LOVE_SIGN_FINGER_STRAIGHT_THRESHOLD)
    middle_ok = is_finger_curled(hand_landmarks_list_param, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
    ring_ok = is_finger_curled(hand_landmarks_list_param, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP)
    pinky_ok = is_finger_extended_and_pointing_up(hand_landmarks_list_param, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP, mp_hands.HandLandmark.PINKY_MCP, LOVE_SIGN_FINGER_STRAIGHT_THRESHOLD)
    thumb_tip = get_landmark(hand_landmarks_list_param, mp_hands.HandLandmark.THUMB_TIP)
    thumb_mcp = get_landmark(hand_landmarks_list_param, mp_hands.HandLandmark.THUMB_MCP)
    thumb_ok = True
    if thumb_tip and thumb_mcp:
        if calculate_landmark_distance(thumb_tip, thumb_mcp) < 0.045: thumb_ok = False
    return index_ok and middle_ok and ring_ok and pinky_ok and thumb_ok

def check_closed_fist_gesture(hand_landmarks_list_param):
    curled_count = 0
    for tip_id, pip_id in [(mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
                          (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
                          (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
                          (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)]:
        if is_finger_curled(hand_landmarks_list_param, tip_id, pip_id):
            curled_count += 1
    return curled_count >= 3


# --- Streamlit App Running Logic ---
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

if not st.session_state.is_running:
    with col_main_content:
        st.warning("Navigation is stopped. Click 'Start Navigation' to begin.")
    st.stop()

cap = cv2.VideoCapture(0)

# ===================================================================
# --- Main Application Loop ---
# This contains the full, unmodified logic from your script
# ===================================================================
while cap.isOpened() and st.session_state.is_running:
    success, image = cap.read()
    if not success:
        with col_main_content:
            st.warning("Cannot open camera. Please ensure your webcam is active.")
        st.session_state.is_running = False
        break

    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_image.flags.writeable = False
    results = hands.process(rgb_image)
    image.flags.writeable = True

    img_h, img_w, _ = image.shape
    current_time = time.time()
    gesture_label = ""
    discrete_action_performed_this_frame = False
    continuous_action_active_this_frame = False

    if results.multi_hand_landmarks:
        num_hands = len(results.multi_hand_landmarks)
        primary_hand_landmarks = results.multi_hand_landmarks[0].landmark

        for hand_lms_iter in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_lms_iter, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())

        index_tip_primary = get_landmark(primary_hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP)
        thumb_tip = get_landmark(primary_hand_landmarks, mp_hands.HandLandmark.THUMB_TIP)
        middle_finger_tip = get_landmark(primary_hand_landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP)
        ring_finger_tip = get_landmark(primary_hand_landmarks, mp_hands.HandLandmark.RING_FINGER_TIP)
        pinky_tip = get_landmark(primary_hand_landmarks, mp_hands.HandLandmark.PINKY_TIP)
        wrist_primary = get_landmark(primary_hand_landmarks, mp_hands.HandLandmark.WRIST)

        can_evaluate_all_gestures = (thumb_tip and index_tip_primary and middle_finger_tip and ring_finger_tip and pinky_tip and wrist_primary)

        if can_evaluate_all_gestures:
            # --- PRIORITY 0: Mode Toggle (Pinky-Thumb) ---
            if (current_time - last_action_time > action_cooldown_time) and not is_dragging and not is_pinching_for_action:
                dist_pinky_thumb = calculate_landmark_distance(thumb_tip, pinky_tip)
                if dist_pinky_thumb < PINKY_THUMB_PINCH_THRESHOLD:
                    dist_idx_thumb_disamb = calculate_landmark_distance(thumb_tip, index_tip_primary)
                    dist_mid_thumb_disamb = calculate_landmark_distance(thumb_tip, middle_finger_tip)
                    dist_ring_thumb_disamb = calculate_landmark_distance(thumb_tip, ring_finger_tip)
                    if (dist_idx_thumb_disamb > pinch_action_threshold * 1.1) and (dist_mid_thumb_disamb > MIDDLE_THUMB_PINCH_THRESHOLD * 1.1) and (dist_ring_thumb_disamb > MIDDLE_THUMB_PINCH_THRESHOLD * 1.1):
                        if current_active_mode == "NONE": current_active_mode = "SCROLL"; gesture_label = "Scroll Mode ARMED"
                        elif current_active_mode == "SCROLL": current_active_mode = "ENTER_BACKSPACE"; gesture_label = "Enter/Backspace Mode ARMED"; prev_fist_y_for_scroll = None
                        elif current_active_mode == "ENTER_BACKSPACE": current_active_mode = "ZOOM"; gesture_label = "Zoom Mode ARMED"
                        elif current_active_mode == "ZOOM": current_active_mode = "VOLUME"; gesture_label = "Volume Mode ARMED"; is_actively_zooming = False; initial_two_hand_zoom_dist = None; prev_fist_y_for_volume = None
                        elif current_active_mode == "VOLUME": current_active_mode = "NONE"; gesture_label = "All Modes OFF"; prev_fist_y_for_volume = None
                        last_action_time = current_time; discrete_action_performed_this_frame = True

            # --- PRIORITY 1: Two-Hand Double Click ---
            if num_hands == 2 and not discrete_action_performed_this_frame and (current_time - last_action_time > action_cooldown_time):
                hand1_lm = results.multi_hand_landmarks[0].landmark
                hand2_lm = results.multi_hand_landmarks[1].landmark
                h1_idx_tip = get_landmark(hand1_lm, mp_hands.HandLandmark.INDEX_FINGER_TIP)
                h2_idx_tip = get_landmark(hand2_lm, mp_hands.HandLandmark.INDEX_FINGER_TIP)
                if h1_idx_tip and h2_idx_tip:
                    if calculate_landmark_distance(h1_idx_tip, h2_idx_tip) < DOUBLE_CLICK_TWO_INDEX_THRESHOLD:
                        pyautogui.doubleClick(); gesture_label = "Double Click (2H Idx)"
                        last_action_time = current_time; discrete_action_performed_this_frame = True
                        if is_actively_zooming: is_actively_zooming = False; initial_two_hand_zoom_dist = None

            # --- PRIORITY 2: Handle Active Modes ---
            if not discrete_action_performed_this_frame:
                is_fist_now = check_closed_fist_gesture(primary_hand_landmarks)

                if current_active_mode == "SCROLL":
                    if is_fist_now:
                        continuous_action_active_this_frame = True
                        current_fist_y = wrist_primary.y
                        if prev_fist_y_for_scroll is None: prev_fist_y_for_scroll = current_fist_y; gesture_label = "Scroll Fist Ready"
                        elif (current_time - last_scroll_action_time > SCROLL_COOLDOWN):
                            delta_y = current_fist_y - prev_fist_y_for_scroll
                            if abs(delta_y) > FIST_Y_SENSITIVITY_FOR_SCROLL:
                                scroll_dir = -1 if delta_y < 0 else 1
                                pyautogui.scroll(scroll_dir * SCROLL_STEP_AMOUNT)
                                gesture_label = "Scroll Up" if scroll_dir > 0 else "Scroll Down"
                                last_scroll_action_time = current_time; prev_fist_y_for_scroll = current_fist_y
                        if not gesture_label: gesture_label = "Scroll Control Active"
                    else:
                        prev_fist_y_for_scroll = None
                        if not gesture_label: gesture_label = "Scroll Mode ON (Show Fist)"

                elif current_active_mode == "ENTER_BACKSPACE":
                    is_love_sign = check_love_sign_gesture(primary_hand_landmarks)
                    if is_love_sign:
                        pyautogui.press('enter'); gesture_label = "Enter"
                        last_action_time = current_time; discrete_action_performed_this_frame = True
                    elif is_fist_now:
                        pyautogui.press('backspace'); gesture_label = "Backspace"
                        last_action_time = current_time; discrete_action_performed_this_frame = True
                    elif not gesture_label: gesture_label = "Enter/Backspace Mode ON"

                elif current_active_mode == "VOLUME":
                    if is_fist_now:
                        continuous_action_active_this_frame = True
                        current_fist_y = wrist_primary.y
                        if prev_fist_y_for_volume is None: prev_fist_y_for_volume = current_fist_y; gesture_label = "Volume Fist Ready"
                        elif (current_time - last_volume_change_time > VOLUME_CHANGE_COOLDOWN):
                            delta_y = current_fist_y - prev_fist_y_for_volume
                            if abs(delta_y) > FIST_Y_SENSITIVITY_FOR_VOLUME:
                                if delta_y < 0:
                                    for _ in range(VOLUME_CHANGE_STEP_COUNT): pyautogui.press('volumeup')
                                    gesture_label = "Volume UP"
                                else:
                                    for _ in range(VOLUME_CHANGE_STEP_COUNT): pyautogui.press('volumedown')
                                    gesture_label = "Volume DOWN"
                                last_volume_change_time = current_time; prev_fist_y_for_volume = current_fist_y
                        if not gesture_label: gesture_label = "Volume Control Active"
                    else:
                        prev_fist_y_for_volume = None
                        if not gesture_label: gesture_label = "Volume Mode ON (Show Fist)"

                elif current_active_mode == "ZOOM":
                    if num_hands == 2:
                        hand1_lm_zoom = results.multi_hand_landmarks[0].landmark
                        hand2_lm_zoom = results.multi_hand_landmarks[1].landmark
                        h1_idx_tip_for_zoom = get_landmark(hand1_lm_zoom, mp_hands.HandLandmark.INDEX_FINGER_TIP)
                        h2_idx_tip_for_zoom = get_landmark(hand2_lm_zoom, mp_hands.HandLandmark.INDEX_FINGER_TIP)
                        if h1_idx_tip_for_zoom and h2_idx_tip_for_zoom:
                            current_idx_dist = calculate_landmark_distance(h1_idx_tip_for_zoom, h2_idx_tip_for_zoom)
                            if not is_actively_zooming:
                                if current_idx_dist > DOUBLE_CLICK_TWO_INDEX_THRESHOLD * 1.5:
                                    is_actively_zooming = True; initial_two_hand_zoom_dist = current_idx_dist; gesture_label = "Zooming Active (Idx)"
                            if is_actively_zooming and initial_two_hand_zoom_dist is not None:
                                delta_dist_zoom = current_idx_dist - initial_two_hand_zoom_dist
                                if abs(delta_dist_zoom) > ZOOM_SENSITIVITY_THRESHOLD and (current_time - last_zoom_scroll_time > ZOOM_SCROLL_COOLDOWN):
                                    zoom_direction = 1 if delta_dist_zoom > 0 else -1
                                    scroll_amount = zoom_direction * ZOOM_SCROLL_MULTIPLIER
                                    pyautogui.keyDown('ctrl'); pyautogui.scroll(scroll_amount); pyautogui.keyUp('ctrl')
                                    gesture_label = "Zoom In (Idx)" if zoom_direction > 0 else "Zoom Out (Idx)"
                                    last_zoom_scroll_time = current_time; initial_two_hand_zoom_dist = current_idx_dist
                                elif not gesture_label: gesture_label = "Zooming (Idx)..."
                            if is_actively_zooming: continuous_action_active_this_frame = True
                        elif is_actively_zooming:
                            is_actively_zooming = False; initial_two_hand_zoom_dist = None; gesture_label = "Zoom Paused (Idx LMs)"
                    elif not gesture_label: gesture_label = "Zoom Armed (Need 2 Hands)"

                # --- PRIORITY 3: Universal Discrete Pinches and Clicks (if no mode action) ---
                if current_active_mode == "NONE" and not discrete_action_performed_this_frame:
                    if (current_time - last_action_time > action_cooldown_time):
                        is_love_sign = check_love_sign_gesture(primary_hand_landmarks)
                        is_fist_for_paste = check_closed_fist_gesture(primary_hand_landmarks)
                        if is_love_sign:
                            pyautogui.hotkey('ctrl', 'c'); gesture_label = "Copy (Ctrl+C)"
                            last_action_time = current_time; discrete_action_performed_this_frame = True
                        elif is_fist_for_paste:
                            pyautogui.hotkey('ctrl', 'v'); gesture_label = "Paste (Ctrl+V)"
                            last_action_time = current_time; discrete_action_performed_this_frame = True

                if not discrete_action_performed_this_frame and not continuous_action_active_this_frame:
                    # Middle-Thumb Right Click
                    dist_thumb_middle = calculate_landmark_distance(thumb_tip, middle_finger_tip)
                    if dist_thumb_middle < MIDDLE_THUMB_PINCH_THRESHOLD and (calculate_landmark_distance(thumb_tip, index_tip_primary) > pinch_action_threshold * 1.05) and (calculate_landmark_distance(thumb_tip, pinky_tip) > PINKY_THUMB_PINCH_THRESHOLD * 1.05 if pinky_tip else True) and not is_pinching_for_action and not is_dragging and (current_time - last_action_time > action_cooldown_time):
                        pyautogui.rightClick(); gesture_label = "Right Click (Middle)"; last_action_time = current_time
                        discrete_action_performed_this_frame = True

                # Index-Thumb Pinch
                if not discrete_action_performed_this_frame and not continuous_action_active_this_frame:
                    is_clearly_index_pinch = (calculate_landmark_distance(thumb_tip, index_tip_primary) < pinch_action_threshold and calculate_landmark_distance(thumb_tip, middle_finger_tip) > MIDDLE_THUMB_PINCH_THRESHOLD * 1.05 and (calculate_landmark_distance(thumb_tip, pinky_tip) > PINKY_THUMB_PINCH_THRESHOLD * 1.05 if pinky_tip else True))
                    if is_pinching_for_action:
                        if is_clearly_index_pinch:
                            if not is_dragging and (current_time - pinch_start_time >= PINCH_DRAG_ACTIVATION_DURATION):
                                is_dragging = True; pyautogui.mouseDown(button='left'); gesture_label = "Drag Mode ON"; last_action_time = current_time
                            elif is_dragging: gesture_label = "Dragging..."
                            else: gesture_label = "Pinch Held (Idx)"
                        else:
                            if is_dragging:
                                is_dragging = False; pyautogui.mouseUp(button='left'); gesture_label = "Drag Mode OFF"; last_action_time = current_time
                            elif (current_time - last_action_time > action_cooldown_time):
                                pyautogui.click(); gesture_label = "Left Click"
                                last_action_time = current_time; discrete_action_performed_this_frame = True
                            is_pinching_for_action = False; pinch_start_time = 0; is_dragging = False
                    elif is_clearly_index_pinch and (current_time - last_action_time > action_cooldown_time / 2.5):
                        is_pinching_for_action = True; pinch_start_time = current_time; gesture_label = "Pinch Detected (Idx)"

        # --- MOUSE MOVEMENT ---
        if index_tip_primary:
            if not continuous_action_active_this_frame and not is_actively_zooming:
                active_range_x = CAM_ACTIVE_X_MAX - CAM_ACTIVE_X_MIN
                active_range_y = CAM_ACTIVE_Y_MAX - CAM_ACTIVE_Y_MIN
                norm_hand_x = (index_tip_primary.x - CAM_ACTIVE_X_MIN) / (active_range_x if active_range_x > 1e-5 else 1.0)
                norm_hand_y = (index_tip_primary.y - CAM_ACTIVE_Y_MIN) / (active_range_y if active_range_y > 1e-5 else 1.0)
                norm_hand_x = max(0.0, min(1.0, norm_hand_x))
                norm_hand_y = max(0.0, min(1.0, norm_hand_y))
                cursor_target_x = int(norm_hand_x * screen_width)
                cursor_target_y = int(norm_hand_y * screen_height)
                current_x = prev_x + (cursor_target_x - prev_x) * smoothing_factor
                current_y = prev_y + (cursor_target_y - prev_y) * smoothing_factor
                pyautogui.moveTo(current_x, current_y, duration=0)
                prev_x, prev_y = current_x, current_y
            elif continuous_action_active_this_frame or is_actively_zooming:
                current_mouse_x, current_mouse_y = pyautogui.position()
                prev_x, prev_y = current_mouse_x, current_mouse_y

        if not gesture_label:
            if is_actively_zooming: gesture_label = "Zooming (Idx)..."
            elif current_active_mode == "ZOOM": gesture_label = "Zoom Armed (Use Idx)"
            elif current_active_mode == "VOLUME":
                if not continuous_action_active_this_frame: gesture_label = "Volume Mode ON (Fist U/D)"
            elif current_active_mode == "SCROLL":
                if not continuous_action_active_this_frame: gesture_label = "Scroll Mode ON (Fist U/D)"
            elif current_active_mode == "ENTER_BACKSPACE": gesture_label = "Enter/Backspace Mode ON"
            elif is_dragging: gesture_label = "Dragging..."
            elif is_pinching_for_action: gesture_label = "Pinch Held (Idx)"
            else: gesture_label = "Mouse Moving / Ready"

    else:
        prev_fist_y_for_scroll = None; prev_fist_y_for_volume = None
        if current_active_mode == "ZOOM" and is_actively_zooming: is_actively_zooming = False; initial_two_hand_zoom_dist = None
        if is_dragging: pyautogui.mouseUp(button='left'); is_dragging = False
        is_pinching_for_action = False; pinch_start_time = 0
        if current_active_mode == "ZOOM": gesture_label = "Zoom Armed (Hand Lost)"
        elif current_active_mode == "VOLUME": gesture_label = "Volume Mode ON (Hand Lost)"
        elif current_active_mode == "SCROLL": gesture_label = "Scroll Mode ON (Hand Lost)"
        elif current_active_mode == "ENTER_BACKSPACE": gesture_label = "Enter/Backspace Mode ON (Hand Lost)"
        else: gesture_label = "No Hand / Ready"

    # Draw gesture label on the OpenCV image
    cv2.putText(image, gesture_label, (20, img_h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

    # Display the video feed
    video_placeholder.image(image, channels="BGR", use_container_width=True)


# --- Cleanup ---
cap.release()
hands.close()
if not st.session_state.get('is_running', False):
    with col_main_content:
        st.success("Navigation has been stopped.")