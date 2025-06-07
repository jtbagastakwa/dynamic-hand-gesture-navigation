import cv2
import mediapipe as mp
import pyautogui
import time
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Get screen dimensions
screen_width, screen_height = pyautogui.size()
pyautogui.FAILSAFE = False

# --- Cursor Control Parameters ---
CAM_ACTIVE_X_MIN = 0.25 
CAM_ACTIVE_X_MAX = 0.75 
CAM_ACTIVE_Y_MIN = 0.25 
CAM_ACTIVE_Y_MAX = 0.75 
if not (0 <= CAM_ACTIVE_X_MIN < CAM_ACTIVE_X_MAX <= 1 and \
        0 <= CAM_ACTIVE_Y_MIN < CAM_ACTIVE_Y_MAX <= 1):
    print("ERROR: Invalid CAM_ACTIVE_... parameters. Reverting to full camera range.")
    CAM_ACTIVE_X_MIN, CAM_ACTIVE_X_MAX = 0.0, 1.0
    CAM_ACTIVE_Y_MIN, CAM_ACTIVE_Y_MAX = 0.0, 1.0

# --- Smoothing and Cooldown ---
prev_x, prev_y = 0, 0
smoothing_factor = 0.2
action_cooldown_time = 0.65 
last_action_time = 0 
gesture_label = ""

# --- Gesture Specific Variables & Thresholds ---
is_pinching_for_action = False 
pinch_start_time = 0            
pinch_action_threshold = 0.045 
PINCH_DRAG_ACTIVATION_DURATION = 0.3 
is_dragging = False 

MIDDLE_THUMB_PINCH_THRESHOLD = 0.04 

# Mode states: "NONE", "SCROLL", "ENTER_BACKSPACE", "ZOOM", "VOLUME"
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
        return tip.y > pip.y # Tip is below the middle knuckle
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

cap = cv2.VideoCapture(0) # Corrected camera index to 0
if not cap.isOpened():
    print("Cannot open camera"); exit()

print("Dynamic Hand Gesture Navigation Started. Press 'ESC' to quit.")
print(f"Mouse active X range: {CAM_ACTIVE_X_MIN*100:.0f}%-{CAM_ACTIVE_X_MAX*100:.0f}% of camera view width")
print(f"Mouse active Y range: {CAM_ACTIVE_Y_MIN*100:.0f}%-{CAM_ACTIVE_Y_MAX*100:.0f}% of camera view height")
print("\nGestures:")
print("- Index Finger (One Hand): Moves Mouse Cursor")
print("- Pinch (Thumb-Pinky): Cycle Mode (None -> Scroll -> Enter/Backspace -> Zoom -> Volume -> None)") 
print("- Two Index Fingers Meet: Double Click") 
print("- Two Index Fingers Apart/Closer (Zoom Mode ON): Zoom In/Out") 
print("- Closed Fist (Scroll Mode ON): Move Up/Down for Scroll")
print("- Closed Fist (Volume Mode ON): Move Up/Down for Volume Up/Down")
print("- Love Sign (Enter/Backspace Mode ON): Press Enter")
print("- Closed Fist (Enter/Backspace Mode ON): Press Backspace")
print("- Love Sign (Normal Mode): Copy (Ctrl+C)")
print("- Closed Fist (Normal Mode): Paste (Ctrl+V)")
print("- Pinch (Thumb-Middle, ANY Mode): Right Click")
print("- Pinch (Thumb-Index, ANY Mode): Left Click or Drag")

while cap.isOpened():
    success, image = cap.read()
    if not success: 
        continue

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
            if (current_time - last_action_time > action_cooldown_time) and \
               not is_dragging and not is_pinching_for_action: 
                dist_pinky_thumb = calculate_landmark_distance(thumb_tip, pinky_tip)
                if dist_pinky_thumb < PINKY_THUMB_PINCH_THRESHOLD:
                    dist_idx_thumb_disamb = calculate_landmark_distance(thumb_tip, index_tip_primary)
                    dist_mid_thumb_disamb = calculate_landmark_distance(thumb_tip, middle_finger_tip)
                    dist_ring_thumb_disamb = calculate_landmark_distance(thumb_tip, ring_finger_tip)
                    if (dist_idx_thumb_disamb > pinch_action_threshold * 1.1) and \
                       (dist_mid_thumb_disamb > MIDDLE_THUMB_PINCH_THRESHOLD * 1.1) and \
                       (dist_ring_thumb_disamb > MIDDLE_THUMB_PINCH_THRESHOLD * 1.1): 
                        if current_active_mode == "NONE":
                            current_active_mode = "SCROLL"
                            gesture_label = "Scroll Mode ARMED"
                        elif current_active_mode == "SCROLL":
                            current_active_mode = "ENTER_BACKSPACE"
                            gesture_label = "Enter/Backspace Mode ARMED"
                            prev_fist_y_for_scroll = None 
                        elif current_active_mode == "ENTER_BACKSPACE":
                            current_active_mode = "ZOOM"
                            gesture_label = "Zoom Mode ARMED"
                        elif current_active_mode == "ZOOM":
                            current_active_mode = "VOLUME"
                            gesture_label = "Volume Mode ARMED"
                            if is_actively_zooming: is_actively_zooming = False; initial_two_hand_zoom_dist = None
                            prev_fist_y_for_volume = None 
                        elif current_active_mode == "VOLUME":
                            current_active_mode = "NONE"
                            gesture_label = "All Modes OFF"
                            prev_fist_y_for_volume = None
                        
                        print(f"{time.strftime('%H:%M:%S')} - Mode: {current_active_mode}")
                        last_action_time = current_time; discrete_action_performed_this_frame = True
            
            # --- PRIORITY 1: Two-Hand Double Click ---
            if num_hands == 2 and not discrete_action_performed_this_frame and \
               (current_time - last_action_time > action_cooldown_time) : 
                hand1_lm = results.multi_hand_landmarks[0].landmark
                hand2_lm = results.multi_hand_landmarks[1].landmark
                h1_idx_tip = get_landmark(hand1_lm, mp_hands.HandLandmark.INDEX_FINGER_TIP)
                h2_idx_tip = get_landmark(hand2_lm, mp_hands.HandLandmark.INDEX_FINGER_TIP)
                if h1_idx_tip and h2_idx_tip:
                    if calculate_landmark_distance(h1_idx_tip, h2_idx_tip) < DOUBLE_CLICK_TWO_INDEX_THRESHOLD:
                        pyautogui.doubleClick(); gesture_label = "Double Click (2H Idx)"; print(f"{time.strftime('%H:%M:%S')} - {gesture_label}")
                        last_action_time = current_time; discrete_action_performed_this_frame = True
                        if is_actively_zooming: is_actively_zooming = False; initial_two_hand_zoom_dist = None
            
            # --- PRIORITY 2: Handle Active Modes ---
            if not discrete_action_performed_this_frame:
                is_fist_now = check_closed_fist_gesture(primary_hand_landmarks)

                if current_active_mode == "SCROLL":
                    if is_fist_now:
                        continuous_action_active_this_frame = True
                        current_fist_y = wrist_primary.y
                        if prev_fist_y_for_scroll is None:
                            prev_fist_y_for_scroll = current_fist_y
                            gesture_label = "Scroll Fist Ready"
                        elif (current_time - last_scroll_action_time > SCROLL_COOLDOWN):
                            delta_y = current_fist_y - prev_fist_y_for_scroll
                            if abs(delta_y) > FIST_Y_SENSITIVITY_FOR_SCROLL:
                                scroll_dir = -1 if delta_y < 0 else 1
                                pyautogui.scroll(scroll_dir * SCROLL_STEP_AMOUNT)
                                gesture_label = "Scroll Up" if scroll_dir > 0 else "Scroll Down"
                                print(f"{time.strftime('%H:%M:%S')} - {gesture_label}")
                                last_scroll_action_time = current_time
                                prev_fist_y_for_scroll = current_fist_y
                        if not gesture_label: gesture_label = "Scroll Control Active"
                    else:
                        prev_fist_y_for_scroll = None 
                        if not gesture_label: gesture_label = "Scroll Mode ON (Show Fist)"
                
                elif current_active_mode == "ENTER_BACKSPACE":
                    is_love_sign = check_love_sign_gesture(primary_hand_landmarks)
                    if is_love_sign:
                        pyautogui.press('enter'); gesture_label = "Enter"; print(f"{time.strftime('%H:%M:%S')} - {gesture_label}")
                        last_action_time = current_time; discrete_action_performed_this_frame = True
                    elif is_fist_now:
                        pyautogui.press('backspace'); gesture_label = "Backspace"; print(f"{time.strftime('%H:%M:%S')} - {gesture_label}")
                        last_action_time = current_time; discrete_action_performed_this_frame = True
                    elif not gesture_label: gesture_label = "Enter/Backspace Mode ON"

                elif current_active_mode == "VOLUME":
                    if is_fist_now:
                        continuous_action_active_this_frame = True 
                        current_fist_y = wrist_primary.y
                        if prev_fist_y_for_volume is None:
                            prev_fist_y_for_volume = current_fist_y
                            gesture_label = "Volume Fist Ready"
                        elif (current_time - last_volume_change_time > VOLUME_CHANGE_COOLDOWN):
                            delta_y = current_fist_y - prev_fist_y_for_volume
                            if abs(delta_y) > FIST_Y_SENSITIVITY_FOR_VOLUME:
                                if delta_y < 0: 
                                    for _ in range(VOLUME_CHANGE_STEP_COUNT): pyautogui.press('volumeup')
                                    gesture_label = "Volume UP"
                                else: 
                                    for _ in range(VOLUME_CHANGE_STEP_COUNT): pyautogui.press('volumedown')
                                    gesture_label = "Volume DOWN"
                                print(f"{time.strftime('%H:%M:%S')} - {gesture_label}")
                                last_volume_change_time = current_time
                                prev_fist_y_for_volume = current_fist_y
                        if not gesture_label: gesture_label = "Volume Control Active"
                    else: 
                        prev_fist_y_for_volume = None 
                        if not gesture_label: gesture_label = "Volume Mode ON (Show Fist)"
                
                # CORRECTED LOGIC BLOCK: Added the missing "elif" for ZOOM mode
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
                                    is_actively_zooming = True; initial_two_hand_zoom_dist = current_idx_dist
                                    gesture_label = "Zooming Active (Idx)"; print(f"{time.strftime('%H:%M:%S')} - Zoom Active")
                            if is_actively_zooming and initial_two_hand_zoom_dist is not None:
                                delta_dist_zoom = current_idx_dist - initial_two_hand_zoom_dist
                                if abs(delta_dist_zoom) > ZOOM_SENSITIVITY_THRESHOLD and \
                                   (current_time - last_zoom_scroll_time > ZOOM_SCROLL_COOLDOWN):
                                    zoom_direction = 1 if delta_dist_zoom > 0 else -1
                                    scroll_amount = zoom_direction * ZOOM_SCROLL_MULTIPLIER
                                    pyautogui.keyDown('ctrl'); pyautogui.scroll(scroll_amount); pyautogui.keyUp('ctrl')
                                    gesture_label = "Zoom In (Idx)" if zoom_direction > 0 else "Zoom Out (Idx)"
                                    print(f"{time.strftime('%H:%M:%S')} - {gesture_label}")
                                    last_zoom_scroll_time = current_time; initial_two_hand_zoom_dist = current_idx_dist
                                elif not gesture_label: gesture_label = "Zooming (Idx)..."
                            if is_actively_zooming : continuous_action_active_this_frame = True
                        elif is_actively_zooming: 
                             is_actively_zooming = False; initial_two_hand_zoom_dist = None; gesture_label = "Zoom Paused (Idx LMs)"; print(f"{time.strftime('%H:%M:%S')} - Zoom Paused (Idx LMs)")
                    elif not gesture_label: gesture_label = "Zoom Armed (Need 2 Hands)"
                
                # --- PRIORITY 3: Universal Discrete Pinches and Clicks (if no mode action) ---
                if current_active_mode == "NONE" and not discrete_action_performed_this_frame:
                    if (current_time - last_action_time > action_cooldown_time):
                        is_love_sign = check_love_sign_gesture(primary_hand_landmarks)
                        is_fist_for_paste = check_closed_fist_gesture(primary_hand_landmarks)
                        if is_love_sign:
                            pyautogui.hotkey('ctrl', 'c'); gesture_label = "Copy (Ctrl+C)"; print(f"{time.strftime('%H:%M:%S')} - {gesture_label}")
                            last_action_time = current_time; discrete_action_performed_this_frame = True
                        elif is_fist_for_paste:
                            pyautogui.hotkey('ctrl', 'v'); gesture_label = "Paste (Ctrl+V)"; print(f"{time.strftime('%H:%M:%S')} - {gesture_label}")
                            last_action_time = current_time; discrete_action_performed_this_frame = True

                if not discrete_action_performed_this_frame and not continuous_action_active_this_frame:
                    # Middle-Thumb Right Click (Now Universal)
                    dist_thumb_middle = calculate_landmark_distance(thumb_tip, middle_finger_tip)
                    if dist_thumb_middle < MIDDLE_THUMB_PINCH_THRESHOLD and \
                       (calculate_landmark_distance(thumb_tip, index_tip_primary) > pinch_action_threshold * 1.05) and \
                       (calculate_landmark_distance(thumb_tip, pinky_tip) > PINKY_THUMB_PINCH_THRESHOLD * 1.05 if pinky_tip else True) and \
                       not is_pinching_for_action and not is_dragging and \
                       (current_time - last_action_time > action_cooldown_time):
                        pyautogui.rightClick(); gesture_label = "Right Click (Middle)"; print(f"{time.strftime('%H:%M:%S')} - Right Click"); last_action_time = current_time
                        discrete_action_performed_this_frame = True 
                
                # Index-Thumb Pinch (Now Universal)
                if not discrete_action_performed_this_frame and not continuous_action_active_this_frame: 
                    is_clearly_index_pinch = (calculate_landmark_distance(thumb_tip, index_tip_primary) < pinch_action_threshold and \
                                             calculate_landmark_distance(thumb_tip, middle_finger_tip) > MIDDLE_THUMB_PINCH_THRESHOLD * 1.05 and \
                                             (calculate_landmark_distance(thumb_tip, pinky_tip) > PINKY_THUMB_PINCH_THRESHOLD * 1.05 if pinky_tip else True))
                    if is_pinching_for_action: 
                        if is_clearly_index_pinch: 
                            if not is_dragging and (current_time - pinch_start_time >= PINCH_DRAG_ACTIVATION_DURATION):
                                is_dragging = True; pyautogui.mouseDown(button='left'); gesture_label = "Drag Mode ON"; print(f"{time.strftime('%H:%M:%S')} - Drag ON"); last_action_time = current_time
                            elif is_dragging: gesture_label = "Dragging..."
                            else: gesture_label = "Pinch Held (Idx)"
                        else: 
                            if is_dragging:
                                is_dragging = False; pyautogui.mouseUp(button='left'); gesture_label = "Drag Mode OFF"; print(f"{time.strftime('%H:%M:%S')} - Drag OFF"); last_action_time = current_time
                            elif (current_time - last_action_time > action_cooldown_time): 
                                pyautogui.click(); gesture_label = "Left Click"; print(f"{time.strftime('%H:%M:%S')} - Left Click"); last_action_time = current_time
                                discrete_action_performed_this_frame = True
                            is_pinching_for_action = False; pinch_start_time = 0; is_dragging = False
                    elif is_clearly_index_pinch and (current_time - last_action_time > action_cooldown_time / 2.5) : 
                        is_pinching_for_action = True; pinch_start_time = current_time; gesture_label = "Pinch Detected (Idx)"
        
        # --- MOUSE MOVEMENT ---
        if index_tip_primary: 
            if not continuous_action_active_this_frame and not is_actively_zooming : 
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
                if continuous_action_active_this_frame : pass 
                else: gesture_label = "Volume Mode ON (Fist U/D)"
            elif current_active_mode == "SCROLL":
                if continuous_action_active_this_frame : pass
                else: gesture_label = "Scroll Mode ON (Fist U/D)"
            elif current_active_mode == "ENTER_BACKSPACE":
                gesture_label = "Enter/Backspace Mode ON"
            elif is_dragging: gesture_label = "Dragging..."
            elif is_pinching_for_action: gesture_label = "Pinch Held (Idx)"
            else: gesture_label = "Mouse Moving / Ready"
            
    else: 
        prev_fist_y_for_scroll = None 
        prev_fist_y_for_volume = None 
        if current_active_mode == "ZOOM" and is_actively_zooming: 
             is_actively_zooming = False; initial_two_hand_zoom_dist = None; print(f"{time.strftime('%H:%M:%S')} - Zoom Paused (Hands Lost)")
        if is_dragging: pyautogui.mouseUp(button='left'); is_dragging = False; print(f"{time.strftime('%H:%M:%S')} - Drag End (Hand Lost)")
        is_pinching_for_action = False; pinch_start_time = 0
        if current_active_mode == "ZOOM": gesture_label = "Zoom Armed (Hand Lost)"
        elif current_active_mode == "VOLUME": gesture_label = "Volume Mode ON (Hand Lost)"
        elif current_active_mode == "SCROLL": gesture_label = "Scroll Mode ON (Hand Lost)"
        elif current_active_mode == "ENTER_BACKSPACE": gesture_label = "Enter/Backspace Mode ON (Hand Lost)"
        else: gesture_label = "No Hand / Ready"

    text_color = (255, 255, 0) 
    if "Zooming (Idx)" in gesture_label or "Zoom In (Idx)" in gesture_label or "Zoom Out (Idx)" in gesture_label : text_color = (0,100,255) 
    elif "Zoom Armed" in gesture_label or "Zoom Paused" in gesture_label: text_color = (100,100,255)
    elif "Zoom Mode ARMED" in gesture_label : text_color = (0, 200, 255)
    elif "Zoom Mode DE-ARMED" in gesture_label: text_color = (0,150,200)
    elif "Volume Mode ON" in gesture_label or "Volume Control Active" in gesture_label or "Volume Fist Ready" in gesture_label : text_color = (255, 69, 0) 
    elif "Volume UP" in gesture_label or "Volume DOWN" in gesture_label : text_color = (255, 100, 0) 
    elif "Volume Mode OFF" in gesture_label: text_color = (210, 105, 30) 
    elif "Scroll Mode ON" in gesture_label or "Scroll Fist Ready" in gesture_label: text_color = (0, 190, 190) 
    elif "Scroll Up" in gesture_label or "Scroll Down" in gesture_label: text_color = (0, 255, 255) 
    elif "Enter/Backspace Mode ON" in gesture_label: text_color = (128, 0, 0) 
    elif "Enter" in gesture_label or "Backspace" in gesture_label : text_color = (255, 99, 71) 
    elif "Copy" in gesture_label: text_color = (0,128,0) 
    elif "Paste" in gesture_label: text_color = (128,0,0) 
    elif "Double Click (2H Idx)" in gesture_label: text_color = (255,100,0) 
    elif "Right Click" in gesture_label: text_color = (255,0,255) 
    elif "Mode ON" in gesture_label or "ing..." in gesture_label or "Click" in gesture_label : text_color = (0, 255, 255)
    elif "OFF" in gesture_label : text_color = (0, 255, 0)
    elif "Lost" in gesture_label or "No Hand" in gesture_label: text_color = (0, 0, 255)
    
    cv2.putText(image, gesture_label, (20, img_h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)
    cv2.imshow('Hand Gesture Navigation', image)

    if cv2.waitKey(5) & 0xFF == 27: # ESC
        if is_dragging: pyautogui.mouseUp(button='left')
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print("Navigation system stopped.")
