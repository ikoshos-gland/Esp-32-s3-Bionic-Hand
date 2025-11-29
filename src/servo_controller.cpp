/**
 * @file servo_controller.cpp
 * @brief Implementation of ServoController class for robotic hand control
 */

#include "servo_controller.h"

// ============================================================================
// GESTURE-TO-SERVO ANGLE MAPPING TABLE
// ============================================================================
//
// HOW TO MODIFY GESTURES:
// -----------------------
// Each gesture has its own INDEPENDENT array entry below.
// You can modify ANY gesture without affecting others.
//
// Example: To change "Open" gesture (Gesture 2):
//   1. Find line with "// Gesture 2: Open"
//   2. Change the 6 numbers: {10, 10, 0, 0, 0, 0} to your desired angles
//   3. Save and rebuild
//
// Even though Rest and Open have the same angles now, they are SEPARATE entries.
// Changing one will NOT affect the other.
//
// SERVO ORDER: [Thumb_Rotation, Thumb_Flex, Index, Middle, Ring, Pinky]
// ANGLE CONVENTION:
//   - 0° = Finger fully extended (open)
//   - 180° = Finger fully closed (fist)
//   - Valid range: 0-180 degrees
//
// GPIO PINS: 10, 11, 12, 13, 14, 47
// ============================================================================

const int ServoController::gesture_angles[NUM_GESTURES][NUM_SERVOS] = {
    // Gesture 0: No Movement - Fingers extended, neutral resting position
    {10,  10,   0,   0,   0,   0},

    // Gesture 1: Wrist Flexion - Neutral mid-position (flexing wrist)
    {45,  45,  45,  45,  45,  45},

    // Gesture 2: Wrist Extension - Fingers slightly extended
    {20,  20,  10,  10,  10,  10},

    // Gesture 3: Wrist Pronation - Thumb rotated inward
    {70,  30,  20,  20,  20,  20},

    // Gesture 4: Wrist Supination - Thumb rotated outward
    {30,  30,  20,  20,  20,  20},

    // Gesture 5: Chuck Grip - Thumb and index/middle touching (tripod pinch)
    {90, 100, 110, 110,  10,  10},

    // Gesture 6: Hand Open - Fully extended hand
    {10,  10,   0,   0,   0,   0}
};

// Constructor
ServoController::ServoController() {
    // Initialize state
    is_moving = false;
    interpolation_start_time = 0;
    interpolation_duration = DEFAULT_INTERPOLATION_DURATION_MS;

    // Initialize position arrays to zero
    for (int i = 0; i < NUM_SERVOS; i++) {
        current_positions[i] = 0;
        target_positions[i] = 0;
        start_positions[i] = 0;
    }
}

// Initialize servo controller
void ServoController::begin() {
    Serial.println("Servo Controller: Initializing...");

    // Attach all servos to their GPIO pins
    for (int i = 0; i < NUM_SERVOS; i++) {
        int channel = servos[i].attach(servo_pins[i]);

        if (channel == 0) {
            Serial.printf("ERROR: Failed to attach servo %d to GPIO %d\n", i, servo_pins[i]);
        } else {
            Serial.printf("✅ Servo %d attached to GPIO %d (channel %d)\n", i, servo_pins[i], channel);
        }
    }

    Serial.println("Servo Controller: Initialization complete");
}

// Move to home position
void ServoController::setHome() {
    const int home_angles[NUM_SERVOS] = {10, 10, 0, 0, 0, 0};

    Serial.println("Servo Controller: Moving to home position...");

    // Move immediately to home position (no interpolation)
    for (int i = 0; i < NUM_SERVOS; i++) {
        servos[i].write(home_angles[i]);
        current_positions[i] = home_angles[i];
        target_positions[i] = home_angles[i];
    }

    is_moving = false;
    Serial.println("Servo Controller: Home position reached");
}

// Move to gesture position
void ServoController::moveToGesture(int gesture_id) {
    // Bounds check
    if (gesture_id < 0 || gesture_id >= NUM_GESTURES) {
        Serial.printf("ERROR: Invalid gesture ID: %d (valid range: 0-%d)\n", gesture_id, NUM_GESTURES - 1);
        return;
    }

    // Get gesture angles
    const int* angles = gesture_angles[gesture_id];

    // Debug output
    const char* gesture_names[] = {"No Movement", "Wrist Flexion", "Wrist Extension",
                                   "Wrist Pronation", "Wrist Supination", "Chuck Grip", "Hand Open"};
    const char* servo_names[] = {"Thumb_Rot", "Thumb_Flex", "Index", "Middle", "Ring", "Pinky"};

    Serial.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Serial.printf("🎯 GESTURE DETECTED: %s (ID: %d)\n", gesture_names[gesture_id], gesture_id);
    Serial.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    Serial.println("Servo Movements:");
    for (int i = 0; i < NUM_SERVOS; i++) {
        int delta = angles[i] - current_positions[i];
        Serial.printf("  [%d] %-11s: %3d° → %3d° (Δ%+4d°)\n",
                     i, servo_names[i], current_positions[i], angles[i], delta);
    }
    Serial.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Initiate interpolated movement
    moveToAngles(angles, DEFAULT_INTERPOLATION_DURATION_MS);
}

// Move to custom angles
void ServoController::moveToAngles(const int angles[NUM_SERVOS], unsigned long duration_ms) {
    // Save start positions for interpolation (use current positions, not target)
    for (int i = 0; i < NUM_SERVOS; i++) {
        start_positions[i] = current_positions[i];
        target_positions[i] = clamp(angles[i], 0, 180);  // Clamp to valid servo range
    }

    // Initialize interpolation state
    interpolation_start_time = millis();
    interpolation_duration = duration_ms;
    is_moving = true;

    Serial.printf("Servo Controller: Starting movement (duration: %lums)\n", duration_ms);
}

// Non-blocking update method
void ServoController::update() {
    if (!is_moving) return;

    unsigned long current_time = millis();
    unsigned long elapsed = current_time - interpolation_start_time;

    // Check if movement complete
    if (elapsed >= interpolation_duration) {
        // Snap to final positions
        for (int i = 0; i < NUM_SERVOS; i++) {
            servos[i].write(target_positions[i]);
            current_positions[i] = target_positions[i];
        }
        is_moving = false;
        Serial.println("✅ Movement complete! All servos reached target positions.\n");
        return;
    }

    // Calculate interpolation progress (0.0 to 1.0)
    float progress = (float)elapsed / (float)interpolation_duration;

    // Update all servos using linear interpolation (LERP)
    for (int i = 0; i < NUM_SERVOS; i++) {
        int delta = target_positions[i] - start_positions[i];
        int new_position = start_positions[i] + (int)(delta * progress);

        servos[i].write(new_position);
        current_positions[i] = new_position;
    }
}

// Check if moving
bool ServoController::isMoving() const {
    return is_moving;
}

// Get current positions
void ServoController::getCurrentPositions(int positions[NUM_SERVOS]) const {
    for (int i = 0; i < NUM_SERVOS; i++) {
        positions[i] = current_positions[i];
    }
}

// Print debug status
void ServoController::printStatus() const {
    Serial.println("=== Servo Controller Status ===");
    Serial.printf("Moving: %s\n", is_moving ? "YES" : "NO");

    Serial.print("Current positions: [");
    for (int i = 0; i < NUM_SERVOS; i++) {
        Serial.printf("%d", current_positions[i]);
        if (i < NUM_SERVOS - 1) Serial.print(", ");
    }
    Serial.println("]");

    Serial.print("Target positions:  [");
    for (int i = 0; i < NUM_SERVOS; i++) {
        Serial.printf("%d", target_positions[i]);
        if (i < NUM_SERVOS - 1) Serial.print(", ");
    }
    Serial.println("]");

    if (is_moving) {
        unsigned long elapsed = millis() - interpolation_start_time;
        Serial.printf("Progress: %lu / %lu ms (%.1f%%)\n",
                     elapsed, interpolation_duration,
                     (float)elapsed / (float)interpolation_duration * 100.0);
    }

    Serial.println("===============================");
}

// Clamp helper
int ServoController::clamp(int value, int min_val, int max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}
