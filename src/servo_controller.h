/**
 * @file servo_controller.h
 * @brief Servo controller for ESP32-S3 robotic hand gesture control
 *
 * Controls 6 servos (Thumb_Rotation, Thumb_Flex, Index, Middle, Ring, Pinky)
 * based on 7 detected hand gestures with smooth interpolated transitions.
 *
 * Hardware Configuration:
 * - Servo 0 (Thumb Rotation): GPIO 10
 * - Servo 1 (Thumb Flex):     GPIO 11
 * - Servo 2 (Index):           GPIO 12
 * - Servo 3 (Middle):          GPIO 13
 * - Servo 4 (Ring):            GPIO 14
 * - Servo 5 (Pinky):           GPIO 47
 */

#ifndef SERVO_CONTROLLER_H
#define SERVO_CONTROLLER_H

#include <Arduino.h>
#include <ESP32Servo.h>

// Configuration constants
#define NUM_SERVOS 6
#define NUM_GESTURES 7
#define DEFAULT_INTERPOLATION_DURATION_MS 500

/**
 * @class ServoController
 * @brief Non-blocking servo controller with smooth interpolation
 *
 * Features:
 * - Smooth LERP interpolation between positions
 * - Non-blocking updates (millis() based timing)
 * - Gesture-to-angle mapping for 11 hand gestures
 * - Bounds checking (0-180°)
 * - Simultaneous movement of all servos
 */
class ServoController {
private:
    // Hardware
    Servo servos[NUM_SERVOS];
    const int servo_pins[NUM_SERVOS] = {10, 11, 12, 13, 14, 47};

    // State management
    int current_positions[NUM_SERVOS];      // Current servo angles (0-180°)
    int target_positions[NUM_SERVOS];       // Target servo angles (0-180°)
    bool is_moving;                         // True if interpolation in progress

    // Interpolation state
    unsigned long interpolation_start_time; // millis() when movement started
    unsigned long interpolation_duration;   // Duration of transition (ms)
    int start_positions[NUM_SERVOS];        // Starting angles for interpolation

    // Gesture-to-servo angle mapping (7 gestures × 6 servos)
    // Format: [Thumb_Rotation, Thumb_Flex, Index, Middle, Ring, Pinky]
    // Convention: 0° = fully extended (open), 180° = fully closed (fist)
    static const int gesture_angles[NUM_GESTURES][NUM_SERVOS];

    // Private helper methods

    /**
     * @brief Update servo positions during interpolation
     *
     * Uses linear interpolation (LERP) to smoothly transition from
     * start positions to target positions. Called from update().
     */
    void updateInterpolation();

    /**
     * @brief Clamp value to specified range
     * @param value Input value
     * @param min_val Minimum allowed value
     * @param max_val Maximum allowed value
     * @return Clamped value
     */
    int clamp(int value, int min_val, int max_val);

public:
    /**
     * @brief Constructor
     */
    ServoController();

    /**
     * @brief Initialize servo controller
     *
     * Attaches all 6 servos to their GPIO pins. Call once in setup().
     * Note: Initialize AFTER TFLite to ensure memory allocation succeeds.
     */
    void begin();

    /**
     * @brief Move servos to safe home position
     *
     * Home position: [10, 10, 0, 0, 0, 0] (fingers extended)
     * Moves immediately without interpolation.
     */
    void setHome();

    /**
     * @brief Move servos to match detected gesture
     *
     * @param gesture_id Gesture ID (0-6):
     *   0: No Movement, 1: Wrist Flexion, 2: Wrist Extension,
     *   3: Wrist Pronation, 4: Wrist Supination, 5: Chuck Grip, 6: Hand Open
     *
     * Triggers smooth interpolated movement over DEFAULT_INTERPOLATION_DURATION_MS.
     * If gesture_id is invalid, prints error and does nothing.
     */
    void moveToGesture(int gesture_id);

    /**
     * @brief Move servos to custom angles with specified duration
     *
     * @param angles Array of 6 servo angles (0-180°)
     * @param duration_ms Duration of transition in milliseconds
     *
     * Automatically clamps angles to 0-180° range.
     * Use for custom movements not in gesture mapping.
     */
    void moveToAngles(const int angles[NUM_SERVOS], unsigned long duration_ms);

    /**
     * @brief Non-blocking update method
     *
     * MUST be called regularly (every 10-20ms) to update servo positions
     * during interpolation. Call from:
     * - loop() for basic functionality
     * - prelim_collection() for smooth 10ms updates (recommended)
     *
     * Does nothing if no movement in progress.
     */
    void update();

    /**
     * @brief Check if servos are currently moving
     * @return true if interpolation in progress, false otherwise
     */
    bool isMoving() const;

    /**
     * @brief Get current servo positions
     * @param positions Output array (size NUM_SERVOS) to store current angles
     */
    void getCurrentPositions(int positions[NUM_SERVOS]) const;

    /**
     * @brief Print debug status to Serial
     *
     * Displays current positions, target positions, and movement status.
     */
    void printStatus() const;
};

#endif // SERVO_CONTROLLER_H
