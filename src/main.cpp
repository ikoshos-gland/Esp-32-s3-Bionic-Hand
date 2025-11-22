/*Libraries*/
//Tensorflow model converted to C++
#include "modeldata.h"
//functions
#include "functions.h"

// MPU6050 sensor object (defined in functions.cpp)
extern Adafruit_MPU6050 mpu;

//Tensorflow custom library for ESP32 (from lib folder)
#include <TensorFlowLite_ESP32.h>
// Use the experimental micro API (which is in lib folder)
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Globals
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Create an area of memory for input, output, and intermediate arrays
constexpr int kTensorArenaSize = 15 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

int this_predict = -1;
int last_predict = -1;

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n\n=== STARTING SETUP ===");
  Serial.println("Initializing I2C...");

  // Initialize I2C for MPU6050
  Wire.begin(I2C_SDA, I2C_SCL);
  Serial.println("I2C initialized");

  // Initialize MPU6050 sensor
  Serial.println("Looking for MPU6050...");
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (1) {
      delay(10);
    }
  }
  Serial.println("MPU6050 Found!");

  // Set up TensorFlow Lite
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model
  model = tflite::GetModel(prosthetic_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema mismatch!");
    return;
  }

  // Pull in all operations
  static tflite::ops::micro::AllOpsResolver resolver;

  // Build interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  // Get input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Setup complete! Starting gesture classification...");
  Serial.println("==================================================");
}



void loop() {
  // Get sensor features
  input->data.f = prelim_collection();

  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed");
    return;
  }

  // Display output
  Serial.println("----------------------------------");
  Serial.println("MODEL OUTPUT (11 gestures):");
  Serial.print("  0:Rest=");
  Serial.print(output->data.f[0], 3);
  Serial.print(" | 1:Fist=");
  Serial.print(output->data.f[1], 3);
  Serial.print(" | 2:Open=");
  Serial.print(output->data.f[2], 3);
  Serial.print(" | 3:Point=");
  Serial.print(output->data.f[3], 3);
  Serial.print(" | 4:Victory=");
  Serial.print(output->data.f[4], 3);
  Serial.print(" | 5:OK=");
  Serial.println(output->data.f[5], 3);
  Serial.print("  6:ThumbUp=");
  Serial.print(output->data.f[6], 3);
  Serial.print(" | 7:ThumbDn=");
  Serial.print(output->data.f[7], 3);
  Serial.print(" | 8:Grasp=");
  Serial.print(output->data.f[8], 3);
  Serial.print(" | 9:Pinch=");
  Serial.print(output->data.f[9], 3);
  Serial.print(" | 10:WristFlex=");
  Serial.println(output->data.f[10], 3);

  // Find highest confidence gesture
  for (int i = 0; i < 11; i++) {
    if (output->data.f[i] > 0.8) this_predict = i;
  }

  // Skip if same as previous
  if(this_predict == last_predict){
    this_predict = -1;
  }

  // Gesture names
  const char* gesture_names[] = {"Rest", "Fist", "Open", "Point", "Victory",
                                  "OK", "ThumbUp", "ThumbDn", "Grasp", "Pinch", "WristFlex"};

  // Handle gesture
  switch(this_predict){
    case -1:
      break;
    case 0: case 1: case 2: case 3: case 4: case 5:
    case 6: case 7: case 8: case 9: case 10:
      Serial.print("GESTURE DETECTED: ");
      Serial.println(gesture_names[this_predict]);
      last_predict = this_predict;
      break;
    default:
      break;
  }
}
