// Recycling classification code, adapted from "classify_images" example in the coralmicro lib
// Original license is displayed below, although the only remaining code from the original
// is the classification functionality


// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstring>
#include <vector>
#include <time.h>

#include "libs/base/i2c.h"
#include "libs/base/filesystem.h"
#include "libs/base/gpio.h"
// #include "libs/base/check.h"
#include "libs/base/led.h"
#include "libs/camera/camera.h"
#include "libs/rpc/rpc_http_server.h"
#include "libs/tensorflow/classification.h"
#include "libs/tensorflow/utils.h"
#include "libs/tpu/edgetpu_manager.h"
#include "libs/tpu/edgetpu_op.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"
#include "third_party/mjson/src/mjson.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_gpio.h"
#include "third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_gpc.h"
#include "third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_snvs_lp.h"

namespace coralmicro {
namespace {
const std::string kModelPath = "/models/garbage_model_F_edgetpu.tflite";
constexpr int kTensorArenaSize = 8 * 1024 * 1024;
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);
const uint8_t LCD_ADDR = 0x3E; // Address of the LCD controller
const uint8_t RGB_ADDR = 0x2D; // Address of the RGB backlight controller

enum waste_classes { // Classes of waste that the model classifies
  CARDBOARD,
  GLASS,
  METAL,
  PAPER,
  PLASTIC,
  TRASH
};

/* 
ClassifyFromCamera

Take picture, perform inference and get classification results

*/

bool ClassifyFromCamera(tflite::MicroInterpreter* interpreter, int model_width, 
                        int model_height, bool bayered,
                        std::vector<tensorflow::Class>* results,
                        std::vector<uint8>* image) {
  CHECK(results != nullptr);
  CHECK(image != nullptr);
  auto* input_tensor = interpreter->input_tensor(0);

  // Note if the model is bayered, the raw data will not be rotated.
  auto format = bayered ? CameraFormat::kRaw : CameraFormat::kRgb;
  CameraFrameFormat fmt{format,
                        CameraFilterMethod::kBilinear,
                        CameraRotation::k270,
                        model_width,
                        model_height,
                        false,
                        image->data()};

  CameraTask::GetSingleton()->Trigger(); // Take image
  if (!CameraTask::GetSingleton()->GetFrame({fmt})) return false;

  TickType_t start, end;
  start = xTaskGetTickCount();
  

  std::memcpy(tflite::GetTensorData<uint8_t>(input_tensor), image->data(),
              image->size());
  if (interpreter->Invoke() != kTfLiteOk) return false;

  *results = tensorflow::GetClassificationResults(interpreter, 0.0f, 1);

  end = xTaskGetTickCount();
  printf("Inference time: %lu ticks\n", (unsigned long)(end - start));

  return true;
}

/* 
ClassifyConsole

Get image, perform inference, display to console

*/

void ClassifyConsole(tflite::MicroInterpreter* interpreter) {
  auto* input_tensor = interpreter->input_tensor(0);
  int model_height = input_tensor->dims->data[1];
  int model_width = input_tensor->dims->data[2];
  // If the model name includes "bayered", provide the raw datastream from the
  // camera.
  auto bayered = kModelPath.find("bayered") != std::string::npos;
  std::vector<uint8_t> image(
      model_width * model_height *
      /*channels=*/(bayered ? 1 : CameraFormatBpp(CameraFormat::kRgb)));
  std::vector<tensorflow::Class> results;
  if (ClassifyFromCamera(interpreter, model_width, model_height, bayered,
                         &results, &image)) {
    printf("%s\r\n", tensorflow::FormatClassificationOutput(results).c_str());
  } else {
    printf("Failed to classify image from camera.\r\n");
  }
}

/* 
Make_LCD_Payload

Takes in a string and returns a message that can be sent over I2C, specifically adding the control
byte that allows for multiple characters to be written to RAM at once

*/
uint8_t *Make_LCD_Payload(const char *text, size_t *len_out)
{
    size_t n = strlen(text);
    uint8_t *buf = (uint8_t *)malloc(n + 1);
    if (!buf) return NULL;

    buf[0] = 0x40;
    memcpy(buf + 1, text, n);

    if (len_out) *len_out = n + 1;
    return buf;                         // caller must free()
}

/* 
ConfigureLCD

Writes the proper commands over I2C to the display to properly start up the display.
This function specifically deals with configuring the LCD controller for the specific
display that it is using, as the controller IC can be used with many different display
formats.

*/
void ConfigureLCD(){
  auto config = I2cGetDefaultConfig(coralmicro::I2c::kI2c1);
  I2cInitController(config);

  uint8_t CONF_SIZE[2] = {0x00, 0x38};
  uint8_t SEL_INSTR_TABLE[2] = {0x00, 0x39};
  uint8_t SET_OSC_FREQ[2] = {0x00, 0x14};
  uint8_t SET_CONTR[2] = {0x00, 0x70};
  uint8_t PWR_ICON_CTRL[2] = {0x00, 0x56};
  uint8_t FOLLOWER_CTRL[2] = {0x00, 0x6C};
  uint8_t DISP_ON[2] = {0x00, 0x0C};
  uint8_t DISP_CLR[2] = {0x00, 0x01};
  
  CHECK(I2cControllerWrite(config, LCD_ADDR,CONF_SIZE,2));
  CHECK(I2cControllerWrite(config, LCD_ADDR,SEL_INSTR_TABLE,2));
  CHECK(I2cControllerWrite(config, LCD_ADDR,SET_OSC_FREQ,2));
  CHECK(I2cControllerWrite(config, LCD_ADDR,SET_CONTR,2));
  CHECK(I2cControllerWrite(config, LCD_ADDR,PWR_ICON_CTRL,2));
  CHECK(I2cControllerWrite(config, LCD_ADDR,FOLLOWER_CTRL,2));
  vTaskDelay(pdMS_TO_TICKS(200));
  CHECK(I2cControllerWrite(config, LCD_ADDR,CONF_SIZE,2));
  CHECK(I2cControllerWrite(config, LCD_ADDR,DISP_ON,2));
  CHECK(I2cControllerWrite(config, LCD_ADDR,DISP_CLR,2));
  vTaskDelay(pdMS_TO_TICKS(200));
}

/* 
ClearLCD

Writes the proper command over I2C to clear the entire screen. Requires a
delay to ensure the operation completes before writing characters to the 
display once more.

*/

void ClearLCD(){
  auto config = I2cGetDefaultConfig(coralmicro::I2c::kI2c1);
  I2cInitController(config);
  uint8_t DISP_CLR[2] = {0x00, 0x01};
  CHECK(I2cControllerWrite(config, LCD_ADDR,DISP_CLR,2));
  vTaskDelay(pdMS_TO_TICKS(200));
}

/* 
WriteLCD

Writes characters to the LCD at specific row, col positions without overwriting
the entire display.

*/

void WriteLCD(uint8_t row, uint8_t col,
  const char *text, size_t len)
{
  auto config = I2cGetDefaultConfig(coralmicro::I2c::kI2c1);
  I2cInitController(config);
  uint8_t base = (row == 0) ? 0x00 : 0x40;  
  uint8_t SET_CURSOR[2] = {0x00, 0x80 | (uint8_t)(base + col)};
  CHECK(I2cControllerWrite(config, LCD_ADDR,SET_CURSOR,2));

  size_t kTransferSize;
  uint8_t *text_to_write = Make_LCD_Payload(text, &kTransferSize);
  CHECK(I2cControllerWrite(config, LCD_ADDR,text_to_write,(int)kTransferSize));
  free(text_to_write);
}

/* 
WriteBacklight

Write a particular color to the backlight over I2C. Note that the address
of the controller may vary depending on the particular unit you receive.

*/

void WriteBacklight(uint8_t r, uint8_t g, uint8_t b){
  auto config = I2cGetDefaultConfig(coralmicro::I2c::kI2c1);
  I2cInitController(config);
  uint8_t SET_R[2] = {0x01, r};
  uint8_t SET_G[2] = {0x02, g};
  uint8_t SET_B[2] = {0x03, b};
  CHECK(I2cControllerWrite(config, RGB_ADDR,SET_R,2));
  CHECK(I2cControllerWrite(config, RGB_ADDR,SET_G,2));
  CHECK(I2cControllerWrite(config, RGB_ADDR,SET_B,2));
}

/* 
ClassifyLCD

Get image, perform inference, display to LCD with previous helper functions.

*/

void ClassifyLCD(tflite::MicroInterpreter* interpreter) {
  auto* input_tensor = interpreter->input_tensor(0);
  int model_height = input_tensor->dims->data[1];
  int model_width = input_tensor->dims->data[2];
  // If the model name includes "bayered", provide the raw datastream from the
  // camera.
  auto bayered = kModelPath.find("bayered") != std::string::npos;
  std::vector<uint8_t> image(
      model_width * model_height *
      /*channels=*/(bayered ? 1 : CameraFormatBpp(CameraFormat::kRgb)));
  std::vector<tensorflow::Class> results;
  if (ClassifyFromCamera(interpreter, model_width, model_height, bayered,
                         &results, &image)) {
    printf("%s\r\n", tensorflow::FormatClassificationOutput(results).c_str());
    if (!results.empty()) {
      ClearLCD();
      const auto& result = results[0];
      char str2[16];
      const char *str1 = "RECYCLING!";
      const char *str3 = "TRASH!";
      switch (result.id)
      {
        case CARDBOARD:
          sprintf(str2, "Cardboard: %.2f", result.score);
          WriteLCD(0, 3, str1, strlen(str1));
          WriteLCD(1, 1, str2, strlen(str2));
          WriteBacklight(0,255,0);
        break;
        case GLASS:
          sprintf(str2, "Glass: %.2f", result.score);
          WriteLCD(0, 3, str1, strlen(str1));
          WriteLCD(1, 2, str2, strlen(str2));
          WriteBacklight(0,0,255);
        break;
        case METAL:
          sprintf(str2, "Metal: %.2f", result.score);
          WriteLCD(0, 3, str1, strlen(str1));
          WriteLCD(1, 2, str2, strlen(str2));
          WriteBacklight(0,0,255);
        break;
        case PAPER:
          sprintf(str2, "Paper: %.2f", result.score);
          WriteLCD(0, 3, str1, strlen(str1));
          WriteLCD(1, 2, str2, strlen(str2));
          WriteBacklight(0,255,0);
        break;
        case PLASTIC:
          sprintf(str2, "Plastic: %.2f", result.score);
          WriteLCD(0, 3, str1, strlen(str1));
          WriteLCD(1, 1, str2, strlen(str2));
          WriteBacklight(0,0,255);
        break;
        case TRASH:
          sprintf(str2, "Trash: %.2f", result.score);
          WriteLCD(0, 5, str3, strlen(str1));
          WriteLCD(1, 2, str2, strlen(str2));
          WriteBacklight(255,0,0);
        break;
        default:
          sprintf(str2, "Trash: %.2f", result.score);
          WriteLCD(0, 5, str3, strlen(str1));
          WriteLCD(1, 2, str2, strlen(str2));
          WriteBacklight(255,0,0);
          break;
      }
      
    }

  } else {
    printf("Failed to classify image from camera.\r\n");
  }
}


[[noreturn]] void Main() {
  // Turn on Status LED to show the board is on.
  LedSet(Led::kStatus, true);

  // Read model file 
  std::vector<uint8_t> model;
  if (!LfsReadFile(kModelPath.c_str(), &model)) {
    printf("ERROR: Failed to load %s\r\n", kModelPath.c_str());
    vTaskSuspend(nullptr);
  }

  // Turn on TPU
  auto tpu_context = EdgeTpuManager::GetSingleton()->OpenDevice();
  if (!tpu_context) {
    printf("ERROR: Failed to get EdgeTpu context\r\n");
    vTaskSuspend(nullptr);
  }

  // Start up TFLite interpreter
  tflite::MicroErrorReporter error_reporter;
  tflite::MicroMutableOpResolver<1> resolver;
  resolver.AddCustom(kCustomOp, RegisterCustomOp());

  tflite::MicroInterpreter interpreter(tflite::GetModel(model.data()), resolver,
                                       tensor_arena, kTensorArenaSize,
                                       &error_reporter);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("ERROR: AllocateTensors() failed\r\n");
    vTaskSuspend(nullptr);
  }

  if (interpreter.inputs().size() != 1) {
    printf("ERROR: Model must have only one input tensor\r\n");
    vTaskSuspend(nullptr);
  }

  // Start camera
  CameraTask::GetSingleton()->SetPower(true);
  CameraTask::GetSingleton()->Enable(CameraMode::kTrigger);


  // Write to LCD to ask user to position waste item unerneath the device where the camera is pointing
  ConfigureLCD();
  const char *init_str_1 = "Place Item Under";
  WriteLCD(0, 0, init_str_1, strlen(init_str_1));
  const char *init_str_2 = "Device:";
  WriteLCD(1, 0, init_str_2, strlen(init_str_2));

  char buf[2];
  for(int i=3; i>0; i--){
    int len = snprintf(buf, sizeof(buf), "%d", i);
    WriteLCD(1, 8, buf, strlen(buf));
    vTaskDelay(pdMS_TO_TICKS(1000));
  }


  // Perform the inference and display result on LCD
  ClassifyLCD(&interpreter);

  // Wait for the user to see the result of classification for 3s
  vTaskDelay(pdMS_TO_TICKS(3000));

  // Disable power to all systems except the Secure Nonvolatile Storage domain (SNVS)
  SNVS->LPCR |= SNVS_LPCR_TOP_MASK;

  // Code beyond this point is unreachable, as the device is completely reset upon recovering from the SNVS shutdown

  GpioConfigureInterrupt(
      Gpio::kUserButton, GpioInterruptMode::kIntModeFalling,
      [handle = xTaskGetCurrentTaskHandle()]() { xTaskResumeFromISR(handle); },
      /*debounce_interval_us=*/50 * 1e3);


  while (true) {
  }
}
}  // namespace
}  // namespace coralmicro

extern "C" void app_main(void* param) {
  (void)param;
  coralmicro::Main();
}
