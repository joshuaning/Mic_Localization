 /*
 * A simple hardware test which receives audio on the A2 analog pin
 * and sends it to the PWM (pin 3) output and DAC (A14 pin) output.
 *
 * This example code is in the public domain.
 */

#include <Audio.h>
#include <Wire.h>
#include <SD.h>
#include <SPI.h>
#include <SerialFlash.h>

// GUItool: begin automatically generated code

const int myInput = AUDIO_INPUT_LINEIN;

AudioInputI2S         audioInput;         // audio shield: mic or line-in
AudioOutputI2S        audioOutput;        // audio shield: headphones & line-out

// Create Audio connections between the components
// Route audio into the left and right filters
AudioConnection c1(audioInput, 0, audioOutput, 0);
AudioConnection c2(audioInput, 1, audioOutput, 1);
AudioControlSGTL5000 audioShield;

// GUItool: end automatically generated code


void setup() {
  // Audio connections require memory to work.  For more
  // detailed information, see the MemoryAndCpuUsage example
  AudioMemory(12);
  // Enable the audio shield, select input, and enable output
  audioShield.inputSelect(myInput);
  audioShield.enable();
  audioShield.volume(0.6);

}

void loop() {
  // Do nothing here.  The Audio flows automatically

  // When AudioInputAnalog is running, analogRead() must NOT be used.
}
