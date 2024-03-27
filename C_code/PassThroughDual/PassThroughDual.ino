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
#include "bandpass_200.h"
#include "lowpass_200.h"


// GUItool: begin automatically generated code

const int myInput = AUDIO_INPUT_LINEIN;

AudioInputI2S         audioInput;         // audio shield: mic or line-in
AudioFilterFIR        BandpassL;
AudioFilterFIR        BandpassR;
AudioOutputI2S        audioOutput;        // audio shield: headphones & line-out
VAD                   vadLeft;

// Create Audio connections between the components
// Route audio into the left and right filters
//AudioConnection c1(audioInput, 0, audioOutput, 0);
//AudioConnection c2(audioInput, 1, audioOutput, 1);
AudioConnection c1(audioInput, 0, BandpassL, 0);
AudioConnection c2(audioInput, 1, BandpassR, 0);
AudioConnection c3(BandpassL, 0, vadLeft, 0);
AudioConnection c4(BandpassR, 0, audioOutput, 1);
AudioConnection c5(vadLeft, 0, audioOutput, 0);

AudioControlSGTL5000 audioShield;

// GUItool: end automatically generated code

struct fir_filter {
  short *coeffs;
  short num_coeffs;    // num_coeffs must be an even number, 4 or higher
};

// index of current filter. Start with the low pass.
//Change to 1 for bandpass
int start_idx = 0;
struct fir_filter fir_list[] = {
  {LP  , 200},   
  {BP  , 200},
  {NULL,   0}
};


void setup() {

  // Audio connections require memory to work.  For more
  // detailed information, see the MemoryAndCpuUsage example
  AudioMemory(100);
  Serial.begin(9600);
  // Enable the audio shield, select input, and enable output
  audioShield.inputSelect(myInput);
  audioShield.enable();
  audioShield.volume(0.6);
  BandpassL.begin(fir_list[start_idx].coeffs, fir_list[start_idx].num_coeffs);
  BandpassR.begin(fir_list[start_idx].coeffs, fir_list[start_idx].num_coeffs);
  vadLeft.begin((const short *)1, 2911495, 10000000);

}

void loop() {
  // Do nothing here.  The Audio flows automatically

  // When AudioInputAnalog is running, analogRead() must NOT be used.
}
