#include <Arduino.h>
#include <AudioStream.h> // github.com/PaulStoffregen/cores/blob/master/teensy4/AudioStream.h
#include "wiener.h"
#include <stdio.h>
#define LED 13
#define INTERVAL 3000

//Constants
const int sampleRate = 44100;
const int blockSize = 128;

//User defined constants
const int BufferSize = 0.5 * sampleRate / blockSize; 	// 0.5 seconds of audio
const int FRAME = int(sampleRate * 0.02); 		//20ms frame


// helper variables
static int count = 0;
static bool switcher = true;
int16_t buff[BufferSize * blockSize];
int counter = 0;
bool full = false;
const int NFFT = 1024;
const float SHIFT = 0.5;
OFFSET =  int(SHIFT*FRAME);

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// TODO: Using matlab to generate hanning window of Frame size
// Precompute the window energy
WINDOW = sg.hann(FRAME)
EW = np.sum(WINDOW)
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

const int frames = int( (BufferSize - FRAME) / OFFSET + 1)
const int framesSamples = blockSize*frames;
int Sbb[NFFT] // matrix of size NFFT x number of channels(1)
const bool recordingNoise = true;






void Wiener::update(void)
{
	Serial.println("Wiener::update");
	count++;
	audio_block_t *block, *b_new;

	block = receiveReadOnly();
	if (!block) return;

	// If there's no coefficient table, give up.  
	if (coeff_p == NULL) {
		release(block);
		Serial.println("No coefficient table");
		return;
	}

	// do passthru
	if (coeff_p == FIR_PASSTHRU) {
		// Just passthrough
		transmit(block);
		release(block);
		Serial.println("Passthru");
		return;
	}

	/*********************************************************************************************/
	//TODO: PERSON 1: Implement the Sbb array processing here
	// Collect data for the Sbb noise matrix
	if (recordingNoise){
		// check if the noise frames have been fully collected
		if (counter * blockSize < framesSamples){  // TODO: fix this part, it is not correct
			memcpy(&(buff[counter*blockSize]), block->data, blockSize*sizeof(int16_t));
		}
		else{
			// end recording noise and start processing sbb matrix
			// Alternatively we can process each frame as it comes in
			// so it does not take too long to compute the noise matrix

			recordingNoise = false;
			//if using counter, don't forget to reset it back to 0
			//counter = 0;

		}
		
		return; // return here so it will not go to the next part of the code
	}
	/*********************************************************************************************/

	// after collecting the Sbb array, we can now start processing the signal
	b_new = allocate(); // allocate memory for the transmit block

	// Collect the signal buffer when the buffer is not full
	if(counter < BufferSize && !full){
		Serial.println("Buffer not full");
		memcpy(&(buff[counter*blockSize]), block->data, blockSize*sizeof(int16_t));
		counter++;
	}
	else{
		Serial.println("Buffer full");
		// if the buffer is full, start processing the signal
		/*********************************************************************************************/
		//TODO: PERSON 2: Implement the wiener processing here
		//Process the signal frame by frame here, transmit the processed data
		//as soon as it is done processing, since the transmited signal size
		//is limited to the block size, you need to store the remaining signal
		//from the processed signal in another buffer and transmit it in the next block
		
		//This code works with the assumption that the signal is coming in at the same rate
		//as we are transmitting it out, which I think is a fair assumption
		/*********************************************************************************************/

		// Code to update the Stored signal buffer
		// It might be safer to use two buffer instead of one buffer:
		// one buffer to store the signal for the wiener filter processing,
		// the other buffer to store the newly arrived signal, this can make
		// sure that we are not overwriting the signal that is not yet processed
		if (counter >= BufferSize){
			full = true;
			counter = 0;
		}
		std::copy(buff + (counter)*blockSize, buff + (counter+1)*blockSize, b_new->data);
		transmit(b_new);
		memcpy(&(buff[counter*blockSize]), block->data, blockSize*sizeof(int16_t));

		//Don't forget to increment the counter
		counter++;
	}
	release(b_new);
	release(block);

	if (count > INTERVAL) {
		switcher = !switcher;
		digitalWrite(LED, switcher);
		count = 0;
	}
}
