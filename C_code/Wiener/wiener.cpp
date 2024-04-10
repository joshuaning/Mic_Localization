#include <Arduino.h>
#include <AudioStream.h> // github.com/PaulStoffregen/cores/blob/master/teensy4/AudioStream.h
#include "wiener.h"
#include <stdio.h>
#define LED 13
#define INTERVAL 3000
//From HannWindow_switch
#include <Audio.h>
#include <Wire.h>
#include <SD.h>
#include <SPI.h>
#include <SerialFlash.h>
#include "hann309.h"
#include "hann882.h"
#include "truth.h"
#include <math.h>


//Constants
const int sampleRate = 44100;
const int blockSamples = 128;		// 2.9 ms at 44100 Hz

//User defined constants
const int bufferBlocks = floor(0.5 * sampleRate / blockSamples); 	// 0.5 seconds of audio
const int FRAME_SAMPLES = int(sampleRate * 0.02); 		//20ms frame, 7 ms for WER

const int bufferSamples = bufferBlocks*blockSamples;


// helper variables
static int count = 0;
static bool switcher = true;
float32_t inputBuffer[bufferSamples];
float32_t outputBuffer[bufferSamples];
int blockCounter = 0;
bool full = false;
const int NFFT = 1024;
const float SHIFT = 0.5;
const int OFFSET =  int(SHIFT*FRAME_SAMPLES);



const int frames = int( (bufferSamples - FRAME_SAMPLES) / OFFSET + 1);
const int framesSamples = blockSamples*frames;
float32_t Sbb[NFFT]; // matrix of size NFFT x number of channels(1)

bool WIENER_PASSTHRU = true;

bool recordingNoise = true;
bool processingSignal = false;

// Hann window code
struct hann_window {
  float32_t *coeffs;
  short num_coeffs;    // num_coeffs must be an even number, 4 or higher
  int32_t EW;
};

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// TODO: Using matlab to generate hanning window of Frame size
// 0.02 (882 points) and 0.007 (309 points)
// Precompute the window energy

// index of initial window
// Change to 0 for 20 ms frame size
//value to updated using Serial input
int cur_idx = 0; // 0 for 20ms frame size, 1 for 7ms frame size
struct hann_window window_list[] = {
  {hann882  , 882, 5046117},  		//20 ms frame size
  {hann309  , 309, 14433866},			//7 ms frame size
  {NULL,   0, 0}
};
float32_t* WINDOW = window_list[cur_idx].coeffs;
int32_t EW = window_list[cur_idx].EW;
// end Hann window code
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


void elementwise_mod_sq(float_32t *pSrc, float_32t *pDst){
	// pSrc is indexed as [R1,C1,R2,C2,...] while pDst will be [R1^2+C1^2, R2^2+C2^2, ...]
	
}

void Wiener::update(void)
{
	Serial.println("Wiener::update");
	count++;
	audio_block_t *block, *b_new;

	block = receiveReadOnly();
	if (!block){
		Serial.print("No blocks");
		return;
	} 

	// if(processingSignal){
	// 	release(block);
	// 	return;
	// }

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
	// Collect data for the Sbb noise array
	if (recordingNoise){

		if(blockCounter == 0){
			//initialize
			memset(Sbb, 0, sizeof(Sbb));
		}
		// check if the noise frames have been fully collected
		if (blockCounter < bufferBlocks){
			Serial.println("Recording noise"); 
			Serial.println(blockCounter);
			Serial.println(bufferBlocks);
			memcpy(&(inputBuffer[blockCounter*blockSamples]), (float32_t *)block->data, blockSamples*sizeof(float32_t));
		}
		else{
			memcpy(&inputBuffer, truth, sizeof(truth));			

			processingSignal = true;
			Serial.println("Noise fully recorded, processing Sbb array");
			// end recording noise and start processing sbb matrix
			for (int i = 0; i < frames; i++){
				int i_min = i * OFFSET;
				
				//Assume WINDOW is the array of hanning window
				float32_t x_framed [NFFT];
				memset(x_framed, 0, sizeof(x_framed));
				float32_t X_framed [NFFT*2];
				float32_t abs [NFFT];
				float32_t sq [NFFT];
				float32_t scaledSbb [NFFT];
				float32_t scaledSq [NFFT];
				float32_t oneOveriPlusOne= 1/(i+1);
				float32_t iOveriPlusOne= i/(i+1);

				//FFT instance:
				arm_cfft_instance_f32 cfft;
				arm_cfft_init_f32(&cfft, NFFT, 1, 1);

				/*
				void arm_mult_q15	(	const q15_t * 	pSrcA,
										const q15_t * 	pSrcB,
										q15_t * 		pDst,
										uint32_t 		blockSize )*/
				arm_mult_f32(WINDOW, inputBuffer + i_min, x_framed,  FRAME_SAMPLES);

				//TODO: Fill indexes 882 to 1023 with 0s

				/*
				void arm_rfft_q15	(	const arm_rfft_instance_q15 * 	S,
										q15_t * 	pSrc,
										q15_t * 	pDst )*/
				arm_cfft_f32(&cfft, x_framed, X_framed);

				//void 	arm_abs_q15 (const q15_t *pSrc, q15_t *pDst, uint32_t blockSize)
				arm_abs_f32(X_framed, abs, NFFT); // abs
				arm_mult_f32(abs, abs, sq,  NFFT); // square
				arm_mult_f32(Sbb, &(iOveriPlusOne), scaledSbb,  NFFT); //This is sus
				arm_mult_f32(sq, &(oneOveriPlusOne), scaledSq,  NFFT); //This is sus

				/*
				void arm_add_q15	(	const q15_t * 	pSrcA,
				const q15_t * 	pSrcB,
				q15_t * 	pDst,
				uint32_t 	blockSize 
				)		*/
				arm_add_f32(scaledSbb, scaledSq, Sbb, NFFT);
				// Print the Sbb array for debug and verification
				Serial.print("[");
				for(int i = 0; i < NFFT; i++){
					Serial.print(Sbb[i]);
					Serial.print(", ");
				}
				Serial.print(']');

				
			}

			//reset the variables
			processingSignal = false;
			recordingNoise = false;
			//if using blockCounter, don't forget to reset it back to 0
			blockCounter = 0;
			Serial.println("Noise array Sbb fully processed");
		}

		blockCounter++;
		transmit(block);
		release(block);
		return; // return here so it will not go to the next part of the code
	}


	//--------------------------------------------------------------------------------------------//

	// after collecting the Sbb array, we can now start processing the signal
	b_new = allocate(); // allocate memory for the transmit block
/*
	//--------------------------------------------------------------------------------------------//
	//TODO: PERSON 2: Implement the wiener processing here.

	//First try processing the whole xframedBuffer at once, similar to the python implementation
	//Transmit the blocks as soon as they are processed.


	// Then 
	//Process the signal frame by frame here, transmit the processed data
	//as soon as it is done processing, since the transmited signal size
	//is limited to the block size, you need to store the remaining signal
	//from the processed signal in another xframedBuffer and transmit it in the next block
	
	//This code works with the assumption that the signal is coming in at the same rate
	//as we are transmitting it out, which I think is a fair assumption
	//--------------------------------------------------------------------------------------------//

	// Code to update the Stored signal xframedBuffer
	// It might be safer to use two xframedBuffer instead of one xframedBuffer:
	// one xframedBuffer to store the signal for the wiener filter processing,
	// the other xframedBuffer to store the newly arrived signal, this can make
	// sure that we are not overwriting the signal that is not yet processed
	bool readyToProcess = blockCounter == bufferBlocks;
	if (readyToProcess){
		processingSignal = true;
		serial.println("Start Processing Signal")

		//FFT instance:
		arm_rfft_instance_q15 irfft;
		arm_rfft_init_q15(&irfft, NFFT, 1, 1);
		arm_rfft_instance_q15 rfft;
		arm_rfft_init_q15(&rfft, NFFT, 0, 1);

		
		q15_t xframedBuffer[FRAME_SAMPLES];
    	std::fill(xframedBuffer, xframedBuffer + NFFT, 0); // ?
		for (int frame = 0; frame < frames; ++frame) {
        int i_min = frame * OFFSET;
        int i_max =  frame * OFFSET + FRAME_SAMPLES; 
		
		// populate the frame
        for (int i = i_min; i < i_max; ++i) {
			xframedBuffer[i-i_min] = inputBuffer[i];
        }

        // Zero padding xframed
        for (int i = i_max - i_min; i < NFFT; ++i) {
            xframedBuffer[i] = 0;
        }

		// Temporal framing with Hanning window
		arm_mult_q15(WINDOW, xframedBuffer, xframedBuffer, FRAME_SAMPLES)

        // FFT
        arm_rfft_q15(&rfft, xframedBuffer, xframedBuffer);

        // make wiener gain array
		 // Assuming the following variables are available:
        // - X_framed: q15_t array of size NFFT
        // - EW: q15_t constant value
        // - Sbb: q15_t array of size NFFT
        // - SNR_post: q15_t array of size NFFT to store the resul
        // Calculate the absolute value of X_framed

		q15_t abs_X_framed[FRAME_SAMPLES];
		q15_t abs_X_framed_squared[FRAME_SAMPLES];
		q15_t temp[FRAME_SAMPLES];
		q15_t snrPost[FRAME_SAMPLES];
		q15_t one_array[FRAME_SAMPLES];
		q15_t SNR_plus_one[FRAME_SAMPLES];
		
        arm_abs_q15(xframedBuffer, abs_X_framed, FRAME_SAMPLES)
        // Square the absolute values
        arm_mult_q15(abs_X_framed, abs_X_framed, abs_X_framed_squared, FRAME_SAMPLES)
        // Divide the squared values by EW
        arm_divide_q15(abs_X_framed_squared, &EW, temp, FRAME_SAMPLES)
        // Divide the result by Sbb
        arm_divide_q15(temp, Sbb, snrPost, FRAME_SAMPLES);
		
		// calculate gain G = SNR/(SNR + 1)
		// Fill the one_array with the constant value 1
		arm_fill_q15(1, one_array, NFFT);

		// Add 1 to each element of SNR_post
		arm_add_q15(SNR_post, one_array, SNR_plus_one, NFFT);

		// Divide SNR_post by (SNR_post + 1) to obtain G
		arm_divide_q15(SNR_post, SNR_plus_one, G, NFFT);

        // Apply gain
		arm_mult_q15(xframedBuffer, G, xframedBuffer, FRAME_SAMPLES)

        // IFFT
        arm_rfft_q15(&irfft, xframedBuffer, xframedBuffer);

		// Estimated signals at each frame normalized by the shift value	
		arm_mult_q15(xframedBuffer, SHIFT, xframedBuffer, FRAME_SAMPLES)

		for(int i = 0; i < FRAME_SAMPLES; ++i){
			outputBuffer[i_min + i] += xframedbuffer[i];
		}
		//----------------------------- END WIENER PROCESSING ------------------------------------//
	}

		processingSignal = false;
		serial.println("End Processing Signal")
		blockCounter = 0;
	}

	// copy the recieved input block into the inputBuffer
	memcpy(&(inputBuffer[blockCounter*blockSamples]), ((q15_t *))block->data, blockSamples*sizeof(q15_t));

	// transmit the relevant blocks worth of samples from outputBuffer (to playback)
	std::copy(outputBuffer + blockCounter*blockSamples, outputBuffer + (blockCounter+1)*blockSamples, b_new->data);
	transmit(b_new);

	for(int i = 0; i < N; ++i) {outputBuffer[i] = 0;} // reset the output buffer
	*/

	// don't forget to increment the blockCounter
	blockCounter++;
	release(b_new);
	release(block);

	if (count > INTERVAL) {
		switcher = !switcher;
		digitalWrite(LED, switcher);
		count = 0;
	}
}
