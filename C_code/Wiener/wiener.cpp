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

//Constants
const int sampleRate = 44100;
const int blockSamples = 128;		// 2.9 ms at 44100 Hz

//User defined constants
const int bufferBlocks = 0.5 * sampleRate / blockSamples; 	// 0.5 seconds of audio
const int FRAME_SAMPLES = int(sampleRate * 0.02); 		//20ms frame, 7 ms for WER

const int bufferSamples = bufferBlocks*blockSamples;


// helper variables
static int count = 0;
static bool switcher = true;
q15_t inputBuffer[bufferSamples];
q15_t outputBuffer[bufferSamples];
int blockCounter = 0;
bool full = false;
const int NFFT = 1024;
const float SHIFT = 0.5;
OFFSET =  int(SHIFT*FRAME_SAMPLES);

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// TODO: Using matlab to generate hanning window of Frame size
// 0.02 (882 points) and 0.007 (309 points)
// Precompute the window energy
WINDOW = sg.hann(FRAME_SAMPLES)
EW = np.sum(WINDOW)
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

const int frames = int( (bufferSamples - FRAME_SAMPLES) / OFFSET + 1)
const int framesSamples = blockSamples*frames;
q15_t Sbb[NFFT] // matrix of size NFFT x number of channels(1)
memset(Sbb, 0, sizeof(q15_array));
WIENER_PASSTHRU = true

bool recordingNoise = true;
bool processingSignal = false;

// Hann window code
struct hann_window {
  short *coeffs;
  short num_coeffs;    // num_coeffs must be an even number, 4 or higher
};

// index of initial window
// Change to 0 for 20 ms frame size
int start_idx = 0;

//value to updated using Serial input
int cur_idx = 0;
struct hann_window window_list[] = {
  {hann882  , 882},  		//20 ms frame size
  {hann309  , 309},			//7 ms frame size
  {NULL,   0}
};
// end Hann window code


//FFT instance:
arm_rfft_instance_q15 irfft;
arm_rfft_init_q15(&irfft, NFFT, 1, 1);
arm_rfft_instance_q15 rfft;
arm_rfft_init_q15(&rfft, NFFT, 0, 1);

//Initializing Empty arrays
for(int i = 0; i < NFFT; i++){
	Sbb[i] = 0;
}

q15_t emptyArrayOfBlockSamples[blockSamples];
for(int i = 0; i < blockSamples; i++){
	emptyArrayOfBlockSamples[i] = 0;
}

void Wiener::update(void)
{
	Serial.println("Wiener::update");
	count++;
	audio_block_t *block, *b_new;

	block = receiveReadOnly();
	if (!block) return;

	if(processingSignal){
		release(block);
		return;
	}

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
		// check if the noise frames have been fully collected
		if (blockCounter < bufferBlocks){
			Serial.println("Recording noise");  
			memcpy(&(inputBuffer[blockCounter*blockSamples]), (q15_t *)block->data, blockSamples*sizeof(q15_t));
		}
		else{
			processingSignal = true;
			Serial.println("Noise fully recorded, processing Sbb array");
			// end recording noise and start processing sbb matrix
			for (int i = 0; i < frames; i++){
				int i_min = i * OFFSET;
				//int i_max = i * OFFSET + FRAME_SAMPLES;
				
				//Assume WINDOW is the array of hanning window
				q15_t x_framed [window_size];
				/*
				
				void arm_mult_q15	(	const q15_t * 	pSrcA,
										const q15_t * 	pSrcB,
										q15_t * 		pDst,
										uint32_t 		blockSize )
				*/
				arm_mult_q15(WINDOW, inputBuffer + i_min, x_framed,  FRAME_SAMPLES);

				/*
				void arm_rfft_q15	(	const arm_rfft_instance_q15 * 	S,
										q15_t * 	pSrc,
										q15_t * 	pDst )	
				*/
				q15_t X_framed [window_size];
				arm_rfft_q15(&rfft, x_framed, X_framed);
				//void 	arm_abs_q15 (const q15_t *pSrc, q15_t *pDst, uint32_t blockSize)
				
				
				
				q15_t abs [window_size];
				q15_t sq [window_size];
				q15_t scaledSbb [window_size];
				q15_t scaledSq [window_size];

				q15_t iplusOne= i+1;
				arm_abs_q15 (X_framed, abs, FRAME_SAMPLES);
				arm_mult_q15(abs, abs, sq,  FRAME_SAMPLES);
				arm_mult_q15(Sbb, &(iplusOne), scaledSbb,  FRAME_SAMPLES); //This is sus
				arm_mult_q15(sq, &(iplusOne), scaledSq,  FRAME_SAMPLES); //This is sus
				
				/*
				void arm_add_q15	(	const q15_t * 	pSrcA,
				const q15_t * 	pSrcB,
				q15_t * 	pDst,
				uint32_t 	blockSize 
				)		
*/
				Sbb = arm_add_q15(scaledSbb, scaledSq, Sbb, NFFT);

		
				
			}


			//reset the variables
			processingSignal = false;
			recordingNoise = false;
			//if using blockCounter, don't forget to reset it back to 0
			blockCounter = 0;

		}
		Serial.println("Noise array Sbb fully processed");
		return; // return here so it will not go to the next part of the code
	}
	/*********************************************************************************************/

	// after collecting the Sbb array, we can now start processing the signal
	b_new = allocate(); // allocate memory for the transmit block

	/*********************************************************************************************/
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
	/*********************************************************************************************/

	// Code to update the Stored signal xframedBuffer
	// It might be safer to use two xframedBuffer instead of one xframedBuffer:
	// one xframedBuffer to store the signal for the wiener filter processing,
	// the other xframedBuffer to store the newly arrived signal, this can make
	// sure that we are not overwriting the signal that is not yet processed
	bool readyToProcess = blockCounter == bufferBlocks;
	if (readyToProcess){
		processingSignal = true;
		serial.println("Start Processing Signal")

		
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
		/*******  END WIENER DATA PROCESSING *******/
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
