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


void elementwise_mod_sq(float32_t *pSrc, float32_t *pDst, uint32_t length) {
    for (uint32_t i = 0; i < length; i += 2) {
        float32_t real = pSrc[i];
        float32_t imag = pSrc[i + 1];
        pDst[i / 2] = real * real + imag * imag;
    }
}

void interpolate_with_zeros(float32_t *pSrc, float32_t *pDst, uint32_t length) {
    for (uint32_t i = 0; i < length; i++) {
        pDst[2 * i] = pSrc[i];
        pDst[2 * i + 1] = 0;
    }
}

void pad_with_zeros(float32_t *pSrc, float32_t *pDst, uint32_t srcLength, uint32_t dstLength) {
    if (srcLength > dstLength) {
        // Sourcfe length is greater than destination length
        // Copy only the first dstLength elements from pSrc to pDst
        memcpy(pDst, pSrc, dstLength * sizeof(float32_t));
    } else {
        // Source length is less than or equal to destination length
        // Copy all elements from pSrc to pDst
        memcpy(pDst, pSrc, srcLength * sizeof(float32_t));

        // Pad the remaining elements in pDst with zeros
        memset(pDst + srcLength, 0, (dstLength - srcLength) * sizeof(float32_t));
    }
}

void wienerFilter(float32_t *inputBuffer, float32_t *outputBuffer, float32_t *Sbb, arm_cfft_instance_f32 *S) {
    uint32_t frames = (bufferSamples - FRAME_SAMPLES) / OFFSET + 1;
    float32_t xframedBuffer[FRAME_SAMPLES];
    float32_t fftBuffer[2 * S->fftLen];
    float32_t G[S->fftLen];

    // Instantiate Hann window and compute EW
    float32_t WINDOW[FRAME_SAMPLES];
    float32_t EW = 0.0f;
    arm_hann_f32(WINDOW, FRAME_SAMPLES);
    for (uint32_t i = 0; i < FRAME_SAMPLES; ++i) {
        EW += WINDOW[i];
    }

    for (uint32_t frame = 0; frame < frames; ++frame) {
        uint32_t i_min = frame * OFFSET;
        uint32_t i_max = frame * OFFSET + FRAME_SAMPLES;

        // Populate the frame
        for (uint32_t i = i_min; i < i_max; ++i) {
            xframedBuffer[i - i_min] = inputBuffer[i];
        }

        // Temporal framing with Hanning window
        arm_mult_f32(WINDOW, xframedBuffer, xframedBuffer, FRAME_SAMPLES);

        // Zero padding and interpolation
        pad_with_zeros(xframedBuffer, fftBuffer, FRAME_SAMPLES, 2 * S->fftLen);

        // FFT
        arm_cfft_f32(S, fftBuffer, 0, 1);

        // Calculate the Wiener gain
        float32_t X_framed_abs_sq[S->fftLen];
        float32_t temp[S->fftLen];
        float32_t snrPost[S->fftLen];
        float32_t one_array[S->fftLen];
        float32_t SNR_plus_one[S->fftLen];

        // Calculate the element-wise modulus squared of X_framed
        elementwise_mod_sq(fftBuffer, X_framed_abs_sq, 2 * S->fftLen);

        // Divide the squared values by EW
        arm_scale_f32(X_framed_abs_sq, 1.0f / EW, temp, S->fftLen);

        // Divide the result by Sbb
        arm_divide_f32(temp, Sbb, snrPost, S->fftLen);

        // Fill the one_array with the constant value 1
        arm_fill_f32(1.0f, one_array, S->fftLen);

        // Add 1 to each element of SNR_post
        arm_add_f32(snrPost, one_array, SNR_plus_one, S->fftLen);

        // Divide SNR_post by (SNR_post + 1) to obtain G
        arm_divide_f32(snrPost, SNR_plus_one, G, S->fftLen);

        // Apply gain
        for (uint32_t i = 0; i < S->fftLen; ++i) {
            fftBuffer[2 * i] *= G[i];
            fftBuffer[2 * i + 1] *= G[i];
        }

        // IFFT
        arm_cfft_f32(S, fftBuffer, 1, 1);

        // Estimated signals at each frame normalized by the shift value
        arm_scale_f32(fftBuffer, 1.0f / S->fftLen, fftBuffer, 2 * S->fftLen);

        // Overlap and add
        for (uint32_t i = 0; i < FRAME_SAMPLES; ++i) {
            outputBuffer[i_min + i] += fftBuffer[2 * i];
        }
    }
}

void welchsPeriodogram(float32_t *x, uint32_t *noise_start, uint32_t *noise_end, float32_t *Sbb, arm_cfft_instance_f32 *S) {
    float32_t WINDOW[FRAME_SAMPLES];
    arm_hann_f32(WINDOW, FRAME_SAMPLES);

    float32_t EW = 0.0f;
    for (uint32_t i = 0; i < FRAME_SAMPLES; i++) {
        EW += WINDOW[i];
    }

    uint32_t frames = (length - FRAME_SAMPLES) / OFFSET + 1; // delete?

    uint32_t noise_frames = (*noise_end - *noise_start - FRAME_SAMPLES) / OFFSET + 1;

    for (uint32_t i = 0; i < S->fftLen; i++) {
        Sbb[i] = 0.0f;
    }

    float32_t x_framed[FRAME_SAMPLES];
    float32_t X_framed[2 * S->fftLen];
    float32_t X_framed_abs_sq[S->fftLen];

    for (uint32_t frame = 0; frame < noise_frames; frame++) {
        uint32_t i_min = frame * OFFSET + *noise_start;
        uint32_t i_max = frame * OFFSET + FRAME_SAMPLES + *noise_start;

        for (uint32_t i = i_min, j = 0; i < i_max; i++, j++) {
            x_framed[j] = x[i] * WINDOW[j];
        }

        pad_with_zeros(x_framed, X_framed, FRAME_SAMPLES, 2 * S->fftLen);
        arm_cfft_f32(S, X_framed, 0, 1);

        elementwise_mod_sq(X_framed, X_framed_abs_sq, 2 * S->fftLen);

        for (uint32_t i = 0; i < S->fftLen; i++) {
            Sbb[i] = frame * Sbb[i] / (frame + 1) + X_framed_abs_sq[i] / (frame + 1);
        }
    }
}

void Wiener::update(void)
{
	Serial.println("Wiener::update");

    // instantiate cfft
    arm_cfft_instance_f32 S;

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
                int i_max = i * OFFSET + FRAME_SAMPLES;
				float32_t x_framed [NFFT*2]; // NFFT*2 because of the complex numbers... we will just interpolate with 0s; //memset(x_framed, 0, sizeof(x_framed)); // do we need this if it gets overwritten anyway?
                welchsPeriodogram(inputBuffer, inputBuffer + i_min, inputBuffer + i_max, Sbb, &S); // TODO are these the right input arguments?
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
	if (readyToProcess) {
    processingSignal = true;
    Serial.println("Start Processing Signal");

    // call Wiener filter
    //wienerFilter(inputBuffer, outputBuffer, bufferSamples, FRAME_SAMPLES, OFFSET, Sbb, &S);


    processingSignal = false;
    Serial.println("End Processing Signal");
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
