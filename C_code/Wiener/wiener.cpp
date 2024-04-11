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




// helper variables
static int count = 0;
static bool switcher = true;
float32_t inputBuffer[Wiener::bufferSamples];
float32_t outputBuffer[Wiener::bufferSamples];
int blockCounter = 0;
bool full = false;
const uint16_t NFFT = 1024;
const float SHIFT = 0.5;
const int OFFSET =  int(SHIFT*Wiener::FRAME_SAMPLES);


const int frames = int( (Wiener::bufferSamples - Wiener::FRAME_SAMPLES) / OFFSET + 1);
const int framesSamples = Wiener::blockSamples*frames;
float32_t Sbb[NFFT]; // matrix of size NFFT x number of channels(1)

bool WIENER_PASSTHRU = true;

bool recordingNoise = true;
bool processingSignal = false;

// // Hann window code
// struct hann_window {
//   float32_t *coeffs;
//   short num_coeffs;    // num_coeffs must be an even number, 4 or higher
//   int32_t EW;
// };

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// TODO: Using matlab to generate hanning window of Frame size
// 0.02 (882 points) and 0.007 (309 points)
// Precompute the window energy

// index of initial window
// Change to 0 for 20 ms frame size
//value to updated using Serial input
// int cur_idx = 0; // 0 for 20ms frame size, 1 for 7ms frame size
// struct hann_window window_list[] = {
//   {hann882  , 882, 5046117},  		//20 ms frame size
//   {hann309  , 309, 14433866},			//7 ms frame size
//   {NULL,   0, 0}
// };
// float32_t* WINDOW = window_list[cur_idx].coeffs;
// int32_t EW = window_list[cur_idx].EW;
// end Hann window code
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//magnitude squared
void Wiener::elementwise_mag_sq(float32_t *pSrc, float32_t *pDst, uint32_t length) {
    // assumes that pSrc is twice the legnth of pDst (output of cfft) 
    for (uint32_t i = 0; i < length; i += 2) {
        float32_t real = pSrc[i];
        float32_t imag = pSrc[i + 1];
        pDst[i / 2] = real * real + imag * imag;
    }
}

//This make real array into complex array
void Wiener::interpolate_with_zeros(float32_t *pSrc, float32_t *pDst, uint32_t length) {
    // assumes that pDst is twice the legnth of the pSrc
    for (uint32_t i = 0; i < length; i++) {
        pDst[2 * i] = pSrc[i];
        pDst[2 * i + 1] = 0;
    }
}

// simulate the nonexistent arm_dividef32() 
void Wiener::elementwise_divide(float32_t *pSrcA, float32_t *pSrcB, float32_t *pDst, uint32_t length) {
    for (uint32_t i = 0; i < length; ++i) {
        pDst[i] = pSrcA[i] / pSrcB[i];
    }
}


void Wiener::pad_with_zeros(float32_t *pSrc, float32_t *pDst, uint32_t srcLength, uint32_t dstLength) {
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


void Wiener::wienerFilter(float32_t *inputBuffer, float32_t *outputBuffer, float32_t *Sbb, arm_cfft_radix4_instance_f32 *S, arm_cfft_radix4_instance_f32 *IS, float32_t *WINDOW, float32_t EW) {
    uint32_t frames = (Wiener::Wiener::bufferSamples - Wiener::FRAME_SAMPLES) / OFFSET + 1;
    float32_t xframedBuffer[Wiener::FRAME_SAMPLES];
    float32_t fftBuffer[2 * NFFT];
    float32_t G[NFFT];

    // Instantiate Hann window and compute EW
    //float32_t WINDOW[Wiener::FRAME_SAMPLES];

    for (uint32_t frame = 0; frame < frames; ++frame) {
        uint32_t i_min = frame * OFFSET;
        uint32_t i_max = frame * OFFSET + Wiener::FRAME_SAMPLES;

        // Populate the frame
        for (uint32_t i = i_min; i < i_max; ++i) {
            xframedBuffer[i - i_min] = inputBuffer[i];
        }

        // Temporal framing with Hanning window
        arm_mult_f32(WINDOW, xframedBuffer, xframedBuffer, Wiener::FRAME_SAMPLES);

        // Zero padding and inteNFFT
        pad_with_zeros(xframedBuffer, fftBuffer, Wiener::FRAME_SAMPLES, 2 * NFFT);

        // FFT
        //arm_cfft_f32(S, fftBuffer, 0, 1);
		arm_cfft_radix4_f32(S, fftBuffer);

        // Calculate the Wiener gain
        float32_t X_framed_abs_sq[NFFT];
        float32_t temp[NFFT];
        float32_t snrPost[NFFT];
        float32_t one_array[NFFT];
        float32_t SNR_plus_one[NFFT];

        // Calculate the element-wise modulus squared of X_framed
        Wiener::elementwise_mag_sq(fftBuffer, X_framed_abs_sq, 2 * NFFT);

        // Divide the squared values by EW
        arm_scale_f32(X_framed_abs_sq, 1.0f / EW, temp, NFFT);

        // Divide the result by Sbb
        Wiener::elementwise_divide(temp, Sbb, snrPost, NFFT);

        // Fill the one_array with the constant value 1
        arm_fill_f32(1.0f, one_array, NFFT);

        // Add 1 to each element of SNR_post
        arm_add_f32(snrPost, one_array, SNR_plus_one, NFFT);

        // Divide SNR_post by (SNR_post + 1) to obtain G
        Wiener::elementwise_divide(snrPost, SNR_plus_one, G, NFFT);

        // Apply gain
        for (uint32_t i = 0; i < NFFT; ++i) {
            fftBuffer[2 * i] *= G[i];
            fftBuffer[2 * i + 1] *= G[i];
        }

        // IFFT
        //arm_cfft_f32(S, fftBuffer, 1, 1);
		arm_cfft_radix4_f32(IS, fftBuffer); // IS is ifft object

        // Estimated signals at each frame normalized by the shift value
        arm_scale_f32(fftBuffer, 1.0f / NFFT, fftBuffer, 2 * NFFT);

        // Overlap and add
        for (uint32_t i = 0; i < Wiener::FRAME_SAMPLES; ++i) {
            outputBuffer[i_min + i] += fftBuffer[2 * i];
        }
    }
}

void Wiener::welchsPeriodogram(float32_t *x, float32_t *Sbb,  arm_cfft_radix4_instance_f32* S, float32_t *WINDOW, float32_t EW) {

    for (uint32_t i = 0; i < NFFT; i++) {
        Sbb[i] = 0.0f;
    }

    float32_t x_framed[Wiener::FRAME_SAMPLES];
    float32_t X_framed[2 * NFFT];
    float32_t X_framed_abs_sq[NFFT];

    for (uint32_t frame = 0; frame < frames; frame++) {
        uint32_t i_min = frame * OFFSET;
        uint32_t i_max = frame * OFFSET + Wiener::FRAME_SAMPLES;

        //arm_mult_f32(WINDOW, x + frame + i_min, xframedBuffer, Wiener::FRAME_SAMPLES);

        for (uint32_t i = i_min, j = 0; i < i_max; i++, j++) {
            x_framed[j] = x[i] * WINDOW[j];
        }

        Wiener::pad_with_zeros(x_framed, X_framed, Wiener::FRAME_SAMPLES, 2 * NFFT);

        //arm_cfft_f32 (const arm_cfft_instance_f32 *S, float32_t *p1, uint8_t ifftFlag, uint8_t bitReverseFlag), p1 has length 2*fftLen (it is complex) 
        //arm_cfft_f32(S, X_framed, 0, 1);
		//arm_cfft_f32(&arm_cfft_sR_f32_len1024, X_framed, 0, 1);
		arm_cfft_radix4_f32(S, X_framed);


        Wiener::elementwise_mag_sq(X_framed, X_framed_abs_sq, 2 * NFFT);

        for (uint32_t i = 0; i < NFFT; i++) {
            Sbb[i] = frame * Sbb[i] / (frame + 1) + X_framed_abs_sq[i] / (frame + 1);
        }
    }
}

void Wiener::update(void){
	Serial.println("Wiener::update");
	float32_t EW = 0;
	float32_t WINDOW [FRAME_SAMPLES];
	for (uint32_t i = 0; i < FRAME_SAMPLES; ++i) {
        	WINDOW[i] = 0.5f - 0.5f * arm_cos_f32(2.0f * PI * i / (FRAME_SAMPLES - 1));
        	EW += WINDOW[i];
    	}

    // instantiate cfft
	//arm_cfft_instance_f32 *S;
	//arm_cfft_init_f32(S, NFFT);
	arm_cfft_radix4_instance_f32 S;	
	arm_cfft_radix4_init_f32(&S, NFFT, 0, 1);
	arm_cfft_radix4_instance_f32 IS;	
	arm_cfft_radix4_init_f32(&IS, NFFT, 1, 1);

	

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
		if (blockCounter < Wiener::bufferBlocks){
			Serial.println("Recording noise"); 
			Serial.println(blockCounter);
			Serial.println(Wiener::bufferBlocks);
			memcpy(&(inputBuffer[blockCounter*Wiener::blockSamples]), (float32_t *)block->data, Wiener::blockSamples*sizeof(float32_t));
		}
		else{
			memcpy(&inputBuffer, truth, sizeof(truth));			

			processingSignal = true;
			Serial.println("Noise fully recorded, processing Sbb array");
			// end recording noise and start processing sbb matrix
			welchsPeriodogram(inputBuffer, Sbb, &S, WINDOW, EW); 
            
            Serial.print("[");
            for(int i = 0; i < NFFT; i++){
                Serial.print(Sbb[i]);
                Serial.print(", ");
            }
            Serial.print(']');


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
	bool readyToProcess = blockCounter == Wiener::bufferBlocks;
	if (readyToProcess) {
    processingSignal = true;
    Serial.println("Start Processing Signal");

    // call Wiener filter
    wienerFilter(inputBuffer, outputBuffer, Sbb, &S, &IS, WINDOW, EW);


    processingSignal = false;
    Serial.println("End Processing Signal");
    blockCounter = 0;
    }

		


	// copy the recieved input block into the inputBuffer
	memcpy(&(inputBuffer[blockCounter*Wiener::blockSamples]), (float32_t *)block->data, Wiener::blockSamples*sizeof(float32_t));

	// transmit the relevant blocks worth of samples from outputBuffer (to playback)
	std::copy(outputBuffer + blockCounter*Wiener::blockSamples, outputBuffer + (blockCounter+1)*Wiener::blockSamples, b_new->data);
	transmit(b_new);

	for(int i = 0; i < bufferSamples; ++i) {outputBuffer[i] = 0;} // reset the output buffer
	

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
