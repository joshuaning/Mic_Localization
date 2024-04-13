#include <Arduino.h>
#include <AudioStream.h> // github.com/PaulStoffregen/cores/blob/master/teensy4/AudioStream.h
#include "wiener.h"
#include <stdio.h>
#define LED 13
#define INTERVAL 3000
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
q15_t outputBuffer_q[Wiener::bufferSamples];
float floatBlock [Wiener::blockSamples];
int blockCounter = 0;
bool full = false;
const uint16_t NFFT = 1024;
const float SHIFT = 0.5;
const int OFFSET =  int(SHIFT*Wiener::FRAME_SAMPLES);


const int frames = int( (Wiener::bufferSamples - Wiener::FRAME_SAMPLES) / OFFSET + 1);
const int framesSamples = Wiener::blockSamples*frames;
float32_t Sbb[NFFT]; // matrix of size NFFT x number of channels(1)

bool PASSTHRU_BUT_PROCESS = false; // for timing checks

bool recordingNoise = true;
bool processingSignal = false;

const uint16_t numMax = 500;
float32_t runningMax = 0;
float32_t noiseMax = 0;
uint16_t MaxIdx = 0;
float32_t historicalMax [numMax]; 
float32_t noiseMargin = 0;
bool firstPass = true;
float32_t stdNoise[Wiener::bufferSamples];

// ----------------------------------- Helper Functions ----------------------------------- //

float find_max_of_History(){
	float32_t outmax = -9999.0;
	for (int i = 0; i<numMax; i++){
		if(outmax < historicalMax[i]){
			outmax = historicalMax[i];
		}
	}
	return outmax;
}

float find_max_of_OutBuffer(){
	float32_t outmax = -9999.0;
	for (int i = 0; i<Wiener::bufferSamples; i++){
		if(outmax < outputBuffer[i]){
			outmax = outputBuffer[i];
		}
	}
	return outmax;
}

void printArr(float32_t *arr, uint32_t sz){
	Serial.print("[");
	for(uint32_t i = 0; i<sz; i++){
		Serial.print(arr[i], 4);
		Serial.print(", ");
	}
	Serial.println("]");
}

void printArr(q15_t *arr, int sz){
	Serial.print("[");
	for(int i = 0; i<sz; i++){
		Serial.print(arr[i]);
		Serial.print(", ");
	}
	Serial.println("]");
}

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
//length is the length of the source array
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
// ----------------------------------- Helper Functions END ----------------------------------- //



// ----------------------------------- Wiener Functions ----------------------------------- //
void Wiener::wienerFilter(float32_t *inputBuffer, float32_t *outputBuffer, float32_t *Sbb, arm_cfft_radix4_instance_f32 *S, arm_cfft_radix4_instance_f32 *IS, float32_t *WINDOW, float32_t EW) {
    uint32_t frames = (Wiener::bufferSamples - Wiener::FRAME_SAMPLES) / OFFSET + 1;
    float32_t xframedBuffer[Wiener::FRAME_SAMPLES];
    float32_t x_framed_interpolated_with_zeros[2*Wiener::FRAME_SAMPLES];
    float32_t fftBuffer[2 * NFFT];
    float32_t G[NFFT];
	Serial.println("Finding max in input signal ");
	float inMax = -9999;
	for(int i = 0; i < Wiener::bufferSamples; i++){
		if(inMax < inputBuffer[i]){
			inMax = inputBuffer[i];
		}
	}

    // Instantiate Hann window and compute EW
    //float32_t WINDOW[Wiener::FRAME_SAMPLES];
	
    for (uint32_t frame = 0; frame < frames; ++frame) {
        uint32_t i_min = frame * OFFSET;
        uint32_t i_max = frame * OFFSET + Wiener::FRAME_SAMPLES;

		Serial.println("inputBuffer [0]");
        //printArr(inputBuffer, bufferSamples);

        // Temporal framing with Hanning window
        // Serial.println("wienerFilter: obtaining the (windowed) frame");
        // arm_mult_f32(WINDOW, xframedBuffer, xframedBuffer, Wiener::FRAME_SAMPLES);
		
		//Serial.println("Window: ");
		//printArr(WINDOW, Wiener::FRAME_SAMPLES);
		
		//Serial.println("constructing xframedBuffer (already windowed): ");
        for (uint32_t i = i_min, j = 0; i < i_max; i++, j++) {
            xframedBuffer[j] = inputBuffer[i] * WINDOW[j];
        }

		Serial.println("xframedBuffer immediately after population(after windowing): [1]");
		//printArr(xframedBuffer, Wiener::FRAME_SAMPLES);

		//Serial.println("wienerFilter: interpolate the frame");
        Wiener::interpolate_with_zeros(xframedBuffer, x_framed_interpolated_with_zeros, Wiener::FRAME_SAMPLES); 
		//Serial.println("interpolated xframedBuffer");
		//printArr(x_framed_interpolated_with_zeros, 2*Wiener::FRAME_SAMPLES);

        // Zero padding and inteNFFT
		//Serial.println("wienerFilter: Zero padding the frame");
        Wiener::pad_with_zeros(x_framed_interpolated_with_zeros, fftBuffer, 2*Wiener::FRAME_SAMPLES, 2*NFFT);
		//Serial.println("interpolated, then padded, xframedBuffer");
		//printArr(fftBuffer, 2*NFFT);
        

        // FFT
		//Serial.println("wienerFilter: FFT the frame");
		arm_cfft_radix4_f32(S, fftBuffer);
		Serial.println("frame post-FFT: [2]");
		//printArr(fftBuffer, 2*NFFT);

        // Calculate the Wiener gain
        float32_t X_framed_abs_sq[NFFT];
        float32_t temp[NFFT];
        float32_t snrPost[NFFT];
        float32_t one_array[NFFT];
        float32_t SNR_plus_one[NFFT];

        // Calculate the element-wise modulus squared of X_framed
		Serial.println("wienerFilter: magnitude square the frame (size returns to NFFT from 2*NFFT): [3]");
		Wiener::elementwise_mag_sq(fftBuffer, X_framed_abs_sq, 2 * NFFT);
        //printArr(X_framed_abs_sq, NFFT);

        // Divide the squared values by EW
		Serial.println("Divide by EW for the frame [4]");
        arm_scale_f32(X_framed_abs_sq, 1.0f / EW, temp, NFFT);
        //printArr(temp, NFFT);

        // Divide the result by Sbb
		Serial.println("Divide result by Sbb [5]");
        Wiener::elementwise_divide(temp, Sbb, snrPost, NFFT);
		//printArr(snrPost, NFFT);

        // Fill the one_array with the constant value 1
        arm_fill_f32(1.0f, one_array, NFFT);

        // Add 1 to each element of SNR_post
        arm_add_f32(snrPost, one_array, SNR_plus_one, NFFT);
		Serial.println("Add SNR_post by 1 [6]");
		//printArr(SNR_plus_one, NFFT);

        // Divide SNR_post by (SNR_post + 1) to obtain G
        Wiener::elementwise_divide(snrPost, SNR_plus_one, G, NFFT);
		Serial.println("Divide SNR_post by (SNR_post + 1) to obtain G [7]");
		//printArr(G, NFFT);

		//Serial.println("G:");
		//printArr(G, NFFT);

        // Apply gain
        for (uint32_t i = 0; i < NFFT; ++i) {
            fftBuffer[2 * i] *= G[i];
            fftBuffer[2 * i + 1] *= G[i];
        }

		Serial.println("Apply gain to fft buffer [8]");
		//printArr(fftBuffer, NFFT*2);

        // IFFT
        //arm_cfft_f32(S, fftBuffer, 1, 1);
		arm_cfft_radix4_f32(IS, fftBuffer); // IS is ifft object
		Serial.println("post-ifft [9]");
		//printArr(fftBuffer, NFFT*2);

        // Estimated signals at each frame normalized by the shift value
		arm_scale_f32(fftBuffer, SHIFT, fftBuffer, 2 * NFFT);
		Serial.println("scaling by shift [10]");
		//printArr(fftBuffer, NFFT*2);

        // Overlap and add

        for (uint32_t i = 0; i < Wiener::FRAME_SAMPLES; ++i) { // loop boundary is FRAME_SAMPLES to truncate zero padding
			outputBuffer[i_min + i] += fftBuffer[2 * i];
        }

		Serial.println("output buffer of the frame [11]");
		//printArr(outputBuffer + i_min,  Wiener::FRAME_SAMPLES);
    }
	Serial.println("Finding max in OutBuffer");
	float32_t outmax = find_max_of_OutBuffer();
	Serial.println("Computing the Historic Max ");
	if(firstPass){
		Serial.println("first pass ");
		historicalMax[MaxIdx] = outmax;
		MaxIdx++;
		firstPass = false;
	}
	else if(inMax > noiseMax + noiseMargin){
		Serial.println("more than 1 std");
		historicalMax[MaxIdx] = outmax;
		MaxIdx++;
		if(MaxIdx == numMax){
			MaxIdx = 0;
		}
	}
	Serial.println("max of history");
	float maxOfHistory = find_max_of_History();
	Serial.println(maxOfHistory);
	
	arm_scale_f32(outputBuffer, 1/maxOfHistory, outputBuffer,Wiener::bufferSamples);
	arm_float_to_q15(outputBuffer, outputBuffer_q, Wiener::bufferSamples);
}

void Wiener::welchsPeriodogram(float32_t *x, float32_t *Sbb,  arm_cfft_radix4_instance_f32* S, float32_t *WINDOW, float32_t EW) {
	//arm_std_f32 (const float32_t *pSrc, uint32_t blockSize, float32_t *pResult)
	Serial.println("The noise margin");
	//arm_std_f32(x, Wiener::blockSamples, stdNoise);
	//Result = sqrt((sumOfSquares - sum2 / blockSize) / (blockSize - 1))
	
	for(int i = 0; i< Wiener::blockSamples; i++){
		noiseMargin += x[i] * x[i];
	}
	noiseMargin = std::sqrt(noiseMargin) * 2;
	Serial.println(noiseMargin);
	Serial.println("Finding max in Noise input ");
	for(int i = 0; i<Wiener::bufferSamples; i++){
		if(noiseMax < x[i] ){
			noiseMax = x[i];
		}
	}
	Serial.println("noiseMax");
	Serial.println(noiseMax);

    for (uint32_t i = 0; i < NFFT; i++) {
        Sbb[i] = 0.0f;
    }

    float32_t x_framed[Wiener::FRAME_SAMPLES];
    float32_t x_framed_interpolated_with_zeros[2*Wiener::FRAME_SAMPLES];
    float32_t X_framed[2 * NFFT];
    float32_t X_framed_abs_sq[NFFT];
	
	arm_fill_f32(0.0f, historicalMax, numMax);

    for (uint32_t frame = 0; frame < frames; frame++) {
        uint32_t i_min = frame * OFFSET;
        uint32_t i_max = frame * OFFSET + Wiener::FRAME_SAMPLES;
        for (uint32_t i = i_min, j = 0; i < i_max; i++, j++) {
            x_framed[j] = x[i] * WINDOW[j];
        }
        Wiener::interpolate_with_zeros(x_framed, x_framed_interpolated_with_zeros, Wiener::FRAME_SAMPLES); 
        Wiener::pad_with_zeros(x_framed_interpolated_with_zeros, X_framed, 2*Wiener::FRAME_SAMPLES, 2*NFFT);
		
		arm_cfft_radix4_f32(S, X_framed);

		/*
		Serial.print("Padded signal for frame ");
		Serial.println(frame);
		Serial.print('[');

		for(int i = 0; i<Wiener::FRAME_SAMPLES; i++){
			Serial.print(x_framed[i]);
			Serial.print(", ");
		}
		Serial.println(']');
		*/

		/*
		Serial.print("frame ");
		Serial.println(frame);
		Serial.print('[');

		for(int i = 0; i< NFFT * 2; i++){
			Serial.print(X_framed[i]);
			Serial.print(", ");
		}
		Serial.println(']');
		*/

        Wiener::elementwise_mag_sq(X_framed, X_framed_abs_sq, 2 * NFFT);
		/*
		Serial.print("Magnitude squared for frame ");
		Serial.println(frame);
		Serial.print('[');

		for(int i = 0; i< NFFT; i++){
			Serial.print(X_framed_abs_sq[i]);
			Serial.print(", ");
		}
		Serial.println(']');
		*/
		
        for (uint32_t i = 0; i < NFFT; i++) {
            Sbb[i] = frame * Sbb[i] / (frame + 1) + X_framed_abs_sq[i] / (frame + 1);
        }
    }
}
// ----------------------------------- Wiener Functions END ----------------------------------- //

void Wiener::update(void){
	int start = micros();
	//Serial.println("Wiener::update");
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
		Serial.print("Wiener.cpp: No blocks");
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
		Serial.println("recording noise");
		if(blockCounter == 0){
			//initialize
			memset(Sbb, 0, sizeof(Sbb));
		}
		// check if the noise frames have been fully collected
		if (blockCounter < Wiener::bufferBlocks){
			// Serial.println("Recording noise"); 
			// Serial.println(blockCounter);
			// Serial.println(Wiener::bufferBlocks);
			//arm_q15_to_float (const q15_t *pSrc, float32_t *pDst, uint32_t blockSize)

			arm_q15_to_float (block->data, floatBlock, Wiener::blockSamples);
			//void *memcpy(void *dest, const void * src, size_t n)
			memcpy(inputBuffer + blockCounter*Wiener::blockSamples, floatBlock, sizeof(floatBlock));
			//std::copy(floatBlock, inputBuffer + blockCounter*Wiener::blockSamples, inputBuffer + (blockCounter+1)*Wiener::blockSamples);

			
			//if (blockCounter % 40 == 0){
			//	Serial.println("block->data: ");
			//	printArr(block->data, Wiener::blockSamples);
			//	Serial.println("printing the newly written portion of the input buffer after conversion to float");
			//	printArr(inputBuffer + blockCounter*Wiener::blockSamples, Wiener::blockSamples);
			//	delay(2000);
			// }
			//memcpy(&(inputBuffer[blockCounter*Wiener::blockSamples]), (float32_t *)block->data, Wiener::blockSamples*sizeof(float32_t));
		}
		else{
			//memcpy(&inputBuffer, truth, sizeof(truth));			

			processingSignal = true;
			Serial.println("Noise fully recorded, processing Sbb array");
			// end recording noise and start processing sbb matrix
			///*
			//Serial.println("(recordingNoise==true) inputBuffer: ");
			//printArr(inputBuffer, bufferSamples);
			//*/

			welchsPeriodogram(inputBuffer, Sbb, &S, WINDOW, EW); 
			// Serial.println("Sbb buffer after Periodogram");
			// printArr(Sbb, NFFT);

            /*
            Serial.print("[");
            for(int i = 0; i < NFFT; i++){
                Serial.print(Sbb[i]);
                Serial.print(", ");
            }
            Serial.println(']');
			*/


			//reset the variables
			processingSignal = false;
			recordingNoise = false;
			//if using blockCounter, don't forget to reset it back to 0
			blockCounter = 0;
			Serial.println("Noise array Sbb fully processed");
			for(int i = 0; i < bufferSamples; ++i) {outputBuffer[i] = 0;} // reset the output buffer
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
		//memcpy(&inputBuffer, truth, sizeof(truth));	
		//Serial.println("readyToProcess");
		processingSignal = true;
		//Serial.println("Start Processing Signal");

		// call Wiener filter
		Serial.println("EW:");
		Serial.println(EW);

		Serial.println("input buffer before passing to Wiener");
		//printArr(inputBuffer, Wiener::bufferSamples);
		wienerFilter(inputBuffer, outputBuffer, Sbb, &S, &IS, WINDOW, EW);
		Serial.println("output buffer after Wiener");
		// printArr(outputBuffer, Wiener::bufferSamples);
		// delay(2000);


		processingSignal = false;
		//Serial.println("End Processing Signal");
		blockCounter = 0;
    }

	


	// copy the recieved input block into the inputBuffer
	//memcpy(&(inputBuffer[blockCounter*Wiener::blockSamples]), (float32_t *)block->data, Wiener::blockSamples*sizeof(float32_t));
	arm_q15_to_float (block->data, floatBlock, Wiener::blockSamples);
	memcpy(inputBuffer + blockCounter*Wiener::blockSamples, floatBlock, sizeof(floatBlock));
	//std::copy(floatBlock, outputBuffer_q + blockCounter*Wiener::blockSamples, outputBuffer_q + (blockCounter+1)*Wiener::blockSamples);

	// transmit the relevant blocks worth of samples from outputBuffer (to playback)

	if(PASSTHRU_BUT_PROCESS){
		Serial.println("Passing Through");
		transmit(block);
	}
	else{
		//Serial.println("Copy output buff to b_new");
		//arm_float_to_q15 (const float32_t *pSrc, q15_t *pDst, uint32_t blockSize)
		//arm_float_to_q15(outputBuffer + blockCounter*Wiener::blockSamples, b_new->data, Wiener::blockSamples);
		std::copy(outputBuffer_q + blockCounter*Wiener::blockSamples, outputBuffer_q + (blockCounter+1)*Wiener::blockSamples, b_new->data);
		Serial.println("b_new q15");
		//printArr(b_new->data, Wiener::blockSamples);
		transmit(b_new);
		
		Serial.println("Done transmitting b_new");
		//std::copy(outputBuffer + blockCounter*Wiener::blockSamples, outputBuffer + (blockCounter+1)*Wiener::blockSamples, b_new->data);
	}
	

	for(int i = 0; i < bufferSamples; ++i) {outputBuffer[i] = 0;} // reset the output buffer
	Serial.println("Done resetting output array");
	

	// don't forget to increment the blockCounter
	blockCounter++;
	release(b_new);
	release(block);

	if (count > INTERVAL) {
		switcher = !switcher;
		digitalWrite(LED, switcher);
		count = 0;
	}
	int end = micros();
	Serial.print("Time to complete update: ");
	Serial.println(end-start);
}
