#include <Arduino.h>
#include <AudioStream.h> // github.com/PaulStoffregen/cores/blob/master/teensy4/AudioStream.h
#include "wiener.h"
#include <stdio.h>
#define LED 13
#define INTERVAL 3000

static int count = 0;
static bool switcher = true;

const int BufferSize = 3 * 44100 / 128;
int16_t buff[BufferSize * 128];

int counter = 0;
bool full = false;

void Wiener::update(void)
{
	Serial.println("Wiener::update");
	count++;
	audio_block_t *block, *b_new, *big;

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
	b_new = allocate();
	big = allocate();
	// get a block for the FIR output
	/*
	
	*/

	/*if(counter == BufferSize){
		full = true;
	}

	if(counter % BufferSize == 0 && full){
		filterBuffer();
	}

	buff[(counter % BufferSize)*128] = b_new;
	std::copy(buff + (counter % BufferSize)*128, buff + ((1+counter) % BufferSize)*128), */

	if(counter < BufferSize && !full){
		Serial.println("Buffer not full");
		if(b_new){
			memcpy(&(buff[counter*128]), block->data, 128*sizeof(int16_t));
			counter++;
		}
		// transmit(b_new); // send the FIR output
		
	}
	else{
		Serial.println("Buffer full");
		if (counter >= BufferSize){
			full = true;
			counter = 0;
		}
		Serial.println("reset counter");
		 // the block that we are ready to transmit
		// for(int i = 0; i < 128; i++){
		// 	Serial.println("Copy data to big");
		// 	big->data[i] = buff[counter*128 + i];
		// }
		std::copy(buff + (counter)*128, buff + (counter+1)*128, big->data);
		transmit(big);
		
		Serial.println("Transmit oldest block");
		if(b_new){
			memcpy(&(buff[counter*128]), block->data, 128*sizeof(int16_t));
		}
		Serial.println("Put new block in buffer");
		/*
		for (int i = 0; i < BufferSize; i++){
			audio_block_t *big;
			std::copy(buff + i*128, buff + (i+1)*128, big->data);
			transmit(big);
			release(big);
		}
		*/
		
		
		// delay(1000);
		// process the whole buffer here
	}
	release(b_new);
	release(block);
	release(big);
	Serial.println("Release b_new and block");
	//arm_fir_fast_q15(&fir_inst, (q15_t *)block->data, (q15_t *)b_new->data, AUDIO_BLOCK_SAMPLES);
	/*
	Serial.println("Wiener::update::energyBasedVAD");
	Serial.println(energyBasedVAD( (int16_t *)block->data, (int16_t *)b_new->data, AUDIO_BLOCK_SAMPLES));
	transmit(b_new); // send the FIR output
	release(b_new);
	*/
	// transmit(b_new); // send the FIR output
	// release(b_new);
	// release(block);


	if (count > INTERVAL) {
		switcher = !switcher;
		digitalWrite(LED, switcher);
		count = 0;
	}
}
