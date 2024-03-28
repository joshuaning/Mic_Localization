#include <Arduino.h>
#include <AudioStream.h> // github.com/PaulStoffregen/cores/blob/master/teensy4/AudioStream.h
#include "vad.h"
#define LED 13
#define INTERVAL 3000

static int count = 0;
static bool switcher = true;

void VAD::update(void)
{
	Serial.println("VAD::update");
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

	// get a block for the FIR output
	b_new = allocate();
	if (b_new) {
		//arm_fir_fast_q15(&fir_inst, (q15_t *)block->data, (q15_t *)b_new->data, AUDIO_BLOCK_SAMPLES);
		Serial.println("VAD::update::energyBasedVAD");
		Serial.println(energyBasedVAD( (int16_t *)block->data, (int16_t *)b_new->data, AUDIO_BLOCK_SAMPLES));
		transmit(b_new); // send the FIR output
		release(b_new);
	}
	release(block);


	if (count > INTERVAL) {
		switcher = !switcher;
		digitalWrite(LED, switcher);
		count = 0;
	}
}
