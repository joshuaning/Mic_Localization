#include <Arduino.h>
#include "vad.h"
#define LED 13
#define INTERVAL 3000

static int count = 0;
static bool switcher = true;
const int ENERGY_THRESHOLD = 2911495;
const int ENERGY_DIFF_THRESHOLD = 10000000;


void AudioFilterFIR::update(void)
{
	count++;
	audio_block_t *block, *b_new;

	block = receiveReadOnly();
	if (!block) return;

	// If there's no coefficient table, give up.  
	if (coeff_p == NULL) {
		release(block);
		return;
	}

	// do passthru
	if (coeff_p == FIR_PASSTHRU) {
		// Just passthrough
		transmit(block);
		release(block);
		return;
	}

	// get a block for the FIR output
	b_new = allocate();
	if (b_new) {
		//arm_fir_fast_q15(&fir_inst, (q15_t *)block->data, (q15_t *)b_new->data, AUDIO_BLOCK_SAMPLES);
		serial.print(energyBasedVAD((q15_t *)block->data, (q15_t *)b_new->data, AUDIO_BLOCK_SAMPLES));
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

int prev_energy = 0;
bool energyBasedVAD(const audio_block_t *pSrc, audio_block_t *pDst, unit32_t blockSize){
    // Simple energy-based Voice Activity Detection
    int current_energy = 0;
	for (int i = 0; i < blockSize; i++) {
		current_energy += (int)pSrc[i] * (int)pSrc[i];
	}
  
	int energy_diff = abs(current_energy - prev_energy);
	prev_energy = current_energy;
	memcpy(pSrc, pDst, blockSize);
	  
	/*Serial.print("Current energy: ");
	Serial.println(current_energy);
	Serial.print("Energy diff: ");
	Serial.println(energy_diff);*/	

    // Return the result of the VAD check
    return energy_diff > ENERGY_DIFF_THRESHOLD && current_energy > ENERGY_THRESHOLD;
}