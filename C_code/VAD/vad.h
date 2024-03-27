#ifndef vad_h_
#define vad_h_

#include <Arduino.h>     // github.com/PaulStoffregen/cores/blob/master/teensy4/Arduino.h
#include <AudioStream.h> // github.com/PaulStoffregen/cores/blob/master/teensy4/AudioStream.h
#include <arm_math.h>    // github.com/PaulStoffregen/cores/blob/master/teensy4/arm_math.h

// Indicates that the code should just pass through the audio
// without any filtering (as opposed to doing nothing at all)
#define FIR_PASSTHRU ((const short *) 1)

class VAD : public AudioStream
{
public:
	int ENERGY_THRESHOLD;
	int ENERGY_DIFF_THRESHOLD;
	VAD(void): AudioStream(1,inputQueueArray), coeff_p(NULL) {
	}
	void begin(const short *cp, const int thresh, const int diff_thresh) {
		coeff_p = cp;
		ENERGY_THRESHOLD = thresh;
		ENERGY_DIFF_THRESHOLD = diff_thresh;
		// Initialize FIR instance (ARM DSP Math Library)
		if (coeff_p && (coeff_p != FIR_PASSTHRU)) {
			//if (arm_fir_init_q15(&fir_inst, n_coeffs, (q15_t *)coeff_p, &StateQ15[0], AUDIO_BLOCK_SAMPLES) != ARM_MATH_SUCCESS) {
				// n_coeffs must be an even number, 4 or larger
				coeff_p = NULL;
			//}
		}
	}
	void end(void) {
		coeff_p = NULL;
	}
	virtual void update(void);
		
	bool energyBasedVAD(const int16_t *pSrc, int16_t *pDst, uint32_t blockSize){
		// Simple energy-based Voice Activity Detection
		int current_energy = 0;
		for (uint32_t i = 0; i < blockSize; i++) {
			current_energy += pSrc[i] * pSrc[i];
			//current_energy += *((pSrc->p) + i) * *((pSrc->p) + i);
		}
	  
		int energy_diff = abs(current_energy - prev_energy);
		prev_energy = current_energy;
		memcpy(pDst, pSrc, blockSize);
		  
		/*Serial.print("Current energy: ");
		Serial.println(current_energy);
		Serial.print("Energy diff: ");
		Serial.println(energy_diff);*/	

		// Return the result of the VAD check
		return energy_diff > ENERGY_DIFF_THRESHOLD && current_energy > ENERGY_THRESHOLD;
	}
private:
	audio_block_t *inputQueueArray[1];
	int prev_energy = 0;

	// pointer to current coefficients or NULL or FIR_PASSTHRU
	const short *coeff_p;

	// ARM DSP Math library filter instance
	//arm_fir_instance_q15 fir_inst;
	//q15_t StateQ15[AUDIO_BLOCK_SAMPLES + FIR_MAX_COEFFS];
};

#endif
