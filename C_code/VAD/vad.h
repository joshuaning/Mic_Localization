#ifndef vad_h_
#define vad_h_

#include <Arduino.h>     // github.com/PaulStoffregen/cores/blob/master/teensy4/Arduino.h
#include <AudioStream.h> // github.com/PaulStoffregen/cores/blob/master/teensy4/AudioStream.h
#include <arm_math.h>    // github.com/PaulStoffregen/cores/blob/master/teensy4/arm_math.h

// Indicates that the code should just pass through the audio
// without any filtering (as opposed to doing nothing at all)
#define FIR_PASSTHRU ((const short *) 1)

#define FIR_MAX_COEFFS 200

class AudioFilterFIR : public AudioStream
{
public:
	AudioFilterFIR(void): AudioStream(1,inputQueueArray), coeff_p(NULL) {
	}
	void begin(const short *cp, int n_coeffs) {
		coeff_p = cp;
		// Initialize FIR instance (ARM DSP Math Library)
		if (coeff_p && (coeff_p != FIR_PASSTHRU) && n_coeffs <= FIR_MAX_COEFFS) {
			if (arm_fir_init_q15(&fir_inst, n_coeffs, (q15_t *)coeff_p,
			  &StateQ15[0], AUDIO_BLOCK_SAMPLES) != ARM_MATH_SUCCESS) {
				// n_coeffs must be an even number, 4 or larger
				coeff_p = NULL;
			}
		}
	}
	void end(void) {
		coeff_p = NULL;
	}
	virtual void update(void);
private:
	audio_block_t *inputQueueArray[1];

	// pointer to current coefficients or NULL or FIR_PASSTHRU
	const short *coeff_p;

	// ARM DSP Math library filter instance
	arm_fir_instance_q15 fir_inst;
	q15_t StateQ15[AUDIO_BLOCK_SAMPLES + FIR_MAX_COEFFS];
};

#endif
