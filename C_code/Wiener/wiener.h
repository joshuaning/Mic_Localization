#ifndef wiener_h_
#define wiener_h_

#include <Arduino.h>     // github.com/PaulStoffregen/cores/blob/master/teensy4/Arduino.h
#include <AudioStream.h> // github.com/PaulStoffregen/cores/blob/master/teensy4/AudioStream.h
#include <arm_math.h>    // github.com/PaulStoffregen/cores/blob/master/teensy4/arm_math.h
// Indicates that the code should just pass through the audio
// without any filtering (as opposed to doing nothing at all)
#define FIR_PASSTHRU ((const short *) 1)

class Wiener : public AudioStream
{
public:
	int ENERGY_THRESHOLD;
	int ENERGY_DIFF_THRESHOLD;

	
	//Constants
	static const int sampleRate = 44100;
	static const int blockSamples = 128;		// 2.9 ms at 44100 Hz

	//static const float32_t EW; // will compute when the window is created in begin()

	//User defined constants
	static const int bufferBlocks = floor(0.5 * sampleRate / blockSamples); 	// 0.5 seconds of audio
	static const int FRAME_SAMPLES = int(sampleRate * 0.02); 		//20ms frame, 7 ms for WER

	static const int bufferSamples = bufferBlocks*blockSamples;

	//static const float32_t WINDOW [FRAME_SAMPLES];

	// static arm_cfft_instance_f32 S;
	// static const int NFFT = 1024;
	
	Wiener(void): AudioStream(1,inputQueueArray), coeff_p(NULL) {



	} 

	void begin(const short *cp, const int thresh, const int diff_thresh) {
		coeff_p = cp;
		ENERGY_THRESHOLD = thresh;
		ENERGY_DIFF_THRESHOLD = diff_thresh;
		memset(vad_history, 0, sizeof(vad_history));
		// Initialize FIR instance (ARM DSP Math Library)
		if ((coeff_p != FIR_PASSTHRU)) {
			//if (arm_fir_init_q15(&fir_inst, n_coeffs, (q15_t *)coeff_p, &StateQ15[0], AUDIO_BLOCK_SAMPLES) != ARM_MATH_SUCCESS) {
				// n_coeffs must be an even number, 4 or larger
				coeff_p = ((const short *) 2);
			//}
		}
		//arm_cfft_init_f32 (arm_cfft_instance_f32 *S, uint16_t fftLen)
		
		

	}
	void end(void) {
		coeff_p = NULL;
	}
	
	virtual void update(void); 

	// Helper functions
    static void elementwise_mag_sq(float32_t *pSrc, float32_t *pDst, uint32_t length);
    static void interpolate_with_zeros(float32_t *pSrc, float32_t *pDst, uint32_t length);
    static void elementwise_divide(float32_t *pSrcA, float32_t *pSrcB, float32_t *pDst, uint32_t length);
    static void pad_with_zeros(float32_t *pSrc, float32_t *pDst, uint32_t srcLength, uint32_t dstLength);
    void wienerFilter(float32_t *inputBuffer, float32_t *outputBuffer, float32_t *Sbb, arm_cfft_radix4_instance_f32 *S, arm_cfft_radix4_instance_f32 *IS, float32_t *WINDOW, float32_t EW);
    void welchsPeriodogram(float32_t *x, float32_t *Sbb, arm_cfft_radix4_instance_f32 *S, float32_t *WINDOW, float32_t EW);
		
	bool energyBasedVAD(const int16_t *pSrc, int16_t *pDst, uint32_t blockSamples){
		// Simple energy-based Voice Activity Detection
		int current_energy = 0;
		for (uint32_t i = 0; i < blockSamples; i++) {
			current_energy += pSrc[i] * pSrc[i];
			//current_energy += *((pSrc->p) + i) * *((pSrc->p) + i);
		}
	  
		int energy_diff = abs(current_energy - prev_energy);
		prev_energy = current_energy;
		
		  
		Serial.print("Current energy: ");
		Serial.println(current_energy);
		Serial.print("Energy diff: ");
		Serial.println(energy_diff);


		if (energy_diff > ENERGY_DIFF_THRESHOLD && current_energy > ENERGY_THRESHOLD){
			memcpy(pDst, pSrc, blockSamples*sizeof(*pDst));
			if (vad_idx == 30) vad_idx = 0;
			vad_history[vad_idx] = true;
			vad_idx++;
		}
		else{
			//count how many elements are true in vad_history
			int count = 0;
			for (int i = 0; i < 30; i++){
				if (vad_history[i] == true){
					count++;
				}
			}
			if (count > 2){
				//if more than 4 elements are true, then it is a voice
				//copy the input to the output
				memcpy(pDst, pSrc, blockSamples*sizeof(*pDst));
			}
			else{
				memset(pDst, 0, blockSamples*sizeof(*pDst));
			}
			if (vad_idx == 30) vad_idx = 0;
			vad_history[vad_idx] = false;
			vad_idx++;
		}
		// Return the result of the VAD check
		return energy_diff > ENERGY_DIFF_THRESHOLD && current_energy > ENERGY_THRESHOLD;
	}
private:
	audio_block_t *inputQueueArray[1];
	int prev_energy = 0;

	// pointer to current coefficients or NULL or FIR_PASSTHRU
	const short *coeff_p;

	bool vad_history[30];
	int vad_idx = 0;

	// ARM DSP Math library filter instance
	//arm_fir_instance_q15 fir_inst;
	//q15_t StateQ15[AUDIO_BLOCK_SAMPLES + FIR_MAX_COEFFS];
};

#endif
