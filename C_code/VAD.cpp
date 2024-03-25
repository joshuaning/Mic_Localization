#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <sndfile.hh>

// Function to compute the energy of an audio signal frame-by-frame
std::vector<double> computeFrameEnergy(const std::vector<double>& signal, int frameLength, int hopLength) {
    std::vector<double> energy;
    for(size_t i = 0; i < signal.size(); i += hopLength) {
        double sum = 0.0;
        for(size_t j = i; j < i + frameLength && j < signal.size(); ++j) {
            sum += signal[j] * signal[j];
        }
        energy.push_back(sum);
    }
    return energy;
}

// Function for a simple voice activity detection (VAD)
std::vector<bool> vad(const std::vector<double>& signal, int frameLength, int hopLength, double energyThreshold) {
    // Normalize signal
    double maxVal = *std::max_element(signal.begin(), signal.end());
    std::vector<double> normalizedSignal(signal.size());
    std::transform(signal.begin(), signal.end(), normalizedSignal.begin(),
                   [&maxVal](double val) { return val / maxVal; });

    // Compute the energy of audio frames
    std::vector<double> energy = computeFrameEnergy(normalizedSignal, frameLength, hopLength);

    // Normalize energy
    double maxEnergy = *std::max_element(energy.begin(), energy.end());
    for (double & e : energy) {
        e /= maxEnergy;
    }

    // Detect voice activity
    std::vector<bool> vadResult(energy.size());
    std::transform(energy.begin(), energy.end(), vadResult.begin(),
                   [energyThreshold](double e) { return e > energyThreshold; });

    return vadResult;
}

// Function to isolate speech
std::vector<double> isolateSpeech(const std::vector<double>& signal, int frameLength, int hopLength, double energyThreshold) {
    std::vector<bool> vadResult = vad(signal, frameLength, hopLength, energyThreshold);
    std::vector<double> speechSignal(signal.size(), 0.0);

    for(size_t i = 0; i < vadResult.size(); i++) {
        if(vadResult[i]) {
            size_t start = i * hopLength;
            size_t end = std::min(start + frameLength, signal.size());
            std::copy(signal.begin() + start, signal.begin() + end, speechSignal.begin() + start);
        }
    }

    return speechSignal;
}

int main() {
    // Open the WAV file
    const char* filePath = "test.wav";
    SF_INFO sfInfo;
    SNDFILE* inFile = sf_open(filePath, SFM_READ, &sfInfo);

    if (inFile == nullptr) {
        std::cout << "Cannot open input file!\n";
        return 1;
    }

    // Read samples
    std::vector<double> signal(sfInfo.frames);
    sf_read_double(inFile, &signal[0], sfInfo.frames);
    sf_close(inFile);

    // Isolate speech from the signal
    std::vector<double> isolatedSpeech = isolateSpeech(signal, 1024, 512, 0.01);

    // Save the isolated speech to a new WAV file
    const char* outputFilePath = "isolated_speech_output.wav";
    SF_INFO sfOutInfo = sfInfo; // Copy input SF_INFO for output file
    SNDFILE* outFile = sf_open(outputFilePath, SFM_WRITE, &sfOutInfo);

    if (outFile == nullptr) {
        std::cout << "Cannot open output file!\n";
        return 1;
    }

    sf_write_double(outFile, &isolatedSpeech[0], sfOutInfo.frames);
    sf_close(outFile);

    std::cout << "Isolated speech saved to " << outputFilePath << std::endl;
    return 0;
}
