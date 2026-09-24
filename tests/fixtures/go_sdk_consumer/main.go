package main

import (
	"encoding/json"
	"fmt"

	speech "github.com/tyemirov/dictator/sdk/go/dictatorspeechv1"
	"google.golang.org/protobuf/proto"
)

func main() {
	request := &speech.SynthesizeSpeechRequest{
		PresetSpeaker: "baya",
		TextFormat:    speech.SynthesisTextFormat_SYNTHESIS_TEXT_FORMAT_SSML,
	}
	wire, err := proto.Marshal(request)
	if err != nil {
		panic(err)
	}
	var decoded speech.SynthesizeSpeechRequest
	if err := proto.Unmarshal(wire, &decoded); err != nil {
		panic(err)
	}
	if !proto.Equal(request, &decoded) {
		panic("synthesis fields changed during protobuf serialization")
	}
	usage := &speech.InputAudioUsage{SampleCount: 32001, SampleRateHz: 16000}
	responses := []proto.Message{
		&speech.TranscribeResponse{InputAudioUsage: usage},
		&speech.GetTranscribeJobResponse{InputAudioUsage: usage},
		&speech.AlignTranscriptResponse{InputAudioUsage: usage},
		&speech.GetAlignTranscriptJobResponse{InputAudioUsage: usage},
		&speech.GetDiarizeAudioJobResponse{InputAudioUsage: usage},
		&speech.RenderSubtitlesResponse{InputAudioUsage: usage},
		&speech.GetRenderSubtitlesJobResponse{InputAudioUsage: usage},
		&speech.ExtractReferenceSampleResponse{InputAudioUsage: usage},
		&speech.GetExtractReferenceSampleJobResponse{InputAudioUsage: usage},
	}
	for _, response := range responses {
		wire, err := proto.Marshal(response)
		if err != nil {
			panic(err)
		}
		retained := response.ProtoReflect().New().Interface()
		if err := proto.Unmarshal(wire, retained); err != nil {
			panic(err)
		}
		if !proto.Equal(response, retained) {
			panic("input audio quantities changed during protobuf serialization")
		}
	}
	evidence := struct {
		PresetSpeaker   string `json:"preset_speaker"`
		TextFormat      string `json:"text_format"`
		InputAudioUsage struct {
			SampleCount   uint64 `json:"sample_count"`
			SampleRateHz  uint32 `json:"sample_rate_hz"`
			ResponseCount int    `json:"response_count"`
		} `json:"input_audio_usage"`
	}{PresetSpeaker: decoded.GetPresetSpeaker(), TextFormat: decoded.GetTextFormat().String()}
	evidence.InputAudioUsage.SampleCount = usage.GetSampleCount()
	evidence.InputAudioUsage.SampleRateHz = usage.GetSampleRateHz()
	evidence.InputAudioUsage.ResponseCount = len(responses)
	payload, err := json.Marshal(evidence)
	if err != nil {
		panic(err)
	}
	fmt.Println(string(payload))
}
