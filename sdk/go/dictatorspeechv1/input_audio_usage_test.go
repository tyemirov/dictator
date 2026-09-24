package dictatorspeechv1_test

import (
	"testing"

	speech "github.com/tyemirov/dictator/sdk/go/dictatorspeechv1"
	"google.golang.org/protobuf/proto"
)

func TestInputAudioUsagePreservesExactQuantitiesAndPresence(t *testing.T) {
	for _, usage := range []*speech.InputAudioUsage{
		nil,
		{SampleCount: 0, SampleRateHz: 16000},
		{SampleCount: 32001, SampleRateHz: 16000},
		{SampleCount: 1<<53 + 1, SampleRateHz: 48000},
	} {
		for _, message := range []proto.Message{
			&speech.TranscribeResponse{InputAudioUsage: usage},
			&speech.GetTranscribeJobResponse{InputAudioUsage: usage},
			&speech.AlignTranscriptResponse{InputAudioUsage: usage},
			&speech.GetAlignTranscriptJobResponse{InputAudioUsage: usage},
			&speech.GetDiarizeAudioJobResponse{InputAudioUsage: usage},
			&speech.RenderSubtitlesResponse{InputAudioUsage: usage},
			&speech.GetRenderSubtitlesJobResponse{InputAudioUsage: usage},
			&speech.ExtractReferenceSampleResponse{InputAudioUsage: usage},
			&speech.GetExtractReferenceSampleJobResponse{InputAudioUsage: usage},
		} {
			encoded, err := proto.Marshal(message)
			if err != nil {
				t.Fatal(err)
			}
			decoded := message.ProtoReflect().New().Interface()
			if err := proto.Unmarshal(encoded, decoded); err != nil {
				t.Fatal(err)
			}
			if !proto.Equal(message, decoded) {
				t.Fatalf("measurement changed during serialization: %v -> %v", message, decoded)
			}
			field := decoded.ProtoReflect().Descriptor().Fields().ByName("input_audio_usage")
			if decoded.ProtoReflect().Has(field) != (usage != nil) {
				t.Fatalf("measurement presence changed: %v", decoded)
			}
		}
	}
}
