package main

import (
	"fmt"

	speech "github.com/tyemirov/dictator/sdk/go/dictatorspeechv1"
	"google.golang.org/protobuf/encoding/protojson"
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
	payload, err := (protojson.MarshalOptions{UseProtoNames: true}).Marshal(&decoded)
	if err != nil {
		panic(err)
	}
	fmt.Println(string(payload))
}
