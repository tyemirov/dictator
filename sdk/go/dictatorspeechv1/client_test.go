package dictatorspeechv1

import (
	"context"
	"errors"
	"io"
	"reflect"
	"strings"
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

var (
	errUploadOpen       = errors.New("upload open failed")
	errUploadSend       = errors.New("upload send failed")
	errUploadClose      = errors.New("upload close failed")
	errTranscribeFailed = errors.New("transcribe failed")
)

type fakeArtifactClient struct {
	stream           *fakeUploadStream
	err              error
	capturedMetadata metadata.MD
}

func (c *fakeArtifactClient) UploadArtifact(ctx context.Context, _ ...grpc.CallOption) (grpc.ClientStreamingClient[UploadArtifactChunk, UploadArtifactResponse], error) {
	c.capturedMetadata, _ = metadata.FromOutgoingContext(ctx)
	if c.err != nil {
		return nil, c.err
	}
	return c.stream, nil
}

func (*fakeArtifactClient) DownloadArtifact(context.Context, *DownloadArtifactRequest, ...grpc.CallOption) (grpc.ServerStreamingClient[DownloadArtifactChunk], error) {
	return nil, errors.New("not implemented")
}

type fakeUploadStream struct {
	grpc.ClientStream
	closeCount int
	sendErr    error
	sendErrAt  int
	sendCount  int
	chunks     []*UploadArtifactChunk
	response   *UploadArtifactResponse
	closeErr   error
}

func (s *fakeUploadStream) Send(chunk *UploadArtifactChunk) error {
	s.sendCount += 1
	if s.sendErrAt == s.sendCount {
		if s.sendErr != nil {
			return s.sendErr
		}
		return errUploadSend
	}
	s.chunks = append(s.chunks, chunk)
	return nil
}

func (s *fakeUploadStream) CloseAndRecv() (*UploadArtifactResponse, error) {
	s.closeCount += 1
	if s.closeErr != nil {
		return nil, s.closeErr
	}
	return s.response, nil
}

type fakeTranscriptionClient struct {
	response         *TranscribeResponse
	err              error
	request          *TranscribeRequest
	capturedMetadata metadata.MD
}

func (c *fakeTranscriptionClient) Transcribe(ctx context.Context, request *TranscribeRequest, _ ...grpc.CallOption) (*TranscribeResponse, error) {
	c.capturedMetadata, _ = metadata.FromOutgoingContext(ctx)
	c.request = request
	if c.err != nil {
		return nil, c.err
	}
	return c.response, nil
}

func (*fakeTranscriptionClient) SubmitTranscribeJob(context.Context, *TranscribeRequest, ...grpc.CallOption) (*SubmitTranscribeJobResponse, error) {
	return nil, errors.New("not implemented")
}

func (*fakeTranscriptionClient) GetTranscribeJob(context.Context, *GetTranscribeJobRequest, ...grpc.CallOption) (*GetTranscribeJobResponse, error) {
	return nil, errors.New("not implemented")
}

func (*fakeTranscriptionClient) CancelTranscribeJob(context.Context, *CancelTranscribeJobRequest, ...grpc.CallOption) (*CancelTranscribeJobResponse, error) {
	return nil, errors.New("not implemented")
}

func (*fakeTranscriptionClient) DiarizeAudio(context.Context, *DiarizeAudioRequest, ...grpc.CallOption) (*DiarizeAudioResponse, error) {
	return nil, errors.New("not implemented")
}

func (*fakeTranscriptionClient) SubmitDiarizeAudioJob(context.Context, *DiarizeAudioRequest, ...grpc.CallOption) (*SubmitDiarizeAudioJobResponse, error) {
	return nil, errors.New("not implemented")
}

func (*fakeTranscriptionClient) GetDiarizeAudioJob(context.Context, *GetDiarizeAudioJobRequest, ...grpc.CallOption) (*GetDiarizeAudioJobResponse, error) {
	return nil, errors.New("not implemented")
}

func (*fakeTranscriptionClient) CancelDiarizeAudioJob(context.Context, *CancelDiarizeAudioJobRequest, ...grpc.CallOption) (*CancelDiarizeAudioJobResponse, error) {
	return nil, errors.New("not implemented")
}

func newTestClient(artifactClient ArtifactServiceClient, transcriptionClient TranscriptionServiceClient) *Client {
	return &Client{
		artifactClient:      artifactClient,
		transcriptionClient: transcriptionClient,
		authToken:           "secret-token",
		authMetadataKey:     DefaultAuthMetadataKey,
		uploadChunkBytes:    3,
	}
}

func TestNewClientNormalizesConfig(t *testing.T) {
	defaultClient := NewClient(nil, ClientConfig{
		AuthToken:        "  token-value  ",
		AuthMetadataKey:  "  ",
		UploadChunkBytes: 0,
	})
	if defaultClient.authToken != "token-value" {
		t.Fatalf("auth token not trimmed: %q", defaultClient.authToken)
	}
	if defaultClient.authMetadataKey != DefaultAuthMetadataKey {
		t.Fatalf("default metadata key mismatch: %q", defaultClient.authMetadataKey)
	}
	if defaultClient.uploadChunkBytes != DefaultUploadArtifactChunkLen {
		t.Fatalf("default chunk length mismatch: %d", defaultClient.uploadChunkBytes)
	}

	customClient := NewClient(nil, ClientConfig{
		AuthMetadataKey:  "authorization",
		UploadChunkBytes: 7,
	})
	if customClient.authMetadataKey != "authorization" {
		t.Fatalf("custom metadata key mismatch: %q", customClient.authMetadataKey)
	}
	if customClient.uploadChunkBytes != 7 {
		t.Fatalf("custom chunk length mismatch: %d", customClient.uploadChunkBytes)
	}
}

func TestUploadArtifactSendsMetadataAuthAndChunkedContent(t *testing.T) {
	stream := &fakeUploadStream{
		response: &UploadArtifactResponse{
			Artifact: &ArtifactRef{ArtifactId: "artifact-123"},
		},
	}
	artifactClient := &fakeArtifactClient{stream: stream}
	client := newTestClient(artifactClient, &fakeTranscriptionClient{})

	artifact, err := client.UploadArtifact(context.Background(), UploadArtifactContentRequest{
		Filename:  " recording.webm ",
		MediaType: " audio/webm ",
		Content:   []byte("abcdefg"),
	})
	if err != nil {
		t.Fatalf("UploadArtifact returned error: %v", err)
	}
	if artifact.GetArtifactId() != "artifact-123" {
		t.Fatalf("artifact id mismatch: %q", artifact.GetArtifactId())
	}
	if got := artifactClient.capturedMetadata.Get(DefaultAuthMetadataKey); !reflect.DeepEqual(got, []string{"secret-token"}) {
		t.Fatalf("auth metadata mismatch: %#v", got)
	}
	if len(stream.chunks) != 4 {
		t.Fatalf("chunk count mismatch: %d", len(stream.chunks))
	}
	if metadataChunk := stream.chunks[0].GetMetadata(); metadataChunk.GetFilename() != "recording.webm" || metadataChunk.GetMediaType() != "audio/webm" {
		t.Fatalf("metadata chunk mismatch: %#v", metadataChunk)
	}
	for index, expected := range []string{"abc", "def", "g"} {
		if got := string(stream.chunks[index+1].GetContent()); got != expected {
			t.Fatalf("content chunk %d mismatch: %q", index, got)
		}
	}
}

func TestUploadArtifactAllowsEmptyAuthAndContent(t *testing.T) {
	stream := &fakeUploadStream{
		response: &UploadArtifactResponse{Artifact: &ArtifactRef{ArtifactId: "empty-artifact"}},
	}
	artifactClient := &fakeArtifactClient{stream: stream}
	client := &Client{
		artifactClient:   artifactClient,
		authMetadataKey:  DefaultAuthMetadataKey,
		uploadChunkBytes: 3,
	}

	artifact, err := client.UploadArtifact(context.Background(), UploadArtifactContentRequest{})
	if err != nil {
		t.Fatalf("UploadArtifact returned error: %v", err)
	}
	if artifact.GetArtifactId() != "empty-artifact" {
		t.Fatalf("artifact id mismatch: %q", artifact.GetArtifactId())
	}
	if got := artifactClient.capturedMetadata.Get(DefaultAuthMetadataKey); len(got) != 0 {
		t.Fatalf("unexpected auth metadata: %#v", got)
	}
	if len(stream.chunks) != 1 {
		t.Fatalf("expected only metadata chunk, got %d chunks", len(stream.chunks))
	}
}

func TestUploadArtifactErrors(t *testing.T) {
	tests := []struct {
		name           string
		artifactClient ArtifactServiceClient
	}{
		{
			name:           "open",
			artifactClient: &fakeArtifactClient{err: errUploadOpen},
		},
		{
			name: "metadata send",
			artifactClient: &fakeArtifactClient{stream: &fakeUploadStream{
				sendErrAt: 1,
			}},
		},
		{
			name: "content send",
			artifactClient: &fakeArtifactClient{stream: &fakeUploadStream{
				sendErrAt: 2,
			}},
		},
		{
			name: "close",
			artifactClient: &fakeArtifactClient{stream: &fakeUploadStream{
				closeErr: errUploadClose,
			}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := newTestClient(tt.artifactClient, &fakeTranscriptionClient{})
			_, err := client.UploadArtifact(context.Background(), UploadArtifactContentRequest{Content: []byte("abc")})
			if err == nil {
				t.Fatal("expected UploadArtifact error")
			}
			if got := err.Error(); !contains(got, "dictator.artifact_upload_failed") {
				t.Fatalf("error prefix mismatch: %q", got)
			}
		})
	}
}

func TestUploadArtifactClosesAndPreservesStatusAfterEarlyEOF(t *testing.T) {
	upstreamErr := status.Error(codes.InvalidArgument, "first upload chunk must contain metadata")
	stream := &fakeUploadStream{
		sendErrAt: 2,
		sendErr:   io.EOF,
		closeErr:  upstreamErr,
	}
	artifactClient := &fakeArtifactClient{stream: stream}
	client := newTestClient(artifactClient, &fakeTranscriptionClient{})

	_, err := client.UploadArtifact(context.Background(), UploadArtifactContentRequest{
		Filename:  "sample.wav",
		MediaType: "audio/wav",
		Content:   []byte("abcdef"),
	})
	if err == nil {
		t.Fatal("expected UploadArtifact error")
	}
	if got := status.Code(err); got != codes.InvalidArgument {
		t.Fatalf("status code mismatch: %v", got)
	}
	if got := err.Error(); !contains(got, "first upload chunk must contain metadata") {
		t.Fatalf("upstream validation detail missing: %q", got)
	}
	if stream.closeCount != 1 {
		t.Fatalf("expected CloseAndRecv once, got %d", stream.closeCount)
	}
}

func TestTranscribeUploadedArtifactSendsRequestWithoutAuthWhenTokenEmpty(t *testing.T) {
	transcriptionClient := &fakeTranscriptionClient{
		response: &TranscribeResponse{Text: "transcript text"},
	}
	client := &Client{
		transcriptionClient: transcriptionClient,
		authMetadataKey:     DefaultAuthMetadataKey,
	}

	response, err := client.TranscribeUploadedArtifact(context.Background(), TranscribeUploadedArtifactRequest{
		ArtifactID:          " artifact-123 ",
		LanguageCode:        " en ",
		ModelSize:           " base ",
		IncludeWordSegments: true,
		AutodetectLanguage:  true,
	})
	if err != nil {
		t.Fatalf("TranscribeUploadedArtifact returned error: %v", err)
	}
	if response.GetText() != "transcript text" {
		t.Fatalf("transcript mismatch: %q", response.GetText())
	}
	if got := transcriptionClient.capturedMetadata.Get(DefaultAuthMetadataKey); len(got) != 0 {
		t.Fatalf("unexpected auth metadata: %#v", got)
	}
	if transcriptionClient.request.GetAudioArtifactId() != "artifact-123" ||
		transcriptionClient.request.GetLanguageCode() != "en" ||
		transcriptionClient.request.GetModelSize() != "base" ||
		!transcriptionClient.request.GetIncludeWordSegments() ||
		!transcriptionClient.request.GetAutodetectLanguage() {
		t.Fatalf("request mismatch: %#v", transcriptionClient.request)
	}
}

func TestTranscribeUploadedArtifactError(t *testing.T) {
	client := newTestClient(&fakeArtifactClient{}, &fakeTranscriptionClient{err: errTranscribeFailed})
	_, err := client.TranscribeUploadedArtifact(context.Background(), TranscribeUploadedArtifactRequest{})
	if err == nil {
		t.Fatal("expected TranscribeUploadedArtifact error")
	}
	if got := err.Error(); !contains(got, "dictator.transcription_failed") {
		t.Fatalf("error prefix mismatch: %q", got)
	}
}

func TestTranscribeAudioUploadsThenTranscribes(t *testing.T) {
	artifactClient := &fakeArtifactClient{stream: &fakeUploadStream{
		response: &UploadArtifactResponse{Artifact: &ArtifactRef{ArtifactId: "artifact-from-upload"}},
	}}
	transcriptionClient := &fakeTranscriptionClient{
		response: &TranscribeResponse{Text: "combined transcript"},
	}
	client := newTestClient(artifactClient, transcriptionClient)

	response, err := client.TranscribeAudio(context.Background(), TranscribeAudioRequest{
		Filename:            "audio.webm",
		MediaType:           "audio/webm",
		Content:             []byte("abcdef"),
		LanguageCode:        "en",
		ModelSize:           "small",
		IncludeWordSegments: true,
		AutodetectLanguage:  true,
	})
	if err != nil {
		t.Fatalf("TranscribeAudio returned error: %v", err)
	}
	if response.Artifact.GetArtifactId() != "artifact-from-upload" {
		t.Fatalf("artifact mismatch: %q", response.Artifact.GetArtifactId())
	}
	if response.Transcription.GetText() != "combined transcript" {
		t.Fatalf("transcription mismatch: %q", response.Transcription.GetText())
	}
	if transcriptionClient.request.GetAudioArtifactId() != "artifact-from-upload" {
		t.Fatalf("transcription artifact id mismatch: %q", transcriptionClient.request.GetAudioArtifactId())
	}
}

func TestTranscribeAudioReturnsUploadError(t *testing.T) {
	client := newTestClient(&fakeArtifactClient{err: errUploadOpen}, &fakeTranscriptionClient{})
	_, err := client.TranscribeAudio(context.Background(), TranscribeAudioRequest{})
	if err == nil {
		t.Fatal("expected upload error")
	}
	if got := err.Error(); !contains(got, "dictator.artifact_upload_failed") {
		t.Fatalf("error prefix mismatch: %q", got)
	}
}

func TestTranscribeAudioReturnsTranscriptionError(t *testing.T) {
	client := newTestClient(
		&fakeArtifactClient{stream: &fakeUploadStream{
			response: &UploadArtifactResponse{Artifact: &ArtifactRef{ArtifactId: "artifact-from-upload"}},
		}},
		&fakeTranscriptionClient{err: errTranscribeFailed},
	)

	_, err := client.TranscribeAudio(context.Background(), TranscribeAudioRequest{})
	if err == nil {
		t.Fatal("expected transcription error")
	}
	if got := err.Error(); !contains(got, "dictator.transcription_failed") {
		t.Fatalf("error prefix mismatch: %q", got)
	}
}

func contains(haystack, needle string) bool {
	return strings.Contains(haystack, needle)
}
