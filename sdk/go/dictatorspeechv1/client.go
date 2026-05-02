package dictatorspeechv1

import (
	"context"
	"fmt"
	"strings"

	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
)

const (
	DefaultAuthMetadataKey        = "x-dictator-token"
	DefaultUploadArtifactChunkLen = 1024 * 1024
)

type ClientConfig struct {
	AuthToken        string
	AuthMetadataKey  string
	UploadChunkBytes int
}

type Client struct {
	artifactClient      ArtifactServiceClient
	transcriptionClient TranscriptionServiceClient
	authToken           string
	authMetadataKey     string
	uploadChunkBytes    int
}

type UploadArtifactContentRequest struct {
	Filename  string
	MediaType string
	Content   []byte
}

type TranscribeUploadedArtifactRequest struct {
	ArtifactID          string
	LanguageCode        string
	ModelSize           string
	IncludeWordSegments bool
	AutodetectLanguage  bool
}

type TranscribeAudioRequest struct {
	Filename            string
	MediaType           string
	Content             []byte
	LanguageCode        string
	ModelSize           string
	IncludeWordSegments bool
	AutodetectLanguage  bool
}

type TranscribeAudioResponse struct {
	Artifact      *ArtifactRef
	Transcription *TranscribeResponse
}

func NewClient(conn grpc.ClientConnInterface, cfg ClientConfig) *Client {
	authMetadataKey := strings.TrimSpace(cfg.AuthMetadataKey)
	if authMetadataKey == "" {
		authMetadataKey = DefaultAuthMetadataKey
	}

	uploadChunkBytes := cfg.UploadChunkBytes
	if uploadChunkBytes <= 0 {
		uploadChunkBytes = DefaultUploadArtifactChunkLen
	}

	return &Client{
		artifactClient:      NewArtifactServiceClient(conn),
		transcriptionClient: NewTranscriptionServiceClient(conn),
		authToken:           strings.TrimSpace(cfg.AuthToken),
		authMetadataKey:     authMetadataKey,
		uploadChunkBytes:    uploadChunkBytes,
	}
}

func (c *Client) UploadArtifact(ctx context.Context, request UploadArtifactContentRequest) (*ArtifactRef, error) {
	artifactStream, err := c.artifactClient.UploadArtifact(c.contextWithAuth(ctx))
	if err != nil {
		return nil, fmt.Errorf("dictator.artifact_upload_failed: %w", err)
	}

	if err := artifactStream.Send(&UploadArtifactChunk{
		Payload: &UploadArtifactChunk_Metadata{
			Metadata: &UploadArtifactMetadata{
				Filename:  strings.TrimSpace(request.Filename),
				MediaType: strings.TrimSpace(request.MediaType),
			},
		},
	}); err != nil {
		return nil, fmt.Errorf("dictator.artifact_upload_failed: %w", err)
	}

	for start := 0; start < len(request.Content); start += c.uploadChunkBytes {
		end := start + c.uploadChunkBytes
		if end > len(request.Content) {
			end = len(request.Content)
		}
		if err := artifactStream.Send(&UploadArtifactChunk{
			Payload: &UploadArtifactChunk_Content{
				Content: request.Content[start:end],
			},
		}); err != nil {
			return nil, fmt.Errorf("dictator.artifact_upload_failed: %w", err)
		}
	}

	artifactResponse, err := artifactStream.CloseAndRecv()
	if err != nil {
		return nil, fmt.Errorf("dictator.artifact_upload_failed: %w", err)
	}
	return artifactResponse.GetArtifact(), nil
}

func (c *Client) TranscribeUploadedArtifact(ctx context.Context, request TranscribeUploadedArtifactRequest) (*TranscribeResponse, error) {
	transcriptionResponse, err := c.transcriptionClient.Transcribe(c.contextWithAuth(ctx), &TranscribeRequest{
		AudioArtifactId:     strings.TrimSpace(request.ArtifactID),
		LanguageCode:        strings.TrimSpace(request.LanguageCode),
		ModelSize:           strings.TrimSpace(request.ModelSize),
		IncludeWordSegments: request.IncludeWordSegments,
		AutodetectLanguage:  request.AutodetectLanguage,
	})
	if err != nil {
		return nil, fmt.Errorf("dictator.transcription_failed: %w", err)
	}
	return transcriptionResponse, nil
}

func (c *Client) TranscribeAudio(ctx context.Context, request TranscribeAudioRequest) (*TranscribeAudioResponse, error) {
	artifact, err := c.UploadArtifact(ctx, UploadArtifactContentRequest{
		Filename:  request.Filename,
		MediaType: request.MediaType,
		Content:   request.Content,
	})
	if err != nil {
		return nil, err
	}

	transcription, err := c.TranscribeUploadedArtifact(ctx, TranscribeUploadedArtifactRequest{
		ArtifactID:          artifact.GetArtifactId(),
		LanguageCode:        request.LanguageCode,
		ModelSize:           request.ModelSize,
		IncludeWordSegments: request.IncludeWordSegments,
		AutodetectLanguage:  request.AutodetectLanguage,
	})
	if err != nil {
		return nil, err
	}

	return &TranscribeAudioResponse{
		Artifact:      artifact,
		Transcription: transcription,
	}, nil
}

func (c *Client) contextWithAuth(ctx context.Context) context.Context {
	if c.authToken == "" {
		return ctx
	}
	return metadata.AppendToOutgoingContext(ctx, c.authMetadataKey, c.authToken)
}
