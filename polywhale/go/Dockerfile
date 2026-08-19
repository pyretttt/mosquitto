FROM golang:alpine AS builder
WORKDIR /app

ENV CGO_ENABLED=0 GOOS=linux GOARCH=amd64

COPY go.mod go.sum ./
RUN go mod download

COPY ./ .

RUN go build -trimpath -ldflags="-s -w" -o polywhale .

FROM alpine:3.23.5 AS runner
WORKDIR /app

RUN addgroup -S appuser \
 && adduser -S -G appuser -H -s /sbin/nologin appuser

COPY --from=builder --chown=appuser:appuser /app/polywhale /app/polywhale

USER appuser

ENTRYPOINT ["/app/polywhale"]