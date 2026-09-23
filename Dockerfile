FROM golang:1.25-bookworm AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -o /enc-dashformer .

FROM debian:bookworm-slim
COPY --from=build /enc-dashformer /usr/local/bin/enc-dashformer
WORKDIR /work
ENTRYPOINT ["enc-dashformer"]
CMD ["--help"]
