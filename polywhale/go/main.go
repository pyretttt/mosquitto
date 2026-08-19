package main

import (
	"time"
	"net/http"
	"polywhale/handlers"
)

func main() {
	InitProbeHandlers()
	mux := http.NewServeMux()
	server := http.Server{
		Addr: "0.0.0.0:8080",
		Handler: mux,
		ReadTimeout:  5 * time.Second,
        WriteTimeout: 10 * time.Second,
	}
	mux.HandleFunc("/healthz", healthzHandler)
	mux.HandleFunc("/ready", readyHandler)
	mux.HandleFunc("/poly/get_user_positions", handlers.GetPositionsInfo)

	preServerHook()

	server.ListenAndServe()
}

func preServerHook() {
	writeHealthyOp := ProbesWriteOp{
		key: "healthy",
		value: "true",
		Output: make(chan bool),
	}
	WriteChannel <- writeHealthyOp
	<-writeHealthyOp.Output
	writeReadyOp := ProbesWriteOp{
		key: "ready",
		value: "true",
		Output: make(chan bool),
	}
	WriteChannel <- writeReadyOp
	<-writeReadyOp.Output
}