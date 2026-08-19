package main

import (
	"net/http"
)

type ProbesReadOp struct {
	key string
	Output chan string
}

type ProbesWriteOp struct {
	key string
	value string
	Output chan bool
}


var (
	readChannel chan ProbesReadOp = make(chan ProbesReadOp)
	WriteChannel chan ProbesWriteOp = make(chan ProbesWriteOp)
	ProbesState = make(map[string]string)
)

func InitProbeHandlers() {
	go func() {
		for {
			select {
			case readOp := <-readChannel:
				readOp.Output <- ProbesState[readOp.key]
			case writeOp := <-WriteChannel:
				ProbesState[writeOp.key] = writeOp.value
				writeOp.Output <- true
			}
		}
	}()
}

func healthzHandler(w http.ResponseWriter, _ *http.Request) {
	readOp := ProbesReadOp{
		key: "healthy",
		Output: make(chan string),
	}
	readChannel <- readOp
	probes := <-readOp.Output
	if probes == "true" {
		w.WriteHeader(http.StatusOK)
	} else {
		w.WriteHeader(http.StatusServiceUnavailable)
	}
}

func readyHandler(w http.ResponseWriter, _ *http.Request) {
	readOp := ProbesReadOp{
		key: "ready",
		Output: make(chan string),
	}
	readChannel <- readOp
	probes := <-readOp.Output
	if probes == "true" {
		w.WriteHeader(http.StatusOK)
	} else {
		w.WriteHeader(http.StatusServiceUnavailable)
	}
}
