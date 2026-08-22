package handlers

import (
	"net/http"
	"time"
)

type responseStatus string

const (
	responseStatusOk responseStatus = "Ok"
	responseStatusFail responseStatus = "Failed"
)

var (
	client = http.Client{
		Timeout: 5 * time.Second,
		Transport: &http.Transport{
			MaxIdleConns:        100,
			MaxIdleConnsPerHost: 10,
			IdleConnTimeout:     80 * time.Second,
		},
	}
)
