package main

import (
	"time"
	"net/http"
	"log"
	"polywhale/handlers"
)

func main() {
	mux := http.NewServeMux()
	server := http.Server{
		Addr: "0.0.0.0:8080",
		Handler: mux,
		ReadTimeout:  5 * time.Second,
        WriteTimeout: 10 * time.Second,
	}

	mux.HandleFunc("/poly/get_user_positions", handlers.GetPositionsInfo)
	log.Println("Server is running on port 8080")
	server.ListenAndServe()
}