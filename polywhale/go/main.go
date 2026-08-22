package main

import (
	"time"
	"net/http"
	"context"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"

	"polywhale/handlers"
)

func main() {
	InitProbeHandlers()
	r := chi.NewRouter()
	r.Use(middleware.Logger)
	server := http.Server{
		Addr: "0.0.0.0:8080",
		Handler: r,
		ReadTimeout:  5 * time.Second,
        WriteTimeout: 10 * time.Second,
	}

	// Routes with no middleware
	r.Group(func(r chi.Router) {
		r.Get("/healthz", healthzHandler)
		r.Get("/ready", readyHandler)
	})

	polyRouter := chi.NewRouter().Group(func(r chi.Router) {
		r.Use(func(h http.Handler) http.Handler {
			return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				query_user := r.URL.Query().Get("user")
				if query_user != "" {
					ctx := context.WithValue(r.Context(), "user", query_user)
					h.ServeHTTP(w, r.WithContext(ctx))
					return
				}

				w.WriteHeader(http.StatusBadRequest)
				w.Write([]byte("user is required"))
			})
		})
		r.Get("/get_user_positions", handlers.GetPositionsInfoHandler)

	})

	r.Mount("/poly", polyRouter)

	preServeHook()

	server.ListenAndServe()
}

func preServeHook() {
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