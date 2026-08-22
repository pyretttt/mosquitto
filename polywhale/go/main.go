package main

import (
	"time"
	"net/http"
	"context"
	"os"
	"database/sql"
	"fmt"
	"log"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	_ "github.com/lib/pq"

	"polywhale/handlers"
)

func main() {
	dbHost := os.Getenv("DB_HOST")
	dbName := os.Getenv("DB_NAME")
	dbPassword := os.Getenv("DB_PASSWORD")
	if dbHost == "" || dbName == "" || dbPassword == "" {
		log.Fatal("DB credentials are wrong")
		panic("DB credentials are wrong")
	}

    db, err := sql.Open("postgres", fmt.Sprintf("host=%s dbname=%s password=%s connect_timeout=5 sslmode=disable", dbHost, dbName, ))
	if err != nil {
		log.Fatal("Failed to connect to DB")
		panic("Failed to connect to DB")
	}
	defer db.Close()

	polyPositionsHandler := handlers.PolyPositionsHandler{Db: db}

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

	// Poly routers
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
				w.Write([]byte("User is required"))
			})
		})
		r.Get("/get_user_positions", (&polyPositionsHandler).GetPositionsInfoHandler)
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