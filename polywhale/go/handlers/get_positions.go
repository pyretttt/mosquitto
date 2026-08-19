package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"errors"
)

type position struct {
	Title string
	Slug string
	EventSlug string
	Outcome string
	EndDate string
	AvgPrice float64
	PositionSize float64
	InitialValue float64
	CurrentValue float64
	CashPnl float64
	PercentPnl float64
	CurPrice float64
}

func handleError(w http.ResponseWriter, response map[string]any, err error) {
	w.WriteHeader(http.StatusBadRequest)
	response["status"] = "Failed"
	response["error"] = err.Error()

	panic(err.Error())
}

func GetPositionsInfo(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	user := r.URL.Query().Get("user")
	var response map[string]any = make(map[string]any)
	response["status"] = "Ok"

	defer func() {
		if r := recover(); r != nil {
			w.WriteHeader(http.StatusBadRequest)
			response["status"] = "Failed"
			jsonData, err := json.Marshal(response)
			if err != nil {
				w.WriteHeader(http.StatusInternalServerError)
				w.Write([]byte(err.Error()))
				return
			}
			w.Write(jsonData)
			return
		}
	}()

	if len(user) == 0 {
		handleError(w, response, errors.New("user is required"))
	}

	req, err := http.NewRequest(
		"GET",
		fmt.Sprintf("https://data-api.polymarket.com/positions?sizeThreshold=1&limit=100&sortBy=TOKENS&sortDirection=DESC&user=%s", user),
		nil,
	)
	if err != nil {
		handleError(w, response, err)
	}

	resp, err := client.Do(req)
	if err != nil {
		handleError(w, response, err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		handleError(w, response, err)
	}

	var positions []position
	err = json.Unmarshal(body, &positions)
	if err != nil {
		handleError(w, response, err)
	}

	response["positions"] = positions

	jsonData, err := json.Marshal(response)
	if err != nil {
		handleError(w, response, err)
	}
	w.Write(jsonData)
	w.WriteHeader(http.StatusOK)
}