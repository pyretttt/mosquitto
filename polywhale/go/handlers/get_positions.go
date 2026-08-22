package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"errors"
	"database/sql"
	"log"
)

type PolyPositionsHandler struct{
	Db *sql.DB
}

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
	response["status"] = responseStatusFail
	response["error"] = err.Error()

	panic(err.Error())
}

func (h *PolyPositionsHandler) GetPositionsInfoHandler(w http.ResponseWriter, r *http.Request) {
	var response map[string]any = make(map[string]any)
	w.Header().Set("Content-Type", "application/json")
	user, ok := r.Context().Value("user").(string)
	if !ok {
		handleError(w, response, errors.New("user is required"))
	}

	response["status"] = responseStatusOk

	defer func() {
		if r := recover(); r != nil {
			w.WriteHeader(http.StatusBadRequest)
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

	req, err := http.NewRequestWithContext(
		r.Context(),
		http.MethodGet,
		"https://data-api.polymarket.com/positions",
		nil,
	)
	if err != nil {
		handleError(w, response, err)
	}
	q := req.URL.Query()
	q.Set("sizeThreshold", "1")
	q.Set("limit", "100")
	q.Set("sortBy", "TOKENS")
	q.Set("sortDirection", "DESC")
	q.Set("user", user)
	req.URL.RawQuery = q.Encode()

	resp, err := client.Do(req)
	if err != nil {
		handleError(w, response, err)
	}
	defer resp.Body.Close()

	var positions []position
	if err := json.NewDecoder(resp.Body).Decode(&positions); err != nil {
		handleError(w, response, err)
	}

	h.SavePositionsToDB(positions)

	response["positions"] = positions

	jsonData, err := json.Marshal(response)
	if err != nil {
		handleError(w, response, err)
	}
	w.Write(jsonData)
	w.WriteHeader(http.StatusOK)
}

func (h *PolyPositionsHandler) SavePositionsToDB(positions []position) {
	jsonData, err := json.Marshal(positions)
	if err != nil {
		log.Fatal(err)
	}
	const query = `
		INSERT INTO user_positions (data)
		VALUES ($1)
		RETURNING id, created_at
	`
	var id int64
	var createdAt string

	err = h.Db.QueryRowContext(
		context.Background(),
		query,
		jsonData,
	).Scan(&id, &createdAt)
	if err != nil {
		log.Fatal(err)
	}
}