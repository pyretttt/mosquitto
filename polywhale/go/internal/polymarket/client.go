package polymarket

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"time"
)

const gammaBaseURL = "https://gamma-api.polymarket.com"

// Client is the Go counterpart of poly-core's PolymarketClient. Only the
// Gamma endpoints used by the TUI are implemented.
type Client struct {
	httpClient *http.Client
	baseURL    string
}

func NewClient() *Client {
	return &Client{
		httpClient: &http.Client{Timeout: 30 * time.Second},
		baseURL:    gammaBaseURL,
	}
}

// Events fetches active, open events ordered newest-first.
func (c *Client) Events(ctx context.Context, limit, offset int) ([]RawEvent, error) {
	query := url.Values{
		"active":    {"true"},
		"closed":    {"false"},
		"limit":     {strconv.Itoa(limit)},
		"offset":    {strconv.Itoa(offset)},
		"ascending": {"false"},
	}
	var events []RawEvent
	if err := c.getJSON(ctx, "/events", query, &events); err != nil {
		return nil, err
	}
	return events, nil
}

// EventsFiltered searches events by free-text query.
func (c *Client) EventsFiltered(ctx context.Context, limit int, searchQuery string, page int) ([]RawEvent, error) {
	query := url.Values{
		"q":              {searchQuery},
		"limit_per_type": {strconv.Itoa(limit)},
		"page":           {strconv.Itoa(page)},
		"ascending":      {"false"},
	}
	var result struct {
		Events []RawEvent `json:"events"`
	}
	if err := c.getJSON(ctx, "/public-search", query, &result); err != nil {
		return nil, err
	}
	return result.Events, nil
}

// EventByID fetches a single event.
func (c *Client) EventByID(ctx context.Context, id string) (RawEvent, error) {
	var event RawEvent
	if err := c.getJSON(ctx, "/events/"+url.PathEscape(id), nil, &event); err != nil {
		return RawEvent{}, err
	}
	return event, nil
}

func (c *Client) getJSON(ctx context.Context, path string, query url.Values, out any) error {
	endpoint := c.baseURL + path
	if len(query) > 0 {
		endpoint += "?" + query.Encode()
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Accept", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("gamma API %s: unexpected status %d", path, resp.StatusCode)
	}
	return json.Unmarshal(body, out)
}

// RawEvent / RawMarket mirror the Gamma API wire format.
type RawEvent struct {
	ID         string      `json:"id"`
	Title      *string     `json:"title"`
	Slug       *string     `json:"slug"`
	Volume24hr *FlexFloat  `json:"volume24hr"`
	Markets    []RawMarket `json:"markets"`
}

type RawMarket struct {
	Question            *string    `json:"question"`
	Slug                *string    `json:"slug"`
	Outcomes            *FlexList  `json:"outcomes"`
	OutcomePrices       *FlexList  `json:"outcomePrices"`
	Volume24hr          *FlexFloat `json:"volume24hr"`
	OneDayPriceChange   *FlexFloat `json:"oneDayPriceChange"`
	Spread              *FlexFloat `json:"spread"`
	UmaResolutionStatus *string    `json:"umaResolutionStatus"`
}

// FlexFloat decodes a JSON number that Gamma may serialize as a number or a
// numeric string.
type FlexFloat float64

func (f *FlexFloat) UnmarshalJSON(data []byte) error {
	var asFloat float64
	if err := json.Unmarshal(data, &asFloat); err == nil {
		*f = FlexFloat(asFloat)
		return nil
	}
	var asString string
	if err := json.Unmarshal(data, &asString); err != nil {
		return fmt.Errorf("FlexFloat: cannot decode %s", string(data))
	}
	parsed, err := strconv.ParseFloat(asString, 64)
	if err != nil {
		return err
	}
	*f = FlexFloat(parsed)
	return nil
}

// FlexList decodes a JSON string array that Gamma frequently double-encodes
// as a JSON string (e.g. "[\"Yes\", \"No\"]").
type FlexList []string

func (l *FlexList) UnmarshalJSON(data []byte) error {
	var asList []string
	if err := json.Unmarshal(data, &asList); err == nil {
		*l = asList
		return nil
	}
	var asString string
	if err := json.Unmarshal(data, &asString); err != nil {
		return fmt.Errorf("FlexList: cannot decode %s", string(data))
	}
	return json.Unmarshal([]byte(asString), (*[]string)(l))
}
