package polymarket

import (
	"context"
	"fmt"
	"strings"

	"polywhale/internal/logging"
)

// Port of poly-tui's top_page_service: maps raw Gamma responses into
// ready-to-render view models with pre-formatted labels.

const EventsPerPage = 30

type SearchType int

const (
	SearchQuery SearchType = iota
	SearchTag
	SearchID
)

func (s SearchType) String() string {
	switch s {
	case SearchTag:
		return "tag"
	case SearchID:
		return "id"
	default:
		return "query"
	}
}

func (s SearchType) Next() SearchType {
	switch s {
	case SearchQuery:
		return SearchTag
	case SearchTag:
		return SearchID
	default:
		return SearchQuery
	}
}

type EventsFilter struct {
	SearchType SearchType
	Query      string
	Page       int
}

type EventsData struct {
	Events     []Event
	NextCursor int
}

type UmaResolutionStatus int

const (
	UmaUnknown UmaResolutionStatus = iota
	UmaPending
	UmaResolved
	UmaFailed
)

func parseUmaResolutionStatus(value string) UmaResolutionStatus {
	switch strings.ToLower(value) {
	case "pending":
		return UmaPending
	case "resolved":
		return UmaResolved
	case "failed":
		return UmaFailed
	default:
		return UmaUnknown
	}
}

type ActivityKind int

const (
	ActivityMuted ActivityKind = iota
	ActivityPositive
	ActivityNegative
	ActivityAccent
	ActivityWarning
)

type Event struct {
	ID           string
	Title        string
	Slug         string
	Bookmarked   bool
	Volume24h    float64
	Markets      []Market
	MarketsCount int

	RankLabel         string
	BookmarkLabel     string
	VolumeLabel       string
	MarketsCountLabel string
}

func NewEvent(id, title, slug string, bookmarked bool, volume24h float64, markets []Market) Event {
	event := Event{
		ID:                id,
		Title:             title,
		Slug:              slug,
		Volume24h:         volume24h,
		Markets:           markets,
		MarketsCount:      len(markets),
		VolumeLabel:       formatVolumeCompact(volume24h),
		MarketsCountLabel: fmt.Sprintf("%dmkt", len(markets)),
	}
	event.SetBookmarked(bookmarked)
	return event
}

func (e *Event) SetRank(rank int) {
	e.RankLabel = fmt.Sprintf("%d", rank)
}

func (e *Event) SetBookmarked(bookmarked bool) {
	e.Bookmarked = bookmarked
	e.BookmarkLabel = bookmarkLabel(bookmarked)
}

type Market struct {
	Title            string
	Slug             string
	Bookmarked       bool
	YesMarketPrice   float64
	NoMarketPrice    float64
	Volume24h        float64
	Movement24h      float64
	Spread           float64
	ResolutionStatus UmaResolutionStatus

	RankLabel     string
	BookmarkLabel string
	YesLabel      string
	NoLabel       string
	VolumeLabel   string
	MovementLabel string
	MovementKind  ActivityKind
	SpreadLabel   string
}

func NewMarket(
	title, slug string,
	bookmarked bool,
	yesPrice, noPrice, volume24h, movement24h, spread float64,
	resolutionStatus UmaResolutionStatus,
) Market {
	market := Market{
		Title:            title,
		Slug:             slug,
		YesMarketPrice:   yesPrice,
		NoMarketPrice:    noPrice,
		Volume24h:        volume24h,
		Movement24h:      movement24h,
		Spread:           spread,
		ResolutionStatus: resolutionStatus,
		YesLabel:         FormatCents(yesPrice),
		NoLabel:          FormatCents(noPrice),
		VolumeLabel:      formatVolumeCompact(volume24h),
		MovementLabel:    formatMovement(movement24h),
		MovementKind:     movementKind(movement24h),
		SpreadLabel:      FormatCents(spread),
	}
	market.SetBookmarked(bookmarked)
	return market
}

func (m *Market) SetBookmarked(bookmarked bool) {
	m.Bookmarked = bookmarked
	m.BookmarkLabel = bookmarkLabel(bookmarked)
}

type SelectedMarket struct {
	Slug           string
	YesMarketPrice float64
	NoMarketPrice  float64
	Spread         float64
	Volume24h      float64
	Liquidity      float64
	OpenInterest   float64
	EndDate        string

	YesLabel          string
	NoLabel           string
	YesQuotesLabel    string
	NoQuotesLabel     string
	VolumeLabel       string
	LiquidityLabel    string
	OpenInterestLabel string
}

func NewSelectedMarket(
	slug string,
	yesPrice, noPrice, yesBid, yesAsk, noBid, noAsk, spread, volume24h, liquidity, openInterest float64,
	endDate string,
) SelectedMarket {
	return SelectedMarket{
		Slug:           slug,
		YesMarketPrice: yesPrice,
		NoMarketPrice:  noPrice,
		Spread:         spread,
		Volume24h:      volume24h,
		Liquidity:      liquidity,
		OpenInterest:   openInterest,
		EndDate:        endDate,
		YesLabel:       FormatCents(yesPrice),
		NoLabel:        FormatCents(noPrice),
		YesQuotesLabel: fmt.Sprintf("  bid %s / ask %s   spread %s",
			FormatCents(yesBid), FormatCents(yesAsk), FormatCents(spread)),
		NoQuotesLabel:     fmt.Sprintf("  bid %s / ask %s", FormatCents(noBid), FormatCents(noAsk)),
		VolumeLabel:       formatDollarCompact(volume24h),
		LiquidityLabel:    formatDollarCompact(liquidity),
		OpenInterestLabel: formatDollarCompact(openInterest),
	}
}

type ChartActivity struct {
	ChartLines []string
	Activities []ActivityEntry
}

type ActivityEntry struct {
	Time  string
	Label string
	Value string
	Kind  ActivityKind
}

// LoadEvents fetches a page of events (optionally filtered) and maps them to
// view models, dropping markets with malformed outcome data.
func (c *Client) LoadEvents(ctx context.Context, nextCursor int, filter *EventsFilter) (EventsData, error) {
	var (
		rawEvents []RawEvent
		err       error
	)
	if filter != nil {
		switch filter.SearchType {
		case SearchID:
			var event RawEvent
			event, err = c.EventByID(ctx, filter.Query)
			rawEvents = []RawEvent{event}
		default:
			// TODO: Implement tag search (falls back to query search, as in Rust).
			rawEvents, err = c.EventsFiltered(ctx, EventsPerPage, filter.Query, filter.Page)
		}
	} else {
		rawEvents, err = c.Events(ctx, EventsPerPage, nextCursor)
	}
	if err != nil {
		return EventsData{}, err
	}

	events := make([]Event, 0, len(rawEvents))
	for _, rawEvent := range rawEvents {
		markets := make([]Market, 0, len(rawEvent.Markets))
		for _, rawMarket := range rawEvent.Markets {
			market, ok := mapMarket(rawMarket)
			if ok {
				markets = append(markets, market)
			}
		}
		logging.Infof("app", "Event Id: %q", rawEvent.ID)
		events = append(events, NewEvent(
			rawEvent.ID,
			stringOr(rawEvent.Title, "N/A"),
			stringOr(rawEvent.Slug, "N/A"),
			false,
			floatOr(rawEvent.Volume24hr, 0),
			markets,
		))
	}
	return EventsData{Events: events, NextCursor: nextCursor + EventsPerPage}, nil
}

func mapMarket(raw RawMarket) (Market, bool) {
	if raw.OutcomePrices == nil || raw.Outcomes == nil || raw.Spread == nil {
		return Market{}, false
	}
	prices := *raw.OutcomePrices
	outcomes := *raw.Outcomes
	if len(prices) < 2 || len(outcomes) == 0 {
		return Market{}, false
	}

	var yesPrice, noPrice string
	switch strings.ToLower(outcomes[0]) {
	case "yes":
		yesPrice, noPrice = prices[0], prices[1]
	case "no":
		yesPrice, noPrice = prices[1], prices[0]
	default:
		logging.Errorf("app", "Unexpected outcome: %v for market %v", outcomes, stringOr(raw.Slug, ""))
		return Market{}, false
	}

	resolutionStatus := UmaUnknown
	if raw.UmaResolutionStatus != nil {
		resolutionStatus = parseUmaResolutionStatus(*raw.UmaResolutionStatus)
	}

	return NewMarket(
		stringOr(raw.Question, "N/A"),
		stringOr(raw.Slug, "N/A"),
		false,
		parseFloatOr(yesPrice, 0),
		parseFloatOr(noPrice, 0),
		floatOr(raw.Volume24hr, 0),
		floatOr(raw.OneDayPriceChange, 0),
		float64(*raw.Spread),
		resolutionStatus,
	), true
}

func stringOr(value *string, fallback string) string {
	if value != nil {
		return *value
	}
	return fallback
}

func floatOr(value *FlexFloat, fallback float64) float64 {
	if value != nil {
		return float64(*value)
	}
	return fallback
}

func parseFloatOr(value string, fallback float64) float64 {
	var parsed FlexFloat
	if err := parsed.UnmarshalJSON([]byte(`"` + value + `"`)); err != nil {
		return fallback
	}
	return float64(parsed)
}

func bookmarkLabel(bookmarked bool) string {
	if bookmarked {
		return "★"
	}
	return ""
}

func FormatCents(value float64) string {
	return fmt.Sprintf("%.3f¢", value)
}

func formatMovement(value float64) string {
	if value >= 0 {
		return fmt.Sprintf("+%.3f¢", value)
	}
	return fmt.Sprintf("%.3f¢", value)
}

func movementKind(value float64) ActivityKind {
	switch {
	case value > 0:
		return ActivityPositive
	case value < 0:
		return ActivityNegative
	default:
		return ActivityMuted
	}
}

func formatVolumeCompact(value float64) string {
	switch {
	case value >= 1_000_000:
		return fmt.Sprintf("%.0fM", value/1_000_000)
	case value >= 1_000:
		return fmt.Sprintf("%.0fk", value/1_000)
	default:
		return fmt.Sprintf("%.0f", value)
	}
}

func formatDollarCompact(value float64) string {
	switch {
	case value >= 1_000_000:
		return fmt.Sprintf("$%.1fM", value/1_000_000)
	case value >= 1_000:
		return fmt.Sprintf("$%.1fk", value/1_000)
	default:
		return fmt.Sprintf("$%.0f", value)
	}
}
