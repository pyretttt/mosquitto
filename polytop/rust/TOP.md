# Polytop Terminal UI Concept

Polytop should feel closer to `btop` than classic `top`: dense, keyboard-first,
pane-based, colorful, and always alive. The main screen should make it possible
to monitor easily available Polymarket market data, not just a list of rows.

All market data in this document should come from public, unauthenticated
Polymarket APIs:

- Gamma API: market and event discovery, titles, tags, end dates, volumes,
  liquidity, outcome prices, best bid/ask, spread, and event comments.
- CLOB API: order books, price history, and market websocket updates.
- Data API: global/per-market open interest and public trade history.
- Local app state: watchlist membership, filters, selected row, latency, and
  connection status.

## Visual Direction

- Background: near-black `#0B1020`
- Pane borders: muted slate `#334155`
- Primary accent: electric cyan `#22D3EE`
- Positive / yes: green `#22C55E`
- Negative / no: red `#EF4444`
- Warning / large move: amber `#F59E0B`
- Muted text: gray `#94A3B8`
- Focused pane border: cyan, optionally bold
- Selected row: deep blue background with bright text
- Recently changed prices: flash green/red for one tick, then fade
- Pane titles live in the border, prefixed by the number key that focuses them

Use Unicode box drawing when available, with ASCII fallback:

- Unicode: `╭─╮ │ ╰─╯ ├ ┤`
- ASCII fallback: `+--+ | +--+`

## Main Page Layout

The bottom command popup is permanently reserved. The main dashboard should never
draw under it.

```text
╭─ POLYTOP ───────────────────────────────────────────────────────────── 17:42:31 ─╮
│ net: online   ws: live   latency: 42ms   refresh: 500ms   mode: observe          │
├─ [1] - Market Snapshot ───────┬─ [2] - Top Markets ──────────────────────────────┤
│                               │                                                  │
│  Loaded 24h Vol $18.4M        │  #  Market                         Yes   No  24h │
│  Global OI      $92.1M        │  1  Will BTC hit 100k in 2026?     63¢  38¢ 842k │
│  Loaded Markets 250           │  2  Fed cuts rates by Sep?         41¢  60¢ 611k │
│  Top Move       +18.2¢        │  3  Lakers win tonight?            55¢  46¢ 570k │
│                               │  4  ETH ETF inflows above $1B?     72¢  29¢ 510k │
│  Top Tags                     │  5  Trump wins popular vote?       49¢  52¢ 421k │
│  Politics      ████████ 42%   │  6  CPI below forecast?            36¢  65¢ 390k │
│  Crypto        ██████░░ 31%   │  7  SpaceX launch this week?       83¢  18¢ 311k │
│  Sports        ████░░░░ 18%   │  8  Oil closes above $90?          22¢  79¢ 280k │
│  Culture       ██░░░░░░  9%   │                                                  │
├─ [3] - Watchlist ─────────────┼─ [4] - Selected Market ──┬─ [5] - Activity ──────┤
│                               │                          │                       │
│  BTC 100k       YES 63¢ +4¢   │ Will BTC hit 100k...     │ 17:42 price +4¢       │
│  Fed cuts       YES 41¢ -2¢   │                          │ 17:41 best bid 62¢    │
│  Lakers         YES 55¢ +1¢   │  yes  63¢  bid 62 / ask64│ 17:40 trade 219 @45¢  │
│  CPI below      YES 36¢ -5¢   │  no   38¢  bid 37 / ask39│ 17:39 spread 2¢       │
│  SpaceX launch  YES 83¢ +8¢   │                          │ 17:38 watchlist add   │
│                               │  volume 24h    $842.1k   │ 17:37 event comment   │
│                               │  liquidity     $184.3k   │                       │
│                               │  end date      2026-12-31│                       │
├─ [6] - Order Book / Depth ───────────────────────────────────────────────────────┤
│  YES bids                     spread 2¢                         YES asks         │
│  62¢  █████████████  18.2k              64¢  ███████░░░░░░  8.4k                 │
│  61¢  ████████░░░░░  11.0k              65¢  ███████████░░  14.7k                │
│  60¢  █████░░░░░░░░  6.6k               66¢  ████░░░░░░░░░  5.2k                 │
├─ Command Popup ──────────────────────────────────────────────────────────────────┤
│  f find market   1-6 focus pane   ↑↓ move   Enter open   w watch   ? help        │
│  Filter: politics volume>100k sort:move                               NORMAL     │
╰──────────────────────────────────────────────────────────────────────────────────╯
```

## Pane Responsibilities

### Header

Shows process-level status:

- App name and current time
- Network status
- WebSocket state
- API latency
- Refresh interval
- Current mode, usually `observe`

This gives confidence that the terminal is live without consuming a full pane.

### Market Snapshot

High-level pulse for the currently loaded market universe:

- Loaded 24h volume, summed from loaded Gamma `volume24hr` values
- Global open interest from Data API `GET /oi`
- Loaded market count after filters and pagination
- Largest recent price move from loaded Gamma price-change fields, or from
  locally cached price history when the field is missing
- Top tag distribution from loaded events/markets

This pane should make the product feel more like a dashboard than a row viewer.

### Top Markets

The main `top`-like feature. Suggested columns:

- `#`: rank after current sorting/filtering
- `Market`: truncated title
- `Yes`: current yes price
- `No`: current no price
- `24h`: market 24h volume

Possible alternate columns toggled with `c`:

- `Liq`: liquidity
- `24h`: 24h volume
- `Move`: price movement
- `End`: resolution date
- `Spread`: ask/bid spread
- `OI`: open interest from Data API for the market condition ID

Rows should support quick visual cues:

- Green/red price flashes on tick updates
- Amber marker for large recent moves
- Dimmed rows for markets ending soon or inactive markets
- Cyan left gutter for the selected row

### Watchlist

User-pinned markets with compact price movement. This should be visible even
while browsing broader market lists.

Interactions:

- `w`: add/remove selected market
- `3`: focus watchlist pane
- `x`: remove from watchlist

### Selected Market

Details for the currently highlighted row:

- Full market title
- Yes/no prices
- Bid/ask spread
- 24h volume
- Liquidity
- End date
- Open interest when loaded from Data API

If no row is selected, show a short onboarding hint.

### Activity

Small event stream:

- Price jumps
- Best bid/ask changes
- Recent public trades
- Spread changes
- Watchlist updates
- Connection warnings
- New event comments

Use color heavily here, but keep messages terse.

### Order Book / Depth

A horizontal market-depth preview for the selected market. It should not try to
replace a full trading screen, only answer: "Is there enough liquidity near this
price?"

Potential modes:

- `depth`: bid and ask levels for the selected outcome token
- `trades`: recent trade tape
- `chart`: tiny sparkline

Mode can be toggled with `d`.

## Data Sources By Pane

### Header

- Current time: local system clock.
- Network status, API latency, refresh interval, and mode: local app state.
- WebSocket state: local state for the CLOB market websocket connection to
  `wss://ws-subscriptions-clob.polymarket.com/ws/market`.

### Market Snapshot

- Loaded markets, 24h volume, liquidity, tags, prices, end dates:
  `GET https://gamma-api.polymarket.com/markets`.
- Event grouping and event-level tags/comments metadata:
  `GET https://gamma-api.polymarket.com/events`.
- Global open interest:
  `GET https://data-api.polymarket.com/oi`.
- Recent price moves: Gamma market price-change fields when present, otherwise
  local comparison against cached CLOB prices or price history.

### Top Markets

- Market title, slug, end date, outcomes, outcome prices, 24h volume,
  liquidity, best bid, best ask, spread, and token IDs:
  `GET https://gamma-api.polymarket.com/markets`.
- Efficient broad discovery by active events:
  `GET https://gamma-api.polymarket.com/events?active=true&closed=false`.
- Live price flashes and bid/ask updates:
  CLOB market websocket at
  `wss://ws-subscriptions-clob.polymarket.com/ws/market`.
- Optional per-market open interest:
  `GET https://data-api.polymarket.com/oi?market=<condition_id>`.

### Watchlist

- Watchlist membership: local app storage.
- Market data for pinned items:
  `GET https://gamma-api.polymarket.com/markets?id=<market_id>`.
- Live price changes for pinned outcome tokens:
  CLOB market websocket at
  `wss://ws-subscriptions-clob.polymarket.com/ws/market`.

### Selected Market

- Full market metadata and summary values:
  `GET https://gamma-api.polymarket.com/markets/<id>` or
  `GET https://gamma-api.polymarket.com/markets/slug/<slug>`.
- Best bid/ask, spread, and full depth for a selected outcome token:
  `GET https://clob.polymarket.com/book?token_id=<token_id>`.
- Open interest:
  `GET https://data-api.polymarket.com/oi?market=<condition_id>`.

### Activity

- Price jumps, best bid/ask changes, spread changes, and latest fills:
  CLOB market websocket at
  `wss://ws-subscriptions-clob.polymarket.com/ws/market`.
- Recent public trade history:
  `GET https://data-api.polymarket.com/trades?market=<condition_id>`.
- Event comments:
  `GET https://gamma-api.polymarket.com/comments?parent_entity_type=Event&parent_entity_id=<event_id>`.
- Watchlist updates and connection warnings: local app events.

### Order Book / Depth

- Depth mode:
  `GET https://clob.polymarket.com/book?token_id=<token_id>`.
- Trades mode:
  `GET https://data-api.polymarket.com/trades?market=<condition_id>`.
- Chart mode:
  `GET https://clob.polymarket.com/prices-history?market=<token_id>&interval=<interval>&fidelity=<minutes>`.
- Live book maintenance:
  CLOB market websocket at
  `wss://ws-subscriptions-clob.polymarket.com/ws/market`.

## Permanent Bottom Popup

The bottom popup should always exist, even when empty. It acts as a command
palette, mode indicator, and contextual help strip.

States:

```text
╭─ Command Popup ──────────────────────────────────────────────────────────────────╮
│ f find market   1-6 focus pane   ↑↓ move   Enter open   w watch   ? help         │
│ Filter: politics volume>100k sort:move                               NORMAL      │
╰──────────────────────────────────────────────────────────────────────────────────╯
```

Find mode:

```text
╭─ Find Market ────────────────────────────────────────────────────────────────────╮
│ query: btc etf inflows_                                                          │
│ Enter apply   Esc cancel   Tab autocomplete                                      │
╰──────────────────────────────────────────────────────────────────────────────────╯
```

Alert mode:

```text
╭─ ALERT ──────────────────────────────────────────────────────────────────────────╮
│ BTC 100k moved +6¢ in 2m. Press Enter to inspect, a to add alert, Esc dismiss.    │
╰──────────────────────────────────────────────────────────────────────────────────╯
```

The popup can be two or three lines tall depending on terminal height. On very
small terminals, preserve the popup and collapse the lower panes first.

## Keyboard Model

- `↑ / ↓`: move selection
- `PgUp / PgDn`: page through markets
- `← / →`: switch selected tab inside focused pane
- `1`-`6`: focus pane by number
- `f`: find market
- `F`: edit filters
- `s`: change sort
- `c`: configure columns
- `w`: toggle watchlist
- `Enter`: open selected market details
- `d`: cycle bottom depth pane mode
- `r`: force refresh
- `?`: help
- `q`: quit

## Sorting And Filtering Ideas

Default sort should use fields that are available from Gamma/CLOB/Data API. A
useful score can combine:

- 24h volume
- Recent price movement
- Liquidity
- Spread tightness
- Watchlist boost
- Ending-soon boost

Filter grammar examples:

```text
category:crypto volume>100k spread<4c
ending<7d liquidity>25k sort:move
watchlist sort:move
```

## Responsive Behavior

- Wide terminal: all panes visible.
- Medium terminal: combine `Watchlist`, `Selected Market`, and `Activity` into
  tabs.
- Narrow terminal: show `Top Markets` plus permanent bottom popup.
- Very short terminal: hide `Market Heat` first, then `Order Book`.
- Bottom popup is never hidden.

## Implementation Notes

- Treat panes as independently focusable widgets.
- Keep layout deterministic so redraws do not jitter.
- Reserve bottom popup height before calculating the main layout.
- Consider a theme object with semantic colors: `positive`, `negative`,
  `accent`, `muted`, `border`, `selected`, `warning`.
- Support both truecolor and 256-color terminals.
- Offer `--ascii` mode for terminals that do not render box drawing well.
