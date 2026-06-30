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
- Local app state: bookmark membership, filters, selected row, latency, and
  connection status.

## Visual Direction

- Background: black `#000000`, matching the flat terminal canvas used by
  `fplot`.
- Pane borders: dark gray / muted slate. Borders should frame content without
  competing with table data.
- Primary accent: cyan for live status glyphs, focused borders, and selection
  gutter.
- Positive / yes: green `#22C55E`
- Negative / no: red `#EF4444`
- Warning / large move: yellow / amber `#F59E0B`
- Muted text: dark gray for helper and status copy
- Table headers: bold yellow, like `fplot`
- Focused pane border: cyan, optionally bold
- Selected row: dark gray background with bold white text and a cyan `▶`
  highlight marker
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
│ ✓  net online   ws live   latency 42ms   refresh 500ms   mode observe            │
├─ [1] - Top Markets: all ─────────────────────────────────────────────────────────┤
│▶ #  ★  Market                                      Yes   No   24h    Move Spread │
│  1  ★  Will BTC hit 100k in 2026?                  63¢  38¢  842k    +4¢    2¢   │
│  2     Fed cuts rates by Sep?                      41¢  60¢  611k    -2¢    3¢   │
│  3     Lakers win tonight?                         55¢  46¢  570k    +1¢    2¢   │
│  4     ETH ETF inflows above $1B?                  72¢  29¢  510k    +6¢    4¢   │
│  5  ★  Trump wins popular vote?                    49¢  52¢  421k    -1¢    2¢   │
│  6     CPI below forecast?                         36¢  65¢  390k    -5¢    5¢   │
│  7     SpaceX launch this week?                    83¢  18¢  311k    +8¢    3¢   │
│  8     Oil closes above $90?                       22¢  79¢  280k    -3¢    4¢   │
├─ [2] - Selected Market: summary ─────────┬─ [3] - Chart + Activity ──────────────┤
│ Will BTC hit 100k in 2026?               │  70¢ ┤                     ╭╮         │
│                                          │  65¢ ┤              ╭──────╯╰─╮       │
│ yes  63¢  bid 62 / ask 64   spread 2¢    │  60¢ ┤      ╭───────╯         ╰╮      │
│ no   38¢  bid 37 / ask 39                │  55¢ ┤ ╭────╯                  ╰─     │
│                                          │      └────────────────────────────    │
│ volume 24h    $842.1k                    │ 17:42 price +4¢                       │
│ liquidity     $184.3k                    │ 17:41 best bid 62¢                    │
│ open interest $2.4M                      │ 17:40 trade 219 @45¢                  │
│ end date      2026-12-31                 │ 17:39 spread 2¢                       │
├─ Command Popup ──────────────────────────────────────────────────────────────────┤
│  f find   1-3 focus   ↑↓ move   b bookmarks/all   w bookmark   Tab chart   ? help│
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

### Top Markets

The main `top`-like feature and the dominant pane on the page. It should occupy
the full frame width below the header so scanning, sorting, and keyboard
navigation all happen in one wide table.

Suggested columns:

- `#`: rank after current sorting/filtering
- `★`: bookmark marker
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

Top Markets owns bookmarked-market browsing instead of using a separate
bookmark pane:

- `w`: add/remove the selected market bookmark
- `b`: switch the table between `all` and `bookmarked` modes
- The pane title should show the active mode, for example
  `[1] - Top Markets: all` or `[1] - Top Markets: bookmarked`
- Bookmark state is local app state and should survive restarts

### Selected Market

Details for the currently highlighted row. This pane sits under Top Markets on
the left half of the frame.

It combines the old selected-market summary and order-book/depth concerns into
one focused pane with tabs:

- `summary`: full market title, yes/no prices, bid/ask spread, 24h volume,
  liquidity, end date, and open interest when loaded from Data API
- `depth`: bid and ask levels for the selected outcome token
- `trades`: recent public trades for the selected market
- `chart`: compact selected-market sparkline when the right pane is expanded to
  another mode

Use `← / →` to switch tabs while this pane is focused.

If no row is selected, show a short onboarding hint.

### Chart + Activity

The right half under Top Markets should provide visual and temporal context for
the selected market. Its default split view shows a small chart above a terse
activity stream.

Chart should show recent selected-market price history as a terminal-friendly
line or sparkline. Activity should include:

- Price jumps
- Best bid/ask changes
- Recent public trades
- Spread changes
- Bookmark updates
- Connection warnings
- New event comments

Use color heavily in activity, but keep messages terse.

Press `Tab` while this pane is focused to cycle:

- `split`: chart above activity
- `chart`: full pane chart
- `activity`: full pane event stream

## Data Sources By Pane

### Header

- Current time: local system clock.
- Network status, API latency, refresh interval, and mode: local app state.
- WebSocket state: local state for the CLOB market websocket connection to
  `wss://ws-subscriptions-clob.polymarket.com/ws/market`.

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
- Bookmark membership and the active `all` / `bookmarked` table mode:
  local app storage.

### Selected Market

- Full market metadata and summary values:
  `GET https://gamma-api.polymarket.com/markets/<id>` or
  `GET https://gamma-api.polymarket.com/markets/slug/<slug>`.
- Best bid/ask, spread, and full depth for a selected outcome token:
  `GET https://clob.polymarket.com/book?token_id=<token_id>`.
- Recent public trade history:
  `GET https://data-api.polymarket.com/trades?market=<condition_id>`.
- Compact chart mode:
  `GET https://clob.polymarket.com/prices-history?market=<token_id>&interval=<interval>&fidelity=<minutes>`.
- Open interest:
  `GET https://data-api.polymarket.com/oi?market=<condition_id>`.

### Chart + Activity

- Chart mode:
  `GET https://clob.polymarket.com/prices-history?market=<token_id>&interval=<interval>&fidelity=<minutes>`.
- Price jumps, best bid/ask changes, spread changes, latest fills, and live
  chart maintenance:
  CLOB market websocket at
  `wss://ws-subscriptions-clob.polymarket.com/ws/market`.
- Recent public trade history:
  `GET https://data-api.polymarket.com/trades?market=<condition_id>`.
- Event comments:
  `GET https://gamma-api.polymarket.com/comments?parent_entity_type=Event&parent_entity_id=<event_id>`.
- Bookmark updates and connection warnings: local app events.

## Permanent Bottom Popup

The bottom popup should always exist, even when empty. It acts as a command
palette, mode indicator, and contextual help strip.

States:

```text
╭─ Command Popup ──────────────────────────────────────────────────────────────────╮
│ f find market   1-3 focus pane   ↑↓ move   b bookmarks/all   w bookmark   ? help │
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
- `1`-`3`: focus pane by number
- `f`: find market
- `F`: edit filters
- `s`: change sort
- `c`: configure columns
- `w`: toggle selected-market bookmark
- `b`: switch Top Markets between all and bookmarked markets
- `Tab`: cycle the Chart + Activity pane between split, full chart, and full
  activity
- `Enter`: open selected market details
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
- Bookmark boost
- Ending-soon boost

Filter grammar examples:

```text
category:crypto volume>100k spread<4c
ending<7d liquidity>25k sort:move
bookmarked sort:move
```

## Responsive Behavior

- Wide terminal: all panes visible.
- Very wide terminal: optionally allow a chart column beside Top Markets, but
  keep the default layout with Top Markets full width.
- Medium terminal: keep Top Markets full width and combine the lower
  `Selected Market` and `Chart + Activity` panes into tabs.
- Narrow terminal: show `Top Markets` plus permanent bottom popup.
- Very short terminal: hide the lower panes first.
- Bottom popup is never hidden.

## Implementation Notes

- Treat panes as independently focusable widgets.
- Keep layout deterministic so redraws do not jitter.
- Reserve bottom popup height before calculating the main layout.
- Consider a theme object with semantic colors: `positive`, `negative`,
  `accent`, `muted`, `border`, `selected`, `warning`.
- Support both truecolor and 256-color terminals.
- Offer `--ascii` mode for terminals that do not render box drawing well.
