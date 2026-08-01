package config

import (
	"fmt"
	"os"
	"strconv"
	"sync"
	"time"
)

// Config mirrors the Rust poly-tui config: a single TICK_RATE knob taken
// from the environment (managed via mise's [env] section).
type Config struct {
	TickRate float64
}

func (c *Config) TickInterval() time.Duration {
	return time.Duration(float64(time.Second) / c.TickRate)
}

var (
	once sync.Once
	cfg  *Config
)

func Get() *Config {
	once.Do(func() {
		tickRate := 30.0
		if raw, ok := os.LookupEnv("TICK_RATE"); ok {
			parsed, err := strconv.ParseFloat(raw, 64)
			if err != nil {
				panic(fmt.Sprintf("TICK_RATE must be a valid number, got %q", raw))
			}
			tickRate = parsed
		}
		cfg = &Config{TickRate: tickRate}
	})
	return cfg
}
